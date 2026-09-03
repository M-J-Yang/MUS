#!/usr/bin/env python3
"""Evaluate the seven frozen-core conditions for an official-split replica.

The core replication intentionally excludes Fold0's exploratory Random,
Gradient, and bootstrap package. It evaluates only Full, NoShift, Utility-75,
Utility-50, Magnitude-50, DropWorst-25, and DropBest-25 with the original
fine-tuned CTC head and no retraining or healing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from usde.ctc import make_processor  # noqa: E402
from usde.metrics import word_error_rate  # noqa: E402
from usde.model import load_ctc_model  # noqa: E402
from usde.shift import ShiftFeatureDataset, collate_shift  # noqa: E402


PROTOCOL = "official_split_local_replica_frozen_core_seven_v1"
CONDITION_ORDER = (
    "full",
    "no_shift",
    "utility75",
    "utility50",
    "magnitude50",
    "drop_worst25",
    "drop_best25",
)


def load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = next(
            (value[key] for key in ("ranking", "tensor", "utility", "features")
             if isinstance(value.get(key), torch.Tensor)),
            None,
        )
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path}: expected a tensor")
    return value


def validate_ranking(value: torch.Tensor, dimension: int, label: str) -> torch.Tensor:
    value = value.to(dtype=torch.long, device="cpu")
    if (
        value.ndim != 1
        or value.numel() != dimension
        or torch.unique(value).numel() != dimension
        or int(value.min()) < 0
        or int(value.max()) >= dimension
    ):
        raise ValueError(f"{label}: expected a complete permutation of {dimension} coordinates")
    return value


def compose(e0: torch.Tensor, delta: torch.Tensor, keep: torch.Tensor | None) -> torch.Tensor:
    if keep is None:
        return e0 + delta
    selected = torch.zeros_like(delta)
    if keep.numel():
        indices = keep.to(device=e0.device, dtype=torch.long)
        selected[..., indices] = delta[..., indices]
    return e0 + selected


def decode(tokenizer: Any, logits: torch.Tensor, length: int) -> str:
    return str(tokenizer.decode(logits[:length].argmax(dim=-1).tolist(), group_tokens=True)).lower().strip()


def scores(references: list[str], hypotheses: list[str]) -> dict[str, Any]:
    return {"wer": float(word_error_rate(references, hypotheses)), "examples": len(references)}


def make_conditions(
    utility: torch.Tensor, magnitude: torch.Tensor
) -> dict[str, torch.Tensor | None]:
    dimension = utility.numel()
    count75 = max(1, round(dimension * 0.75))
    count50 = max(1, round(dimension * 0.50))
    return {
        "full": None,
        "no_shift": torch.empty(0, dtype=torch.long),
        "utility75": utility[:count75],
        "utility50": utility[:count50],
        "magnitude50": magnitude[:count50],
        "drop_worst25": utility[: dimension - max(1, round(dimension * 0.25))],
        "drop_best25": utility[max(1, round(dimension * 0.25)) :],
    }


def evaluate_split(
    model: torch.nn.Module,
    tokenizer: Any,
    manifest: Path,
    cache_root: Path,
    feature_split: str,
    conditions: dict[str, torch.Tensor | None],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    dataset = ShiftFeatureDataset(
        manifest,
        cache_root,
        tokenizer,
        feature_split=feature_split,
        expected_dim=int(getattr(model.config, "hidden_size", 0)),
        load_eft=True,
    )
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate_shift,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    predictions = {name: [] for name in conditions}
    references: list[str] = []
    max_identity_error = 0.0
    max_logit_error = 0.0
    with torch.inference_mode():
        for batch in DataLoader(dataset, **kwargs):
            e0 = batch["e0"].to(device, non_blocking=True)
            eft = batch["eft"].to(device, non_blocking=True)
            delta = batch["delta"].to(device, non_blocking=True)
            max_identity_error = max(max_identity_error, float((e0 + delta - eft).abs().max().cpu()))
            full_logits = model.lm_head(e0 + delta)
            eft_logits = model.lm_head(eft)
            max_logit_error = max(max_logit_error, float((full_logits - eft_logits).abs().max().cpu()))
            lengths = batch["feature_lengths"].tolist()
            for name, keep in conditions.items():
                logits = full_logits if name == "full" else model.lm_head(compose(e0, delta, keep))
                for row, length in zip(logits.cpu(), lengths, strict=True):
                    predictions[name].append(decode(tokenizer, row, int(length)))
            references.extend(batch["transcripts"])
    if max_identity_error > 1e-5 or max_logit_error > 1e-4:
        raise RuntimeError(
            f"identity gate failed: feature={max_identity_error:.6g}, logit={max_logit_error:.6g}"
        )
    return {
        "examples": len(references),
        "identity": {
            "max_abs_error": max_identity_error,
            "max_logit_abs_error": max_logit_error,
            "pass": True,
        },
        "conditions": {name: scores(references, predictions[name]) for name in conditions},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    processor = make_processor(str(args.checkpoint), None)
    model = load_ctc_model(args.checkpoint).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dimension = int(getattr(model.config, "hidden_size", 0))
    utility = validate_ranking(load_tensor(args.utility_ranking), dimension, "utility ranking")
    magnitude = validate_ranking(load_tensor(args.magnitude_ranking), dimension, "magnitude ranking")
    conditions = make_conditions(utility, magnitude)
    split_results: dict[str, Any] = {}
    for split in ("dev", "test"):
        split_results[split] = evaluate_split(
            model,
            processor.tokenizer,
            args.manifest_root / f"{split}.jsonl",
            args.cache_root,
            split,
            conditions,
            args.batch_size,
            args.num_workers,
            device,
        )
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "replica_label": "official-split local replica",
        "checkpoint": str(args.checkpoint),
        "manifest_root": str(args.manifest_root),
        "cache_root": str(args.cache_root),
        "utility_ranking": str(args.utility_ranking),
        "magnitude_ranking": str(args.magnitude_ranking),
        "device": str(device),
        "hidden_dim": dimension,
        "conditions": list(CONDITION_ORDER),
        "splits": split_results,
    }
    json_path = args.output_dir / "core_metrics.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    full = split_results["test"]["conditions"]["full"]["wer"]
    lines = [
        "# Official-split local replica core-seven evaluation",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        "",
        "| Condition | Test WER | Δ vs Full |",
        "|---|---:|---:|",
    ]
    for name in CONDITION_ORDER:
        value = split_results["test"]["conditions"][name]["wer"]
        lines.append(f"| {name} | {100.0 * value:.3f}% | {100.0 * (value - full):+.3f} pp |")
    lines += [
        "",
        f"Identity gate: PASS (test max feature error {split_results['test']['identity']['max_abs_error']:.3g}; "
        f"max logit error {split_results['test']['identity']['max_logit_abs_error']:.3g}).",
        "",
    ]
    (args.output_dir / "core_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"metrics": str(json_path), "summary": str(args.output_dir / "core_summary.md")}, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--utility-ranking", type=Path, required=True)
    parser.add_argument("--magnitude-ranking", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
