#!/usr/bin/env python3
"""Evaluate matched DADS/magnitude displacement-scale controls on D0.

The script reuses the cached E0/Eft/Delta streams and the frozen adapted CTC
head.  For each retention budget it compares the same two, pre-specified
scales for both rankings:

* unit scale: ``alpha = 1``;
* inverse-retention scale: ``alpha = d / K``.

No scale is selected from test (or from dev); the second rule is the same
fixed rule already used by the Random+Rescale control.  The resulting grid is
intended to isolate coordinate ranking from the size of the retained shift.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from usde.ctc import make_processor  # noqa: E402
from usde.metrics import word_error_rate  # noqa: E402
from usde.model import load_ctc_model  # noqa: E402
from usde.shift import ShiftFeatureDataset, collate_shift  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


PROTOCOL = "d0_frozen_head_matched_ranking_scale_controls_v1"
RETENTIONS = (25, 50, 75)
METHODS = ("DADS", "Magnitude")
SCALES = ("unit", "inverse_retention")


@dataclass(frozen=True)
class Condition:
    key: str
    method: str
    keep: torch.Tensor | None
    scale: float = 1.0


def load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        for key in ("ranking", "tensor", "utility", "features"):
            if isinstance(value.get(key), torch.Tensor):
                value = value[key]
                break
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path}: expected a tensor")
    return value


def validate_ranking(value: torch.Tensor, dimension: int, label: str) -> torch.Tensor:
    if value.ndim != 1 or value.numel() != dimension:
        raise ValueError(f"{label}: expected a ranking of {dimension} coordinates")
    value = value.to(dtype=torch.long, device="cpu")
    if (
        torch.unique(value).numel() != dimension
        or int(value.min()) < 0
        or int(value.max()) >= dimension
    ):
        raise ValueError(f"{label}: expected a complete coordinate permutation")
    return value


def make_loader(
    manifest: Path,
    cache_root: Path,
    tokenizer: Any,
    split: str,
    hidden_dim: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = ShiftFeatureDataset(
        manifest,
        cache_root,
        tokenizer,
        feature_split=split,
        expected_dim=hidden_dim,
        load_eft=True,
    )
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate_shift,
        "pin_memory": False,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def compose(
    e0: torch.Tensor,
    delta: torch.Tensor,
    keep: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    if keep is None:
        return e0 + delta
    selected = torch.zeros_like(delta)
    if keep.numel():
        indices = keep.to(device=e0.device, dtype=torch.long)
        selected[..., indices] = delta[..., indices] * float(scale)
    return e0 + selected


def decode(tokenizer: Any, logits: torch.Tensor, length: int) -> str:
    ids = logits.argmax(dim=-1) if logits.ndim == 2 else logits
    return str(tokenizer.decode(ids[:length].tolist(), group_tokens=True)).lower().strip()


def build_conditions(
    dimension: int,
    rankings: dict[str, torch.Tensor],
    retentions: tuple[int, ...],
) -> list[Condition]:
    conditions: list[Condition] = []
    for retention in retentions:
        count = max(1, round(dimension * retention / 100))
        scales = {
            "unit": 1.0,
            "inverse_retention": dimension / count,
        }
        for method, ranking_name in (("DADS", "utility"), ("Magnitude", "magnitude")):
            keep = rankings[ranking_name][:count]
            for scale_name in SCALES:
                conditions.append(Condition(
                    key=f"{method.lower()}_{retention}_{scale_name}",
                    method=method,
                    keep=keep,
                    scale=scales[scale_name],
                ))
    return conditions


def evaluate_split(
    head: torch.nn.Module,
    tokenizer: Any,
    manifest: Path,
    cache_root: Path,
    split: str,
    hidden_dim: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    conditions: list[Condition],
) -> dict[str, Any]:
    loader = make_loader(
        manifest, cache_root, tokenizer, split, hidden_dim, batch_size, num_workers
    )
    all_conditions = [
        Condition("full", "Full", None, 1.0),
        Condition("no_shift", "No shift", torch.empty(0, dtype=torch.long), 1.0),
        *conditions,
    ]
    hypotheses = {condition.key: [] for condition in all_conditions}
    references: list[str] = []
    max_feature_error = max_logit_error = 0.0
    with torch.inference_mode():
        for batch in loader:
            e0 = batch["e0"].to(device, non_blocking=True)
            eft = batch["eft"].to(device, non_blocking=True)
            delta = batch["delta"].to(device, non_blocking=True)
            references.extend(batch["transcripts"])
            max_feature_error = max(max_feature_error, float((e0 + delta - eft).abs().max().cpu()))
            eft_logits = head(eft)
            full_logits = head(e0 + delta)
            max_logit_error = max(max_logit_error, float((eft_logits - full_logits).abs().max().cpu()))
            for condition in all_conditions:
                logits = head(compose(e0, delta, condition.keep, condition.scale))
                for row, length in zip(logits.cpu(), batch["feature_lengths"], strict=True):
                    hypotheses[condition.key].append(decode(tokenizer, row, int(length)))
    measured = {
        key: {"wer": float(word_error_rate(references, values)), "examples": len(references)}
        for key, values in hypotheses.items()
    }
    return {
        "examples": len(references),
        "identity": {
            "max_abs_error": max_feature_error,
            "max_logit_abs_error": max_logit_error,
            "pass": max_feature_error <= 1e-5 and max_logit_error <= 1e-4,
        },
        "measured": measured,
    }


def summarize(measured: dict[str, dict[str, Any]], retentions: tuple[int, ...]) -> dict[str, Any]:
    return {
        method: {
            str(retention): {
                scale: measured[f"{method.lower()}_{retention}_{scale}"]
                for scale in SCALES
            }
            for retention in retentions
        }
        for method in METHODS
    }


def format_wer(row: dict[str, Any]) -> str:
    return f"{float(row['wer']) * 100:.3f}%"


def write_summary(result: dict[str, Any], path: Path) -> None:
    test = result["splits"]["test"]
    lines = [
        "# D0 matched ranking-scale controls",
        "",
        "All conditions reuse cached E0/Eft/Delta features and the frozen adapted CTC head.",
        "No encoder retraining, head retraining, or scale selection is performed.",
        "The inverse-retention scale is the pre-specified alpha=d/K rule used by Random+Rescale.",
        "",
        "| Retained | Scale | DADS / Utility | Magnitude |",
        "|---:|---|---:|---:|",
    ]
    for retention in result["retentions_percent"]:
        if retention in (0, 100):
            continue
        for scale in SCALES:
            lines.append(
                f"| {retention}% | {scale} | "
                f"{format_wer(test['summary']['DADS'][str(retention)][scale])} | "
                f"{format_wer(test['summary']['Magnitude'][str(retention)][scale])} |"
            )
    lines += [
        "",
        f"Full-shift test WER: {format_wer(test['measured']['full'])}.",
        f"No-shift test WER: {format_wer(test['measured']['no_shift'])}.",
        "",
        "Identity gate: " + ("PASS" if test["identity"]["pass"] else "FAIL")
        + f" (max feature error {test['identity']['max_abs_error']:.3g}; "
        f"max logit error {test['identity']['max_logit_abs_error']:.3g}).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.num_workers < 0 or args.log_every < 1:
        raise ValueError("batch-size must be positive; num-workers non-negative; log-every positive")
    retentions = tuple(args.retentions)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite to reuse it")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("dev", "test"):
        if not (args.manifest_root / f"{name}.jsonl").is_file():
            raise FileNotFoundError(args.manifest_root / f"{name}.jsonl")
    ranking_dir = args.ranking_dir
    rankings = {
        "utility": validate_ranking(
            load_tensor(ranking_dir / "utility_ranking.pt"), args.hidden_dim, "utility ranking"
        ),
        "magnitude": validate_ranking(
            load_tensor(ranking_dir / "magnitude_ranking.pt"), args.hidden_dim, "magnitude ranking"
        ),
    }

    processor = make_processor(str(args.checkpoint), None)
    model = load_ctc_model(args.checkpoint).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_dim = int(getattr(model.config, "hidden_size", 0))
    if hidden_dim != args.hidden_dim or not hasattr(model, "lm_head"):
        raise ValueError(f"checkpoint hidden size {hidden_dim} does not match --hidden-dim {args.hidden_dim}")

    conditions = build_conditions(hidden_dim, rankings, retentions)
    splits: dict[str, Any] = {}
    for split in ("dev", "test"):
        measured = evaluate_split(
            model.lm_head,
            processor.tokenizer,
            args.manifest_root / f"{split}.jsonl",
            args.cache_root,
            split,
            hidden_dim,
            args.batch_size,
            args.num_workers,
            device,
            conditions,
        )
        splits[split] = {
            "examples": measured["examples"],
            "identity": measured["identity"],
            "measured": measured["measured"],
            "summary": summarize(measured["measured"], retentions),
        }
        print({
            "split": split,
            "examples": measured["examples"],
            "full_wer": measured["measured"]["full"]["wer"],
            "identity_pass": measured["identity"]["pass"],
        }, flush=True)

    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "checkpoint": str(args.checkpoint),
        "manifest_root": str(args.manifest_root),
        "cache_root": str(args.cache_root),
        "ranking_dir": str(ranking_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "hidden_dim": hidden_dim,
        "retentions_percent": list(retentions),
        "scales": {
            "unit": {"alpha": 1.0, "selection": "none"},
            "inverse_retention": {"alpha": "d/K", "selection": "none"},
        },
        "scale_selection": "none; both pre-specified scales are reported on dev and test",
        "same_scale_grid_for_methods": True,
        "splits": splits,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(result, args.output_dir / "summary.md")
    print(json.dumps({"metrics": str(metrics_path), "summary": str(args.output_dir / "summary.md")}, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--ranking-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--retentions", type=int, nargs="+", default=list(RETENTIONS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(set(args.retentions)) != len(args.retentions) or any(
        value < 1 or value > 99 for value in args.retentions
    ):
        parser.error("retentions must contain unique percentages between 1 and 99")
    run(args)


if __name__ == "__main__":
    main()
