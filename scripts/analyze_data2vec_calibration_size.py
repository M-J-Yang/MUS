#!/usr/bin/env python3
"""Measure Taylor-ranking stability as the attribution calibration set grows."""

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

from usde.ctc import make_processor, read_records  # noqa: E402
from usde.metrics import word_error_rate  # noqa: E402
from usde.model import load_ctc_model  # noqa: E402
from usde.shift import ShiftFeatureDataset, collate_shift, ctc_taylor_batch_sums  # noqa: E402


PROTOCOL = "official_fold0_data2vec_calibration_size_v1"
DEFAULT_SIZES = (128, 256, 512, 1024, 1640)
DEFAULT_SEEDS = (1337, 2027, 31415)


def load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = next(
            (value[key] for key in ("ranking", "tensor", "utility")
             if isinstance(value.get(key), torch.Tensor)),
            None,
        )
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"{path}: expected a one-dimensional tensor")
    return value.to(dtype=torch.long, device="cpu")


def validate_ranking(value: torch.Tensor, dimension: int, label: str) -> torch.Tensor:
    if (
        value.numel() != dimension
        or torch.unique(value).numel() != dimension
        or int(value.min()) < 0
        or int(value.max()) >= dimension
    ):
        raise ValueError(f"{label}: expected a complete permutation of {dimension}")
    return value


def make_loader(
    manifest: Path,
    cache_root: Path,
    tokenizer: Any,
    split: str,
    hidden_dim: int,
    batch_size: int,
    num_workers: int,
    max_examples: int | None = None,
    sample_seed: int = 1337,
    load_eft: bool = False,
) -> DataLoader:
    dataset = ShiftFeatureDataset(
        manifest,
        cache_root,
        tokenizer,
        feature_split=split,
        expected_dim=hidden_dim,
        max_examples=max_examples,
        sample_seed=sample_seed,
        load_eft=load_eft,
    )
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate_shift,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def compute_ranking(
    head: torch.nn.Module,
    tokenizer: Any,
    manifest: Path,
    cache_root: Path,
    split: str,
    hidden_dim: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_examples: int,
    sample_seed: int,
) -> tuple[torch.Tensor, int]:
    loader = make_loader(
        manifest, cache_root, tokenizer, split, hidden_dim, batch_size,
        num_workers, max_examples=max_examples, sample_seed=sample_seed,
    )
    score_sum = torch.zeros(hidden_dim, dtype=torch.float64)
    frame_count = 0
    for batch in loader:
        e0 = batch["e0"].to(device, non_blocking=True)
        delta = batch["delta"].to(device, non_blocking=True)
        scores, frames, _ = ctc_taylor_batch_sums(
            head,
            e0,
            delta,
            batch["targets"].to(device, non_blocking=True),
            batch["feature_lengths"].to(device),
            batch["target_lengths"].to(device),
            int(tokenizer.pad_token_id),
        )
        score_sum += scores.double().cpu()
        frame_count += frames
    if frame_count < 1:
        raise RuntimeError("calibration subset produced no valid frames")
    return torch.argsort(score_sum / frame_count, descending=True, stable=True), frame_count


def compose(e0: torch.Tensor, delta: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    selected = torch.zeros_like(delta)
    if keep.numel():
        selected[..., keep.to(device=e0.device, dtype=torch.long)] = delta[..., keep.to(device=e0.device, dtype=torch.long)]
    return e0 + selected


def decode(tokenizer: Any, logits: torch.Tensor, length: int) -> str:
    return str(tokenizer.decode(logits[:length].argmax(dim=-1).tolist(), group_tokens=True)).lower().strip()


def evaluate_rankings(
    head: torch.nn.Module,
    tokenizer: Any,
    manifest: Path,
    cache_root: Path,
    split: str,
    hidden_dim: int,
    rankings: dict[str, torch.Tensor],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, float]:
    loader = make_loader(
        manifest, cache_root, tokenizer, split, hidden_dim, batch_size, num_workers,
        load_eft=True,
    )
    names = ["full", "no_shift"] + [
        key for key in rankings if key not in {"full", "no_shift"}
    ]
    hypotheses = {name: [] for name in names}
    references: list[str] = []
    max_feature_error = max_logit_error = 0.0
    with torch.inference_mode():
        for batch in loader:
            e0 = batch["e0"].to(device, non_blocking=True)
            eft = batch["eft"].to(device, non_blocking=True)
            delta = batch["delta"].to(device, non_blocking=True)
            max_feature_error = max(max_feature_error, float((e0 + delta - eft).abs().max().cpu()))
            full_logits = head(e0 + delta)
            eft_logits = head(eft)
            max_logit_error = max(max_logit_error, float((full_logits - eft_logits).abs().max().cpu()))
            lengths = batch["feature_lengths"].tolist()
            logits_by_name: dict[str, torch.Tensor] = {
                "full": full_logits,
                "no_shift": head(e0),
            }
            for name, keep in rankings.items():
                if name in {"full", "no_shift"}:
                    continue
                logits_by_name[name] = head(compose(e0, delta, keep))
            for name in names:
                for row, length in zip(logits_by_name[name].cpu(), lengths, strict=True):
                    hypotheses[name].append(decode(tokenizer, row, int(length)))
            references.extend(batch["transcripts"])
    if max_feature_error > 1e-5 or max_logit_error > 1e-4:
        raise RuntimeError(
            f"identity gate failed: feature={max_feature_error:.6g}, logit={max_logit_error:.6g}"
        )
    return {
        "identity_max_feature_error": max_feature_error,
        "identity_max_logit_error": max_logit_error,
        **{name + "_wer": float(word_error_rate(references, hypotheses[name])) for name in names},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--full-utility-ranking", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite")
    if any(size < 1 for size in args.calibration_sizes) or any(seed < 0 for seed in args.seeds):
        raise ValueError("calibration sizes must be positive and seeds non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    processor = make_processor(str(args.checkpoint), None)
    model = load_ctc_model(args.checkpoint).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_dim = int(getattr(model.config, "hidden_size", 0))
    if hidden_dim < 1 or not hasattr(model, "lm_head"):
        raise ValueError("checkpoint must provide hidden_size and lm_head")
    full_ranking = validate_ranking(
        load_tensor(args.full_utility_ranking), hidden_dim, "full utility ranking"
    )
    utility_manifest = args.manifest_root / "train_utility.jsonl"
    utility_total = len(read_records(utility_manifest))
    results: list[dict[str, Any]] = []
    for size in args.calibration_sizes:
        for seed in args.seeds:
            actual_size = min(size, utility_total)
            ranking, frames = compute_ranking(
                model.lm_head, processor.tokenizer,
                utility_manifest, args.cache_root,
                "train_utility", hidden_dim, args.batch_size, args.num_workers,
                device, actual_size, seed,
            )
            count50 = max(1, round(hidden_dim * 0.50))
            count75 = max(1, round(hidden_dim * 0.75))
            count25 = max(1, round(hidden_dim * 0.25))
            conditions = {
                "utility75": ranking[:count75],
                "utility50": ranking[:count50],
                "drop_worst25": ranking[: hidden_dim - count25],
                "drop_best25": ranking[count25:],
            }
            measured = evaluate_rankings(
                model.lm_head, processor.tokenizer,
                args.manifest_root / "test.jsonl", args.cache_root, "test", hidden_dim,
                conditions, args.batch_size, args.num_workers, device,
            )
            top10 = max(1, round(hidden_dim * 0.10))
            top25 = max(1, round(hidden_dim * 0.25))
            top10_overlap = len(set(ranking[:top10].tolist()) & set(full_ranking[:top10].tolist())) / top10
            top25_overlap = len(set(ranking[:top25].tolist()) & set(full_ranking[:top25].tolist())) / top25
            row = {
                "calibration_size_requested": size,
                "calibration_size_used": actual_size,
                "seed": seed,
                "valid_frames": frames,
                "top10_overlap_with_full": top10_overlap,
                "top25_overlap_with_full": top25_overlap,
                "ranking_path": str(args.output_dir / f"utility_size{size}_seed{seed}_ranking.pt"),
                **measured,
            }
            torch.save(ranking.cpu(), row["ranking_path"])
            results.append(row)
            print({key: row[key] for key in ("calibration_size_requested", "seed", "top10_overlap_with_full", "utility50_wer")}, flush=True)
    payload = {
        "protocol": PROTOCOL,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest_root / "train_utility.jsonl"),
        "test_manifest": str(args.manifest_root / "test.jsonl"),
        "cache_root": str(args.cache_root),
        "full_utility_ranking": str(args.full_utility_ranking),
        "calibration_sizes": args.calibration_sizes,
        "seeds": args.seeds,
        "hidden_dim": hidden_dim,
        "head_training_after_cache": "none",
        "identity_gate": "E0 + Delta == Eft; logits(E0 + Delta) == logits(Eft)",
        "results": results,
    }
    (args.output_dir / "calibration_size_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    full_test = next(row["full_wer"] for row in results)
    lines = [
        "# Data2Vec calibration-size analysis",
        "",
        "Taylor Utility is recomputed on deterministic random subsets of `train_utility`; the test set and frozen CTC head are unchanged.",
        "",
        "| Calibration utterances | Seed | Top-10 overlap | Top-25 overlap | Utility-75 WER | Utility-50 WER | DropWorst-25 WER | DropBest-25 WER |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['calibration_size_used']} | {row['seed']} | {row['top10_overlap_with_full']:.3f} | "
            f"{row['top25_overlap_with_full']:.3f} | {100 * row['utility75_wer']:.3f}% | "
            f"{100 * row['utility50_wer']:.3f}% | {100 * row['drop_worst25_wer']:.3f}% | "
            f"{100 * row['drop_best25_wer']:.3f}% |"
        )
    lines += ["", f"Reference full-shift test WER: {100 * full_test:.3f}%.", ""]
    (args.output_dir / "calibration_size_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
