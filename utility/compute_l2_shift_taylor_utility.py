#!/usr/bin/env python3
"""Rank fine-tuning-induced representation coordinates with CTC Taylor utility.

The fine-tuned model supplies only its frozen original ``lm_head``. Utility is
computed on the held-out ``train_utility`` split from
``abs(Delta * dL_CTC/dDelta)`` and no CTC head is trained or re-fit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from usde.ctc import make_processor  # noqa: E402
from usde.model import load_ctc_model  # noqa: E402
from usde.shift import ShiftFeatureDataset, collate_shift, ctc_taylor_batch_sums  # noqa: E402


PROTOCOL = "single_model_finetuning_shift_taylor_utility_v1"


def _atomic_torch_save(value: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value.detach().cpu(), temporary)
    temporary.replace(path)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float | None:
    if values_a.shape != values_b.shape or values_a.ndim != 1:
        raise ValueError("Spearman inputs must be one-dimensional arrays of equal shape")
    ranks_a = _rankdata(values_a)
    ranks_b = _rankdata(values_b)
    centered_a = ranks_a - ranks_a.mean()
    centered_b = ranks_b - ranks_b.mean()
    denominator = float(np.sqrt(np.dot(centered_a, centered_a) * np.dot(centered_b, centered_b)))
    value = float(np.dot(centered_a, centered_b) / denominator) if denominator else float("nan")
    return value if math.isfinite(value) else None


def compute(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite to reuse it")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = make_processor(args.checkpoint, None)
    model = load_ctc_model(args.checkpoint).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not hasattr(model, "lm_head"):
        raise AttributeError("fine-tuned checkpoint has no lm_head")
    hidden_dim = int(getattr(model.config, "hidden_size", 0))
    blank = processor.tokenizer.pad_token_id
    if blank is None:
        raise ValueError("fine-tuned tokenizer has no CTC blank/pad token")

    dataset = ShiftFeatureDataset(
        args.manifest,
        args.cache_root,
        processor.tokenizer,
        feature_split=args.feature_split,
        expected_dim=hidden_dim,
        max_examples=args.max_utterances,
        sample_seed=args.seed,
        load_eft=False,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "collate_fn": collate_shift,
        "pin_memory": device.type == "cuda",
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    score_sum = torch.zeros(hidden_dim, dtype=torch.float64)
    magnitude_sum = torch.zeros(hidden_dim, dtype=torch.float64)
    frame_count = 0
    processed = 0
    total_loss = 0.0
    for batch_index, batch in enumerate(loader, start=1):
        e0 = batch["e0"].to(device, non_blocking=True)
        delta = batch["delta"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        scores, batch_frames, batch_loss = ctc_taylor_batch_sums(
            model.lm_head,
            e0,
            delta,
            targets,
            batch["feature_lengths"].to(device),
            batch["target_lengths"].to(device),
            int(blank),
        )
        score_sum += scores.double().cpu()
        valid = torch.arange(e0.shape[1], device=device).unsqueeze(0) < batch["feature_lengths"].to(device).unsqueeze(1)
        magnitude_sum += (delta.abs() * valid.unsqueeze(-1)).sum(dim=(0, 1)).double().cpu()
        frame_count += batch_frames
        batch_size = len(batch["utt_ids"])
        processed += batch_size
        total_loss += float(batch_loss.cpu()) * batch_size
        if batch_index == 1 or processed % args.log_every < batch_size or processed == len(dataset):
            print({"processed": processed, "total": len(dataset), "batches": batch_index}, flush=True)

    if frame_count == 0:
        raise RuntimeError("utility pass produced no valid frames")
    utility = score_sum / frame_count
    magnitude = magnitude_sum / frame_count
    ranking = torch.argsort(utility, descending=True, stable=True)
    score_path = args.output_dir / f"{args.output_stem}.pt"
    ranking_path = args.output_dir / f"{args.output_stem}_ranking.pt"
    stats_path = args.output_dir / f"{args.output_stem}_stats.json"
    _atomic_torch_save(utility, score_path)
    _atomic_torch_save(ranking, ranking_path)
    utility_np = utility.numpy()
    magnitude_np = magnitude.numpy()
    stats: dict[str, Any] = {
        "protocol": PROTOCOL,
        "utility": "mean_over_valid_frames(abs(Delta * dL_CTC/dDelta))",
        "head_training": "none; original fine-tuned lm_head frozen",
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "feature_root": str(args.cache_root),
        "feature_split": args.feature_split,
        "device": str(device),
        "seed": args.seed,
        "hidden_dim": hidden_dim,
        "blank_id": int(blank),
        "num_utterances": processed,
        "num_valid_frames": frame_count,
        "mean_ctc_loss": total_loss / max(processed, 1),
        "score_path": str(score_path),
        "ranking_path": str(ranking_path),
        "spearman_magnitude_utility": spearman(utility_np, magnitude_np),
        "top_10_utility": ranking[:10].tolist(),
        "utility": utility.tolist(),
        "magnitude": magnitude.tolist(),
    }
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: stats[key] for key in ("num_utterances", "num_valid_frames", "mean_ctc_loss", "score_path", "ranking_path")}, sort_keys=True), flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="held-out train_utility manifest")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--feature-split", default="train_utility")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="utility_shift_taylor")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.log_every < 1:
        raise ValueError("log-every must be positive")
    compute(args)


if __name__ == "__main__":
    main()
