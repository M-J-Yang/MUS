#!/usr/bin/env python3
"""Compute CTC-loss Taylor importance for cached Delta coordinates.

The FullDelta CTC teacher is frozen. For each utility example this pass
computes the gradient of the per-example CTC loss with respect to the cached
Delta and accumulates ``abs(Delta[t, i] * d L_CTC / d Delta[t, i])`` over all
valid frames. No forced alignment or frame-level competitor is used.
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from usde.stage4 import CachedFeatureDataset, collate, load_vocab  # noqa: E402
from utility.compute_utility import load_full_delta_teacher  # noqa: E402


PROTOCOL = "stage5_ctc_taylor_delta_utility_v1"


def ctc_taylor_batch_sums(
    model: torch.nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    feature_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    reference_dim: int,
    blank: int,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Return coordinate Taylor sums, valid-frame count, and mean loss.

    ``features`` is a padded ``[B, T, reference + delta]`` batch. The
    gradient is taken with respect to Delta only. CTC is evaluated with
    unreduced losses, then each utterance is divided by its target length,
    matching PyTorch ``CTCLoss(reduction='mean')`` while preserving an exact
    per-example gradient before summing the batch.
    """

    if features.ndim != 3:
        raise ValueError(f"features must have shape [B,T,D], got {tuple(features.shape)}")
    if targets.ndim != 1:
        raise ValueError(
            f"targets must be a flattened one-dimensional tensor, got {tuple(targets.shape)}"
        )
    if feature_lengths.ndim != 1 or target_lengths.ndim != 1:
        raise ValueError("feature_lengths and target_lengths must be one-dimensional")
    if feature_lengths.numel() != features.shape[0] or target_lengths.numel() != features.shape[0]:
        raise ValueError("length tensors must have one entry per batch element")
    if reference_dim < 1 or reference_dim >= features.shape[-1]:
        raise ValueError("reference_dim must leave at least one Delta coordinate")
    if torch.any(feature_lengths < 1) or torch.any(feature_lengths > features.shape[1]):
        raise ValueError("feature_lengths must be positive and no longer than the padded batch")
    if torch.any(target_lengths < 1):
        raise ValueError("target_lengths must be positive")

    # Keep the reference stream outside the attribution graph. Delta is a
    # fresh leaf so autograd returns exactly dL/dDelta, independent of model
    # parameter gradients or any caller-side requires_grad state.
    reference = features[..., :reference_dim].detach()
    delta = features[..., reference_dim:].detach().requires_grad_(True)
    logits = model(torch.cat((reference, delta), dim=-1))
    if logits.ndim != 3:
        raise ValueError(f"model must return [B,T,V] logits, got {tuple(logits.shape)}")
    log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
    criterion = torch.nn.CTCLoss(blank=blank, reduction="none", zero_infinity=True)
    raw_losses = criterion(log_probs, targets, feature_lengths, target_lengths)
    if raw_losses.ndim != 1 or raw_losses.shape[0] != features.shape[0]:
        raise ValueError("CTC reduction='none' must return one loss per utterance")
    normalized_losses = raw_losses / target_lengths.to(
        device=raw_losses.device, dtype=raw_losses.dtype
    )
    loss = normalized_losses.sum()
    (gradient,) = torch.autograd.grad(loss, delta)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("CTC loss gradient with respect to Delta is non-finite")

    valid = (
        torch.arange(features.shape[1], device=features.device).unsqueeze(0)
        < feature_lengths.to(device=features.device).unsqueeze(1)
    )
    contribution = (delta * gradient).abs().masked_fill(~valid.unsqueeze(-1), 0.0)
    sums = contribution.sum(dim=(0, 1)).detach()
    frame_count = int(feature_lengths.sum().detach().cpu())
    mean_loss = normalized_losses.mean().detach()
    if not torch.isfinite(sums).all() or not torch.isfinite(mean_loss):
        raise FloatingPointError("CTC Taylor utility contains non-finite values")
    return sums, frame_count, mean_loss


def _atomic_torch_save(value: torch.Tensor, path: Path) -> None:
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


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if values_a.shape != values_b.shape or values_a.ndim != 1:
        raise ValueError("Spearman inputs must be one-dimensional arrays of equal shape")
    ranks_a = _rankdata(values_a)
    ranks_b = _rankdata(values_b)
    centered_a = ranks_a - ranks_a.mean()
    centered_b = ranks_b - ranks_b.mean()
    denominator = float(
        np.sqrt(np.dot(centered_a, centered_a) * np.dot(centered_b, centered_b))
    )
    return float(np.dot(centered_a, centered_b) / denominator) if denominator else float("nan")


def top_k_overlap(values_a: torch.Tensor, values_b: torch.Tensor, k: int) -> float:
    if values_a.ndim != 1 or values_b.ndim != 1 or values_a.shape != values_b.shape:
        raise ValueError("ranking inputs must be one-dimensional arrays of equal shape")
    count = min(max(int(k), 1), values_a.numel())
    top_a = set(torch.argsort(values_a, descending=True, stable=True)[:count].tolist())
    top_b = set(torch.argsort(values_b, descending=True, stable=True)[:count].tolist())
    return len(top_a & top_b) / count


def _json_value(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def compute_utility(args: argparse.Namespace) -> dict[str, Any]:
    if args.reference_dim < 1 or args.delta_dim < 1:
        raise ValueError("reference-dim and delta-dim must be positive")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers must be non-negative")
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite to reuse it")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(args.vocab)
    blank = next(
        int(index) for token, index in vocab.items() if token in {"<pad>", "<blank>"}
    )
    model, checkpoint = load_full_delta_teacher(
        args.checkpoint,
        vocab,
        args.reference_dim,
        args.delta_dim,
        device,
    )
    dataset = CachedFeatureDataset(
        args.manifest,
        args.feature_root,
        "full_delta",
        vocab,
        feature_split=args.feature_split,
        expected_dim=args.reference_dim,
        max_examples=args.max_utterances,
    )
    if not len(dataset):
        raise ValueError("utility dataset is empty")
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    score_sum = torch.zeros(args.delta_dim, dtype=torch.float64)
    magnitude_sum = torch.zeros(args.delta_dim, dtype=torch.float64)
    frame_count = 0
    processed = 0
    total_loss = 0.0
    for batch_index, batch in enumerate(loader, start=1):
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        scores, batch_frames, batch_loss = ctc_taylor_batch_sums(
            model,
            features,
            targets,
            batch["feature_lengths"],
            batch["target_lengths"],
            args.reference_dim,
            blank,
        )
        score_sum += scores.double().cpu()
        valid = (
            torch.arange(features.shape[1], device=device).unsqueeze(0)
            < batch["feature_lengths"].to(device).unsqueeze(1)
        )
        delta = features[..., args.reference_dim:]
        magnitude_sum += (delta.abs() * valid.unsqueeze(-1)).sum(dim=(0, 1)).double().cpu()
        frame_count += batch_frames
        batch_size = len(batch["utt_ids"])
        processed += batch_size
        total_loss += float(batch_loss.cpu()) * batch_size
        if batch_index == 1 or processed % args.log_every < batch_size:
            print(
                {"processed": processed, "total": len(dataset), "batches": batch_index},
                flush=True,
            )

    if frame_count == 0:
        raise RuntimeError("CTC Taylor utility pass produced no valid frames")
    utility = score_sum / frame_count
    magnitude = magnitude_sum / frame_count
    ranking = torch.argsort(utility, descending=True, stable=True)
    output_stem = getattr(args, "output_stem", "utility_v4")
    if not output_stem or Path(output_stem).name != output_stem:
        raise ValueError("output-stem must be a non-empty filename stem without path separators")
    score_path = args.output_dir / f"{output_stem}.pt"
    ranking_path = args.output_dir / f"{output_stem}_ranking.pt"
    stats_path = args.output_dir / (
        "stats_v4.json" if output_stem == "utility_v4" else f"{output_stem}_stats.json"
    )
    _atomic_torch_save(utility, score_path)
    _atomic_torch_save(ranking, ranking_path)

    utility_np = utility.numpy()
    magnitude_np = magnitude.numpy()
    stats: dict[str, Any] = {
        "protocol": PROTOCOL,
        "aggregation": "E_frame,utterance[abs(Delta * dL_CTC/dDelta)]",
        "loss": "per-example CTCLoss(reduction='none') divided by target length",
        "alignment_backend": "none",
        "competitor": "none",
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "feature_root": str(args.feature_root),
        "feature_split": args.feature_split,
        "vocab": str(args.vocab),
        "device": str(device),
        "reference_dim": args.reference_dim,
        "delta_dim": args.delta_dim,
        "vocab_size": len(vocab),
        "blank_id": blank,
        "num_utterances": processed,
        "num_batches": math.ceil(processed / args.batch_size),
        "num_valid_frames": frame_count,
        "mean_ctc_loss": total_loss / max(processed, 1),
        "score_path": str(score_path),
        "ranking_path": str(ranking_path),
        "spearman_magnitude_utility_v4": _json_value(spearman(utility_np, magnitude_np)),
        "overlap_at_256": top_k_overlap(utility, magnitude, 256),
        "overlap_at_512": top_k_overlap(utility, magnitude, 512),
        "top_utility_v4": ranking[:10].tolist(),
        "utility_v4": utility.tolist(),
        "magnitude_all_frames": magnitude.tolist(),
        "checkpoint_protocol": checkpoint.get("protocol"),
        "checkpoint_epoch": checkpoint.get("epoch"),
    }
    stats_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "num_utterances": processed,
                "num_valid_frames": frame_count,
                "mean_ctc_loss": stats["mean_ctc_loss"],
                "spearman_magnitude_utility_v4": stats["spearman_magnitude_utility_v4"],
                "overlap_at_256": stats["overlap_at_256"],
                "overlap_at_512": stats["overlap_at_512"],
                "top_utility_v4": stats["top_utility_v4"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/runs/stage4/full_delta/best.pt")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/arctic_step2/l2/train_utility.jsonl")
    )
    parser.add_argument(
        "--feature-root", type=Path, default=Path("artifacts/features/stage3_l2")
    )
    parser.add_argument("--feature-split", default="train")
    parser.add_argument("--vocab", type=Path, default=Path("assets/ctc_vocab/vocab.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage5"))
    parser.add_argument(
        "--output-stem",
        default="utility_v4",
        help="filename stem for score/ranking outputs; default preserves Stage 5 names",
    )
    parser.add_argument("--reference-dim", type=int, default=1024)
    parser.add_argument("--delta-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_utterances is not None and args.max_utterances < 1:
        raise ValueError("max-utterances must be positive")
    if args.log_every < 1:
        raise ValueError("log-every must be positive")
    compute_utility(args)


if __name__ == "__main__":
    main()
