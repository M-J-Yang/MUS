#!/usr/bin/env python3
"""Evaluate DropBest/DropWorst ablations on a frozen FullDelta CTC head.

The intervention is applied to the cached Delta stream at evaluation time:
selected Delta coordinates are set to zero, while the already-trained
FullDelta linear CTC head and all reference coordinates remain unchanged.
No retraining is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from usde.stage4 import (  # noqa: E402
    BASE_DIM,
    CachedFeatureDataset,
    collate,
    greedy_wer,
    load_vocab,
)
from utility.compute_utility import load_full_delta_teacher  # noqa: E402


PROTOCOL = "stage7a_full_delta_drop_ablation_v1"


def _load_tensor(path: Path, label: str) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = value.get(label, value.get("ranking", value.get("tensor")))
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path}: expected a tensor")
    return value


def _validate_ranking(ranking: torch.Tensor, delta_dim: int) -> torch.Tensor:
    if ranking.ndim != 1 or ranking.numel() != delta_dim:
        raise ValueError(
            f"ranking must be a one-dimensional permutation of {delta_dim} coordinates"
        )
    ranking = ranking.to(dtype=torch.long, device="cpu")
    if (
        torch.unique(ranking).numel() != delta_dim
        or int(ranking.min()) < 0
        or int(ranking.max()) >= delta_dim
    ):
        raise ValueError("ranking must be a complete permutation of Delta coordinates")
    return ranking


def mask_delta_coordinates(
    features: torch.Tensor,
    indices: torch.Tensor,
    reference_dim: int,
) -> torch.Tensor:
    """Return cached [reference; Delta] features with selected Delta columns zeroed."""

    if features.ndim != 3 or reference_dim < 1 or reference_dim >= features.shape[-1]:
        raise ValueError("features must be [B,T,D] with a non-empty reference and Delta stream")
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    delta_dim = features.shape[-1] - reference_dim
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= delta_dim):
        raise ValueError(f"indices must be in [0, {delta_dim})")
    masked = features.clone()
    if indices.numel():
        masked[..., reference_dim + indices.to(features.device)] = 0.0
    return masked


class MaskedFullDelta(nn.Module):
    """Frozen FullDelta head preceded by a fixed Delta-coordinate mask."""

    def __init__(
        self,
        model: nn.Module,
        indices: torch.Tensor,
        reference_dim: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("masked_indices", indices.to(dtype=torch.long), persistent=True)
        self.reference_dim = int(reference_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(mask_delta_coordinates(features, self.masked_indices, self.reference_dim))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _atomic_torch_save(value: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value.detach().cpu(), temporary)
    temporary.replace(path)


def _loader(
    manifest: Path,
    feature_root: Path,
    vocab: dict[str, int],
    split: str,
    reference_dim: int,
    delta_dim: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = CachedFeatureDataset(
        manifest,
        feature_root,
        "full_delta",
        vocab,
        feature_split=split,
        expected_dim=reference_dim,
        expected_auxiliary_dim=delta_dim,
    )
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate,
        "pin_memory": False,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    vocab: dict[str, int],
    device: torch.device,
) -> float:
    return greedy_wer(model, loader, vocab, device)  # type: ignore[arg-type]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.k < 1:
        raise ValueError("k must be positive")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")

    vocab = load_vocab(args.vocab)
    full_delta, checkpoint = load_full_delta_teacher(
        args.checkpoint,
        vocab,
        args.reference_dim,
        args.delta_dim,
        device,
    )
    ranking = _validate_ranking(_load_tensor(args.ranking, "ranking"), args.delta_dim)
    if args.k > args.delta_dim:
        raise ValueError(f"k must be no larger than delta-dim={args.delta_dim}")
    drop_best = ranking[: args.k].clone()
    drop_worst = ranking[-args.k :].clone()
    if torch.equal(drop_best, drop_worst):
        raise ValueError("DropBest and DropWorst coordinate sets unexpectedly coincide")

    dev_loader = _loader(
        args.dev_manifest,
        args.feature_root,
        vocab,
        args.dev_feature_split,
        args.reference_dim,
        args.delta_dim,
        args.batch_size,
        args.num_workers,
    )
    test_loader = _loader(
        args.test_manifest,
        args.feature_root,
        vocab,
        args.test_feature_split,
        args.reference_dim,
        args.delta_dim,
        args.batch_size,
        args.num_workers,
    )

    baseline_model = full_delta
    drop_best_model = MaskedFullDelta(full_delta, drop_best, args.reference_dim).to(device).eval()
    drop_worst_model = MaskedFullDelta(full_delta, drop_worst, args.reference_dim).to(device).eval()
    measured = {
        "FullDelta": {
            "dev_wer": _evaluate(baseline_model, dev_loader, vocab, device),
            "test_wer": _evaluate(baseline_model, test_loader, vocab, device),
        },
        "DropBest-v4": {
            "dev_wer": _evaluate(drop_best_model, dev_loader, vocab, device),
            "test_wer": _evaluate(drop_best_model, test_loader, vocab, device),
        },
        "DropWorst-v4": {
            "dev_wer": _evaluate(drop_worst_model, dev_loader, vocab, device),
            "test_wer": _evaluate(drop_worst_model, test_loader, vocab, device),
        },
    }
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "checkpoint": str(args.checkpoint),
        "ranking": str(args.ranking),
        "feature_root": str(args.feature_root),
        "vocab": str(args.vocab),
        "dev_manifest": str(args.dev_manifest),
        "test_manifest": str(args.test_manifest),
        "device": str(device),
        "reference_dim": args.reference_dim,
        "delta_dim": args.delta_dim,
        "k": args.k,
        "fraction_of_delta": args.k / args.delta_dim,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_dev_wer": checkpoint.get("dev_wer"),
        "checkpoint_test_wer": checkpoint.get("test_wer"),
        "drop_best_indices_path": str(args.output_dir / "drop_best_indices.pt"),
        "drop_worst_indices_path": str(args.output_dir / "drop_worst_indices.pt"),
        "measured": measured,
        "deltas_vs_full_delta": {
            name: {
                metric: values[metric] - measured["FullDelta"][metric]
                for metric in ("dev_wer", "test_wer")
            }
            for name, values in measured.items()
            if name != "FullDelta"
        },
    }
    _atomic_torch_save(drop_best, args.output_dir / "drop_best_indices.pt")
    _atomic_torch_save(drop_worst, args.output_dir / "drop_worst_indices.pt")
    _save_json(args.output_dir / "metrics.json", result)
    print(
        json.dumps(
            {
                "DropBest-v4_test_wer": measured["DropBest-v4"]["test_wer"],
                "DropWorst-v4_test_wer": measured["DropWorst-v4"]["test_wer"],
                "FullDelta_test_wer": measured["FullDelta"]["test_wer"],
                "k": args.k,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/runs/stage4/full_delta/best.pt"))
    parser.add_argument("--ranking", type=Path, default=Path("results/stage5/utility_v4_ranking.pt"))
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/features/stage3_l2"))
    parser.add_argument("--dev-manifest", type=Path, default=Path("manifests/arctic_step2/l2/dev.jsonl"))
    parser.add_argument("--test-manifest", type=Path, default=Path("manifests/arctic_step2/l2/test.jsonl"))
    parser.add_argument("--dev-feature-split", default="dev")
    parser.add_argument("--test-feature-split", default="test")
    parser.add_argument("--vocab", type=Path, default=Path("assets/ctc_vocab/vocab.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage7a/drop_ablation_v4_k256"))
    parser.add_argument("--reference-dim", type=int, default=BASE_DIM)
    parser.add_argument("--delta-dim", type=int, default=BASE_DIM)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
