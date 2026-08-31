#!/usr/bin/env python3
"""Audit a completed Stage 3 ``wavlm_ft``/``delta`` cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _records(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        values = {str(json.loads(line)["utt_id"]) for line in handle if line.strip()}
    if not values:
        raise ValueError(f"{path}: no records")
    return values


def audit_split(cache_root: Path, manifest_path: Path, split: str, expected_dim: int) -> dict[str, Any]:
    expected = _records(manifest_path)
    reference_dir = cache_root / split / "wavlm_ft"
    delta_dir = cache_root / split / "delta"
    reference_ids = {path.stem for path in reference_dir.glob("*.pt")}
    delta_ids = {path.stem for path in delta_dir.glob("*.pt")}
    if reference_ids != expected or delta_ids != expected:
        raise ValueError(
            f"{split}: cache/manifest mismatch; "
            f"reference_missing={len(expected - reference_ids)}, "
            f"delta_missing={len(expected - delta_ids)}, "
            f"reference_extra={len(reference_ids - expected)}, "
            f"delta_extra={len(delta_ids - expected)}"
        )

    min_frames = None
    max_frames = 0
    dimensions: set[tuple[int, int]] = set()
    total_frames = 0
    for index, utt_id in enumerate(sorted(expected), start=1):
        reference = torch.load(reference_dir / f"{utt_id}.pt", map_location="cpu")
        delta = torch.load(delta_dir / f"{utt_id}.pt", map_location="cpu")
        if not isinstance(reference, torch.Tensor) or not isinstance(delta, torch.Tensor):
            raise ValueError(f"{split}/{utt_id}: cache entries must be tensors")
        if reference.ndim != 2 or delta.ndim != 2 or reference.shape != delta.shape:
            raise ValueError(f"{split}/{utt_id}: shape mismatch ref={tuple(reference.shape)} delta={tuple(delta.shape)}")
        if reference.shape[1] != expected_dim or not torch.isfinite(reference).all() or not torch.isfinite(delta).all():
            raise ValueError(f"{split}/{utt_id}: invalid dimension or non-finite value")
        if delta.abs().sum() <= 0:
            raise ValueError(f"{split}/{utt_id}: Delta is identically zero")
        frames = int(reference.shape[0])
        min_frames = frames if min_frames is None else min(min_frames, frames)
        max_frames = max(max_frames, frames)
        total_frames += frames
        dimensions.add(tuple(reference.shape))
        if index == 1 or index % 5000 == 0:
            print({"split": split, "audited": index, "total": len(expected), "utt_id": utt_id}, flush=True)
    return {
        "split": split,
        "utterances": len(expected),
        "total_frames": total_frames,
        "min_frames": min_frames,
        "max_frames": max_frames,
        "dimensions": sorted([list(value) for value in dimensions]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--expected-dim", type=int, default=1024)
    args = parser.parse_args()
    reports = [audit_split(args.cache_root, args.manifest_root / f"{split}.jsonl", split, args.expected_dim) for split in args.splits]
    report = {"cache_root": str(args.cache_root), "expected_dim": args.expected_dim, "splits": reports}
    output = args.cache_root / "cache_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
