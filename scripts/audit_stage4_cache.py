#!/usr/bin/env python3
"""Audit the three aligned streams required by Stage 4 A/B/C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path}: expected a tensor")
    return value


def _records(path: Path) -> set[str]:
    rows = {str(json.loads(line)["utt_id"]) for line in path.open(encoding="utf-8") if line.strip()}
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def audit_split(cache_root: Path, manifest_path: Path, split: str, expected_dim: int) -> dict[str, Any]:
    expected = _records(manifest_path)
    directories = {stream: cache_root / split / stream for stream in ("wavlm_ft", "w2v2_ft", "delta")}
    ids = {stream: {path.stem for path in directory.glob("*.pt")} for stream, directory in directories.items()}
    for stream, observed in ids.items():
        if observed != expected:
            raise ValueError(
                f"{split}/{stream}: cache/manifest mismatch; missing={len(expected - observed)}, "
                f"extra={len(observed - expected)}"
            )
    min_frames: int | None = None
    max_frames = 0
    total_frames = 0
    for index, utt_id in enumerate(sorted(expected), start=1):
        reference = _load(directories["wavlm_ft"] / f"{utt_id}.pt")
        fine_tuned = _load(directories["w2v2_ft"] / f"{utt_id}.pt")
        delta = _load(directories["delta"] / f"{utt_id}.pt")
        if any(value.ndim != 2 for value in (reference, fine_tuned, delta)):
            raise ValueError(f"{split}/{utt_id}: all streams must be [T,D]")
        if not (reference.shape == fine_tuned.shape == delta.shape):
            raise ValueError(
                f"{split}/{utt_id}: frame shape mismatch; ref={tuple(reference.shape)}, "
                f"w2v2_ft={tuple(fine_tuned.shape)}, delta={tuple(delta.shape)}"
            )
        if reference.shape[1] != expected_dim or not all(torch.isfinite(value).all() for value in (reference, fine_tuned, delta)):
            raise ValueError(f"{split}/{utt_id}: invalid dimension or non-finite value")
        if delta.abs().sum() <= 0:
            raise ValueError(f"{split}/{utt_id}: Delta is identically zero")
        frames = int(reference.shape[0])
        min_frames = frames if min_frames is None else min(min_frames, frames)
        max_frames = max(max_frames, frames)
        total_frames += frames
        if index == 1 or index % 5000 == 0:
            print({"split": split, "audited": index, "total": len(expected), "utt_id": utt_id}, flush=True)
    return {
        "split": split,
        "utterances": len(expected),
        "total_frames": total_frames,
        "min_frames": min_frames,
        "max_frames": max_frames,
        "dimensions": [expected_dim],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--expected-dim", type=int, default=1024)
    args = parser.parse_args()
    report = {
        "protocol": "stage4_three_stream_cache_audit_v1",
        "cache_root": str(args.cache_root),
        "expected_dim": args.expected_dim,
        "streams": ["wavlm_ft", "w2v2_ft", "delta"],
        "splits": [audit_split(args.cache_root, args.manifest_root / f"{split}.jsonl", split, args.expected_dim) for split in args.splits],
    }
    output = args.cache_root / "stage4_cache_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
