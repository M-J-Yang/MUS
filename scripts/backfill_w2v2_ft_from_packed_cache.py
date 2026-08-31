#!/usr/bin/env python3
"""Materialize ``w2v2_ft`` from the existing packed Stage 2 cache.

The packed layer-24 cache already contains ``w2v2_ft``, ``w2v2_pt``,
``wavlm_ft``, and ``delta`` for train/dev.  This migration writes only the
missing ``w2v2_ft`` files into the Stage 3 directory and validates the other
streams without changing them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor(item: Any, key: str, path: Path, expected_dim: int) -> torch.Tensor:
    value = item.get(key) if isinstance(item, dict) else None
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != expected_dim:
        raise ValueError(f"{path}: {key} must be a [T,{expected_dim}] tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{path}: {key} contains non-finite values")
    return value


def _valid_existing(path: Path, shape: torch.Size) -> bool:
    if not path.is_file():
        return False
    try:
        value = _load(path)
    except (OSError, RuntimeError, EOFError, ValueError):
        return False
    return isinstance(value, torch.Tensor) and value.shape == shape and torch.isfinite(value).all().item()


def _atomic_save(value: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value.cpu().contiguous(), temporary)
    temporary.replace(path)


def convert_split(
    source_root: Path,
    manifest_path: Path,
    output_root: Path,
    split: str,
    layer: int,
    expected_dim: int,
    skip_existing: bool,
) -> dict[str, Any]:
    records = [json.loads(line) for line in manifest_path.open(encoding="utf-8") if line.strip()]
    if not records:
        raise ValueError(f"{manifest_path}: no records")
    source_dir = source_root / split
    reference_dir = output_root / split / "wavlm_ft"
    delta_dir = output_root / split / "delta"
    target_dir = output_root / split / "w2v2_ft"
    converted = skipped = 0
    frame_lengths: list[int] = []
    for index, row in enumerate(records, start=1):
        utt_id = str(row["utt_id"])
        source_path = source_dir / f"{utt_id}.pt"
        item = _load(source_path)
        if not isinstance(item, dict) or str(item.get("utt_id")) != utt_id or int(item.get("layer", -1)) != layer:
            raise ValueError(f"{source_path}: unexpected packed payload metadata")
        reference = _tensor(item, "wavlm_ft", source_path, expected_dim)
        pretrained = _tensor(item, "w2v2_pt", source_path, expected_dim)
        fine_tuned = _tensor(item, "w2v2_ft", source_path, expected_dim)
        stored_delta = _tensor(item, "delta", source_path, expected_dim)
        if pretrained.shape != fine_tuned.shape or reference.shape != fine_tuned.shape:
            raise ValueError(f"{source_path}: packed streams do not share shape")
        if not torch.equal(fine_tuned - pretrained, stored_delta):
            raise ValueError(f"{source_path}: packed delta invariant failed")
        existing_reference = _load(reference_dir / f"{utt_id}.pt")
        existing_delta = _load(delta_dir / f"{utt_id}.pt")
        if not isinstance(existing_reference, torch.Tensor) or not isinstance(existing_delta, torch.Tensor):
            raise ValueError(f"{split}/{utt_id}: existing Stage 3 ref/delta entries must be tensors")
        if existing_reference.shape != reference.shape or existing_delta.shape != stored_delta.shape:
            raise ValueError(f"{split}/{utt_id}: packed and Stage 3 ref/delta shapes disagree")
        target_path = target_dir / f"{utt_id}.pt"
        if skip_existing and _valid_existing(target_path, fine_tuned.shape):
            skipped += 1
        else:
            _atomic_save(fine_tuned, target_path)
            converted += 1
        frame_lengths.append(int(fine_tuned.shape[0]))
        if index == 1 or index % 1000 == 0:
            print({"split": split, "processed": index, "total": len(records), "utt_id": utt_id}, flush=True)
    report = {
        "protocol": "stage3_w2v2_ft_packed_backfill_v1",
        "source_root": str(source_root),
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "split": split,
        "layer": layer,
        "utterances": len(records),
        "converted": converted,
        "skipped": skipped,
        "min_frames": min(frame_lengths),
        "max_frames": max(frame_lengths),
        "stream_written": "w2v2_ft",
        "streams_reused": ["wavlm_ft", "delta"],
        "w2v2_pt_recomputed": False,
        "delta_recomputed": False,
    }
    (output_root / split).mkdir(parents=True, exist_ok=True)
    (output_root / split / "w2v2_ft_backfill_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--expected-dim", type=int, default=1024)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    reports = [
        convert_split(
            args.source_root,
            args.manifest_root / f"{split}.jsonl",
            args.output_root,
            split,
            args.layer,
            args.expected_dim,
            args.skip_existing,
        )
        for split in args.splits
    ]
    report = {"protocol": "stage3_w2v2_ft_packed_backfill_v1", "reports": reports}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "w2v2_ft_packed_backfill_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
