#!/usr/bin/env python3
"""Convert the old combined final-layer cache into the Stage 3 cache layout.

This is an offline migration.  It never loads an SSL model and recomputes
``delta = w2v2_ft - w2v2_pt`` from the source payload before writing:

    <output>/<split>/wavlm_ft/<utt_id>.pt
    <output>/<split>/delta/<utt_id>.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _atomic_save(tensor: torch.Tensor, path: Path, dtype: torch.dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(tensor.to(dtype=dtype).cpu().contiguous(), temporary)
    temporary.replace(path)


def _valid_target(path: Path, expected_frames: int | None, expected_dim: int | None) -> bool:
    try:
        value = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, EOFError, ValueError):
        return False
    return bool(
        isinstance(value, torch.Tensor)
        and value.ndim == 2
        and (expected_frames is None or value.shape[0] == expected_frames)
        and (expected_dim is None or value.shape[1] == expected_dim)
        and torch.isfinite(value).all()
    )


def _validate_tensor(name: str, value: Any, utt_id: str, expected_dim: int | None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{utt_id}: {name} must be a finite [T,D] tensor")
    if expected_dim is not None and value.shape[1] != expected_dim:
        raise ValueError(f"{utt_id}: {name} has dimension {value.shape[1]}, expected {expected_dim}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{utt_id}: {name} contains non-finite values")
    return value


def convert_split(
    source_root: Path,
    manifest_path: Path,
    output_root: Path,
    split: str,
    layer: int,
    storage_dtype: torch.dtype,
    expected_dim: int | None,
    skip_existing: bool,
) -> dict[str, Any]:
    records = []
    with manifest_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"{manifest_path}: no records")

    source_dir = source_root / split
    reference_dir = output_root / split / "wavlm_ft"
    delta_dir = output_root / split / "delta"
    frame_lengths: list[int] = []
    dimensions: set[tuple[int, int]] = set()
    converted = 0
    skipped = 0

    for index, record in enumerate(records, start=1):
        utt_id = str(record["utt_id"])
        source_path = source_dir / f"{utt_id}.pt"
        reference_path = reference_dir / f"{utt_id}.pt"
        delta_path = delta_dir / f"{utt_id}.pt"
        if skip_existing and _valid_target(reference_path, None, expected_dim) and _valid_target(delta_path, None, expected_dim):
            skipped += 1
            reference = torch.load(reference_path, map_location="cpu")
            frame_lengths.append(int(reference.shape[0]))
            dimensions.add(tuple(reference.shape))
            continue
        if not source_path.is_file():
            raise FileNotFoundError(f"{split}/{utt_id}: missing source cache {source_path}")

        item = torch.load(source_path, map_location="cpu")
        if not isinstance(item, dict):
            raise ValueError(f"{split}/{utt_id}: source payload is not a dict")
        if str(item.get("utt_id")) != utt_id:
            raise ValueError(f"{split}/{utt_id}: payload utt_id={item.get('utt_id')!r}")
        if int(item.get("layer", -1)) != layer:
            raise ValueError(f"{split}/{utt_id}: payload layer={item.get('layer')}, expected {layer}")

        e_pt = _validate_tensor("w2v2_pt", item.get("w2v2_pt"), utt_id, expected_dim)
        e_ft = _validate_tensor("w2v2_ft", item.get("w2v2_ft"), utt_id, expected_dim)
        e_ref = _validate_tensor("wavlm_ft", item.get("wavlm_ft"), utt_id, expected_dim)
        stored_delta = _validate_tensor("delta", item.get("delta"), utt_id, expected_dim)
        if e_pt.shape != e_ft.shape:
            raise ValueError(f"{split}/{utt_id}: W2V2 shape mismatch {tuple(e_pt.shape)} vs {tuple(e_ft.shape)}")
        delta = e_ft - e_pt
        if e_ref.shape != delta.shape:
            raise ValueError(f"{split}/{utt_id}: reference/Delta shape mismatch {tuple(e_ref.shape)} vs {tuple(delta.shape)}")
        if not torch.equal(delta, stored_delta):
            raise ValueError(f"{split}/{utt_id}: stored Delta differs from w2v2_ft - w2v2_pt")
        if delta.abs().sum() <= 0:
            raise ValueError(f"{split}/{utt_id}: Delta is identically zero")

        _atomic_save(e_ref, reference_path, storage_dtype)
        _atomic_save(delta, delta_path, storage_dtype)
        frame_lengths.append(int(delta.shape[0]))
        dimensions.add(tuple(delta.shape))
        converted += 1
        if index == 1 or index % 1000 == 0:
            print({"split": split, "processed": index, "total": len(records), "utt_id": utt_id}, flush=True)

    report = {
        "split": split,
        "source_root": str(source_root),
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "layer": layer,
        "utterances": len(records),
        "converted": converted,
        "skipped": skipped,
        "min_frames": min(frame_lengths),
        "max_frames": max(frame_lengths),
        "dimensions": sorted([list(value) for value in dimensions]),
        "storage_dtype": str(storage_dtype).removeprefix("torch."),
    }
    (output_root / split).mkdir(parents=True, exist_ok=True)
    (output_root / split / "conversion_report.json").write_text(
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
    parser.add_argument("--storage-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.layer < 0 or args.expected_dim < 1:
        raise ValueError("layer must be non-negative and expected-dim must be positive")
    dtype = torch.float16 if args.storage_dtype == "float16" else torch.float32
    reports = []
    for split in args.splits:
        reports.append(convert_split(
            args.source_root,
            args.manifest_root / f"{split}.jsonl",
            args.output_root,
            split,
            args.layer,
            dtype,
            args.expected_dim,
            args.skip_existing,
        ))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "conversion_report.json").write_text(
        json.dumps({"source_root": str(args.source_root), "splits": reports}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
