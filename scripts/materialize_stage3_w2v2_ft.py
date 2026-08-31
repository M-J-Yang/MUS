#!/usr/bin/env python3
"""Backfill only the cached W2V2 fine-tuned stream needed by Stage 4 B.

The existing Stage 3 cache already contains ``wavlm_ft`` and ``delta``.  This
script computes only ``w2v2_ft`` from the frozen public checkpoint, verifies
its frame shape against those two existing streams, and writes CPU tensors
atomically.  It never recomputes or overwrites ``w2v2_pt`` or ``delta``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_stage3_features import _encode, extract_last_hidden  # noqa: E402
from usde.ctc import read_records, resolve_audio_path  # noqa: E402
from usde.features import load_audio, load_encoder  # noqa: E402


def _load_tensor(path: Path, label: str) -> torch.Tensor:
    try:
        item = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        item = torch.load(path, map_location="cpu")
    if isinstance(item, dict) and isinstance(item.get(label), torch.Tensor):
        item = item[label]
    if not isinstance(item, torch.Tensor):
        raise ValueError(f"{path}: expected a tensor cache entry")
    return item


def _atomic_save(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(tensor.cpu().contiguous(), temporary)
    temporary.replace(path)


def materialize(
    manifest_path: Path,
    output_root: Path,
    split: str,
    w2v2_ft_id: str,
    layer: int,
    device_name: str,
    max_utterances: int | None,
    skip_existing: bool,
    dry_run: bool,
    expected_dim: int,
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
) -> dict[str, Any]:
    records = read_records(manifest_path)
    if max_utterances is not None:
        if max_utterances < 1:
            raise ValueError("max-utterances must be positive")
        records = records[:max_utterances]
    reference_dir = output_root / split / "wavlm_ft"
    delta_dir = output_root / split / "delta"
    output_dir = output_root / split / "w2v2_ft"
    frame_lengths: list[int] = []
    for position, row in enumerate(records, start=1):
        utt_id = str(row["utt_id"])
        if Path(utt_id).name != utt_id:
            raise ValueError(f"{utt_id}: utt_id must not contain path separators")
        reference_path = reference_dir / f"{utt_id}.pt"
        delta_path = delta_dir / f"{utt_id}.pt"
        output_path = output_dir / f"{utt_id}.pt"
        reference = _load_tensor(reference_path, "wavlm_ft")
        delta = _load_tensor(delta_path, "delta")
        if reference.ndim != 2 or delta.ndim != 2 or reference.shape != delta.shape:
            raise ValueError(
                f"{split}/{utt_id}: existing ref/delta mismatch; "
                f"ref={tuple(reference.shape)}, delta={tuple(delta.shape)}"
            )
        if reference.shape[1] != expected_dim or not torch.isfinite(reference).all() or not torch.isfinite(delta).all():
            raise ValueError(f"{split}/{utt_id}: invalid existing ref/delta cache")
        if skip_existing and output_path.is_file():
            cached = _load_tensor(output_path, "w2v2_ft")
            if cached.shape == reference.shape and torch.isfinite(cached).all():
                frame_lengths.append(int(cached.shape[0]))
                continue

        audio_path = resolve_audio_path(str(row["wav_path"]), manifest_path)
        audio = load_audio(str(audio_path))
        state = extract_last_hidden(model, _encode(processor, audio), layer, device)
        if state.shape != reference.shape:
            raise ValueError(
                f"{split}/{utt_id}: w2v2_ft/ref shape mismatch; "
                f"w2v2_ft={tuple(state.shape)}, ref={tuple(reference.shape)}"
            )
        if not torch.isfinite(state).all():
            raise ValueError(f"{split}/{utt_id}: w2v2_ft contains non-finite values")
        if not dry_run:
            _atomic_save(state, output_path)
        frame_lengths.append(int(state.shape[0]))
        if position == 1 or position % 100 == 0:
            print({"split": split, "processed": position, "total": len(records), "utt_id": utt_id}, flush=True)
    if not frame_lengths:
        raise ValueError(f"{split}: no records selected")
    return {
        "manifest": str(manifest_path),
        "split": split,
        "layer": layer,
        "utterances": len(records),
        "min_frames": min(frame_lengths),
        "max_frames": max(frame_lengths),
        "dry_run": dry_run,
        "cache_root": None if dry_run else str(output_root),
        "stream_written": "w2v2_ft",
        "streams_reused": ["wavlm_ft", "delta"],
        "w2v2_pt_recomputed": False,
        "delta_recomputed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--split", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--w2v2-ft", required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-dim", type=int, default=1024)
    args = parser.parse_args()
    if len(args.manifest) != len(args.split):
        raise ValueError("--manifest and --split must have the same number of values")
    device = torch.device(args.device)
    model, processor = load_encoder(args.w2v2_ft, device)
    if int(processor.sampling_rate) != 16000:
        raise ValueError("w2v2_ft feature extractor must use 16-kHz audio")
    reports = [
        materialize(
            manifest_path=manifest,
            output_root=args.output_root,
            split=split,
            w2v2_ft_id=args.w2v2_ft,
            layer=args.layer,
            device_name=args.device,
            max_utterances=args.max_utterances,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            expected_dim=args.expected_dim,
            model=model,
            processor=processor,
            device=device,
        )
        for manifest, split in zip(args.manifest, args.split, strict=True)
    ]
    report = {"protocol": "stage3_w2v2_ft_backfill_v1", "reports": reports}
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "w2v2_ft_backfill_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
