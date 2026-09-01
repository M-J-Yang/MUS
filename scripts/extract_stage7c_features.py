#!/usr/bin/env python3
"""Batch-extract Stage 7C CMU reference and L2-adaptation Delta features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from extract_step2_layers import batch_hidden_states  # noqa: E402
from usde.features import load_audio, load_encoder  # noqa: E402
from usde.manifest import load_jsonl  # noqa: E402


def _valid_tensor(path: Path) -> bool:
    try:
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            value = torch.load(path, map_location="cpu")
        return bool(
            isinstance(value, torch.Tensor)
            and value.ndim == 2
            and value.shape[1] == 1024
            and torch.isfinite(value).all()
        )
    except (OSError, RuntimeError, EOFError, ValueError):
        return False


def _atomic_save(value: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value.cpu().contiguous(), temporary)
    temporary.replace(path)


def extract(args: argparse.Namespace) -> dict[str, Any]:
    if args.layer < 1 or args.layer > 24 or args.reference_layer < 1 or args.reference_layer > 24:
        raise ValueError("layer and reference-layer must be in [1, 24]")
    if args.batch_size < 1 or args.start_index < 0:
        raise ValueError("batch-size must be positive and start-index non-negative")
    records = load_jsonl(args.manifest)
    selected = records[args.start_index:] if args.max_utterances is None else records[args.start_index : args.start_index + args.max_utterances]
    if not selected:
        raise ValueError("selected extraction range is empty")

    device = torch.device(args.device)
    wavlm, wavlm_processor = load_encoder(str(args.wavlm_ft), device)
    w2v2_ft, w2v2_processor = load_encoder(str(args.w2v2_ft), device)
    w2v2_pt, w2v2_pt_processor = load_encoder(str(args.w2v2_pt), device)
    processors = (wavlm_processor, w2v2_processor, w2v2_pt_processor)
    if {int(processor.sampling_rate) for processor in processors} != {16000}:
        raise ValueError("all processors must use 16-kHz audio")

    split_root = args.output_root / args.split
    reference_root = split_root / "wavlm_ft"
    delta_root = split_root / "delta"
    frame_lengths: list[int] = []
    delta_means: list[float] = []
    skipped = 0
    written = 0
    for batch_start in range(0, len(selected), args.batch_size):
        batch_records = selected[batch_start : batch_start + args.batch_size]
        output_paths = [
            (reference_root / f"{row['utt_id']}.pt", delta_root / f"{row['utt_id']}.pt")
            for row in batch_records
        ]
        if args.skip_existing and all(_valid_tensor(ref) and _valid_tensor(delta) for ref, delta in output_paths):
            skipped += len(batch_records)
            for ref_path, _delta_path in output_paths:
                frame_lengths.append(int(torch.load(ref_path, map_location="cpu", weights_only=True).shape[0]))
            continue

        audios = [load_audio(row["audio_path"]) for row in batch_records]
        wavlm_batch = batch_hidden_states(wavlm, wavlm_processor, audios, device)
        w2v2_ft_batch = batch_hidden_states(w2v2_ft, w2v2_processor, audios, device)
        w2v2_pt_batch = batch_hidden_states(w2v2_pt, w2v2_pt_processor, audios, device)
        for row, wavlm_states, w2v2_ft_states, w2v2_pt_states, paths in zip(
            batch_records, wavlm_batch, w2v2_ft_batch, w2v2_pt_batch, output_paths, strict=True
        ):
            reference = wavlm_states[args.reference_layer]
            delta = w2v2_ft_states[args.layer] - w2v2_pt_states[args.layer]
            if reference.shape != delta.shape or reference.shape[1] != 1024:
                raise ValueError(
                    f"{row['utt_id']}: frame/dimension mismatch; "
                    f"reference={tuple(reference.shape)}, delta={tuple(delta.shape)}"
                )
            if not torch.isfinite(reference).all() or not torch.isfinite(delta).all():
                raise ValueError(f"{row['utt_id']}: encoder produced non-finite features")
            _atomic_save(reference, paths[0])
            _atomic_save(delta, paths[1])
            frame_lengths.append(int(reference.shape[0]))
            delta_means.append(float(delta.abs().mean()))
            written += 1
        print({"processed": min(batch_start + len(batch_records), len(selected)), "total": len(selected)}, flush=True)

    report = {
        "protocol": "stage7c_cmu_feature_cache_v1",
        "manifest": str(args.manifest),
        "output_root": str(args.output_root),
        "split": args.split,
        "wavlm_ft": str(args.wavlm_ft),
        "w2v2_ft_l2": str(args.w2v2_ft),
        "w2v2_pt": str(args.w2v2_pt),
        "layer": args.layer,
        "reference_layer": args.reference_layer,
        "device": str(device),
        "batch_size": args.batch_size,
        "utterances": len(selected),
        "written": written,
        "skipped_existing": skipped,
        "min_frames": min(frame_lengths),
        "max_frames": max(frame_lengths),
        "mean_delta_abs_written": sum(delta_means) / len(delta_means) if delta_means else None,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (split_root / "extraction_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/stage7c/cmu.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/features/stage7c_cmu"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--wavlm-ft", type=Path, default=Path("checkpoints/wavlm_myst_fullfinetune"))
    parser.add_argument("--w2v2-ft", type=Path, default=Path("checkpoints/w2v2_myst_fullfinetune"))
    parser.add_argument("--w2v2-pt", type=Path, default=Path("checkpoints/w2v2_large_lv60_pretrained"))
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--reference-layer", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.max_utterances is not None and args.max_utterances < 1:
        raise ValueError("max-utterances must be positive")
    report = extract(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
