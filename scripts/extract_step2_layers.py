#!/usr/bin/env python3
"""Batch-extract shared WavLM reference and W2V2 Delta tensors for Step 2.

Each encoder is run once per padded batch. A file contains only the shared
reference layer and one W2V2 Delta layer, so the audit does not duplicate the
fine-tuned/pretrained streams. The test split should be extracted only after
layer selection is frozen.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from usde.features import hidden_states, load_audio, load_encoder
from usde.manifest import load_jsonl


def batch_hidden_states(model: torch.nn.Module, processor: Any, audios: list[Any], device: torch.device) -> tuple[tuple[torch.Tensor, ...], ...]:
    inputs = processor(audios, sampling_rate=processor.sampling_rate, return_tensors="pt", padding=True)
    attention_mask = inputs.get("attention_mask")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True, return_dict=True)
    if attention_mask is None:
        lengths = [output.hidden_states[0].shape[1]] * len(audios)
    else:
        lengths_tensor = model._get_feat_extract_output_lengths(attention_mask.sum(-1))
        lengths = [int(value) for value in lengths_tensor]
    return tuple(
        tuple(state[row, :length].to(dtype=torch.float32).cpu().contiguous() for state in output.hidden_states)
        for row, length in enumerate(lengths)
    )


def existing_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        item = torch.load(path, map_location="cpu")
        reference, delta = item.get("wavlm_ft"), item.get("delta")
        return reference is not None and delta is not None and reference.ndim == 2 and reference.shape == delta.shape and reference.shape[1] == 1024 and torch.isfinite(reference).all() and torch.isfinite(delta).all()
    except (OSError, RuntimeError, EOFError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wavlm-ft", type=Path, required=True)
    parser.add_argument("--w2v2-ft", type=Path, required=True)
    parser.add_argument("--w2v2-pt", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(1, 25)))
    parser.add_argument("--reference-layer", type=int, default=24)
    parser.add_argument("--storage-dtype", choices=("float32", "float16"), default="float16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    layers = sorted(set(args.layers))
    if not layers or min(layers) < 1 or max(layers) > 24:
        raise ValueError(f"layers must be in [1, 24], got {layers}")
    if args.reference_layer not in range(1, 25):
        raise ValueError("reference-layer must be in [1, 24]")
    if args.batch_size < 1 or args.start_index < 0:
        raise ValueError("batch-size must be positive and start-index non-negative")
    storage_dtype = torch.float16 if args.storage_dtype == "float16" else torch.float32
    device = torch.device(args.device)
    wavlm, wavlm_processor = load_encoder(str(args.wavlm_ft), device)
    w2v2_ft, w2v2_processor = load_encoder(str(args.w2v2_ft), device)
    w2v2_pt, w2v2_pt_processor = load_encoder(str(args.w2v2_pt), device)
    processors = (wavlm_processor, w2v2_processor, w2v2_pt_processor)
    if {processor.sampling_rate for processor in processors} != {16000}:
        raise ValueError("all processors must use 16-kHz audio")
    all_records = load_jsonl(args.manifest)
    records = all_records[args.start_index:] if args.max_utterances is None else all_records[args.start_index : args.start_index + args.max_utterances]
    if not records:
        raise ValueError("selected extraction range is empty")
    for layer in layers:
        (args.output_root / f"layer_{layer:02d}").mkdir(parents=True, exist_ok=True)
    frame_lengths: list[int] = []
    shift_sums = {layer: 0.0 for layer in layers}
    shift_frames = {layer: 0 for layer in layers}
    for batch_start in range(0, len(records), args.batch_size):
        batch_records = records[batch_start : batch_start + args.batch_size]
        output_paths = [args.output_root / f"layer_{layer:02d}" / f"{row['utt_id']}.pt" for layer in layers for row in batch_records]
        if args.skip_existing and all(existing_valid(path) for path in output_paths):
            frame_lengths.extend([0] * len(batch_records))
            continue
        audios = [load_audio(row["audio_path"]) for row in batch_records]
        wavlm_batch = batch_hidden_states(wavlm, wavlm_processor, audios, device)
        w2v2_ft_batch = batch_hidden_states(w2v2_ft, w2v2_processor, audios, device)
        w2v2_pt_batch = batch_hidden_states(w2v2_pt, w2v2_pt_processor, audios, device)
        for row, wavlm_states, w2v2_ft_states, w2v2_pt_states in zip(batch_records, wavlm_batch, w2v2_ft_batch, w2v2_pt_batch, strict=True):
            reference = wavlm_states[args.reference_layer]
            for layer in layers:
                delta = w2v2_ft_states[layer] - w2v2_pt_states[layer]
                if reference.shape != delta.shape or reference.shape[1] != 1024:
                    raise ValueError(f"{row['utt_id']}/layer_{layer}: frame or dimension mismatch: reference={tuple(reference.shape)}, delta={tuple(delta.shape)}")
                if not torch.isfinite(reference).all() or not torch.isfinite(delta).all():
                    raise ValueError(f"{row['utt_id']}/layer_{layer}: non-finite feature")
                output_path = args.output_root / f"layer_{layer:02d}" / f"{row['utt_id']}.pt"
                temporary_path = output_path.with_name(output_path.name + ".tmp")
                torch.save({"utt_id": row["utt_id"], "reference_layer": args.reference_layer, "layer": layer, "wavlm_ft": reference.to(storage_dtype), "delta": delta.to(storage_dtype)}, temporary_path)
                temporary_path.replace(output_path)
            frame_lengths.append(reference.shape[0])
        print(json.dumps({"processed": min(batch_start + len(batch_records), len(records)), "total": len(records), "last_utt": batch_records[-1]["utt_id"]}, sort_keys=True), flush=True)
    nonzero_frames = [value for value in frame_lengths if value]
    report = {"manifest": str(args.manifest), "utterances": len(records), "layers": layers, "reference_layer": args.reference_layer, "storage_dtype": args.storage_dtype, "batch_size": args.batch_size, "start_index": args.start_index, "min_frames": min(nonzero_frames) if nonzero_frames else None, "max_frames": max(nonzero_frames) if nonzero_frames else None, "shift_magnitude": {str(layer): shift_sums[layer] / shift_frames[layer] for layer in layers if shift_frames[layer]}}
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_name = "extraction_report.json" if args.start_index == 0 and args.max_utterances is None else f"extraction_report_{args.start_index}_{len(records)}.json"
    (args.output_root / report_name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
