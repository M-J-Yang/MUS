#!/usr/bin/env python3
"""Extract frozen final-layer reference and representation-Delta features.

The script intentionally has a small progression gate:

* ``--dry-run --max-utterances 1`` checks one utterance and prints shapes;
* ``--dry-run --max-utterances 20`` checks a small manifest prefix;
* without ``--dry-run`` it writes only ``wavlm_ft`` and ``delta`` tensors.

W2V2-pt and W2V2-ft receive the exact same feature-extractor output.  This is
important because the subtraction is frame-wise and is meant to measure the
fine-tuning-induced representation shift, not preprocessing differences.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from usde.ctc import read_records, resolve_audio_path  # noqa: E402
from usde.features import load_audio, load_encoder  # noqa: E402


def _select_layer(states: tuple[torch.Tensor, ...], layer: int) -> torch.Tensor:
    """Select a hidden state, where -1 means the final transformer layer."""

    if not states:
        raise ValueError("model returned no hidden states")
    index = layer if layer >= 0 else len(states) + layer
    if index < 0 or index >= len(states):
        raise ValueError(f"requested layer {layer}, but model returned {len(states)} hidden states")
    return states[index]


def extract_last_hidden(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    layer: int,
    device: torch.device,
) -> torch.Tensor:
    """Return one model's selected hidden state as a CPU ``[T, D]`` tensor."""

    model_inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model(**model_inputs, output_hidden_states=True, return_dict=True)
    state = _select_layer(tuple(output.hidden_states), layer).squeeze(0)
    if state.ndim != 2:
        raise ValueError(f"expected hidden state [T,D], got {tuple(state.shape)}")
    return state.to(dtype=torch.float32, device="cpu").contiguous()


def _encode(processor: Any, audio: Any) -> dict[str, torch.Tensor]:
    encoded = processor(
        audio,
        sampling_rate=processor.sampling_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return {key: value for key, value in encoded.items() if torch.is_tensor(value)}


def extract_one(
    audio_path: Path,
    wavlm: torch.nn.Module,
    wavlm_processor: Any,
    w2v2_ft: torch.nn.Module,
    w2v2_processor: Any,
    w2v2_pt: torch.nn.Module,
    layer: int,
    device: torch.device,
) -> dict[str, torch.Tensor | float]:
    """Run the three encoders once and return debug streams plus Delta."""

    audio = load_audio(str(audio_path))

    # Encode W2V2 once.  Both W2V2 checkpoints must see identical tensors.
    w2v2_inputs = _encode(w2v2_processor, audio)
    e_ft = extract_last_hidden(w2v2_ft, w2v2_inputs, layer, device)
    e_pt = extract_last_hidden(w2v2_pt, w2v2_inputs, layer, device)
    e_ref = extract_last_hidden(
        wavlm,
        _encode(wavlm_processor, audio),
        layer,
        device,
    )

    if e_pt.shape != e_ft.shape:
        raise ValueError(
            f"{audio_path}: W2V2 frame/dimension mismatch: "
            f"pt={tuple(e_pt.shape)}, ft={tuple(e_ft.shape)}"
        )
    if e_ref.shape[0] != e_ft.shape[0]:
        raise ValueError(
            f"{audio_path}: reference/Delta frame mismatch: "
            f"ref={tuple(e_ref.shape)}, delta={tuple(e_ft.shape)}"
        )
    streams = {"e_pt": e_pt, "e_ft": e_ft, "e_ref": e_ref}
    if not all(torch.isfinite(value).all() for value in streams.values()):
        raise ValueError(f"{audio_path}: encoder produced non-finite hidden states")

    delta = e_ft - e_pt
    if not torch.isfinite(delta).all():
        raise ValueError(f"{audio_path}: Delta contains non-finite values")
    delta_abs_mean = float(delta.abs().mean())
    if delta_abs_mean <= 0.0:
        raise ValueError(f"{audio_path}: Delta is identically zero")
    return {
        "e_pt": e_pt,
        "e_ft": e_ft,
        "e_ref": e_ref,
        "delta": delta,
        "delta_abs_mean": delta_abs_mean,
        "delta_abs_max": float(delta.abs().max()),
    }


def _atomic_save(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(tensor.cpu(), temporary)
    temporary.replace(path)


def _valid_cached_pair(reference_path: Path, delta_path: Path) -> bool:
    try:
        reference = torch.load(reference_path, map_location="cpu")
        delta = torch.load(delta_path, map_location="cpu")
        return bool(
            isinstance(reference, torch.Tensor)
            and isinstance(delta, torch.Tensor)
            and reference.ndim == 2
            and delta.ndim == 2
            and reference.shape[0] == delta.shape[0]
            and torch.isfinite(reference).all()
            and torch.isfinite(delta).all()
            and delta.abs().sum() > 0
        )
    except (OSError, RuntimeError, EOFError, ValueError):
        return False


def extract_split(
    manifest_path: Path,
    output_root: Path,
    wavlm_ft_id: str,
    w2v2_ft_id: str,
    w2v2_pt_id: str,
    layer: int,
    device_name: str,
    split: str | None = None,
    start_index: int = 0,
    max_utterances: int | None = None,
    skip_existing: bool = False,
    dry_run: bool = False,
    debug_output_dir: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Extract a manifest range, optionally writing the final cache."""

    if start_index < 0:
        raise ValueError("start-index must be non-negative")
    if max_utterances is not None and max_utterances < 1:
        raise ValueError("max-utterances must be positive")
    records = read_records(manifest_path)
    selected = records[start_index:] if max_utterances is None else records[start_index : start_index + max_utterances]
    if not selected:
        raise ValueError("selected extraction range is empty")

    device = torch.device(device_name)
    wavlm, wavlm_processor = load_encoder(wavlm_ft_id, device)
    w2v2_ft, w2v2_processor = load_encoder(w2v2_ft_id, device)
    w2v2_pt, w2v2_pt_processor = load_encoder(w2v2_pt_id, device)
    processors = (wavlm_processor, w2v2_processor, w2v2_pt_processor)
    if {int(processor.sampling_rate) for processor in processors} != {16000}:
        raise ValueError("Stage 3 requires all feature extractors to use 16-kHz audio")

    split_name = split or manifest_path.stem
    reference_dir = output_root / split_name / "wavlm_ft"
    delta_dir = output_root / split_name / "delta"
    layer_name = "last" if layer < 0 else str(layer)
    frame_lengths: list[int] = []
    delta_means: list[float] = []

    for position, row in enumerate(selected, start=1):
        utt_id = str(row["utt_id"])
        if Path(utt_id).name != utt_id:
            raise ValueError(f"{utt_id}: utt_id must not contain path separators")
        audio_path = resolve_audio_path(str(row["wav_path"]), manifest_path)
        reference_path = reference_dir / f"{utt_id}.pt"
        delta_path = delta_dir / f"{utt_id}.pt"
        if skip_existing and _valid_cached_pair(reference_path, delta_path):
            reference = torch.load(reference_path, map_location="cpu")
            delta = torch.load(delta_path, map_location="cpu")
            frame_lengths.append(int(reference.shape[0]))
            delta_means.append(float(delta.abs().mean()))
            continue

        item = extract_one(
            audio_path,
            wavlm,
            wavlm_processor,
            w2v2_ft,
            w2v2_processor,
            w2v2_pt,
            layer,
            device,
        )
        e_ref = item["e_ref"]
        delta = item["delta"]
        assert isinstance(e_ref, torch.Tensor) and isinstance(delta, torch.Tensor)
        frame_lengths.append(int(delta.shape[0]))
        delta_means.append(float(item["delta_abs_mean"]))

        print(
            f"[{position}/{len(selected)}] {utt_id}\n"
            f"pt    = {tuple(item['e_pt'].shape)}\n"
            f"ft    = {tuple(item['e_ft'].shape)}\n"
            f"ref   = {tuple(e_ref.shape)}\n"
            f"delta = {tuple(delta.shape)}\n"
            f"delta_mean = {item['delta_abs_mean']:.6f}\n"
            f"delta_max  = {item['delta_abs_max']:.6f}",
            flush=True,
        )

        if not dry_run:
            _atomic_save(e_ref, reference_path)
            _atomic_save(delta, delta_path)
        if debug_output_dir is not None:
            debug_dir = debug_output_dir / split_name
            _atomic_save(item["e_pt"], debug_dir / "w2v2_pt" / f"{utt_id}.pt")
            _atomic_save(item["e_ft"], debug_dir / "w2v2_ft" / f"{utt_id}.pt")
            _atomic_save(e_ref, debug_dir / "wavlm_ft" / f"{utt_id}.pt")
            _atomic_save(delta, debug_dir / "delta" / f"{utt_id}.pt")

    report = {
        "manifest": str(manifest_path),
        "split": split_name,
        "layer": layer,
        "layer_name": layer_name,
        "utterances": len(selected),
        "start_index": start_index,
        "dry_run": dry_run,
        "min_frames": min(frame_lengths),
        "max_frames": max(frame_lengths),
        "mean_delta_abs": sum(delta_means) / len(delta_means),
        "cache_root": None if dry_run else str(output_root),
    }
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        (report_path or (output_root / split_name / "extraction_report.json")).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="TSV or JSONL manifest")
    parser.add_argument("--output-root", type=Path, required=True, help="cache root, e.g. features")
    parser.add_argument("--wavlm-ft", required=True)
    parser.add_argument("--w2v2-ft", required=True)
    parser.add_argument("--w2v2-pt", required=True)
    parser.add_argument("--layer", type=int, default=-1, help="hidden-state index; -1 is final")
    parser.add_argument("--split", default=None, help="output split name; defaults to manifest stem")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="run QA without writing final cache")
    parser.add_argument("--debug-output-dir", type=Path, default=None, help="optionally save all four debug streams")
    parser.add_argument("--report-path", type=Path, default=None, help="override report path for parallel shard runs")
    args = parser.parse_args()
    report = extract_split(
        manifest_path=args.manifest,
        output_root=args.output_root,
        wavlm_ft_id=args.wavlm_ft,
        w2v2_ft_id=args.w2v2_ft,
        w2v2_pt_id=args.w2v2_pt,
        layer=args.layer,
        device_name=args.device,
        split=args.split,
        start_index=args.start_index,
        max_utterances=args.max_utterances,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        debug_output_dir=args.debug_output_dir,
        report_path=args.report_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
