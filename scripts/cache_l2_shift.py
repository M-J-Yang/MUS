#!/usr/bin/env python3
"""Cache E0, Eft, and Delta for one corrected Fold 0 W2V2 split.

Both models receive the same unpadded waveform through the same pretrained
processor. The cache stores the final encoder representation before the CTC
head, so later reconstruction and pruning never train another head.
"""

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

from usde.ctc import load_audio, make_processor, read_records, resolve_audio_path  # noqa: E402
from usde.model import load_ctc_model  # noqa: E402
from usde.shift import reconstruction_error, validate_shift_tensors  # noqa: E402


PROTOCOL = "single_model_finetuning_shift_cache_v1"


def _atomic_save(value: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value.detach().cpu().contiguous(), temporary)
    temporary.replace(path)


def _select_hidden(output: Any, layer: int) -> torch.Tensor:
    states = output.hidden_states
    if states is None or not len(states):
        raise ValueError("model returned no hidden states")
    index = layer if layer >= 0 else len(states) + layer
    if index < 0 or index >= len(states):
        raise ValueError(f"requested layer {layer}, but model returned {len(states)} hidden states")
    state = states[index].squeeze(0)
    if state.ndim != 2:
        raise ValueError(f"expected hidden state [T,D], got {tuple(state.shape)}")
    return state.to(dtype=torch.float32, device="cpu").contiguous()


def _encode(processor: Any, waveform: torch.Tensor, sample_rate: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = processor(
        waveform.numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}


def _hidden(
    model: torch.nn.Module,
    processor: Any,
    waveform: torch.Tensor,
    sample_rate: int,
    layer: int,
    device: torch.device,
) -> torch.Tensor:
    inputs = _encode(processor, waveform, sample_rate, device)
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True, return_dict=True)
    return _select_hidden(output, layer)


def _valid_cache(paths: dict[str, Path], expected_dim: int | None) -> tuple[bool, dict[str, Any] | None]:
    try:
        tensors = {
            stream: torch.load(path, map_location="cpu", weights_only=True)
            for stream, path in paths.items()
        }
        validate_shift_tensors(tensors["e0"], tensors["eft"], tensors["delta"], expected_dim)
        report = reconstruction_error(tensors["e0"], tensors["eft"], tensors["delta"])
        return bool(report["allclose"]), report
    except (OSError, RuntimeError, EOFError, TypeError, ValueError, KeyError):
        return False, None


def cache_split(args: argparse.Namespace) -> dict[str, Any]:
    if args.start_index < 0:
        raise ValueError("start-index must be non-negative")
    if args.max_utterances is not None and args.max_utterances < 1:
        raise ValueError("max-utterances must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")

    records = read_records(args.manifest)
    selected = records[args.start_index:]
    if args.max_utterances is not None:
        selected = selected[: args.max_utterances]
    if not selected:
        raise ValueError("selected extraction range is empty")

    processor0 = make_processor(args.pretrained_model, None)
    processor_ft = make_processor(args.fine_tuned_model, None)
    if processor0.tokenizer.get_vocab() != processor_ft.tokenizer.get_vocab():
        raise ValueError("pretrained and fine-tuned processors do not have identical vocabularies")
    if int(processor0.feature_extractor.sampling_rate) != 16000:
        raise ValueError("shift extraction requires a 16-kHz pretrained processor")

    model0 = load_ctc_model(args.pretrained_model).to(device).eval()
    model_ft = load_ctc_model(args.fine_tuned_model).to(device).eval()
    for model in (model0, model_ft):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    dim0 = int(getattr(model0.config, "hidden_size", 0))
    dim_ft = int(getattr(model_ft.config, "hidden_size", 0))
    if dim0 != dim_ft or dim0 < 1:
        raise ValueError(f"hidden dimensions differ: pretrained={dim0}, fine_tuned={dim_ft}")
    if int(getattr(model0.config, "vocab_size", 0)) != int(getattr(model_ft.config, "vocab_size", 0)):
        raise ValueError("pretrained and fine-tuned CTC heads have different vocabulary sizes")

    split_root = args.output_root / args.split
    stream_dirs = {stream: split_root / stream for stream in ("e0", "eft", "delta")}
    frame_lengths: list[int] = []
    delta_means: list[float] = []
    identity_max_errors: list[float] = []
    skipped = 0
    for position, row in enumerate(selected, start=1):
        utt_id = str(row["utt_id"])
        if Path(utt_id).name != utt_id:
            raise ValueError(f"{utt_id}: utt_id must not contain path separators")
        paths = {stream: stream_dirs[stream] / f"{utt_id}.pt" for stream in stream_dirs}
        if args.skip_existing and all(path.is_file() for path in paths.values()):
            valid, identity = _valid_cache(paths, dim0)
            if valid and identity is not None:
                frame_lengths.append(int(torch.load(paths["e0"], map_location="cpu", weights_only=True).shape[0]))
                delta_means.append(float(torch.load(paths["delta"], map_location="cpu", weights_only=True).abs().mean()))
                identity_max_errors.append(float(identity["max_abs_error"]))
                skipped += 1
                continue

        waveform, sample_rate = load_audio(resolve_audio_path(row["wav_path"], args.manifest))
        e0 = _hidden(model0, processor0, waveform, sample_rate, args.layer, device)
        eft = _hidden(model_ft, processor_ft, waveform, sample_rate, args.layer, device)
        delta = eft - e0
        validate_shift_tensors(e0, eft, delta, dim0)
        if float(delta.abs().sum()) <= 0.0:
            raise ValueError(f"{utt_id}: Delta is identically zero")
        identity = reconstruction_error(e0, eft, delta, args.identity_atol, args.identity_rtol)
        if not identity["allclose"]:
            raise ValueError(f"{utt_id}: E0 + Delta != Eft: {identity}")
        for stream, value in (("e0", e0), ("eft", eft), ("delta", delta)):
            _atomic_save(value, paths[stream])
        frame_lengths.append(int(e0.shape[0]))
        delta_means.append(float(delta.abs().mean()))
        identity_max_errors.append(float(identity["max_abs_error"]))
        if position == 1 or position % args.log_every == 0 or position == len(selected):
            print({"processed": position, "total": len(selected), "utt_id": utt_id, "frames": int(e0.shape[0])}, flush=True)

    report = {
        "protocol": PROTOCOL,
        "manifest": str(args.manifest),
        "split": args.split,
        "pretrained_model": args.pretrained_model,
        "fine_tuned_model": args.fine_tuned_model,
        "layer": args.layer,
        "hidden_dim": dim0,
        "utterances": len(selected),
        "written": len(selected) - skipped,
        "skipped_valid": skipped,
        "start_index": args.start_index,
        "min_frames": min(frame_lengths),
        "max_frames": max(frame_lengths),
        "mean_delta_abs": sum(delta_means) / len(delta_means),
        "max_identity_abs_error": max(identity_max_errors),
        "identity_atol": args.identity_atol,
        "identity_rtol": args.identity_rtol,
        "cache_root": str(args.output_root),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (split_root / "extraction_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", required=True, help="cache split name, e.g. train_utility, dev, or test")
    parser.add_argument("--pretrained-model", required=True, help="facebook/wav2vec2-large-960h or local snapshot")
    parser.add_argument("--fine-tuned-model", required=True, help="saved corrected Fold 0 checkpoint")
    parser.add_argument("--layer", type=int, default=-1, help="hidden-state index; -1 is the final layer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--identity-atol", type=float, default=1e-5)
    parser.add_argument("--identity-rtol", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.log_every < 1:
        raise ValueError("log-every must be positive")
    cache_split(args)


if __name__ == "__main__":
    main()
