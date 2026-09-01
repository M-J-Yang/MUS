"""Deterministic final-layer SSL feature extraction for Step 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel

from usde.manifest import load_jsonl


def load_audio(path: str, expected_sample_rate: int = 16000) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != expected_sample_rate:
        raise ValueError(f"{path}: expected {expected_sample_rate} Hz, got {sample_rate} Hz")
    if audio.shape[1] != 1:
        raise ValueError(f"{path}: expected mono audio, got {audio.shape[1]} channels")
    return audio[:, 0]


def hidden_states(model: AutoModel, processor: AutoFeatureExtractor, audio: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    inputs = processor(audio, sampling_rate=processor.sampling_rate, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True, return_dict=True)
    return tuple(state.squeeze(0).to(dtype=torch.float32).cpu().contiguous() for state in output.hidden_states)


def hidden_state(model: AutoModel, processor: AutoFeatureExtractor, audio: np.ndarray, layer: int, device: torch.device) -> torch.Tensor:
    states = hidden_states(model, processor, audio, device)
    if layer < 0 or layer >= len(states):
        raise ValueError(f"requested transformer layer {layer}, but model returned {len(states)} hidden states")
    return states[layer]


def _load_checkpoint_state(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load a local checkpoint state dict, including the CTC wrapper keys."""
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        return dict(load_file(str(safetensors_path), device="cpu"))
    checkpoint = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{model_dir}: unsupported checkpoint format")
    return checkpoint


def _load_legacy_compatible_encoder(model_dir: Path) -> AutoModel:
    """Load checkpoints written by newer PyTorch/Transformers runtimes.

    The local fine-tuned checkpoints store weight normalization as
    ``parametrizations.weight.original{0,1}``, whereas the pinned Transformers
    runtime exposes the same parameters as ``weight_g`` and ``weight_v``.
    Loading through ``from_pretrained`` otherwise reports missing parameters
    and initializes them to zero, which can make every hidden state NaN.
    """
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModel.from_config(config)
    model_keys = set(model.state_dict())
    prefix = f"{model.base_model_prefix}."
    state: dict[str, torch.Tensor] = {}
    for key, value in _load_checkpoint_state(model_dir).items():
        if key.startswith(prefix):
            key = key[len(prefix):]
        candidates = [key]
        if ".parametrizations.weight.original0" in key:
            candidates.append(key.replace(".parametrizations.weight.original0", ".weight_g"))
        if ".parametrizations.weight.original1" in key:
            candidates.append(key.replace(".parametrizations.weight.original1", ".weight_v"))
        if ".weight_g" in key:
            candidates.append(key.replace(".weight_g", ".parametrizations.weight.original0"))
        if ".weight_v" in key:
            candidates.append(key.replace(".weight_v", ".parametrizations.weight.original1"))
        target = next((candidate for candidate in candidates if candidate in model_keys), None)
        if target is not None:
            state[target] = value
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{model_dir}: incompatible base checkpoint keys; " 
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return model


def load_encoder(model_id: str, device: torch.device) -> tuple[AutoModel, AutoFeatureExtractor]:
    processor = AutoFeatureExtractor.from_pretrained(model_id)
    model_path = Path(model_id)
    if model_path.is_dir() and (model_path / "model.safetensors").is_file():
        checkpoint_keys = _load_checkpoint_state(model_path)
        has_new_weight_norm_keys = any(
            ".parametrizations.weight.original" in key for key in checkpoint_keys
        )
    else:
        has_new_weight_norm_keys = False
    model = (
        _load_legacy_compatible_encoder(model_path)
        if has_new_weight_norm_keys
        else AutoModel.from_pretrained(model_id)
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Fail at extraction time instead of writing corrupt feature artifacts.
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError(f"{model_id}: encoder contains non-finite parameters")
    return model, processor


def extract_split(
    manifest_path: Path,
    output_dir: Path,
    wavlm_ft_id: str,
    w2v2_ft_id: str,
    w2v2_pt_id: str,
    layer: int,
    device: str,
    skip_existing: bool = False,
    start_index: int = 0,
    max_utterances: int | None = None,
) -> dict[str, Any]:
    """Write one self-contained tensor record per utterance.

    The three streams are deliberately extracted on the same unpadded waveform.
    A frame mismatch is a hard error: Step 1 never inserts an alignment module
    or silently truncates a stream.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    wavlm, wavlm_processor = load_encoder(wavlm_ft_id, torch_device)
    w2v2_ft, w2v2_processor = load_encoder(w2v2_ft_id, torch_device)
    w2v2_pt, w2v2_pt_processor = load_encoder(w2v2_pt_id, torch_device)
    if {wavlm_processor.sampling_rate, w2v2_processor.sampling_rate, w2v2_pt_processor.sampling_rate} != {16000}:
        raise ValueError("Step 1 requires all processors to use 16-kHz audio")

    records = load_jsonl(manifest_path)
    if start_index < 0:
        raise ValueError("start-index must be non-negative")
    records = records[start_index:] if max_utterances is None else records[start_index : start_index + max_utterances]
    if not records:
        raise ValueError("selected extraction range is empty")
    frame_lengths: list[int] = []
    for index, record in enumerate(records, start=1):
        utterance_id = record["utt_id"]
        output_path = output_dir / f"{utterance_id}.pt"
        if skip_existing and output_path.is_file():
            try:
                item = torch.load(output_path, map_location="cpu")
                reference, delta = item.get("wavlm_ft"), item.get("delta")
                valid = reference is not None and delta is not None and reference.ndim == 2 and reference.shape == delta.shape and reference.shape[1] == 1024 and torch.isfinite(reference).all() and torch.isfinite(delta).all()
            except (OSError, RuntimeError, EOFError, ValueError):
                valid = False
            if valid:
                frame_lengths.append(reference.shape[0])
                continue
        audio = load_audio(record["audio_path"])
        wavlm_ft = hidden_state(wavlm, wavlm_processor, audio, layer, torch_device)
        w2v2_ft_state = hidden_state(w2v2_ft, w2v2_processor, audio, layer, torch_device)
        w2v2_pt_state = hidden_state(w2v2_pt, w2v2_pt_processor, audio, layer, torch_device)
        lengths = {wavlm_ft.shape[0], w2v2_ft_state.shape[0], w2v2_pt_state.shape[0]}
        dimensions = {wavlm_ft.shape[1], w2v2_ft_state.shape[1], w2v2_pt_state.shape[1]}
        if len(lengths) != 1 or dimensions != {1024}:
            raise ValueError(f"{utterance_id}: frame/dimension mismatch: lengths={lengths}, dims={dimensions}")
        if not torch.isfinite(wavlm_ft).all() or not torch.isfinite(w2v2_ft_state).all() or not torch.isfinite(w2v2_pt_state).all():
            raise ValueError(f"{utterance_id}: encoder produced non-finite hidden states")
        delta = w2v2_ft_state - w2v2_pt_state
        if not torch.allclose(delta, w2v2_ft_state - w2v2_pt_state, atol=0.0, rtol=0.0):
            raise AssertionError(f"{utterance_id}: delta invariant failed")
        payload = {
            "utt_id": utterance_id,
            "layer": layer,
            "wavlm_ft": wavlm_ft,
            "w2v2_ft": w2v2_ft_state,
            "w2v2_pt": w2v2_pt_state,
            "delta": delta,
        }
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(output_path)
        frame_lengths.append(wavlm_ft.shape[0])
        if index == 1 or index % 100 == 0:
            print({"processed": index, "total": len(records), "utt_id": utterance_id}, flush=True)
    return {"utterances": len(records), "min_frames": min(frame_lengths), "max_frames": max(frame_lengths), "layer": layer}
