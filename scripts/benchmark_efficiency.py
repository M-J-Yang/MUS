#!/usr/bin/env python3
"""Benchmark deployment cost for the frozen-head selective-reuse experiments.

The benchmark deliberately separates three paths:

``full``
    One adapted encoder followed by its CTC head.
``selective_dual_<method>``
    Reference encoder and adapted encoder, followed by a 50% representation
    mask and the adapted CTC head.  DADS, Magnitude, and Random are measured
    with the same graph; only the coordinate mask changes.
``selective_cached_<method>``
    The two final-layer representations are already cached; only composition
    and the adapted CTC head are timed.  The three methods again use the same
    graph.  This is useful for offline reuse, but it is not an end-to-end
    speech inference result.

FLOPs are analytical estimates from the executed module shapes.  Timing and
GPU memory are measured on the supplied hardware.  The script uses the test
manifest only to obtain representative utterance lengths; it does not decode
or alter any paper WER result.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from usde.ctc import load_audio, make_processor, read_records, resolve_audio_path  # noqa: E402
from usde.model import load_ctc_model  # noqa: E402


PROTOCOL = "selective_reuse_efficiency_benchmark_v1"
DEFAULT_RETENTION = 50


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[name]


def bytes_per_dtype(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def move_inputs(
    inputs: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        target_dtype = dtype if value.is_floating_point() else value.dtype
        moved[key] = value.to(device=device, dtype=target_dtype, non_blocking=True)
    return moved


def base_model(model: nn.Module) -> nn.Module:
    getter = getattr(model, "get_base_model", None)
    if getter is not None:
        return getter()
    base = getattr(model, "base_model", None)
    if isinstance(base, nn.Module):
        return base
    raise TypeError(f"{type(model).__name__} does not expose a base model")


def model_parameter_breakdown(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    encoder = sum(parameter.numel() for parameter in base_model(model).parameters())
    head_module = getattr(model, "lm_head", None)
    head = sum(parameter.numel() for parameter in head_module.parameters()) if head_module else 0
    return {
        "total": int(total),
        "encoder": int(encoder),
        "ctc_head": int(head),
        "other": int(total - encoder - head),
    }


def load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        for key in ("ranking", "tensor", "utility", "features"):
            if isinstance(value.get(key), torch.Tensor):
                value = value[key]
                break
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path}: expected a tensor")
    return value


def load_keep_indices(path: Path | None, dimension: int, retention: int) -> torch.Tensor:
    count = max(1, round(dimension * retention / 100.0))
    if path is None:
        return torch.arange(count, dtype=torch.long)
    ranking = load_tensor(path).to(dtype=torch.long, device="cpu")
    if (
        ranking.ndim != 1
        or ranking.numel() != dimension
        or torch.unique(ranking).numel() != dimension
        or int(ranking.min()) < 0
        or int(ranking.max()) >= dimension
    ):
        raise ValueError(f"{path}: expected a permutation of {dimension} coordinates")
    return ranking[:count].contiguous()


def load_inputs(
    manifest: Path,
    processor: Any,
    num_samples: int,
    seed: int,
) -> tuple[list[dict[str, torch.Tensor]], list[float], list[str]]:
    records = read_records(manifest)
    if not records:
        raise ValueError(f"{manifest}: no records")
    durations: list[float] = []
    paths: list[Path] = []
    for row in records:
        path = resolve_audio_path(row["wav_path"], manifest)
        waveform, sample_rate = load_audio(path)
        durations.append(float(waveform.numel()) / float(sample_rate))
        paths.append(path)
    order = sorted(range(len(records)), key=lambda index: durations[index])
    target = durations[order[len(order) // 2]]
    # Include the duration-median utterance and then spread the remaining
    # examples across the duration range.  This avoids a speed result driven
    # by one unusually short or long sentence.
    median_position = len(order) // 2
    positions = {median_position}
    if num_samples > 1:
        positions.update(
            round(index * (len(order) - 1) / (num_samples - 1))
            for index in range(num_samples)
        )
    positions = sorted(positions)[: max(1, min(num_samples, len(order)))]
    selected = [order[position] for position in positions]

    inputs: list[dict[str, torch.Tensor]] = []
    selected_durations: list[float] = []
    selected_ids: list[str] = []
    for index in selected:
        waveform, sample_rate = load_audio(paths[index])
        encoded = processor.feature_extractor(
            waveform.numpy(),
            sampling_rate=sample_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        values = encoded["input_values"]
        mask = encoded.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(values, dtype=torch.long)
        inputs.append({"input_values": values, "attention_mask": mask})
        selected_durations.append(durations[index])
        selected_ids.append(str(records[index]["utt_id"]))
    return inputs, selected_durations, selected_ids


def _shape_from_base_output(output: Any) -> tuple[int, int, int]:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, (tuple, list)):
        hidden = output[0]
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise RuntimeError("could not recover [batch, frames, hidden] from base-model output")
    return tuple(int(value) for value in hidden.shape)


def estimate_module_flops(module: nn.Module, inputs: dict[str, torch.Tensor]) -> dict[str, int]:
    """Estimate Conv1d/Linear and attention matmul FLOPs for one forward."""

    conv_flops = 0
    linear_flops = 0
    base_shape: tuple[int, int, int] | None = None
    hooks: list[Any] = []

    def conv_hook(layer: nn.Conv1d) -> Callable[..., None]:
        def hook(_module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            nonlocal conv_flops
            if not isinstance(output, torch.Tensor) or not args or not isinstance(args[0], torch.Tensor):
                return
            batch, channels, frames = (int(value) for value in output.shape)
            conv_flops += 2 * batch * channels * frames * layer.kernel_size[0] * (
                layer.in_channels // layer.groups
            )
        return hook

    def linear_hook(layer: nn.Linear) -> Callable[..., None]:
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            nonlocal linear_flops
            if isinstance(output, torch.Tensor):
                linear_flops += 2 * int(output.numel()) * int(layer.in_features)
        return hook

    def root_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        nonlocal base_shape
        base_shape = _shape_from_base_output(output)

    hooks.append(module.register_forward_hook(root_hook))
    for layer in module.modules():
        if isinstance(layer, nn.Conv1d):
            hooks.append(layer.register_forward_hook(conv_hook(layer)))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_hook(layer)))
    with torch.inference_mode():
        module(**inputs, return_dict=True)
    for hook in hooks:
        hook.remove()
    if base_shape is None:
        raise RuntimeError("base-model forward hook did not capture an output")
    batch, frames, hidden = base_shape
    config = getattr(module, "config", None)
    layers = int(getattr(config, "num_hidden_layers", 0))
    attention = 4 * batch * layers * frames * frames * hidden
    return {
        "conv": int(conv_flops),
        "linear": int(linear_flops),
        "attention_matmul": int(attention),
        "total": int(conv_flops + linear_flops + attention),
        "batch": batch,
        "frames": frames,
        "hidden": hidden,
    }


def head_flops(model: nn.Module, frames: int, batch: int = 1) -> int:
    head = getattr(model, "lm_head", None)
    if not isinstance(head, nn.Linear):
        raise TypeError("expected a linear CTC lm_head")
    return int(2 * batch * frames * head.in_features * head.out_features)


def prepare_features(
    reference: nn.Module,
    adapted: nn.Module,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        reference_features = base_model(reference)(**inputs, return_dict=True).last_hidden_state
        adapted_features = base_model(adapted)(**inputs, return_dict=True).last_hidden_state
    return reference_features, adapted_features


def compose_features(
    reference_features: torch.Tensor,
    adapted_features: torch.Tensor,
    keep: torch.Tensor,
) -> torch.Tensor:
    mixed = reference_features.clone()
    mixed[..., keep] = adapted_features[..., keep]
    return mixed


def run_timed(
    mode: str,
    adapted: nn.Module,
    reference: nn.Module | None,
    samples: list[dict[str, torch.Tensor]],
    durations: list[float],
    keep: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    cached_features: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, float | int]:
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup must be non-negative")
    gpu_samples = [move_inputs(sample, device, dtype) for sample in samples]
    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    if mode == "selective_cached":
        if cached_features is not None:
            cached = cached_features
        elif reference is not None:
            with torch.inference_mode():
                for sample in gpu_samples:
                    cached.append(prepare_features(reference, adapted, sample))
        else:
            raise ValueError("selective_cached requires cached features or a reference model")

    def forward(index: int) -> None:
        if mode == "full":
            adapted(**gpu_samples[index], return_dict=True)
        elif mode == "selective_dual":
            if reference is None:
                raise ValueError("selective_dual requires a reference model")
            reference_features, adapted_features = prepare_features(
                reference, adapted, gpu_samples[index]
            )
            adapted.lm_head(compose_features(reference_features, adapted_features, keep))
        elif mode == "selective_cached":
            reference_features, adapted_features = cached[index]
            adapted.lm_head(compose_features(reference_features, adapted_features, keep))
        else:
            raise ValueError(f"unknown mode: {mode}")

    with torch.inference_mode():
        for _ in range(warmup):
            for index in range(len(gpu_samples)):
                forward(index)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(repeats):
            for index in range(len(gpu_samples)):
                forward(index)
        synchronize(device)
        elapsed = time.perf_counter() - start

    total_audio = float(sum(durations)) * repeats
    return {
        "samples": len(samples),
        "repeats": repeats,
        "elapsed_seconds": elapsed,
        "milliseconds_per_utterance": 1000.0 * elapsed / (len(samples) * repeats),
        "audio_seconds_per_second": total_audio / elapsed if elapsed else math.inf,
        "real_time_factor": elapsed / total_audio if total_audio else math.inf,
        "peak_allocated_gib": (
            float(torch.cuda.max_memory_allocated(device)) / (1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "peak_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device)) / (1024**3)
            if device.type == "cuda"
            else 0.0
        ),
    }


def memory_gib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.memory_allocated(device)) / (1024**3)


def format_number(value: float) -> str:
    return f"{value:.3f}"


def benchmark_one(
    entry: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    num_samples: int,
    warmup: int,
    repeats: int,
    retention: int,
    seed: int,
) -> dict[str, Any]:
    adapted_path = project_path(entry["adapted"])
    reference_path = project_path(entry["reference"]) if entry.get("reference") else None
    manifest = project_path(entry["manifest"])
    raw_rankings = entry.get("rankings")
    if raw_rankings is None:
        raw_rankings = {"dads": entry.get("ranking")}
    if not isinstance(raw_rankings, dict) or not raw_rankings:
        raise ValueError(f"{entry['name']}: rankings must be a non-empty mapping")
    ranking_paths = {
        str(name): project_path(value) if value else None
        for name, value in raw_rankings.items()
    }

    processor = make_processor(str(adapted_path), None)
    inputs, durations, utt_ids = load_inputs(manifest, processor, num_samples, seed)
    total_duration = sum(durations)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    adapted = load_ctc_model(adapted_path).to(device=device, dtype=dtype).eval()
    adapted_params = model_parameter_breakdown(adapted)
    load_memory_full = memory_gib(device)
    hidden_dim = int(getattr(adapted.config, "hidden_size", 0))
    keeps = {
        name: load_keep_indices(path, hidden_dim, retention).to(device)
        for name, path in ranking_paths.items()
    }
    default_keep = next(iter(keeps.values()))

    representative_index = min(range(len(durations)), key=lambda index: abs(durations[index] - sorted(durations)[len(durations) // 2]))
    representative_inputs = move_inputs(inputs[representative_index], device, dtype)
    full_base_flops = estimate_module_flops(base_model(adapted), representative_inputs)
    representative_frames = full_base_flops["frames"]
    full_flops = full_base_flops["total"] + head_flops(adapted, representative_frames, full_base_flops["batch"])
    full_timing = run_timed(
        "full", adapted, None, inputs, durations, default_keep, device, dtype, warmup, repeats
    )
    full_row: dict[str, Any] = {
        "mode": "full",
        "label": "full adapted encoder + CTC head",
        "parameters": adapted_params,
        "loaded_parameter_memory_gib": load_memory_full,
        "flops": {
            "representative_frames": representative_frames,
            "representative_audio_seconds": durations[representative_index],
            "forward_flops": int(full_flops),
            "forward_gflops": full_flops / 1e9,
            "gflops_per_audio_second": full_flops / 1e9 / durations[representative_index],
        },
        "timing": full_timing,
    }

    if reference_path is None:
        result = {
            "name": entry["name"],
            "adapted": str(adapted_path),
            "reference": None,
            "manifest": str(manifest),
            "dtype": dtype_name,
            "device": str(device),
            "retention_percent": retention,
            "samples": {"count": len(inputs), "utterance_ids": utt_ids, "audio_seconds": total_duration},
            "rows": [full_row],
        }
        del adapted
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    # The reference lm_head is not part of the selective deployment path. It
    # is replaced before timing so measured memory follows the actual graph.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    reference = load_ctc_model(reference_path).to(device=device, dtype=dtype).eval()
    reference_head = getattr(reference, "lm_head", None)
    if isinstance(reference_head, nn.Module):
        reference.lm_head = nn.Identity()
        del reference_head
    reference_params = model_parameter_breakdown(reference)
    dual_loaded_memory = memory_gib(device)

    ref_base_flops = estimate_module_flops(base_model(reference), representative_inputs)
    mask_flops = 3 * ref_base_flops["batch"] * ref_base_flops["frames"] * hidden_dim
    dual_flops = (
        full_base_flops["total"]
        + ref_base_flops["total"]
        + head_flops(adapted, representative_frames, full_base_flops["batch"])
        + mask_flops
    )
    selective_parameters = {
        "total": adapted_params["total"] + reference_params["encoder"],
        "encoder": adapted_params["encoder"] + reference_params["encoder"],
        "ctc_head": adapted_params["ctc_head"],
        "other": adapted_params["other"],
        "reference_encoder_only": reference_params["encoder"],
    }
    selective_flops = {
        "representative_frames": representative_frames,
        "representative_audio_seconds": durations[representative_index],
        "forward_flops": int(dual_flops),
        "forward_gflops": dual_flops / 1e9,
        "gflops_per_audio_second": dual_flops / 1e9 / durations[representative_index],
        "mask_composition_flops": int(mask_flops),
    }
    dual_rows: list[dict[str, Any]] = []
    for name, keep in keeps.items():
        dual_timing = run_timed(
            "selective_dual", adapted, reference, inputs, durations, keep,
            device, dtype, warmup, repeats,
        )
        dual_rows.append({
            "mode": f"selective_dual_{name}",
            "label": f"{name}: reference + adapted encoders + {retention}% mask + adapted CTC head",
            "parameters": selective_parameters,
            "loaded_parameter_memory_gib": dual_loaded_memory,
            "flops": selective_flops,
            "timing": dual_timing,
        })

    # Cache one representation pair per timed input.  Its memory is explicit
    # in the output because this path trades compute for storage.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.inference_mode():
        for sample in inputs:
            cached.append(prepare_features(reference, adapted, move_inputs(sample, device, dtype)))
    cache_bytes = sum((left.numel() + right.numel()) * left.element_size() for left, right in cached)
    cached_head = nn.Module()
    cached_head.add_module("lm_head", adapted.lm_head)
    del reference, adapted
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    cached_flops = head_flops(cached_head, representative_frames, full_base_flops["batch"]) + mask_flops
    cached_parameters = {
        "total": adapted_params["ctc_head"],
        "encoder": 0,
        "ctc_head": adapted_params["ctc_head"],
        "other": 0,
    }
    cached_flop_record = {
        "representative_frames": representative_frames,
        "representative_audio_seconds": durations[representative_index],
        "forward_flops": int(cached_flops),
        "forward_gflops": cached_flops / 1e9,
        "gflops_per_audio_second": cached_flops / 1e9 / durations[representative_index],
        "mask_composition_flops": int(mask_flops),
    }
    cached_rows: list[dict[str, Any]] = []
    for name, keep in keeps.items():
        cached_timing = run_timed(
            "selective_cached", cached_head, None, inputs, durations, keep,
            device, dtype, warmup, repeats, cached_features=cached,
        )
        cached_rows.append({
            "mode": f"selective_cached_{name}",
            "label": f"{name}: cached representations + {retention}% mask + adapted CTC head (offline)",
            "parameters": cached_parameters,
            "loaded_parameter_memory_gib": memory_gib(device),
            "feature_cache_bytes": int(cache_bytes),
            "flops": cached_flop_record,
            "timing": cached_timing,
        })
    result = {
        "name": entry["name"],
        "adapted": str(adapted_path),
        "reference": str(reference_path),
        "manifest": str(manifest),
        "rankings": {name: str(path) if path else None for name, path in ranking_paths.items()},
        "dtype": dtype_name,
        "device": str(device),
        "retention_percent": retention,
        "hidden_dim": hidden_dim,
        "retained_coordinates": {name: int(keep.numel()) for name, keep in keeps.items()},
        "samples": {"count": len(inputs), "utterance_ids": utt_ids, "audio_seconds": total_duration},
        "rows": [full_row, *dual_rows, *cached_rows],
    }
    del cached, cached_head
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Selective-reuse efficiency benchmark",
        "",
        f"Protocol: `{result['protocol']}`  ",
        f"Device: `{result['device']}`; dtype: `{result['dtype']}`; retention: `{result['retention_percent']}%`  ",
        "",
        "`full` is one adapted encoder plus its CTC head. The DADS, Magnitude, and Random selective rows use the same online graph and differ only in mask contents; their structural costs are therefore identical. Cached rows exclude encoder computation and are offline post-processing measurements, not end-to-end speed claims.",
        "",
        "| Experiment | Path | Parameters (M) | GFLOPs / representative utterance | GFLOPs / audio s | ms / utt | audio s / s | RTF | Peak allocated GiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in result["experiments"]:
        for row in experiment["rows"]:
            params = row["parameters"]["total"] / 1e6
            flops = row["flops"]
            timing = row["timing"]
            lines.append(
                f"| {experiment['name']} | {row['mode']} | {params:.2f} | "
                f"{flops['forward_gflops']:.2f} | {flops['gflops_per_audio_second']:.2f} | "
                f"{timing['milliseconds_per_utterance']:.2f} | {timing['audio_seconds_per_second']:.2f} | "
                f"{timing['real_time_factor']:.3f} | {timing['peak_allocated_gib']:.2f} |"
            )
    lines += [
        "",
        "Interpretation: DADS changes selection quality, not the computation graph. The current 1024-dimensional mask does not structurally prune the encoder, so DADS has no parameter/FLOP/memory advantage over matched Magnitude or Random masks. The online selective graph has two encoder forwards; cached rows describe only a separate offline post-processing stage.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--retention", type=int, default=DEFAULT_RETENTION)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_samples < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("num-samples must be positive; warmup non-negative; repeats positive")
    if not 1 <= args.retention <= 99:
        parser.error("retention must be between 1 and 99")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    entries = config.get("experiments", config.get("models"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{config_path}: expected a non-empty experiments list")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    if args.dtype != "fp32" and device.type != "cuda":
        raise ValueError("bf16/fp16 benchmark requires CUDA")
    dtype = dtype_from_name(args.dtype)
    experiments = [
        benchmark_one(
            entry,
            device,
            dtype,
            args.dtype,
            args.num_samples,
            args.warmup,
            args.repeats,
            args.retention,
            args.seed,
        )
        for entry in entries
    ]
    result = {
        "protocol": PROTOCOL,
        "config": str(config_path),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": args.dtype,
        "batch_size": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "retention_percent": args.retention,
        "seed": args.seed,
        "experiments": experiments,
        "limitations": [
            "FLOPs are analytical Conv1d/Linear/attention-matmul estimates and omit small elementwise and normalization terms.",
            "Timing excludes audio file I/O and feature-extractor preprocessing; it measures model inference on preloaded waveforms.",
            "Cached rows are offline feature reuse and must not be presented as end-to-end ASR latency.",
            "At matched retention, DADS, Magnitude, and Random have the same parameter count, FLOPs, memory graph, and asymptotic latency.",
            "Parameter count and online FLOPs do not decrease under the current unstructured coordinate mask.",
        ],
    }
    json_path = args.output_dir / "efficiency_metrics.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, args.output_dir / "efficiency_metrics.md")
    print(json.dumps({"json": str(json_path), "markdown": str(args.output_dir / 'efficiency_metrics.md')}, sort_keys=True))


if __name__ == "__main__":
    main()
