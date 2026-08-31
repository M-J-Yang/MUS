#!/usr/bin/env python3
"""Headwise Selective Attention (HSA) merge for Whisper checkpoints.

This script is self-contained and does not depend on the training/evaluation
helpers in this repository. It implements the headwise variant used in the
extended compositional domain-adaptation work.

Merge semantics
---------------
Let B be the pretrained base model, M1 the primary/anchor fine-tuned model,
and M2 the secondary fine-tuned model. Their task vectors are:

    tv1 = M1 - B
    tv2 = M2 - B

For the top-K encoder self-attention Q/K/V heads selected by task-vector
magnitude in encoder layer i, HSA applies:

    B + lambda_i * tv1 + (1 - lambda_i) * tv2

where lambda_i = lambda ** (alpha * (i + 1) / L) for Hugging Face's
zero-based encoder layer index i. The +1 maps the implementation index to the
paper's 1-based layer position, so the final encoder layer reaches
lambda ** alpha.

For encoder self-attention Q/K/V heads outside the top-K set, this script
keeps the primary model's task vector:

    B + tv1

This is intentionally different from the older experimental script, where
non-selected heads were masked to zero and therefore reverted to the base
model. The HSA selection/interpolation is applied only to encoder-layer
query/key/value projections; all other parameters keep the primary model's
task vector.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import torch
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


HSA_TARGET_PATTERNS = (
    r".*\.encoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)\.(weight|bias)$",
)
ENCODER_LAYER_PATTERN = re.compile(r".*\.encoder\.layers\.(\d+)\.")


def normalize_fraction(value: float) -> float:
    """Accept either 0.6 or 60 for 60%."""
    value = float(value)
    if value > 1.0:
        value = value / 100.0
    if value < 0.0 or value > 1.0:
        raise ValueError(f"k-percent must be in [0, 1] or [0, 100], got {value}")
    return value


def parse_lambdas(single_lambda: float | None, lambda_list: str | None) -> list[float]:
    if lambda_list:
        values = [float(item.strip()) for item in lambda_list.split(",") if item.strip()]
    elif single_lambda is not None:
        values = [float(single_lambda)]
    else:
        values = [0.5]

    for value in values:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"lambda must be in [0, 1], got {value}")
    return values


def validate_alpha(value: float) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"alpha must be non-negative, got {value}")
    return value


def lambda_tag(value: float) -> str:
    scaled = value * 10.0
    if math.isclose(scaled, round(scaled), rel_tol=0.0, abs_tol=1e-8):
        return f"lamda_{int(round(scaled))}"
    text = f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"lamda_{text}"


def alpha_tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"alpha_{text}"


def is_attention_param(name: str, compiled_patterns: Iterable[re.Pattern]) -> bool:
    return any(pattern.match(name) for pattern in compiled_patterns)


def encoder_layer_index(name: str) -> int | None:
    match = ENCODER_LAYER_PATTERN.match(name)
    if not match:
        return None
    return int(match.group(1))


def effective_lambda(name: str, lamda: float, alpha: float, num_encoder_layers: int) -> float:
    """Layer-wise lambda_i for a zero-based Hugging Face encoder layer index."""
    layer_idx = encoder_layer_index(name)
    if layer_idx is None:
        return float(lamda)
    if num_encoder_layers <= 0:
        raise ValueError("num_encoder_layers must be positive")
    exponent = float(alpha) * (float(layer_idx + 1) / float(num_encoder_layers))
    if float(lamda) == 0.0 and exponent == 0.0:
        return 1.0
    if float(lamda) == 0.0:
        return 0.0
    return float(lamda) ** exponent


def load_processor_and_feature_extractor(path_or_id: str):
    """Load Whisper processor/feature extractor from a model path or its parent."""
    candidates = [path_or_id]
    path = Path(path_or_id)
    if path.exists():
        candidates.append(str(path.parent))

    last_error = None
    feature_extractor = None
    processor = None
    for candidate in candidates:
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(candidate)
            processor = WhisperProcessor.from_pretrained(candidate)
            break
        except Exception as exc:  # Hugging Face raises several config/tokenizer exceptions.
            last_error = exc

    if processor is None or feature_extractor is None:
        raise RuntimeError(f"Could not load processor/feature extractor from {path_or_id}") from last_error

    processor.current_processor = feature_extractor
    processor.feature_extractor = feature_extractor
    return processor, feature_extractor


def load_whisper_model(path_or_id: str, device: str):
    config = AutoConfig.from_pretrained(path_or_id)
    model = WhisperForConditionalGeneration.from_pretrained(path_or_id, config=config).to(device)
    model.eval()
    return model, config


def param_dict(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {name: param for name, param in model.named_parameters()}


def cloned_base_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return {name: param.detach().clone() for name, param in model.named_parameters()}


def infer_num_heads(config, override: int | None) -> int:
    if override is not None:
        return int(override)
    for attr in ("encoder_attention_heads", "decoder_attention_heads", "num_attention_heads"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer number of attention heads; pass --num-heads.")


def infer_encoder_layers(config, override: int | None) -> int:
    if override is not None:
        return int(override)
    for attr in ("encoder_layers", "num_hidden_layers"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer encoder layer count; pass --num-encoder-layers.")


def choose_top_heads(scores: torch.Tensor, k_fraction: float) -> torch.Tensor:
    head_count = scores.numel()
    if head_count == 0 or k_fraction <= 0.0:
        return torch.zeros(head_count, dtype=torch.bool, device=scores.device)
    if k_fraction >= 1.0:
        return torch.ones(head_count, dtype=torch.bool, device=scores.device)

    keep = max(1, int(round(k_fraction * head_count)))
    threshold = torch.topk(scores, keep, largest=True).values.min()
    return scores >= threshold


def head_mask_for_param(
    name: str,
    proxy: torch.Tensor,
    num_heads: int,
    k_fraction: float,
) -> torch.Tensor | None:
    """Build an element mask that selects top-K heads for one encoder Q/K/V tensor."""
    if proxy.ndim == 2:
        out_features, in_features = proxy.shape

        # Encoder q/k/v projections are grouped by output rows. We deliberately
        # do not support out_proj column grouping here because HSA is scoped to
        # encoder-layer query/key/value matrices in the paper.
        if out_features % num_heads != 0:
            raise ValueError(f"{name}: out_features={out_features} not divisible by heads={num_heads}")
        head_dim = out_features // num_heads
        channel_scores = proxy.norm(p=2, dim=1)
        scores = channel_scores.view(num_heads, head_dim).pow(2).sum(dim=1).sqrt()
        chosen = choose_top_heads(scores, k_fraction).to(proxy.dtype)
        row_mask = chosen.view(-1, 1).repeat(1, head_dim).reshape(out_features)
        return row_mask.view(-1, 1).expand_as(proxy)

    if proxy.ndim == 1:
        features = proxy.shape[0]
        if features % num_heads != 0:
            raise ValueError(f"{name}: features={features} not divisible by heads={num_heads}")
        head_dim = features // num_heads
        scores = proxy.abs().view(num_heads, head_dim).pow(2).sum(dim=1).sqrt()
        chosen = choose_top_heads(scores, k_fraction).to(proxy.dtype)
        return chosen.view(-1, 1).repeat(1, head_dim).reshape(features)

    return None


def build_hsa_params(
    base_params: dict[str, torch.Tensor],
    model1_params: dict[str, torch.nn.Parameter],
    model2_params: dict[str, torch.nn.Parameter],
    lamda: float,
    alpha: float,
    num_encoder_layers: int,
    k_fraction: float,
    num_heads: int,
    hsa_target_patterns: Iterable[re.Pattern],
    attention_scale: float,
    non_attention_scale: float,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Return merged parameters and a small accounting summary."""
    merged_params: dict[str, torch.Tensor] = {}
    summary = {
        "hsa_target_tensors": 0,
        "model1_passthrough_tensors": 0,
        "selected_head_elements": 0,
        "total_head_elements": 0,
        "layer_scaled_tensors": 0,
    }

    with torch.no_grad():
        for name, base_value in base_params.items():
            p1 = model1_params.get(name)
            p2 = model2_params.get(name)

            if p1 is None:
                merged_params[name] = base_value.clone()
                continue

            tv1 = p1.detach() - base_value

            if not is_attention_param(name, hsa_target_patterns):
                merged_params[name] = base_value + float(non_attention_scale) * tv1
                summary["model1_passthrough_tensors"] += 1
                continue

            summary["hsa_target_tensors"] += 1
            if p2 is None:
                merged_params[name] = base_value + float(attention_scale) * tv1
                continue

            tv2 = p2.detach() - base_value
            proxy = tv1.abs() + tv2.abs()
            mask = head_mask_for_param(name, proxy, num_heads=num_heads, k_fraction=k_fraction)

            if mask is None:
                merged_params[name] = base_value + float(attention_scale) * tv1
                continue

            mask = mask.to(dtype=base_value.dtype, device=base_value.device)
            lamda_i = effective_lambda(
                name,
                lamda=lamda,
                alpha=alpha,
                num_encoder_layers=num_encoder_layers,
            )
            selected_delta = lamda_i * tv1 + (1.0 - lamda_i) * tv2
            fallback_delta = tv1
            merged_delta = mask * selected_delta + (1.0 - mask) * fallback_delta
            merged_params[name] = base_value + float(attention_scale) * merged_delta

            summary["selected_head_elements"] += int(mask.sum().item())
            summary["total_head_elements"] += int(mask.numel())
            summary["layer_scaled_tensors"] += 1

    return merged_params, summary


def copy_params_to_model(params: dict[str, torch.Tensor], model: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in params:
                param.copy_(params[name])


def save_bundle(model, config, processor, feature_extractor, out_path: Path) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    config.save_pretrained(out_path)
    processor.save_pretrained(out_path)
    feature_extractor.save_pretrained(out_path)
    print(f"[SAVE] {out_path}")


def hsa_merge(
    pretrained_model: str,
    model1_path: str,
    model2_path: str,
    out_path: Path,
    lambdas: list[float],
    alpha: float,
    k_fraction: float,
    num_heads: int | None,
    num_encoder_layers: int | None,
    device: str,
    processor_source: str,
    save_lambda_subdirs: bool,
    attention_scale: float,
    non_attention_scale: float,
) -> None:
    print(f"[LOAD] base:   {pretrained_model}")
    base_model, base_config = load_whisper_model(pretrained_model, device=device)
    print(f"[LOAD] model1: {model1_path}")
    model1, _ = load_whisper_model(model1_path, device=device)
    print(f"[LOAD] model2: {model2_path}")
    model2, _ = load_whisper_model(model2_path, device=device)

    if processor_source == "model1":
        processor, feature_extractor = load_processor_and_feature_extractor(model1_path)
    elif processor_source == "model2":
        processor, feature_extractor = load_processor_and_feature_extractor(model2_path)
    else:
        processor, feature_extractor = load_processor_and_feature_extractor(pretrained_model)

    heads = infer_num_heads(base_config, num_heads)
    encoder_layers = infer_encoder_layers(base_config, num_encoder_layers)
    k_fraction = normalize_fraction(k_fraction)
    hsa_target_patterns = tuple(re.compile(pattern) for pattern in HSA_TARGET_PATTERNS)

    base_params = cloned_base_params(base_model)
    model1_params = param_dict(model1)
    model2_params = param_dict(model2)

    print(f"[INFO] heads={heads}, encoder_layers={encoder_layers}, k={k_fraction:.3f}, lambdas={lambdas}, alpha={alpha:.4f}")
    print("[INFO] HSA targets encoder self-attention q/k/v projections only.")
    print("[INFO] selected-head coefficient uses lambda_i = lambda ** (alpha * (encoder_layer_index + 1) / encoder_layers).")
    print("[INFO] model1 is the anchor: non-selected target heads and all other tensors keep tv1.")

    for lamda in lambdas:
        merged_params, summary = build_hsa_params(
            base_params=base_params,
            model1_params=model1_params,
            model2_params=model2_params,
            lamda=lamda,
            alpha=alpha,
            num_encoder_layers=encoder_layers,
            k_fraction=k_fraction,
            num_heads=heads,
            hsa_target_patterns=hsa_target_patterns,
            attention_scale=attention_scale,
            non_attention_scale=non_attention_scale,
        )
        copy_params_to_model(merged_params, base_model)

        if len(lambdas) > 1 or save_lambda_subdirs:
            current_out = out_path / f"{lambda_tag(lamda)}_{alpha_tag(alpha)}"
        else:
            current_out = out_path

        save_bundle(base_model, base_config, processor, feature_extractor, current_out)
        print(
            "[INFO] lambda={:.4f}; hsa_target_tensors={}; model1_passthrough_tensors={}; "
            "layer_scaled_tensors={}; selected_elements={}/{}".format(
                lamda,
                summary["hsa_target_tensors"],
                summary["model1_passthrough_tensors"],
                summary["layer_scaled_tensors"],
                summary["selected_head_elements"],
                summary["total_head_elements"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-contained Headwise Selective Attention merge for Whisper.")
    parser.add_argument(
        "--pretrained-model",
        "--model-class",
        default="openai/whisper-small.en",
        help="Pretrained base model id/path used to form task vectors.",
    )
    parser.add_argument(
        "--model1",
        required=True,
        help="Primary/anchor fine-tuned model. Non-selected heads keep this model's task vector.",
    )
    parser.add_argument(
        "--model2",
        required=True,
        help="Secondary fine-tuned model used only in selected top-K encoder Q/K/V heads.",
    )
    parser.add_argument("--out-path", "--outdir", required=True, type=Path, help="Output model directory.")
    parser.add_argument(
        "--lambda",
        "--lamda",
        dest="lamda",
        type=float,
        default=None,
        help="Base weight on model1 task vector inside selected heads before layer-wise alpha scaling. Defaults to 0.5.",
    )
    parser.add_argument(
        "--lambdas",
        default=None,
        help="Comma-separated lambda values. If provided, saves one subdirectory per value.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Layer-wise exponential scale. Uses lambda_i = lambda ** (alpha * (i + 1) / L) for zero-based encoder layer index i. Default: 0.2.",
    )
    parser.add_argument(
        "--k-percent",
        type=float,
        default=0.6,
        help="Fraction or percentage of heads selected per encoder Q/K/V tensor, e.g. 0.6 or 60.",
    )
    parser.add_argument("--num-heads", type=int, default=None, help="Override attention-head count.")
    parser.add_argument("--num-encoder-layers", type=int, default=None, help="Override encoder layer count L.")
    parser.add_argument("--device", default="cpu", help="Torch device for loading/merging. Default: cpu.")
    parser.add_argument(
        "--processor-source",
        choices=("model1", "model2", "pretrained"),
        default="model1",
        help="Where to copy tokenizer/processor files from.",
    )
    parser.add_argument(
        "--save-lambda-subdirs",
        action="store_true",
        help="Save even a single lambda run under out-path/lamda_*.",
    )
    parser.add_argument(
        "--attention-scale",
        type=float,
        default=1.0,
        help="Scale applied to HSA target encoder Q/K/V task-vector updates.",
    )
    parser.add_argument(
        "--non-attention-scale",
        type=float,
        default=1.0,
        help="Scale applied to model1 task vectors for parameters outside the HSA target set.",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    hsa_merge(
        pretrained_model=args.pretrained_model,
        model1_path=args.model1,
        model2_path=args.model2,
        out_path=args.out_path,
        lambdas=parse_lambdas(args.lamda, args.lambdas),
        alpha=validate_alpha(args.alpha),
        k_fraction=args.k_percent,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        device=args.device,
        processor_source=args.processor_source,
        save_lambda_subdirs=args.save_lambda_subdirs,
        attention_scale=args.attention_scale,
        non_attention_scale=args.non_attention_scale,
    )


if __name__ == "__main__":
    main()
