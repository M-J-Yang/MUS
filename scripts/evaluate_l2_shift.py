#!/usr/bin/env python3
"""Evaluate fine-tuned CTC, reconstruction, and masked-shift representations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from usde.ctc import AudioCTCDataset, CTCDataCollator, make_processor  # noqa: E402
from usde.metrics import word_error_rate  # noqa: E402
from usde.model import load_ctc_model  # noqa: E402
from usde.shift import ShiftFeatureDataset, collate_shift, reconstruction_error, select_delta  # noqa: E402


PROTOCOL = "single_model_finetuning_shift_reconstruction_v1"


def _decode(tokenizer: Any, values: torch.Tensor, length: int) -> str:
    ids = values.argmax(dim=-1) if values.ndim == 2 else values
    return str(tokenizer.decode(ids[:length].tolist(), group_tokens=True)).lower().strip()


def _scores(references: list[str], hypotheses: list[str]) -> dict[str, Any]:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    return {
        "wer": float(word_error_rate(references, hypotheses)),
        "examples": len(references),
    }


def _load_ranking(path: Path | None) -> torch.Tensor | None:
    if path is None:
        return None
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = value.get("ranking", value.get("tensor"))
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"{path}: expected a one-dimensional ranking tensor")
    return value.to(dtype=torch.long, device="cpu")


def _output_lengths(model: torch.nn.Module, attention_mask: torch.Tensor) -> torch.Tensor:
    method = getattr(model, "_get_feat_extract_output_lengths", None)
    if method is None:
        raise AttributeError("fine-tuned model has no feature-extractor output-length helper")
    return method(attention_mask.sum(dim=-1).to(dtype=torch.long)).to(device=attention_mask.device)


def _evaluate_direct(
    model: torch.nn.Module,
    processor: Any,
    manifest: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_examples: int | None,
    seed: int,
) -> tuple[dict[str, Any], list[str], list[str]]:
    dataset = AudioCTCDataset(manifest, processor, max_examples=max_examples, sample_seed=seed)
    # Compare against the same unpadded waveform used during caching. A
    # padded batch can change convolutional boundary frames for shorter audio,
    # making a correct Eft cache appear inconsistent with direct inference.
    loader_kwargs: dict[str, Any] = {
        "batch_size": 1,
        "shuffle": False,
        "collate_fn": CTCDataCollator(processor),
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    references: list[str] = []
    hypotheses: list[str] = []
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            values = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            output = model(**values, return_dict=True)
            lengths = _output_lengths(model, values["attention_mask"]).cpu()
            for row, length in zip(output.logits.cpu(), lengths, strict=True):
                hypotheses.append(_decode(processor.tokenizer, row, int(length)))
            batch_records = dataset.records[cursor : cursor + len(output.logits)]
            references.extend(str(row["transcript"]).lower().strip() for row in batch_records)
            cursor += len(output.logits)
    return _scores(references, hypotheses), references, hypotheses


def _evaluate_cached(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: ShiftFeatureDataset,
    loader: DataLoader,
    device: torch.device,
    ranking: torch.Tensor | None,
    retain_fraction: float,
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, Any]]:
    references: list[str] = []
    predictions = {"eft_cache": [], "full_delta_reconstruction": [], "no_shift_e0": []}
    if not 0.0 < retain_fraction <= 1.0:
        raise ValueError("retain-fraction must be in (0, 1]")
    max_identity_error = 0.0
    max_reconstruction_logit_error = 0.0
    retained_predictions: list[str] = []
    keep_indices: torch.Tensor | None = None
    with torch.inference_mode():
        for batch in loader:
            e0 = batch["e0"].to(device, non_blocking=True)
            eft = batch["eft"].to(device, non_blocking=True)
            delta = batch["delta"].to(device, non_blocking=True)
            if ranking is not None:
                if ranking.numel() != delta.shape[-1] or torch.unique(ranking).numel() != ranking.numel() or int(ranking.min()) < 0 or int(ranking.max()) >= delta.shape[-1]:
                    raise ValueError("ranking must be a complete permutation of hidden coordinates")
                keep_count = max(1, int(round(delta.shape[-1] * retain_fraction)))
                keep_indices = ranking[:keep_count]
            full = select_delta(e0, delta, None)
            no_shift = select_delta(e0, delta, torch.empty(0, dtype=torch.long, device=device))
            retained = select_delta(e0, delta, keep_indices) if keep_indices is not None else None
            identity = reconstruction_error(e0[0].cpu(), eft[0].cpu(), delta[0].cpu())
            for index in range(1, e0.shape[0]):
                identity_item = reconstruction_error(e0[index].cpu(), eft[index].cpu(), delta[index].cpu())
                identity["max_abs_error"] = max(identity["max_abs_error"], identity_item["max_abs_error"])
                identity["allclose"] = bool(identity["allclose"] and identity_item["allclose"])
            max_identity_error = max(max_identity_error, float(identity["max_abs_error"]))
            eft_logits = model.lm_head(eft)
            full_logits = model.lm_head(full)
            no_shift_logits = model.lm_head(no_shift)
            max_reconstruction_logit_error = max(max_reconstruction_logit_error, float((eft_logits - full_logits).abs().max()))
            lengths = batch["feature_lengths"].tolist()
            for eft_row, full_row, no_shift_row, length in zip(eft_logits.cpu(), full_logits.cpu(), no_shift_logits.cpu(), lengths, strict=True):
                predictions["eft_cache"].append(_decode(tokenizer, eft_row, int(length)))
                predictions["full_delta_reconstruction"].append(_decode(tokenizer, full_row, int(length)))
                predictions["no_shift_e0"].append(_decode(tokenizer, no_shift_row, int(length)))
            if retained is not None:
                retained_logits = model.lm_head(retained)
                for row, length in zip(retained_logits.cpu(), lengths, strict=True):
                    retained_predictions.append(_decode(tokenizer, row, int(length)))
            references.extend(batch["transcripts"])

    measured = {name: _scores(references, values) for name, values in predictions.items()}
    if retained_predictions:
        measured["retained_delta"] = _scores(references, retained_predictions)
        predictions["retained_delta"] = retained_predictions
    diagnostics = {
        "max_identity_abs_error": max_identity_error,
        "max_reconstruction_logit_abs_error": max_reconstruction_logit_error,
        "identity_pass": max_identity_error <= 1e-5,
        "reconstruction_logits_pass": max_reconstruction_logit_error <= 1e-4,
    }
    return measured, predictions, diagnostics


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    processor = make_processor(str(args.checkpoint), None)
    model = load_ctc_model(args.checkpoint).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    ranking = _load_ranking(args.ranking)
    direct, direct_refs, direct_hypotheses = _evaluate_direct(
        model, processor, args.manifest, args.batch_size, args.num_workers, device, args.max_examples, args.seed
    )
    dataset = ShiftFeatureDataset(
        args.manifest,
        args.cache_root,
        processor.tokenizer,
        feature_split=args.feature_split,
        expected_dim=int(getattr(model.config, "hidden_size", 0)),
        max_examples=args.max_examples,
        sample_seed=args.seed,
        load_eft=True,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "collate_fn": collate_shift,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    cached, cached_predictions, diagnostics = _evaluate_cached(
        model, processor.tokenizer, dataset, DataLoader(dataset, **loader_kwargs), device, ranking, args.retain_fraction
    )
    prediction_match = sum(
        left == right for left, right in zip(direct_hypotheses, cached_predictions["eft_cache"], strict=True)
    ) / max(len(direct_hypotheses), 1)
    full_wer = cached["full_delta_reconstruction"]["wer"]
    no_shift_wer = cached["no_shift_e0"]["wer"]
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "cache_root": str(args.cache_root),
        "feature_split": args.feature_split,
        "device": str(device),
        "hidden_dim": int(getattr(model.config, "hidden_size", 0)),
        "selection": "top Taylor-utility coordinates" if ranking is not None else None,
        "retain_fraction": args.retain_fraction if ranking is not None else None,
        "direct_fine_tuned": direct,
        "cached": cached,
        "direct_vs_eft_prediction_match_rate": prediction_match,
        "diagnostics": diagnostics,
        "representation_gain": {
            "no_shift_wer": no_shift_wer,
            "full_shift_wer": full_wer,
            "absolute_gain": no_shift_wer - full_wer,
            "retained_gain": None if "retained_delta" not in cached else no_shift_wer - cached["retained_delta"]["wer"],
            "retained_gain_fraction": None if "retained_delta" not in cached or no_shift_wer == full_wer else (no_shift_wer - cached["retained_delta"]["wer"]) / (no_shift_wer - full_wer),
        },
    }
    result["wer_identity_pass"] = bool(
        direct["wer"] == cached["eft_cache"]["wer"] == cached["full_delta_reconstruction"]["wer"]
    )
    result["prediction_identity_pass"] = bool(prediction_match == 1.0)
    result["gate_pass"] = bool(
        result["wer_identity_pass"]
        and result["prediction_identity_pass"]
        and diagnostics["identity_pass"]
        and diagnostics["reconstruction_logits_pass"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["gate_pass"]:
        raise RuntimeError("shift reconstruction gate failed; inspect the written evaluation report")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--feature-split", required=True)
    parser.add_argument("--ranking", type=Path, default=None)
    parser.add_argument("--retain-fraction", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
