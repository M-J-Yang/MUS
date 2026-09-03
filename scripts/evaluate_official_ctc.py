#!/usr/bin/env python3
"""Independent greedy WER evaluation for the official L2-ARCTIC manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCTC, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from usde.metrics import word_error_rate
from usde.supcon import OfficialSupConDataset, SupConDataCollator


def load_ctc_model(checkpoint: Path) -> AutoModelForCTC:
    # Support legacy and Transformers>=4.57 parametrized weight-norm checkpoints.
    safe_path = checkpoint / 'model.safetensors'
    if not safe_path.exists():
        return AutoModelForCTC.from_pretrained(str(checkpoint))

    from safetensors.torch import load_file

    model = AutoModelForCTC.from_config(AutoConfig.from_pretrained(str(checkpoint)))
    state = load_file(str(safe_path), device='cpu')
    remapped = {}
    for key, value in state.items():
        key = key.replace(
            '.parametrizations.weight.original0', '.weight_g'
        ).replace(
            '.parametrizations.weight.original1', '.weight_v'
        )
        remapped[key] = value
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'checkpoint load mismatch: missing={missing}, unexpected={unexpected}')
    return model


def evaluate(checkpoint: Path, manifest: Path, batch_size: int, workers: int, device: torch.device, max_duration_s: float) -> float:
    processor = AutoProcessor.from_pretrained(str(checkpoint))
    model = load_ctc_model(checkpoint).to(device).eval()
    dataset = OfficialSupConDataset(
        manifest,
        processor,
        max_duration_s=max_duration_s,
        supcon_enabled=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=SupConDataCollator(processor, include_metadata=False),
        num_workers=workers,
        pin_memory=True,
    )
    predictions: list[str] = []
    references: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            model_inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**model_inputs, return_dict=True).logits
            predicted_ids = np.argmax(logits.cpu().numpy(), axis=-1)
            label_ids = np.where(labels.numpy() == -100, processor.tokenizer.pad_token_id, labels.numpy())
            predictions.extend(str(text).lower().strip() for text in processor.tokenizer.batch_decode(predicted_ids))
            references.extend(
                str(text).lower().strip()
                for text in processor.tokenizer.batch_decode(label_ids, group_tokens=False)
            )
    return float(word_error_rate(references, predictions))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-duration-s", type=float, default=10.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if args.dev_manifest is None and args.test_manifest is None:
        raise ValueError("provide --dev-manifest and/or --test-manifest")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results: dict[str, object] = {"checkpoint": str(args.checkpoint), "batch_size": args.batch_size}
    if args.dev_manifest is not None:
        results["dev_wer"] = evaluate(args.checkpoint, args.dev_manifest, args.batch_size, args.workers, device, args.max_duration_s)
    if args.test_manifest is not None:
        results["test_wer"] = evaluate(args.checkpoint, args.test_manifest, args.batch_size, args.workers, device, args.max_duration_s)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
