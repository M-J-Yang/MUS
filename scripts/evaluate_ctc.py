#!/usr/bin/env python3
"""Greedy CTC WER evaluation for a saved Stage 2 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCTC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from usde.ctc import AudioCTCDataset, CTCDataCollator, make_processor
from usde.metrics import word_error_rate


def evaluate(model: torch.nn.Module, dataset: AudioCTCDataset, processor, batch_size: int, device: torch.device) -> float:
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=CTCDataCollator(processor))
    predictions: list[str] = []
    references: list[str] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            model_inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**model_inputs, return_dict=True).logits
            predicted_ids = np.argmax(logits.cpu().numpy(), axis=-1)
            label_ids = labels.numpy()
            label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
            predictions.extend(str(text).lower().strip() for text in processor.tokenizer.batch_decode(predicted_ids))
            references.extend(str(text).lower().strip() for text in processor.tokenizer.batch_decode(label_ids, group_tokens=False))
    return float(word_error_rate(references, predictions))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dev_tsv", "--dev-tsv", dest="dev_tsv", type=Path, default=None)
    parser.add_argument("--test_tsv", "--test-tsv", dest="test_tsv", type=Path, default=None)
    parser.add_argument("--vocab_dir", "--vocab-dir", dest="vocab_dir", type=Path, default=Path("assets/ctc_vocab"))
    parser.add_argument("--audio_root", "--audio-root", dest="audio_root", type=Path, default=None)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_json", "--output-json", dest="output_json", type=Path, default=None)
    parser.add_argument("--model_key", "--model-key", dest="model_key", default=None)
    args = parser.parse_args()
    if args.dev_tsv is None and args.test_tsv is None:
        raise ValueError("provide at least --dev_tsv or --test_tsv")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = make_processor(str(args.checkpoint), args.vocab_dir)
    model = AutoModelForCTC.from_pretrained(args.checkpoint).to(device)
    results: dict[str, float | str] = {"checkpoint": str(args.checkpoint)}
    for name, manifest in (("dev_wer", args.dev_tsv), ("test_wer", args.test_tsv)):
        if manifest is not None:
            dataset = AudioCTCDataset(manifest, processor, args.audio_root)
            results[name] = evaluate(model, dataset, processor, args.batch_size, device)
    if args.output_json is not None:
        payload: dict[str, object] = {}
        if args.output_json.is_file():
            payload = json.loads(args.output_json.read_text(encoding="utf-8"))
        key = args.model_key or args.checkpoint.name
        payload[key] = results
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
