#!/usr/bin/env python3
"""Run the short WavLM head-only CTC overfit diagnostic."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCTC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from usde.ctc import AudioCTCDataset, CTCDataCollator, make_processor
from usde.metrics import word_error_rate


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def decode_batch(model, processor, batch: dict[str, torch.Tensor]) -> tuple[list[str], float]:  # type: ignore[no-untyped-def]
    with torch.inference_mode():
        logits = model(**{key: value for key, value in batch.items() if key != "labels"}, return_dict=True).logits
    ids = logits.argmax(-1)
    blank_ratio = float((ids == model.config.pad_token_id).float().mean())
    return [str(text).lower().strip() for text in processor.tokenizer.batch_decode(ids)], blank_ratio


def backbone_module(model):  # type: ignore[no-untyped-def]
    """Return the SSL backbone for WavLM or Wav2Vec2 CTC models."""
    for name in ("wavlm", "wav2vec2"):
        if hasattr(model, name):
            return getattr(model, name)
    raise AttributeError(f"{model.__class__.__name__} has no known SSL backbone")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", "--model-name", dest="model_name", default="microsoft/wavlm-base-plus")
    parser.add_argument("--train_tsv", "--train-tsv", dest="train_tsv", type=Path, default=Path("data/train.tsv"))
    parser.add_argument("--vocab_dir", "--vocab-dir", dest="vocab_dir", type=Path, default=Path("assets/ctc_vocab"))
    parser.add_argument("--output_json", "--output-json", dest="output_json", type=Path, default=Path("artifacts/runs/stage2_wavlm_tiny_overfit/diagnostic.json"))
    parser.add_argument("--num_examples", "--num-examples", dest="num_examples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=5e-4)
    parser.add_argument("--freeze_backbone", "--freeze-backbone", dest="freeze_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--omit_attention_mask", "--omit-attention-mask", dest="omit_attention_mask", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    processor = make_processor(args.model_name, args.vocab_dir)
    config = AutoConfig.from_pretrained(
        args.model_name,
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
        layerdrop=0.0,
    )
    model = AutoModelForCTC.from_pretrained(
        args.model_name,
        config=config,
        ignore_mismatched_sizes=True,
    ).to(device)
    backbone = backbone_module(model)
    if args.freeze_backbone:
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
    model.train()
    if args.freeze_backbone:
        backbone.eval()
    model.lm_head.train()

    dataset = AudioCTCDataset(
        args.train_tsv,
        processor,
        max_examples=args.num_examples,
        sample_seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=CTCDataCollator(processor),
    )
    first_batch = move_batch(next(iter(loader)), device)
    if args.omit_attention_mask:
        first_batch.pop("attention_mask", None)
    label_ids = first_batch["labels"][0]
    label_ids = label_ids[label_ids != -100]
    with torch.inference_mode():
        first_output = model(**first_batch, return_dict=True)
    target_lengths = (first_batch["labels"] != -100).sum(-1)
    if first_output.logits.shape[1] < int(target_lengths.max()):
        raise ValueError(
            f"invalid CTC alignment: output_frames={first_output.logits.shape[1]} target_max={int(target_lengths.max())}"
        )
    print("sampling_rate=", processor.feature_extractor.sampling_rate)
    print("do_normalize=", processor.feature_extractor.do_normalize)
    print("return_attention_mask=", processor.feature_extractor.return_attention_mask)
    print("tokenizer_pad=", processor.tokenizer.pad_token_id, "model_pad=", model.config.pad_token_id)
    print("label=", repr(processor.tokenizer.decode(label_ids.tolist())))
    print("valid_label_ratio=", float((first_batch["labels"] != -100).float().mean()))
    print("logits=", tuple(first_output.logits.shape), "target_lengths=", target_lengths.tolist())

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    history: list[dict[str, object]] = []
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            batch = move_batch(batch, device)
            if args.omit_attention_mask:
                batch.pop("attention_mask", None)
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch, return_dict=True)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.lm_head.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step == 1 or step % 100 == 0:
                predictions, blank_ratio = decode_batch(model, processor, batch)
                record = {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": float(output.loss.detach().cpu()),
                    "blank_ratio": blank_ratio,
                    "prediction": predictions[0],
                }
                history.append(record)
                print(record, flush=True)

    model.eval()
    predictions: list[str] = []
    references: list[str] = []
    blank_ratios: list[float] = []
    for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=CTCDataCollator(processor)):
        batch = move_batch(batch, device)
        labels = batch.pop("labels")
        if args.omit_attention_mask:
            batch.pop("attention_mask", None)
        decoded, blank_ratio = decode_batch(model, processor, {**batch, "labels": labels})
        label_ids = labels.detach().cpu().numpy()
        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
        references.extend(str(text).lower().strip() for text in processor.tokenizer.batch_decode(label_ids, group_tokens=False))
        predictions.extend(decoded)
        blank_ratios.append(blank_ratio)
    final_wer = word_error_rate(references, predictions)
    summary = {
        "model_name": args.model_name,
        "examples": len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "backbone_frozen": args.freeze_backbone,
        "spec_augment_disabled": True,
        "attention_mask_omitted": args.omit_attention_mask,
        "precision": "fp32",
        "final_train_wer": final_wer,
        "final_blank_ratio_mean": float(np.mean(blank_ratios)),
        "history": history,
        "success": bool(final_wer < 0.5 and np.mean(blank_ratios) < 0.99),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
