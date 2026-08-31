#!/usr/bin/env python3
"""Train/evaluate the frozen final-layer Delta W2V2 concat CTC baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from usde.fusion import ConcatLinearCTC, FrozenFusionDataset, collate, load_vocab
from usde.text import BLANK, decode


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    errors = words = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref, hyp = reference.split(), hypothesis.split()
        table = list(range(len(hyp) + 1))
        for i, token in enumerate(ref, start=1):
            next_row = [i]
            for j, predicted in enumerate(hyp, start=1):
                next_row.append(min(table[j] + 1, next_row[j - 1] + 1, table[j - 1] + (token != predicted)))
            table = next_row
        errors += table[-1]
        words += len(ref)
    return errors / max(words, 1)


@torch.inference_mode()
def evaluate(model: ConcatLinearCTC, loader: DataLoader, vocab: dict[str, int], device: torch.device) -> float:
    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []
    for batch in loader:
        logits = model(batch["features"].to(device))
        for row, length in zip(logits.argmax(dim=-1).cpu(), batch["feature_lengths"], strict=True):
            hypotheses.append(decode(row[: int(length)].tolist(), vocab))
        references.extend(batch["transcripts"])
    return word_error_rate(references, hypotheses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--dev-features", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pure-linear", action="store_true", help="Disable the original source LayerNorm+Dropout wrapper.")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    vocab = load_vocab(args.vocab)
    blank_id = vocab[BLANK]
    train_loader = DataLoader(FrozenFusionDataset(args.train_manifest, args.train_features, vocab), batch_size=args.batch_size, shuffle=True, collate_fn=collate, pin_memory=True)
    dev_loader = DataLoader(FrozenFusionDataset(args.dev_manifest, args.dev_features, vocab), batch_size=args.batch_size, shuffle=False, collate_fn=collate, pin_memory=True)
    model = ConcatLinearCTC(len(vocab), source_compatible=not args.pure_linear).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CTCLoss(blank=blank_id, zero_infinity=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_wer = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["features"].to(device))
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = criterion(log_probs, batch["targets"].to(device), batch["feature_lengths"], batch["target_lengths"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_wer = evaluate(model, dev_loader, vocab, device)
        epoch_result = {"epoch": epoch, "train_ctc_loss": sum(losses) / len(losses), "dev_wer": dev_wer}
        history.append(epoch_result)
        if dev_wer < best_wer:
            best_wer = dev_wer
            torch.save({"model": model.state_dict(), "vocab": vocab, "source_compatible": not args.pure_linear, "epoch": epoch, "dev_wer": dev_wer}, args.output_dir / "fusion_best.pt")
        print(json.dumps(epoch_result, sort_keys=True))
    (args.output_dir / "train_report.json").write_text(json.dumps({"best_dev_wer": best_wer, "history": history}, indent=2) + "\n")


if __name__ == "__main__":
    main()
