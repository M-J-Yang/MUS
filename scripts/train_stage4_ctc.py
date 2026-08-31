#!/usr/bin/env python3
"""Train one of the three frozen-feature Stage 4 linear CTC baselines."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from usde.stage4 import (  # noqa: E402
    BASE_DIM,
    CONDITIONS,
    CachedFeatureDataset,
    LinearCTC,
    blank_id,
    collate,
    greedy_wer,
    load_vocab,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_checkpoint(model: LinearCTC, path: Path, payload: dict[str, Any]) -> None:
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    checkpoint = dict(payload)
    checkpoint["model"] = state
    checkpoint["classifier_weight"] = state["linear.weight"]
    checkpoint["classifier_bias"] = state["linear.bias"]
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"{path}: invalid Stage 4 checkpoint")
    return payload


def _default_train_manifest() -> Path:
    teacher = Path("manifests/arctic_step2/l2/train_teacher.jsonl")
    return teacher if teacher.is_file() else Path("manifests/arctic_step2/l2/train.jsonl")


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("epochs/batch-size/learning-rate must be positive and weight-decay non-negative")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    vocab = load_vocab(args.vocab)
    train_manifest = args.train_manifest or _default_train_manifest()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite_output_dir:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite-output-dir to reuse it")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CachedFeatureDataset(
        train_manifest,
        args.feature_root,
        args.condition,
        vocab,
        feature_split=args.train_feature_split,
        expected_dim=args.expected_dim,
        max_examples=args.max_train_examples,
        sample_seed=args.seed,
    )
    dev_dataset = CachedFeatureDataset(
        args.dev_manifest,
        args.feature_root,
        args.condition,
        vocab,
        feature_split=args.dev_feature_split,
        expected_dim=args.expected_dim,
        max_examples=args.max_dev_examples,
        sample_seed=args.seed,
    )
    test_dataset = None
    if args.test_manifest is not None:
        test_dataset = CachedFeatureDataset(
            args.test_manifest,
            args.feature_root,
            args.condition,
            vocab,
            feature_split=args.test_feature_split,
            expected_dim=args.expected_dim,
            max_examples=args.max_test_examples,
            sample_seed=args.seed,
        )

    input_dim = args.expected_dim if args.condition == "ref" else args.expected_dim * 2
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, shuffle=True, generator=train_generator, **loader_kwargs)
    dev_loader = DataLoader(dev_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs) if test_dataset is not None else None

    model = LinearCTC(input_dim, len(vocab)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CTCLoss(blank=blank_id(vocab), zero_infinity=True)
    best_wer = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    checkpoint_payload = {
        "protocol": "stage4_three_condition_linear_ctc_v1",
        "condition": args.condition,
        "feature_root": str(args.feature_root),
        "base_dim": args.expected_dim,
        "input_dim": input_dim,
        "vocab": vocab,
        "blank_id": blank_id(vocab),
        "source_compatible": False,
        "layer": args.layer,
        "seed": args.seed,
        "train_manifest": str(train_manifest),
        "dev_manifest": str(args.dev_manifest),
        "train_feature_split": args.train_feature_split,
        "dev_feature_split": args.dev_feature_split,
    }
    config = dict(vars(args))
    config.update({"resolved_train_manifest": train_manifest, "device": str(device), "input_dim": input_dim, "vocab_size": len(vocab)})
    _save_json(args.output_dir / "config.json", config)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            features = batch["features"].to(device, non_blocking=True)
            logits = model(features)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = criterion(
                log_probs,
                batch["targets"].to(device, non_blocking=True),
                batch["feature_lengths"],
                batch["target_lengths"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"epoch {epoch}: CTC loss became non-finite")
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_wer = greedy_wer(model, dev_loader, vocab, device)
        result = {
            "epoch": epoch,
            "train_ctc_loss": sum(losses) / max(len(losses), 1),
            "dev_wer": dev_wer,
            "train_examples": len(train_dataset),
            "dev_examples": len(dev_dataset),
        }
        history.append(result)
        if dev_wer < best_wer:
            best_wer = dev_wer
            best_epoch = epoch
            _save_checkpoint(
                model,
                args.output_dir / "best.pt",
                dict(checkpoint_payload, epoch=epoch, dev_wer=dev_wer),
            )
        metrics = {
            **checkpoint_payload,
            "best_dev_wer": best_wer,
            "best_epoch": best_epoch,
            "history": history,
            "test_wer": None,
        }
        _save_json(args.output_dir / "metrics.json", metrics)
        print(json.dumps(result, sort_keys=True), flush=True)

    if best_epoch == 0:
        raise RuntimeError("training completed without saving a best checkpoint")
    best_checkpoint = _load_checkpoint(args.output_dir / "best.pt")
    model.load_state_dict(best_checkpoint["model"])
    test_wer = greedy_wer(model, test_loader, vocab, device) if test_loader is not None else None
    metrics = {
        **checkpoint_payload,
        "best_dev_wer": best_wer,
        "best_epoch": best_epoch,
        "history": history,
        "test_wer": test_wer,
        "test_examples": None if test_dataset is None else len(test_dataset),
    }
    _save_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps({"condition": args.condition, "best_dev_wer": best_wer, "test_wer": test_wer, "best_epoch": best_epoch}, sort_keys=True), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/features/stage3_l2"))
    parser.add_argument("--train-manifest", type=Path, default=None, help="defaults to train_teacher.jsonl when present")
    parser.add_argument("--dev-manifest", type=Path, default=Path("manifests/arctic_step2/l2/dev.jsonl"))
    parser.add_argument("--test-manifest", type=Path, default=Path("manifests/arctic_step2/l2/test.jsonl"))
    parser.add_argument("--train-feature-split", default="train")
    parser.add_argument("--dev-feature-split", default="dev")
    parser.add_argument("--test-feature-split", default="test")
    parser.add_argument("--vocab", type=Path, default=Path("assets/ctc_vocab/vocab.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--expected-dim", type=int, default=BASE_DIM)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-dev-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
