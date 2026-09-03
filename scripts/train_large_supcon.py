#!/usr/bin/env python3
"""Official-protocol Wav2Vec2-Large CTC + transcript-grouped SupCon training."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from usde.metrics import word_error_rate
from usde.supcon import (
    OfficialSupConDataset,
    SupConDataCollator,
    SupConTrainerMixin,
    TranscriptGroupedBatchSampler,
    W2V2SupCon,
)


class SupConTrainer(SupConTrainerMixin, Trainer):
    """Trainer using the public repo's loss while retaining HF checkpointing."""


def patch_accelerate_compat() -> None:
    """Bridge Transformers 4.37 Trainer to the installed Accelerate 1.x API."""

    from accelerate import Accelerator, DataLoaderConfiguration

    if "dispatch_batches" in inspect.signature(Accelerator.__init__).parameters:
        return
    import transformers.trainer as trainer_module

    class CompatibleAccelerator(Accelerator):
        def __init__(self, *args, dispatch_batches=None, split_batches=None, **kwargs):  # type: ignore[no-untyped-def]
            if "dataloader_config" not in kwargs:
                kwargs["dataloader_config"] = DataLoaderConfiguration(
                    dispatch_batches=dispatch_batches,
                    split_batches=bool(split_batches),
                )
            super().__init__(*args, **kwargs)

    trainer_module.Accelerator = CompatibleAccelerator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_fn(processor):  # type: ignore[no-untyped-def]
    tokenizer = processor.tokenizer

    def compute_metrics(pred):  # type: ignore[no-untyped-def]
        logits = pred.predictions[0] if isinstance(pred.predictions, (tuple, list)) else pred.predictions
        predicted_ids = np.argmax(logits, axis=-1)
        labels = np.where(pred.label_ids == -100, tokenizer.pad_token_id, pred.label_ids)
        predictions = [str(text).lower().strip() for text in tokenizer.batch_decode(predicted_ids)]
        references = [str(text).lower().strip() for text in tokenizer.batch_decode(labels, group_tokens=False)]
        return {"wer": float(word_error_rate(references, predictions))}

    return compute_metrics


def metadata() -> dict[str, Any]:
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
        "rank": int(os.environ.get("RANK", "-1")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


def freeze_head(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True


def freeze_joint(model: W2V2SupCon) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
    if not hasattr(model.base, "freeze_feature_encoder"):
        raise AttributeError(f"{type(model.base)} has no freeze_feature_encoder()")
    model.base.freeze_feature_encoder()


def train_head_warmup(
    base: torch.nn.Module,
    processor,
    train_dataset: OfficialSupConDataset,
    dev_dataset: OfficialSupConDataset,
    output_dir: Path,
    args: argparse.Namespace,
) -> Trainer:
    freeze_head(base)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / (args.head_batch_size * world_size)))
    max_steps = steps_per_epoch if args.smoke_steps is None else min(steps_per_epoch, args.smoke_steps)
    eval_steps = 1 if args.smoke_steps is not None else max(1, steps_per_epoch // 10)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        learning_rate=3e-6,
        warmup_steps=max_steps,
        per_device_train_batch_size=args.head_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=1,
        max_grad_norm=0.5,
        fp16=False,
        bf16=False,
        group_by_length=False,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=args.best_metric,
        greater_is_better=False,
        remove_unused_columns=True,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        logging_steps=eval_steps,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(
        model=base,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=SupConDataCollator(processor, include_metadata=False),
        compute_metrics=metric_fn(processor),
    )
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(
            json.dumps(
                {
                    "phase": "head_warmup",
                    "train_examples": len(train_dataset),
                    "dev_examples": len(dev_dataset),
                    "steps_per_epoch": steps_per_epoch,
                    "max_steps": max_steps,
                    "eval_steps": eval_steps,
                    "batch_size": args.head_batch_size,
                    "learning_rate": 3e-6,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(str(output_dir))
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(output_dir))
    return trainer


def train_supcon(
    base: torch.nn.Module,
    processor,
    train_dataset: OfficialSupConDataset,
    dev_dataset: OfficialSupConDataset,
    output_dir: Path,
    args: argparse.Namespace,
) -> SupConTrainer:
    model = W2V2SupCon(base, processor, proj_dim=args.proj_dim)
    freeze_joint(model)
    if args.gradient_checkpointing:
        model.base.gradient_checkpointing_enable()
    sampler = TranscriptGroupedBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        group_size=args.group_size,
        samples_per_group=args.samples_per_group,
        seed=args.seed,
        drop_last=True,
        distributed=True,
    )
    eval_steps = 1 if args.smoke_steps is not None else max(1, len(sampler) // 2)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.smoke_steps if args.smoke_steps is not None else -1,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.scheduler,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        gradient_checkpointing=False,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=args.best_metric,
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        logging_steps=1 if args.smoke_steps is not None else max(1, eval_steps),
        ddp_find_unused_parameters=False,
        tf32=args.tf32,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    trainer = SupConTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=SupConDataCollator(processor),
        compute_metrics=metric_fn(processor),
        callbacks=callbacks,
        supcon_lambda=args.supcon_lambda,
        supcon_temp=args.supcon_temp,
        supcon_ramp_ratio=args.supcon_ramp_ratio,
        batch_sampler=sampler,
    )
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(
            json.dumps(
                {
                    "phase": "supcon_joint",
                    "train_examples": len(train_dataset),
                    "dev_examples": len(dev_dataset),
                    "batch_size": args.batch_size,
                    "group_size": args.group_size,
                    "samples_per_group": args.samples_per_group,
                    "local_batches_per_epoch": len(sampler),
                    "eval_steps": eval_steps,
                    "max_steps": args.smoke_steps,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "supcon_lambda": args.supcon_lambda,
                    "supcon_temp": args.supcon_temp,
                    "supcon_ramp_ratio": args.supcon_ramp_ratio,
                    "projection_dim": args.proj_dim,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(str(output_dir))
    return trainer


def run(args: argparse.Namespace) -> None:
    patch_accelerate_compat()
    if not torch.cuda.is_available():
        raise RuntimeError("SupCon training requires CUDA")
    set_seed(args.seed)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite_output_dir:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite-output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(str(args.pretrained_path))
    base = AutoModelForCTC.from_pretrained(str(args.pretrained_path))
    train_dataset = OfficialSupConDataset(
        args.train_manifest,
        processor,
        max_duration_s=args.max_duration_s,
        supcon_enabled=True,
        max_examples=args.max_train_examples,
        sample_seed=args.seed,
    )
    dev_dataset = OfficialSupConDataset(
        args.dev_manifest,
        processor,
        max_duration_s=args.max_duration_s,
        supcon_enabled=False,
        max_examples=args.max_dev_examples,
        sample_seed=args.seed,
    )
    test_dataset = OfficialSupConDataset(
        args.test_manifest,
        processor,
        max_duration_s=args.max_duration_s,
        supcon_enabled=False,
        max_examples=args.max_test_examples,
        sample_seed=args.seed,
    )
    grouped_ids = [int(value) for value in train_dataset.supcon_ids]
    group_counts: dict[int, int] = {}
    for value in grouped_ids:
        group_counts[value] = group_counts.get(value, 0) + 1
    config_payload: dict[str, Any] = {
        "protocol": "official_robust_atc_asr_arctic_8fold_0_supcon_repeated_v1",
        "pretrained_path": str(args.pretrained_path),
        "train_manifest": str(args.train_manifest),
        "dev_manifest": str(args.dev_manifest),
        "test_manifest": str(args.test_manifest),
        "seed": args.seed,
        "max_duration_s": args.max_duration_s,
        "head_warmup": {
            "epochs": 1,
            "learning_rate": 3e-6,
            "batch_size": args.head_batch_size,
            "precision": "fp32",
            "max_grad_norm": 0.5,
            "best_metric": args.best_metric,
        },
        "supcon": {
            "lambda": args.supcon_lambda,
            "temperature": args.supcon_temp,
            "ramp_ratio": args.supcon_ramp_ratio,
            "projection_dim": args.proj_dim,
            "batch_size_per_rank": args.batch_size,
            "group_size": args.group_size,
            "samples_per_group": args.samples_per_group,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "scheduler": args.scheduler,
            "precision": args.precision,
            "best_metric": args.best_metric,
        },
        "dataset": {
            "train": len(train_dataset),
            "dev": len(dev_dataset),
            "test": len(test_dataset),
            "unique_supcon_ids": len(set(grouped_ids)),
            "groups_ge_2": sum(1 for value in group_counts.values() if value >= 2),
            "groups_singleton": sum(1 for value in group_counts.values() if value == 1),
        },
        "gpu": metadata(),
        "smoke_steps": args.smoke_steps,
    }
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        (args.output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Official protocol uses a fresh CTC head warm-up because the split differs
    # materially from the previously trained local Fold0. After an interrupted
    # joint run, --skip-head-warmup lets us reuse the already saved warm-up.
    warmup_dir = args.output_dir / "head_warmup"
    if args.skip_head_warmup:
        if not (warmup_dir / "config.json").exists():
            raise FileNotFoundError(f"--skip-head-warmup requested but no saved warm-up at {warmup_dir}")
        warmup_best = None
    else:
        warmup_trainer = train_head_warmup(base, processor, train_dataset, dev_dataset, warmup_dir, args)
        warmup_best = warmup_trainer.state.best_metric
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    best_base = AutoModelForCTC.from_pretrained(str(warmup_dir))
    best_processor = AutoProcessor.from_pretrained(str(warmup_dir))
    joint_dir = args.output_dir / "supcon_joint"
    joint_trainer = train_supcon(best_base, best_processor, train_dataset, dev_dataset, joint_dir, args)
    best_model = joint_trainer.model
    if hasattr(best_model, "module"):
        best_model = best_model.module
    best_model = best_model  # type: ignore[assignment]
    best_base = best_model.base
    if joint_trainer.is_world_process_zero():
        best_base.save_pretrained(str(args.output_dir))
        best_processor.save_pretrained(str(args.output_dir))
        torch.save(best_model.proj.state_dict(), args.output_dir / "supcon_proj.pt")
        summary = {
            **config_payload,
            "head_warmup_best_metric_name": args.best_metric,
            "head_warmup_best_metric": warmup_best,
            "supcon_best_checkpoint": joint_trainer.state.best_model_checkpoint,
            "supcon_best_metric_name": args.best_metric,
            "supcon_best_metric": joint_trainer.state.best_metric,
            "supcon_global_step": joint_trainer.state.global_step,
            "supcon_epoch": joint_trainer.state.epoch,
            "supcon_train_metrics": joint_trainer.state.log_history[-1] if joint_trainer.state.log_history else {},
        }
        (args.output_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-path", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head-batch-size", type=int, default=4)
    # Public ALTERNATIVE_CONFIG[24] = (group_size=6, samples_per_group=4,
    # grad_acc=1). The 10-second crop keeps this larger grouping stable on
    # our 24-GiB cards while preserving real repeated-transcript positives.
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--samples-per-group", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--best-metric", choices=("wer", "loss"), default="wer")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--supcon-lambda", type=float, default=0.05)
    parser.add_argument("--supcon-temp", type=float, default=0.1)
    parser.add_argument("--supcon-ramp-ratio", type=float, default=0.1)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-duration-s", type=float, default=10.0)
    parser.add_argument("--dataloader-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--skip-head-warmup", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-dev-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    args = parser.parse_args()
    if args.batch_size != args.group_size * args.samples_per_group:
        raise ValueError("--batch-size must equal --group-size * --samples-per-group")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 requested but CUDA bf16 is unsupported")
    if min(args.head_batch_size, args.batch_size, args.group_size, args.samples_per_group, args.eval_batch_size) < 1:
        raise ValueError("batch and group settings must be positive")
    if args.smoke_steps is not None and args.smoke_steps < 1:
        raise ValueError("--smoke-steps must be positive")
    run(args)


if __name__ == "__main__":
    main()
