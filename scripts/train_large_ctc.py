#!/usr/bin/env python3
"""Two-stage Large CTC fine-tuning for the formal L2-ARCTIC UT-8 protocol."""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCTC,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer import LengthGroupedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from usde.ctc import AudioCTCDataset, CTCDataCollator, make_processor
from usde.metrics import word_error_rate


class LengthAwareTrainer(Trainer):
    """Use cached waveform lengths for length grouping without decoding twice."""

    def _get_train_sampler(self):  # type: ignore[no-untyped-def]
        if self.train_dataset is not None and self.args.group_by_length:
            lengths = getattr(self.train_dataset, "input_lengths", None)
            if lengths is not None:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    lengths=lengths,
                    model_input_name="input_values",
                )
        return super()._get_train_sampler()


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


def freeze_for_phase(model: torch.nn.Module, phase: str) -> dict[str, int]:
    """Configure head-only warm-up or joint FT while keeping conv frozen."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    if phase == "head":
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True
    elif phase == "joint":
        if not hasattr(model, "freeze_feature_encoder"):
            raise AttributeError(f"{model.__class__.__name__} has no freeze_feature_encoder()")
        model.freeze_feature_encoder()
        for name, parameter in model.named_parameters():
            if "feature_extractor" not in name:
                parameter.requires_grad = True
    else:
        raise ValueError(f"unknown phase: {phase}")

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    conv_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "feature_extractor" in name and parameter.requires_grad
    )
    if phase == "joint" and conv_trainable:
        raise RuntimeError("convolutional feature encoder is not frozen")
    if trainable <= 0:
        raise RuntimeError(f"{phase}: no trainable parameters")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen, "conv_trainable_parameters": conv_trainable}


def build_metrics(processor):  # type: ignore[no-untyped-def]
    tokenizer = processor.tokenizer

    def compute_metrics(eval_prediction):  # type: ignore[no-untyped-def]
        logits = eval_prediction.predictions[0] if isinstance(eval_prediction.predictions, tuple) else eval_prediction.predictions
        predicted_ids = np.argmax(logits, axis=-1)
        label_ids = np.where(eval_prediction.label_ids == -100, tokenizer.pad_token_id, eval_prediction.label_ids)
        predictions = [str(text).lower().strip() for text in tokenizer.batch_decode(predicted_ids)]
        references = [str(text).lower().strip() for text in tokenizer.batch_decode(label_ids, group_tokens=False)]
        return {"wer": float(word_error_rate(references, predictions))}

    return compute_metrics


def make_args(
    output_dir: Path,
    args: argparse.Namespace,
    num_train_epochs: float,
    load_best_model_at_end: bool,
    smoke: bool,
    phase: str,
) -> TrainingArguments:
    if phase not in {"head", "joint"}:
        raise ValueError(f"unknown phase: {phase}")
    learning_rate = args.head_learning_rate if phase == "head" else args.learning_rate
    per_device_batch_size = (
        args.head_per_device_batch_size if phase == "head" else args.per_device_batch_size
    )
    gradient_accumulation_steps = (
        args.head_gradient_accumulation_steps
        if phase == "head"
        else args.gradient_accumulation_steps
    )
    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        # Match the public recipe: ramp the head for the complete warm-up
        # epoch, then use the paper's linear warm-up ratio for joint FT.
        warmup_ratio=1.0 if phase == "head" else args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear" if phase == "head" else "cosine",
        per_device_train_batch_size=per_device_batch_size,
        # Batch-1 evaluation avoids padded tail frames changing WER and makes
        # checkpoint selection agree with the direct evaluator.
        per_device_eval_batch_size=args.eval_per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=False if phase == "head" else args.joint_precision == "fp16",
        bf16=False if phase == "head" else args.joint_precision == "bf16",
        gradient_checkpointing=False if phase == "head" else args.gradient_checkpointing,
        max_grad_norm=0.5 if phase == "head" else 1.0,
        group_by_length=True,
        evaluation_strategy="steps" if smoke else "epoch",
        eval_steps=1 if smoke else None,
        save_strategy="steps" if smoke else "epoch",
        save_steps=1 if smoke else 1,
        save_total_limit=2,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        tf32=args.tf32 if phase == "joint" else False,
        logging_strategy="steps",
        logging_steps=1 if smoke else 50,
    )
    if smoke:
        kwargs["max_steps"] = args.smoke_steps
    return TrainingArguments(**kwargs)


def build_trainer(
    model: torch.nn.Module,
    processor,
    train_dataset: AudioCTCDataset,
    dev_dataset: AudioCTCDataset,
    output_dir: Path,
    args: argparse.Namespace,
    phase: str,
) -> LengthAwareTrainer:
    smoke = args.smoke_test
    phase_args = make_args(
        output_dir,
        args,
        1.0 if phase == "head" or smoke else args.joint_max_epochs,
        load_best_model_at_end=True,
        smoke=smoke,
        phase=phase,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if phase == "joint" and not smoke else []
    return LengthAwareTrainer(
        model=model,
        args=phase_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=CTCDataCollator(processor),
        compute_metrics=build_metrics(processor),
        callbacks=callbacks,
    )


def gpu_metadata() -> dict[str, Any]:
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "local_rank": int(__import__("os").environ.get("LOCAL_RANK", "-1")),
        "world_size": int(__import__("os").environ.get("WORLD_SIZE", "1")),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    patch_accelerate_compat()
    if not torch.cuda.is_available():
        raise RuntimeError("formal Large CTC training requires CUDA")
    if args.per_device_batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("batch sizes must be positive")
    set_seed(args.seed)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite_output_dir:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite-output-dir to reuse it")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    init_path = args.pretrained_path or args.model_id
    if args.vocab_dir is None:
        # Formal transfer path: retain the model's original processor, vocab,
        # and ASR-trained lm_head. A shape mismatch is a hard error rather than
        # an implicit new CTC head.
        processor = make_processor(str(init_path), None)
        config = AutoConfig.from_pretrained(
            str(init_path),
            ctc_loss_reduction=args.ctc_loss_reduction,
            ctc_zero_infinity=args.ctc_zero_infinity,
            layerdrop=0.0,
        )
        pretrained_vocab_size = int(getattr(config, "vocab_size", 0))
        if pretrained_vocab_size != len(processor.tokenizer):
            raise ValueError(
                f"{init_path}: config vocab_size={pretrained_vocab_size} does not match "
                f"pretrained tokenizer size={len(processor.tokenizer)}"
            )
        model = AutoModelForCTC.from_pretrained(
            str(init_path),
            config=config,
            ignore_mismatched_sizes=False,
        )
        pretrained_ctc_head = True
    else:
        # Legacy project-local vocabulary path retained for existing diagnostics.
        processor = make_processor(str(init_path), args.vocab_dir)
        config = AutoConfig.from_pretrained(
            str(init_path),
            vocab_size=len(processor.tokenizer),
            pad_token_id=processor.tokenizer.pad_token_id,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            layerdrop=0.0,
        )
        model = AutoModelForCTC.from_pretrained(
            str(init_path),
            config=config,
            ignore_mismatched_sizes=True,
        )
        pretrained_ctc_head = False
    model.config.layerdrop = 0.0
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset = AudioCTCDataset(
        args.train_manifest,
        processor,
        max_examples=args.max_train_examples,
        sample_seed=args.seed,
    )
    dev_dataset = AudioCTCDataset(
        args.dev_manifest,
        processor,
        max_examples=args.max_dev_examples,
        sample_seed=args.seed,
    )
    test_dataset = AudioCTCDataset(
        args.test_manifest,
        processor,
        max_examples=args.max_test_examples,
        sample_seed=args.seed,
    ) if args.test_manifest is not None else None

    config_payload = {
        "protocol": "l2_arctic_unseen_transcript_ut8_fold0_large_ctc_v1",
        "model_id": args.model_id,
        "pretrained_path": str(init_path),
        "pretrained_ctc_head": pretrained_ctc_head,
        "train_manifest": str(args.train_manifest),
        "dev_manifest": str(args.dev_manifest),
        "test_manifest": None if args.test_manifest is None else str(args.test_manifest),
        "vocab_dir": str(args.vocab_dir),
        "seed": args.seed,
        "head_warmup_epochs": 1,
        "head_learning_rate": args.head_learning_rate,
        "head_per_device_batch_size": args.head_per_device_batch_size,
        "head_gradient_accumulation_steps": args.head_gradient_accumulation_steps,
        "joint_max_epochs": args.joint_max_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "scheduler": "cosine",
        "per_device_batch_size": args.per_device_batch_size,
        "eval_per_device_batch_size": args.eval_per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "global_effective_batch_size": args.per_device_batch_size * args.gradient_accumulation_steps * int(__import__("os").environ.get("WORLD_SIZE", "1")),
        "head_global_effective_batch_size": args.head_per_device_batch_size * args.head_gradient_accumulation_steps * int(__import__("os").environ.get("WORLD_SIZE", "1")),
        "head_precision": "fp32",
        "joint_precision": args.joint_precision,
        "ctc_loss_reduction": args.ctc_loss_reduction,
        "ctc_zero_infinity": args.ctc_zero_infinity,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_clip_norm": 1.0,
        "layerdrop": 0.0,
        "feature_encoder_frozen": True,
        "selection_metric": "dev_wer",
        "decode": "greedy_ctc",
        "smoke_test": args.smoke_test,
        "smoke_steps": args.smoke_steps if args.smoke_test else None,
        "train_examples": len(train_dataset),
        "dev_examples": len(dev_dataset),
        "test_examples": None if test_dataset is None else len(test_dataset),
        "gpu": gpu_metadata(),
    }
    if int(__import__("os").environ.get("LOCAL_RANK", "0")) == 0:
        (args.output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    head_stats = freeze_for_phase(model, "head")
    head_trainer = build_trainer(model, processor, train_dataset, dev_dataset, args.output_dir / "head_warmup", args, "head")
    head_result = head_trainer.train()
    head_trainer.save_model(str(args.output_dir / "head_warmup"))
    if head_trainer.is_world_process_zero():
        processor.save_pretrained(str(args.output_dir / "head_warmup"))

    joint_stats = freeze_for_phase(model, "joint")
    joint_trainer = build_trainer(model, processor, train_dataset, dev_dataset, args.output_dir / "joint", args, "joint")
    joint_result = joint_trainer.train()
    joint_trainer.save_model(str(args.output_dir / "joint"))
    if joint_trainer.is_world_process_zero():
        processor.save_pretrained(str(args.output_dir / "joint"))

    # The joint trainer has already restored the best dev checkpoint.  Save a
    # stable root checkpoint for later feature extraction/evaluation.
    joint_trainer.save_model(str(args.output_dir))
    if joint_trainer.is_world_process_zero():
        processor.save_pretrained(str(args.output_dir))
    test_metrics = joint_trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test") if test_dataset is not None else {}

    state = joint_trainer.state
    summary = {
        **config_payload,
        "head": {
            **head_stats,
            "best_checkpoint": head_trainer.state.best_model_checkpoint,
            "best_dev_wer": head_trainer.state.best_metric,
            "global_step": head_trainer.state.global_step,
            "epoch": head_trainer.state.epoch,
            "train_metrics": head_result.metrics,
        },
        "joint": {
            **joint_stats,
            "best_checkpoint": state.best_model_checkpoint,
            "best_dev_wer": state.best_metric,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "max_epochs": args.joint_max_epochs,
            "early_stopped": bool(state.global_step < state.max_steps),
            "train_metrics": joint_result.metrics,
        },
        "best_checkpoint": str(args.output_dir),
        "best_dev_wer": state.best_metric,
        "test_wer": test_metrics.get("test_wer"),
        "test_metrics": test_metrics,
    }
    if joint_trainer.is_world_process_zero():
        (args.output_dir / ("smoke_summary.json" if args.smoke_test else "training_summary.json")).write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True, default=str), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--pretrained-path", type=Path, default=None)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument(
        "--vocab-dir",
        type=Path,
        default=None,
        help="legacy custom vocab; omit to inherit the pretrained processor and CTC head",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=3e-6)
    parser.add_argument("--head-per-device-batch-size", type=int, default=4)
    parser.add_argument("--head-gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--eval-per-device-batch-size", type=int, default=1)
    parser.add_argument("--joint-precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--ctc-loss-reduction", choices=("mean", "sum"), default="mean")
    parser.add_argument("--ctc-zero-infinity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--joint-max-epochs", type=int, default=40)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-dev-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()
    if args.smoke_steps < 1:
        raise ValueError("--smoke-steps must be positive")
    if args.head_learning_rate <= 0 or args.learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.head_per_device_batch_size < 1 or args.head_gradient_accumulation_steps < 1:
        raise ValueError("head batch settings must be positive")
    if args.eval_per_device_batch_size < 1:
        raise ValueError("eval batch size must be positive")
    if args.joint_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("--joint-precision bf16 requires CUDA bf16 support")
    run(args)


if __name__ == "__main__":
    main()
