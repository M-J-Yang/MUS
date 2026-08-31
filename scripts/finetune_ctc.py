#!/usr/bin/env python3
"""Fine-tune a Wav2Vec2- or WavLM-style SSL checkpoint with character CTC."""

from __future__ import annotations

import argparse
import json
import inspect
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCTC, Trainer, TrainingArguments
from transformers.trainer import LengthGroupedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from usde.ctc import AudioCTCDataset, CTCDataCollator, make_processor
from usde.metrics import word_error_rate


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


class LengthAwareTrainer(Trainer):
    """Use cached audio lengths so grouping does not decode the whole corpus once up front."""

    def __init__(self, *args, head_learning_rate: float | None = None, blank_logit_penalty: float = 0.0, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.head_learning_rate = head_learning_rate
        self.blank_logit_penalty = blank_logit_penalty

    def compute_loss(self, model, inputs, return_outputs=False):  # type: ignore[no-untyped-def]
        if self.blank_logit_penalty <= 0:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        blank_probability = outputs.logits.log_softmax(dim=-1)[..., model.config.pad_token_id].exp()
        loss = outputs.loss + self.blank_logit_penalty * blank_probability.mean()
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):  # type: ignore[no-untyped-def]
        """Optionally give the freshly initialized CTC head a separate LR."""
        if self.optimizer is not None:
            return self.optimizer
        if self.head_learning_rate is None:
            return super().create_optimizer()
        head_parameters = []
        backbone_parameters = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            (head_parameters if name.startswith("lm_head.") else backbone_parameters).append(parameter)
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(
            [
                {"params": backbone_parameters, "lr": self.args.learning_rate},
                {"params": head_parameters, "lr": self.head_learning_rate},
            ],
            **optimizer_kwargs,
        )
        return self.optimizer

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


def choose_precision(requested: str, device: torch.device) -> tuple[bool, bool]:
    if requested == "fp16":
        if device.type != "cuda":
            raise ValueError("--precision fp16 requires a CUDA device")
        return True, False
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("--precision bf16 requires a GPU with bfloat16 support")
        return False, True
    return False, False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_tsv", "--train-tsv", dest="train_tsv", type=Path, required=True)
    parser.add_argument("--dev_tsv", "--dev-tsv", dest="dev_tsv", type=Path, required=True)
    parser.add_argument("--vocab_dir", "--vocab-dir", dest="vocab_dir", type=Path, default=Path("assets/ctc_vocab"))
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--audio_root", "--audio-root", dest="audio_root", type=Path, default=None)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=3e-5)
    parser.add_argument("--head_learning_rate", "--head-learning-rate", dest="head_learning_rate", type=float, default=None)
    parser.add_argument("--warmup_ratio", "--warmup-ratio", dest="warmup_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", "--warmup-steps", dest="warmup_steps", type=int, default=None)
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=0.01)
    parser.add_argument("--per_device_batch_size", "--per-device-batch-size", dest="per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", "--gradient-accumulation-steps", dest="gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--gradient_checkpointing", "--gradient-checkpointing", dest="gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="CUDA device such as cuda:0; defaults to CUDA when available")
    parser.add_argument("--max_train_examples", "--max-train-examples", dest="max_train_examples", type=int, default=None)
    parser.add_argument("--max_dev_examples", "--max-dev-examples", dest="max_dev_examples", type=int, default=None)
    parser.add_argument("--disable_spec_augment", "--disable-spec-augment", dest="disable_spec_augment", action="store_true")
    parser.add_argument("--layerdrop", type=float, default=None)
    parser.add_argument("--eval_steps", "--eval-steps", dest="eval_steps", type=int, default=None)
    parser.add_argument("--force_input_normalize", "--force-input-normalize", dest="force_input_normalize", action="store_true")
    parser.add_argument("--ctc_head_init_std", "--ctc-head-init-std", dest="ctc_head_init_std", type=float, default=None)
    parser.add_argument("--ctc_head_blank_bias", "--ctc-head-blank-bias", dest="ctc_head_blank_bias", type=float, default=None)
    parser.add_argument("--blank_logit_penalty", "--blank-logit-penalty", dest="blank_logit_penalty", type=float, default=0.0)
    parser.add_argument("--overwrite_output_dir", "--overwrite-output-dir", dest="overwrite_output_dir", action="store_true")
    args = parser.parse_args()

    patch_accelerate_compat()
    if args.per_device_batch_size < 1 or args.gradient_accumulation_steps < 1 or args.epochs <= 0:
        raise ValueError("batch sizes and epochs must be positive")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fp16, bf16 = choose_precision(args.precision, device)
    set_seed(args.seed)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite_output_dir:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite_output_dir to reuse it")

    processor = make_processor(args.model_name, args.vocab_dir)
    if args.force_input_normalize:
        processor.feature_extractor.do_normalize = True
    vocab_size = len(processor.tokenizer)
    config_kwargs = dict(
        vocab_size=vocab_size,
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
    )
    if args.disable_spec_augment:
        config_kwargs.update(mask_time_prob=0.0, mask_feature_prob=0.0, layerdrop=0.0)
    elif args.layerdrop is not None:
        config_kwargs["layerdrop"] = args.layerdrop
    config = AutoConfig.from_pretrained(
        args.model_name,
        **config_kwargs,
    )
    model = AutoModelForCTC.from_pretrained(
        args.model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )
    if args.ctc_head_init_std is not None:
        if args.ctc_head_init_std <= 0:
            raise ValueError("--ctc_head_init_std must be positive")
        torch.nn.init.normal_(model.lm_head.weight, mean=0.0, std=args.ctc_head_init_std)
        torch.nn.init.zeros_(model.lm_head.bias)
    if args.ctc_head_blank_bias is not None:
        model.lm_head.bias.data[processor.tokenizer.pad_token_id] = args.ctc_head_blank_bias
    if not hasattr(model, "freeze_feature_encoder"):
        raise AttributeError(f"{model.__class__.__name__} does not expose freeze_feature_encoder()")
    model.freeze_feature_encoder()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset = AudioCTCDataset(
        args.train_tsv, processor, args.audio_root, max_examples=args.max_train_examples, sample_seed=args.seed
    )
    dev_dataset = AudioCTCDataset(
        args.dev_tsv, processor, args.audio_root, max_examples=args.max_dev_examples, sample_seed=args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.0 if args.warmup_steps is not None else args.warmup_ratio,
        warmup_steps=args.warmup_steps or 0,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        group_by_length=True,
        evaluation_strategy="steps" if args.eval_steps is not None else "epoch",
        eval_steps=args.eval_steps,
        save_strategy="steps" if args.eval_steps is not None else "epoch",
        save_steps=args.eval_steps or 500,
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        no_cuda=device.type == "cpu",
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )

    tokenizer = processor.tokenizer

    def compute_metrics(eval_prediction):  # type: ignore[no-untyped-def]
        logits = eval_prediction.predictions[0] if isinstance(eval_prediction.predictions, tuple) else eval_prediction.predictions
        predicted_ids = np.argmax(logits, axis=-1)
        label_ids = np.where(eval_prediction.label_ids == -100, tokenizer.pad_token_id, eval_prediction.label_ids)
        predictions = [str(text).lower().strip() for text in tokenizer.batch_decode(predicted_ids)]
        references = [str(text).lower().strip() for text in tokenizer.batch_decode(label_ids, group_tokens=False)]
        return {"wer": float(word_error_rate(references, predictions))}

    trainer = LengthAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=CTCDataCollator(processor),
        compute_metrics=compute_metrics,
        head_learning_rate=args.head_learning_rate,
        blank_logit_penalty=args.blank_logit_penalty,
    )
    train_result = trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    trainer.save_state()
    summary = {
        "model_name": args.model_name,
        "output_dir": str(args.output_dir),
        "train_examples": len(train_dataset),
        "dev_examples": len(dev_dataset),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_wer": trainer.state.best_metric,
        "train_metrics": train_result.metrics,
        "vocab_size": vocab_size,
        "feature_encoder_frozen": True,
        "head_learning_rate": args.head_learning_rate,
        "warmup_steps": args.warmup_steps,
        "layerdrop": args.layerdrop,
        "eval_steps": args.eval_steps,
        "spec_augment_disabled": args.disable_spec_augment,
        "input_normalize_forced": args.force_input_normalize,
        "ctc_head_init_std": args.ctc_head_init_std,
        "ctc_head_blank_bias": args.ctc_head_blank_bias,
        "blank_logit_penalty": args.blank_logit_penalty,
        "effective_batch_size": args.per_device_batch_size * args.gradient_accumulation_steps,
        "sample_rate": 16000,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
