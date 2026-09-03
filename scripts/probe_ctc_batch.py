#!/usr/bin/env python3
"""Find the largest safe per-device CTC batch on a target GPU.

The probe uses the longest utterances in the manifest and performs one AdamW
update, so optimizer state and activation memory are both included.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForCTC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from usde.ctc import AudioCTCDataset, CTCDataCollator, make_processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidates", default="1,2,3,4,5,6")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the CTC batch probe requires CUDA")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("target GPU does not support bf16")

    processor = make_processor(str(args.model), None)
    config = AutoConfig.from_pretrained(
        str(args.model), ctc_loss_reduction="mean", ctc_zero_infinity=True, layerdrop=0.0
    )
    model = AutoModelForCTC.from_pretrained(str(args.model), config=config).to(device)
    model.config.ctc_loss_reduction = "mean"
    model.config.ctc_zero_infinity = True
    model.freeze_feature_encoder()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=1e-5, weight_decay=0.01)
    dataset = AudioCTCDataset(args.manifest, processor)
    collator = CTCDataCollator(processor)
    order = sorted(range(len(dataset)), key=lambda i: dataset.input_lengths[i], reverse=True)
    candidates = [int(x) for x in args.candidates.split(",") if x.strip()]
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.precision)
    print(f"device={device} model={args.model} examples={len(dataset)}", flush=True)
    print(f"longest_seconds={dataset.input_lengths[order[0]] / 16000:.2f}", flush=True)

    best = 0
    for batch_size in candidates:
        if batch_size < 1 or batch_size > len(order):
            continue
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        batch = collator([dataset[i] for i in order[:batch_size]])
        batch = {key: value.to(device) for key, value in batch.items()}
        try:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss={loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            reserved = torch.cuda.max_memory_reserved(device) / 2**30
            print(f"batch={batch_size} status=PASS loss={float(loss):.4f} peak_alloc_gb={peak:.2f} peak_reserved_gb={reserved:.2f}", flush=True)
            best = batch_size
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            print(f"batch={batch_size} status=OOM peak_alloc_gb={peak:.2f}", flush=True)
            break
        finally:
            del batch
            if "loss" in locals():
                del loss
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            gc.collect()
    print(f"recommended_max_batch={best}", flush=True)


if __name__ == "__main__":
    main()
