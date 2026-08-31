# Step 1 environment and checkpoint validation

**Date:** 2026-08-28 (Asia/Shanghai)

## Step 1-A.2 — isolated environment — COMPLETE

**Method.** Created `/home/zbzb/.conda/envs/MUS` as a separate Python 3.11
virtual environment. It has its own interpreter and local `site-packages`.
Author-locked core packages were installed into this environment only, using a
PyPI mirror after the default PyPI TLS connection failed.

**Observed result.** `MUS` resolves PyTorch 2.5.1+cu124, torchaudio 2.5.1+cu124,
torchvision 0.20.1, Transformers 4.47.1, Datasets 3.3.2, Evaluate 0.4.3,
jiwer 3.0.5, SoundFile 0.13.0, Accelerate 1.3.0, and W&B 0.20.1. CUDA 12.4
and an RTX 3090 are visible. `py311` was never modified.

**Effect.** Every Step 1 command must explicitly use
`/home/zbzb/.conda/envs/MUS/bin/python`; model and pip caches are under the
project, not the inherited environment.

## Step 1-A.3 — official checkpoint acquisition — COMPLETE

**Method.** Downloaded full local snapshots through the official Hugging Face
repository IDs. The local network proxy terminated TLS; direct connections to
the Hugging Face mirror were used after clearing proxy variables.

**Observed result.** WavLM-ft and W2V2-ft each provide a 24-layer, 1024-D,
16-kHz CTC model with vocabulary size 42. Meta W2V2-pt is a 24-layer, 1024-D,
