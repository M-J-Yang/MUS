#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home/zbzb/.conda/envs/MUS/bin/python}
DEVICE=${STAGE7C_DEVICE:-cuda:0}
BATCH_SIZE=${STAGE7C_BATCH_SIZE:-8}

cd "$ROOT"

PYTHONPATH="$ROOT/src" "$PYTHON" scripts/prepare_stage7c_matched.py \
  --l2-manifest manifests/arctic_step2/l2/train_utility.jsonl \
  --cmu-manifest data/processed/arctic/cmu_manifest.jsonl \
  --output-dir manifests/stage7c

PYTHONPATH="$ROOT/src:$ROOT/scripts" "$PYTHON" scripts/extract_stage7c_features.py \
  --manifest manifests/stage7c/cmu.jsonl \
  --output-root artifacts/features/stage7c_cmu \
  --wavlm-ft checkpoints/wavlm_myst_fullfinetune \
  --w2v2-ft checkpoints/w2v2_myst_fullfinetune \
  --w2v2-pt checkpoints/w2v2_large_lv60_pretrained \
  --layer 24 \
  --reference-layer 24 \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --skip-existing

PYTHONPATH="$ROOT/src:$ROOT" "$PYTHON" utility/compute_ctc_taylor_utility.py \
  --checkpoint artifacts/runs/stage4/full_delta/best.pt \
  --manifest manifests/stage7c/l2.jsonl \
  --feature-root artifacts/features/stage3_l2 \
  --feature-split train \
  --vocab assets/ctc_vocab/vocab.json \
  --output-dir results/stage7c \
  --output-stem utility_v4_l2 \
  --reference-dim 1024 \
  --delta-dim 1024 \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --log-every 200 \
  --overwrite

PYTHONPATH="$ROOT/src:$ROOT" "$PYTHON" utility/compute_ctc_taylor_utility.py \
  --checkpoint artifacts/runs/stage4/full_delta/best.pt \
  --manifest manifests/stage7c/cmu.jsonl \
  --feature-root artifacts/features/stage7c_cmu \
  --feature-split train \
  --vocab assets/ctc_vocab/vocab.json \
  --output-dir results/stage7c \
  --output-stem utility_v4_cmu \
  --reference-dim 1024 \
  --delta-dim 1024 \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --log-every 200 \
  --overwrite

PYTHONPATH="$ROOT/src:$ROOT" "$PYTHON" analysis/compare_stage7c_control.py \
  --output results/stage7c/matched_text_control.json
