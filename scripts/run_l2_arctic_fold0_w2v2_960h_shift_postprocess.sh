#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zbzb/.conda/envs/py311/bin/python}"
PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT="artifacts/runs/l2_arctic_ut8/fold0/w2v2_large_960h"
CACHE_ROOT="artifacts/features/l2_arctic_ut8/fold0/w2v2_large_960h_shift"
UTILITY_DIR="artifacts/results/l2_arctic_ut8/fold0/w2v2_large_960h_shift_utility"
RESULT_DIR="artifacts/results/l2_arctic_ut8/fold0/w2v2_large_960h_shift"

if [[ ! -f "$CHECKPOINT/training_summary.json" ]]; then
  printf 'missing completed formal training summary: %s\n' "$CHECKPOINT/training_summary.json" >&2
  exit 1
fi

for split in train_utility dev test; do
  PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" scripts/cache_l2_shift.py \
    --manifest "manifests/l2_arctic_ut8/fold0/${split}.jsonl" \
    --split "$split" \
    --output-root "$CACHE_ROOT" \
    --pretrained-model checkpoints/wav2vec2_large_960h_pretrained \
    --fine-tuned-model "$CHECKPOINT" \
    --device "${SHIFT_DEVICE:-cuda:0}" \
    --skip-existing
done

PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" utility/compute_l2_shift_taylor_utility.py \
  --checkpoint "$CHECKPOINT" \
  --manifest manifests/l2_arctic_ut8/fold0/train_utility.jsonl \
  --cache-root "$CACHE_ROOT" \
  --feature-split train_utility \
  --output-dir "$UTILITY_DIR" \
  --device "${SHIFT_DEVICE:-cuda:0}" \
  --overwrite

PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" scripts/evaluate_l2_shift.py \
  --checkpoint "$CHECKPOINT" \
  --manifest manifests/l2_arctic_ut8/fold0/dev.jsonl \
  --cache-root "$CACHE_ROOT" \
  --feature-split dev \
  --ranking "$UTILITY_DIR/utility_shift_taylor_ranking.pt" \
  --output "$RESULT_DIR/dev_metrics.json" \
  --device "${SHIFT_DEVICE:-cuda:0}" \
  --overwrite

PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" scripts/evaluate_l2_shift.py \
  --checkpoint "$CHECKPOINT" \
  --manifest manifests/l2_arctic_ut8/fold0/test.jsonl \
  --cache-root "$CACHE_ROOT" \
  --feature-split test \
  --ranking "$UTILITY_DIR/utility_shift_taylor_ranking.pt" \
  --output "$RESULT_DIR/test_metrics.json" \
  --device "${SHIFT_DEVICE:-cuda:0}" \
  --overwrite
