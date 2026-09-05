#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/zbzb/.conda/envs/py311/bin/python}"
PYTHONPATH_ROOT="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-checkpoints/data2vec_audio_large_960h}"
FINE_TUNED_MODEL="${FINE_TUNED_MODEL:-artifacts/runs/l2_arctic_official_ut8/fold0/data2vec_audio_large_960h_ctc_formal_b4}"
MANIFEST_ROOT="${MANIFEST_ROOT:-manifests/l2_arctic_official_ut8/fold0}"
CACHE_ROOT="${CACHE_ROOT:-artifacts/features/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift}"
UTILITY_DIR="${UTILITY_DIR:-artifacts/results/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift_utility}"
RESULT_DIR="${RESULT_DIR:-artifacts/results/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift_empirical_package}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CACHE_LOG_EVERY="${CACHE_LOG_EVERY:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

for required in \
  "$PRETRAINED_MODEL/config.json" \
  "$PRETRAINED_MODEL/pytorch_model.bin" \
  "$FINE_TUNED_MODEL/config.json" \
  "$FINE_TUNED_MODEL/model.safetensors" \
  "$MANIFEST_ROOT/train.jsonl" \
  "$MANIFEST_ROOT/dev.jsonl" \
  "$MANIFEST_ROOT/test.jsonl"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required Data2Vec resource: $required" >&2
    exit 1
  fi
done

if [[ ! -f "$MANIFEST_ROOT/train_utility.jsonl" ]]; then
  PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" scripts/prepare_step2_utility_split.py \
    --train-manifest "$MANIFEST_ROOT/train.jsonl" \
    --teacher-out "$MANIFEST_ROOT/train_teacher.jsonl" \
    --utility-out "$MANIFEST_ROOT/train_utility.jsonl" \
    --utility-every 10
fi

for split in train_utility dev test; do
  PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" scripts/cache_l2_shift.py \
    --manifest "$MANIFEST_ROOT/$split.jsonl" \
    --split "$split" \
    --output-root "$CACHE_ROOT" \
    --pretrained-model "$PRETRAINED_MODEL" \
    --fine-tuned-model "$FINE_TUNED_MODEL" \
    --device "$DEVICE" \
    --log-every "$CACHE_LOG_EVERY" \
    --skip-existing
done

PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" utility/compute_l2_shift_taylor_utility.py \
  --checkpoint "$FINE_TUNED_MODEL" \
  --manifest "$MANIFEST_ROOT/train_utility.jsonl" \
  --cache-root "$CACHE_ROOT" \
  --feature-split train_utility \
  --output-dir "$UTILITY_DIR" \
  --device "$DEVICE" \
  --overwrite

PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" scripts/evaluate_official_shift_package.py \
  --checkpoint "$FINE_TUNED_MODEL" \
  --manifest-root "$MANIFEST_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --utility-dir "$UTILITY_DIR" \
  --output-dir "$RESULT_DIR" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --overwrite
