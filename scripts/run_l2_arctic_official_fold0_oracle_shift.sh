#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/zbzb/.conda/envs/py311/bin/python}"
PYTHONPATH_ROOT="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-checkpoints/wav2vec2_large_960h_pretrained}"
ORACLE_CHECKPOINT="${ORACLE_CHECKPOINT:-artifacts/oracles/wav2vec2-large-l2-arctic-supcon-repeated-8fold-0}"
MANIFEST_ROOT="${MANIFEST_ROOT:-manifests/l2_arctic_official_ut8/fold0}"
CACHE_ROOT="${CACHE_ROOT:-artifacts/features/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift}"
UTILITY_DIR="${UTILITY_DIR:-artifacts/results/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift_utility}"
RESULT_DIR="${RESULT_DIR:-artifacts/results/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CACHE_LOG_EVERY="${CACHE_LOG_EVERY:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

for required in "$BASE_CHECKPOINT/config.json" "$ORACLE_CHECKPOINT/config.json" "$ORACLE_CHECKPOINT/model.safetensors" "$MANIFEST_ROOT/train.jsonl" "$MANIFEST_ROOT/dev.jsonl" "$MANIFEST_ROOT/test.jsonl"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required official-oracle resource: $required" >&2
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
    --pretrained-model "$BASE_CHECKPOINT" \
    --fine-tuned-model "$ORACLE_CHECKPOINT" \
    --device "$DEVICE" \
    --log-every "$CACHE_LOG_EVERY" \
    --skip-existing
done

PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" utility/compute_l2_shift_taylor_utility.py \
  --checkpoint "$ORACLE_CHECKPOINT" \
  --manifest "$MANIFEST_ROOT/train_utility.jsonl" \
  --cache-root "$CACHE_ROOT" \
  --feature-split train_utility \
  --output-dir "$UTILITY_DIR" \
  --device "$DEVICE" \
  --overwrite

RANKING="$UTILITY_DIR/utility_shift_taylor_ranking.pt"
for fraction in 1.0 0.75 0.5; do
  case "$fraction" in
    1.0) label=100 ;;
    0.75) label=75 ;;
    0.5) label=50 ;;
    *) echo "unsupported retention fraction: $fraction" >&2; exit 1 ;;
  esac
  for split in dev test; do
    PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" scripts/evaluate_l2_shift.py \
      --checkpoint "$ORACLE_CHECKPOINT" \
      --manifest "$MANIFEST_ROOT/$split.jsonl" \
      --cache-root "$CACHE_ROOT" \
      --feature-split "$split" \
      --ranking "$RANKING" \
      --retain-fraction "$fraction" \
      --output "$RESULT_DIR/${split}_${label}.json" \
      --batch-size "$EVAL_BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --device "$DEVICE" \
      --overwrite
  done
done

PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" scripts/summarize_shift_pruning.py \
  --results-dir "$RESULT_DIR" \
  --output-json "$RESULT_DIR/pruning_summary.json" \
  --output-markdown "$RESULT_DIR/pruning_summary.md"
