#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/zbzb/.conda/envs/py311/bin/python}"
PYTHONPATH_ROOT="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ORACLE_CHECKPOINT="${ORACLE_CHECKPOINT:-artifacts/oracles/wav2vec2-large-l2-arctic-supcon-repeated-8fold-0}"
MANIFEST_ROOT="${MANIFEST_ROOT:-manifests/l2_arctic_official_ut8/fold0}"
CACHE_ROOT="${CACHE_ROOT:-artifacts/features/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift}"
UTILITY_DIR="${UTILITY_DIR:-artifacts/results/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift_utility}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/results/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift_empirical_package}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"

for required in \
  "$ORACLE_CHECKPOINT/config.json" \
  "$ORACLE_CHECKPOINT/model.safetensors" \
  "$MANIFEST_ROOT/train_utility.jsonl" \
  "$MANIFEST_ROOT/dev.jsonl" \
  "$MANIFEST_ROOT/test.jsonl" \
  "$UTILITY_DIR/utility_shift_taylor_ranking.pt" \
  "$CACHE_ROOT/train_utility/e0" \
  "$CACHE_ROOT/dev/e0" \
  "$CACHE_ROOT/test/e0"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required formal resource: $required" >&2
    echo "Run scripts/run_l2_arctic_official_fold0_oracle_shift.sh first." >&2
    exit 1
  fi
done

PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" scripts/evaluate_official_shift_package.py \
  --checkpoint "$ORACLE_CHECKPOINT" \
  --manifest-root "$MANIFEST_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --utility-dir "$UTILITY_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --reuse-rankings \
  --overwrite
