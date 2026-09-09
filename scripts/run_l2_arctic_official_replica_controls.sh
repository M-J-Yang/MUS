#!/usr/bin/env bash
set -euo pipefail

# Complete the source-matched W1/W2 control package without encoder retraining.
# Usage: FOLD=1 bash scripts/run_l2_arctic_official_replica_controls.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/zbzb/.conda/envs/py311/bin/python}"
PYTHONPATH_ROOT="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
FOLD="${FOLD:-1}"
CHECKPOINT="${CHECKPOINT:-artifacts/runs/l2_arctic_official_ut8/fold${FOLD}/w2v2_large_960h_supcon_local_replica_full_gc}"
MANIFEST_ROOT="${MANIFEST_ROOT:-manifests/l2_arctic_official_ut8/fold${FOLD}}"
CACHE_ROOT="${CACHE_ROOT:-artifacts/features/l2_arctic_official_ut8/fold${FOLD}/w2v2_large_960h_oracle_shift_local_replica}"
UTILITY_DIR="${UTILITY_DIR:-artifacts/results/l2_arctic_official_ut8/fold${FOLD}/w2v2_large_960h_oracle_shift_local_replica_utility}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/results/l2_arctic_official_ut8/fold${FOLD}/w2v2_large_960h_oracle_shift_local_replica_controls}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [[ "$FOLD" != "1" && "$FOLD" != "2" ]]; then
  echo "FOLD must be 1 or 2 for W1/W2 controls" >&2
  exit 2
fi

for required in \
  "$CHECKPOINT/config.json" \
  "$CHECKPOINT/model.safetensors" \
  "$MANIFEST_ROOT/train_utility.jsonl" \
  "$MANIFEST_ROOT/dev.jsonl" \
  "$MANIFEST_ROOT/test.jsonl" \
  "$UTILITY_DIR/utility_shift_taylor_ranking.pt" \
  "$CACHE_ROOT/train_utility/e0" \
  "$CACHE_ROOT/dev/e0" \
  "$CACHE_ROOT/test/e0"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required cached control resource: $required" >&2
    exit 1
  fi
done

PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" scripts/evaluate_official_shift_package.py \
  --checkpoint "$CHECKPOINT" \
  --manifest-root "$MANIFEST_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --utility-dir "$UTILITY_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --retentions 50 75 \
  --deletions \
  --reuse-rankings \
  --overwrite
