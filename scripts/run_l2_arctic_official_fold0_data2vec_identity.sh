#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/zbzb/.conda/envs/py311/bin/python}"
CHECKPOINT="${CHECKPOINT:-artifacts/runs/l2_arctic_official_ut8/fold0/data2vec_audio_large_960h_ctc_formal_b4}"
MANIFEST_ROOT="${MANIFEST_ROOT:-manifests/l2_arctic_official_ut8/fold0}"
CACHE_ROOT="${CACHE_ROOT:-artifacts/features/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift}"
RANKING="${RANKING:-artifacts/results/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift_utility/utility_shift_taylor_ranking.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/results/l2_arctic_official_ut8/fold0/data2vec_identity}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"

for required in \
  "$CHECKPOINT/config.json" \
  "$CHECKPOINT/model.safetensors" \
  "$RANKING" \
  "$MANIFEST_ROOT/dev.jsonl" \
  "$MANIFEST_ROOT/test.jsonl"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required identity resource: $required" >&2
    exit 1
  fi
done

for split in dev test; do
  PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" scripts/evaluate_l2_shift.py \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST_ROOT/$split.jsonl" \
    --cache-root "$CACHE_ROOT" \
    --feature-split "$split" \
    --ranking "$RANKING" \
    --retain-fraction 0.5 \
    --output "$OUTPUT_ROOT/$split.json" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --overwrite
done
