#!/usr/bin/env bash
set -euo pipefail

# Compare matched DADS/magnitude displacement scales using cached D0 features.
# No encoder or CTC head is retrained.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/zbzb/.conda/envs/py311/bin/python}"
PYTHONPATH_ROOT="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT="${CHECKPOINT:-artifacts/runs/l2_arctic_official_ut8/fold0/data2vec_audio_large_960h_ctc_formal_b4}"
MANIFEST_ROOT="${MANIFEST_ROOT:-manifests/l2_arctic_official_ut8/fold0}"
CACHE_ROOT="${CACHE_ROOT:-artifacts/features/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift}"
RANKING_DIR="${RANKING_DIR:-artifacts/results/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift_empirical_package/rankings}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/results/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift_scale_controls}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"

for required in \
  "$CHECKPOINT/config.json" \
  "$CHECKPOINT/model.safetensors" \
  "$MANIFEST_ROOT/dev.jsonl" \
  "$MANIFEST_ROOT/test.jsonl" \
  "$RANKING_DIR/utility_ranking.pt" \
  "$RANKING_DIR/magnitude_ranking.pt" \
  "$CACHE_ROOT/dev/e0" \
  "$CACHE_ROOT/dev/eft" \
  "$CACHE_ROOT/dev/delta" \
  "$CACHE_ROOT/test/e0" \
  "$CACHE_ROOT/test/eft" \
  "$CACHE_ROOT/test/delta"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required cached scale-control resource: $required" >&2
    exit 1
  fi
done

PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" scripts/evaluate_d0_scale_controls.py \
  --checkpoint "$CHECKPOINT" \
  --manifest-root "$MANIFEST_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --ranking-dir "$RANKING_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --retentions 25 50 75 \
  --overwrite
