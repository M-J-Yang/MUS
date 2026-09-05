#!/usr/bin/env bash
set -euo pipefail

# Data2Vec-Audio-Large-960h Fold0 replication.  The inherited processor and
# pretrained CTC head are intentionally retained by omitting --vocab-dir.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/zbzb/.conda/envs/py311/bin/python}"
PYTHONPATH_ROOT="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
MODEL_ID="${MODEL_ID:-facebook/data2vec-audio-large-960h}"
PRETRAINED_PATH="${PRETRAINED_PATH:-checkpoints/data2vec_audio_large_960h}"
MANIFEST_ROOT="${MANIFEST_ROOT:-manifests/l2_arctic_official_ut8/fold0}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/runs/l2_arctic_official_ut8/fold0/data2vec_audio_large_960h_ctc_formal_b4}"

for required in \
  "$PRETRAINED_PATH/config.json" \
  "$PRETRAINED_PATH/pytorch_model.bin" \
  "$PRETRAINED_PATH/preprocessor_config.json" \
  "$PRETRAINED_PATH/vocab.json" \
  "$MANIFEST_ROOT/train.jsonl" \
  "$MANIFEST_ROOT/dev.jsonl" \
  "$MANIFEST_ROOT/test.jsonl"; do
  if [[ ! -f "$required" ]]; then
    echo "missing Data2Vec replication resource: $required" >&2
    exit 1
  fi
done

ARGS=(
  scripts/train_large_ctc.py
  --model-id "$MODEL_ID"
  --pretrained-path "$PRETRAINED_PATH"
  --train-manifest "$MANIFEST_ROOT/train.jsonl"
  --dev-manifest "$MANIFEST_ROOT/dev.jsonl"
  --test-manifest "$MANIFEST_ROOT/test.jsonl"
  --output-dir "$OUTPUT_DIR"
  --head-learning-rate "${HEAD_LEARNING_RATE:-3e-6}"
  --head-per-device-batch-size "${HEAD_BATCH_SIZE:-4}"
  --head-gradient-accumulation-steps "${HEAD_GRAD_ACCUM:-1}"
  --learning-rate "${LEARNING_RATE:-1e-5}"
  --weight-decay "${WEIGHT_DECAY:-0.01}"
  --warmup-ratio "${WARMUP_RATIO:-0.1}"
  --per-device-batch-size "${BATCH_SIZE:-4}"
  --gradient-accumulation-steps "${GRAD_ACCUM:-1}"
  --eval-per-device-batch-size "${EVAL_BATCH_SIZE:-1}"
  --joint-precision "${PRECISION:-bf16}"
  --joint-max-epochs "${JOINT_MAX_EPOCHS:-40}"
  --early-stopping-patience "${PATIENCE:-5}"
  --dataloader-num-workers "${DATALOADER_WORKERS:-4}"
  --seed "${SEED:-1337}"
  --tf32
  --gradient-checkpointing
)
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  ARGS+=(
    --smoke-test
    --smoke-steps "${SMOKE_STEPS:-2}"
    --max-train-examples "${MAX_TRAIN_EXAMPLES:-16}"
    --max-dev-examples "${MAX_DEV_EXAMPLES:-8}"
    --max-test-examples "${MAX_TEST_EXAMPLES:-8}"
  )
fi
if [[ -n "${HEAD_WARMUP_CHECKPOINT:-}" ]]; then
  ARGS+=(--skip-head-warmup-from "$HEAD_WARMUP_CHECKPOINT")
fi
if [[ "${OVERWRITE_OUTPUT_DIR:-0}" == "1" ]]; then
  ARGS+=(--overwrite-output-dir)
fi

PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" "${ARGS[@]}"
