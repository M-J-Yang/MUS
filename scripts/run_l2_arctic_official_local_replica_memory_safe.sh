#!/usr/bin/env bash
set -euo pipefail

# Memory-safe companion to run_l2_arctic_official_local_replica.sh.  It keeps
# the same frozen recipe and enables activation checkpointing when the
# 24-GiB cards cannot hold the joint Wav2Vec2-Large step without it.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FOLD="${FOLD:?set FOLD=1 or FOLD=2}"
if [[ "$FOLD" != "1" && "$FOLD" != "2" ]]; then
  echo "FOLD must be 1 or 2" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/home/zbzb/.conda/envs/py311/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/zbzb/.conda/envs/py311/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29561}"
PRETRAINED_PATH="${PRETRAINED_PATH:-checkpoints/wav2vec2_large_960h_pretrained}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/runs/l2_arctic_official_ut8/fold${FOLD}/w2v2_large_960h_supcon_local_replica_full_gc}"
MANIFEST_ROOT="manifests/l2_arctic_official_ut8/fold${FOLD}"

for required in \
  "$PRETRAINED_PATH/config.json" \
  "$MANIFEST_ROOT/train.jsonl" \
  "$MANIFEST_ROOT/dev.jsonl" \
  "$MANIFEST_ROOT/test.jsonl" \
  "$MANIFEST_ROOT/train_utility.jsonl"; do
  if [[ ! -f "$required" ]]; then
    echo "missing replica resource: $required" >&2
    exit 1
  fi
done

ARGS=(
  --nnodes 1 --node_rank 0
  --master_addr "${MASTER_ADDR:-127.0.0.1}"
  --master_port "$MASTER_PORT"
  --nproc_per_node "$NPROC_PER_NODE"
  "$ROOT/scripts/train_large_supcon.py"
  --pretrained-path "$PRETRAINED_PATH"
  --train-manifest "$MANIFEST_ROOT/train.jsonl"
  --dev-manifest "$MANIFEST_ROOT/dev.jsonl"
  --test-manifest "$MANIFEST_ROOT/test.jsonl"
  --output-dir "$OUTPUT_DIR"
  --head-batch-size "${HEAD_BATCH_SIZE:-4}"
  --batch-size "${BATCH_SIZE:-24}"
  --group-size "${GROUP_SIZE:-6}"
  --samples-per-group "${SAMPLES_PER_GROUP:-4}"
  --gradient-accumulation-steps "${GRAD_ACCUM:-1}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-1}"
  --learning-rate "${LEARNING_RATE:-1e-5}"
  --weight-decay "${WEIGHT_DECAY:-0.0}"
  --warmup-ratio "${WARMUP_RATIO:-0.0}"
  --scheduler "${SCHEDULER:-linear}"
  --precision "${PRECISION:-bf16}"
  --supcon-lambda "${SUPCON_LAMBDA:-0.05}"
  --supcon-temp "${SUPCON_TEMP:-0.1}"
  --supcon-ramp-ratio "${SUPCON_RAMP_RATIO:-0.1}"
  --proj-dim "${PROJ_DIM:-256}"
  --epochs "${EPOCHS:-40}"
  --patience "${PATIENCE:-5}"
  --max-duration-s "${MAX_DURATION_S:-10.0}"
  --dataloader-workers "${DATALOADER_WORKERS:-8}"
  --seed "${SEED:-1337}"
  --tf32 --gradient-checkpointing
)
if [[ "${SKIP_HEAD_WARMUP:-0}" == "1" ]]; then
  ARGS+=(--skip-head-warmup)
fi
if [[ "${OVERWRITE_OUTPUT_DIR:-0}" == "1" ]]; then
  ARGS+=(--overwrite-output-dir)
fi

PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$TORCHRUN_BIN" "${ARGS[@]}"
