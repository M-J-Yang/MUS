#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/zbzb/.conda/envs/py311/bin/torchrun}"
PRETRAINED_PATH="${PRETRAINED_PATH:-checkpoints/wav2vec2_large_960h_pretrained}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/runs/l2_arctic_official_ut8/fold0/w2v2_large_960h_supcon_l005}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29551}"
BATCH_SIZE="${BATCH_SIZE:-24}"
GROUP_SIZE="${GROUP_SIZE:-6}"
SAMPLES_PER_GROUP="${SAMPLES_PER_GROUP:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
PRECISION="${PRECISION:-bf16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"

if [[ "$GRADIENT_CHECKPOINTING" == "true" ]]; then
  GC_ARGS=(--gradient-checkpointing)
else
  GC_ARGS=(--no-gradient-checkpointing)
fi

PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$TORCHRUN_BIN" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="$MASTER_PORT" \
  --nproc_per_node="$NPROC_PER_NODE" \
  "$ROOT/scripts/train_large_supcon.py" \
  --pretrained-path "$PRETRAINED_PATH" \
  --train-manifest manifests/l2_arctic_official_ut8/fold0/train.jsonl \
  --dev-manifest manifests/l2_arctic_official_ut8/fold0/dev.jsonl \
  --test-manifest manifests/l2_arctic_official_ut8/fold0/test.jsonl \
  --output-dir "$OUTPUT_DIR" \
  --head-batch-size "${HEAD_BATCH_SIZE:-4}" \
  --batch-size "$BATCH_SIZE" \
  --group-size "$GROUP_SIZE" \
  --samples-per-group "$SAMPLES_PER_GROUP" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-1}" \
  --learning-rate "${LEARNING_RATE:-1e-5}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --warmup-ratio "${WARMUP_RATIO:-0.0}" \
  --scheduler "${SCHEDULER:-linear}" \
  --precision "$PRECISION" \
  --supcon-lambda "${SUPCON_LAMBDA:-0.05}" \
  --supcon-temp "${SUPCON_TEMP:-0.1}" \
  --supcon-ramp-ratio "${SUPCON_RAMP_RATIO:-0.1}" \
  --proj-dim "${PROJ_DIM:-256}" \
  --epochs "${EPOCHS:-40}" \
  --patience "${PATIENCE:-5}" \
  --max-duration-s "${MAX_DURATION_S:-10.0}" \
  --dataloader-workers "${DATALOADER_NUM_WORKERS:-8}" \
  --seed "${SEED:-1337}" \
  --tf32 \
  "${GC_ARGS[@]}"
