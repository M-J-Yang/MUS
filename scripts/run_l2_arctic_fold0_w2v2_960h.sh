#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/zbzb/.conda/envs/py311/bin/torchrun}"
MODEL_ID="${MODEL_ID:-facebook/wav2vec2-large-960h}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/runs/l2_arctic_ut8/fold0/w2v2_large_960h}"
HEAD_BATCH_SIZE="${HEAD_BATCH_SIZE:-4}"
HEAD_GRAD_ACCUM="${HEAD_GRAD_ACCUM:-1}"
JOINT_BATCH_SIZE="${JOINT_BATCH_SIZE:-2}"
JOINT_GRAD_ACCUM="${JOINT_GRAD_ACCUM:-2}"
JOINT_PRECISION="${JOINT_PRECISION:-bf16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
DATALOADER_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
PRETRAINED_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING}" == "true" ]]; then
  GC_ARGS=(--gradient-checkpointing)
else
  GC_ARGS=(--no-gradient-checkpointing)
fi
if [[ -n "${PRETRAINED_PATH:-}" ]]; then
  PRETRAINED_ARGS=(--pretrained-path "$PRETRAINED_PATH")
fi

PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$TORCHRUN_BIN" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29533}" \
  --nproc_per_node="${NPROC_PER_NODE:-4}" \
  "$ROOT/scripts/train_large_ctc.py" \
  --model-id "$MODEL_ID" \
  "${PRETRAINED_ARGS[@]}" \
  --train-manifest manifests/l2_arctic_ut8/fold0/train.jsonl \
  --dev-manifest manifests/l2_arctic_ut8/fold0/dev.jsonl \
  --test-manifest manifests/l2_arctic_ut8/fold0/test.jsonl \
  --output-dir "$OUTPUT_DIR" \
  --head-learning-rate 3e-6 \
  --head-per-device-batch-size "$HEAD_BATCH_SIZE" \
  --head-gradient-accumulation-steps "$HEAD_GRAD_ACCUM" \
  --learning-rate 1e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --per-device-batch-size "$JOINT_BATCH_SIZE" \
  --gradient-accumulation-steps "$JOINT_GRAD_ACCUM" \
  --joint-max-epochs 40 \
  --early-stopping-patience 5 \
  --joint-precision "$JOINT_PRECISION" \
  --eval-per-device-batch-size "$EVAL_BATCH_SIZE" \
  --ctc-loss-reduction mean \
  --ctc-zero-infinity \
  --tf32 \
  "${GC_ARGS[@]}" \
  --seed 1337 \
  --dataloader-num-workers "$DATALOADER_WORKERS"
