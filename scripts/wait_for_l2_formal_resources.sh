#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS="$ROOT/checkpoints/wav2vec2_large_960h_pretrained/pytorch_model.bin"
OUTPUT="$ROOT/artifacts/runs/l2_arctic_ut8/fold0/w2v2_large_960h"
EXPECTED_BYTES=1262009187
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"

while true; do
  size=0
  if [[ -f "$WEIGHTS" ]]; then
    size="$(stat -c '%s' "$WEIGHTS")"
  fi
  wavlm_running=0
  if pgrep -f 'train_large_ctc.py.*--output-dir artifacts/runs/l2_arctic_ut8/fold0/wavlm_large' >/dev/null; then
    wavlm_running=1
  fi
  gpu_free_ready=0
  if free_mib_values="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; then
    gpu_free_ready=1
    while IFS= read -r free_mib; do
      if [[ "$free_mib" -lt "$MIN_FREE_MIB" ]]; then
        gpu_free_ready=0
        break
      fi
    done <<< "$free_mib_values"
  fi
  if [[ "$size" -eq "$EXPECTED_BYTES" && "$wavlm_running" -eq 0 && "$gpu_free_ready" -eq 1 ]]; then
    break
  fi
  sleep 30
done

if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -print -quit | grep -q .; then
  printf 'refusing to overwrite non-empty formal output: %s\n' "$OUTPUT" >&2
  exit 1
fi

cd "$ROOT"
export PRETRAINED_PATH="checkpoints/wav2vec2_large_960h_pretrained"
bash scripts/run_l2_arctic_fold0_w2v2_960h.sh
bash scripts/run_l2_arctic_fold0_w2v2_960h_shift_postprocess.sh
