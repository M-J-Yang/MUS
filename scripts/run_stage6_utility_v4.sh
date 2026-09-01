#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/zbzb/.conda/envs/py311/bin/python}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

for k in 256 512; do
    output_dir="results/stage6/utility_v4/k${k}"
    if [[ -f "$output_dir/metrics.json" ]] && grep -Eq '"test_wer"[[:space:]]*:[[:space:]]*[0-9]' "$output_dir/metrics.json"; then
        echo "Skipping completed utility_v4 K=$k"
        continue
    fi
    "$PYTHON_BIN" scripts/train_selected_delta.py \
        --selection utility_v4 \
        --k "$k" \
        --ranking results/stage5/utility_v4_ranking.pt \
        --output-dir "$output_dir" \
        --device "${DEVICE:-cuda:0}" \
        --seed 1337 \
        --selection-seed 42 \
        --num-workers "${NUM_WORKERS:-8}" \
        --overwrite-output-dir
done

"$PYTHON_BIN" analysis/collect_stage6_results.py
