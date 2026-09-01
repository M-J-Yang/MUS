#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/zbzb/.conda/envs/py311/bin/python}"
RUN_SCRIPT="scripts/train_selected_delta.py"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

gpu_is_free() {
    local gpu="$1"
    local pids
    local stats
    local utilization
    local memory

    pids="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')"
    stats="$(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    utilization="${stats%%,*}"
    memory="${stats##*,}"
    [[ -z "$pids" && "${utilization:-100}" -lt 5 && "${memory:-999999}" -lt 1024 ]]
}

echo "Stage 6 runner started at $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Waiting for one of physical GPU 1, 2, or 3 to be fully free."

while true; do
    for gpu in 1 2 3; do
        if ! gpu_is_free "$gpu"; then
            continue
        fi

        echo "Using physical GPU $gpu at $(date '+%Y-%m-%d %H:%M:%S %Z')"
        export CUDA_VISIBLE_DEVICES="$gpu"

        for k in 256 512; do
            for selection in random magnitude utility; do
                output_dir="results/stage6/$selection/k$k"
                metrics_path="$output_dir/metrics.json"

                if [[ -f "$metrics_path" ]]; then
                    echo "Skipping existing $output_dir"
                    continue
                fi

                args=(
                    "$PYTHON_BIN"
                    "$RUN_SCRIPT"
                    --selection "$selection"
                    --k "$k"
                    --output-dir "$output_dir"
                    --device cuda:0
                    --seed 1337
                    --selection-seed 42
                    --num-workers 8
                )
                if [[ "$selection" == "magnitude" ]]; then
                    args+=(--ranking results/stage5/magnitude_ranking.pt)
                elif [[ "$selection" == "utility" ]]; then
                    args+=(--ranking results/stage5/utility_ranking.pt)
                fi

                echo "Starting $selection K=$k at $(date '+%Y-%m-%d %H:%M:%S %Z')"
                "${args[@]}"
            done
        done

        "$PYTHON_BIN" analysis/collect_stage6_results.py
        echo "Stage 6 completed at $(date '+%Y-%m-%d %H:%M:%S %Z')"
        exit 0
    done

    echo "No free target GPU at $(date '+%Y-%m-%d %H:%M:%S %Z'); checking again in 1800s."
    sleep 1800
done
