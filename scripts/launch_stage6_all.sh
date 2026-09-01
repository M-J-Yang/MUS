#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/zbzb/.conda/envs/py311/bin/python}"
RUN_SCRIPT="scripts/train_selected_delta.py"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
mkdir -p results/stage6/logs

declare -a JOB_PIDS=()
declare -a JOB_NAMES=()

start_job() {
    local selection="$1"
    local k="$2"
    local gpu="$3"
    local output_dir="results/stage6/$selection/k$k"
    local metrics_path="$output_dir/metrics.json"
    local log_path="results/stage6/logs/${selection}_k${k}.log"
    local -a args

    if [[ -f "$metrics_path" ]] && grep -Eq '"test_wer"[[:space:]]*:[[:space:]]*[0-9]' "$metrics_path"; then
        echo "Skipping completed $selection K=$k"
        return 0
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
        --overwrite-output-dir
    )
    if [[ "$selection" == "magnitude" ]]; then
        args+=(--ranking results/stage5/magnitude_ranking.pt)
    elif [[ "$selection" == "utility" ]]; then
        args+=(--ranking results/stage5/utility_ranking.pt)
    fi

    echo "Starting $selection K=$k on physical GPU $gpu; log=$log_path"
    {
        echo "Started $(date '+%Y-%m-%d %H:%M:%S %Z') on physical GPU $gpu"
        CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}"
        status=$?
        echo "Finished $(date '+%Y-%m-%d %H:%M:%S %Z') with exit code $status"
        exit "$status"
    } >"$log_path" 2>&1 &

    JOB_PIDS+=("$!")
    JOB_NAMES+=("$selection K=$k GPU=$gpu")
}

wait_batch() {
    local failures=0
    local index
    for index in "${!JOB_PIDS[@]}"; do
        if wait "${JOB_PIDS[$index]}"; then
            echo "Completed ${JOB_NAMES[$index]}"
        else
            echo "FAILED ${JOB_NAMES[$index]}; inspect results/stage6/logs"
            failures=$((failures + 1))
        fi
    done
    JOB_PIDS=()
    JOB_NAMES=()
    return "$failures"
}

echo "Stage 6 parallel launcher started at $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "GPU mapping: physical 0, 1, 2, 3; user authorized all four GPUs."

# First wave: four independent runs, one per physical GPU.
start_job random 256 0
start_job magnitude 256 1
start_job utility 256 2
start_job random 512 3
wait_batch || exit 1

# Second wave: the remaining two runs.
start_job magnitude 512 0
start_job utility 512 1
wait_batch || exit 1

"$PYTHON_BIN" analysis/collect_stage6_results.py
echo "All Stage 6 runs completed at $(date '+%Y-%m-%d %H:%M:%S %Z')"
