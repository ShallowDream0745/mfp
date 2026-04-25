#!/bin/bash

set -u

log_dir="training_logs"
mkdir -p "${log_dir}"

# =========================
# Experiment config
# =========================

DM_CONTROL_TASKS=(
    "humanoid-run"
    # "humanoid-walk"
    # "dog-run"
)
ENV_TYPE="dm_control"

SEEDS=(11)
# SEEDS=(11 12 13)

UPDATE_FLOW=true
EXTRA="flow-raw"

# =========================
# Resource config
# =========================

VISIBLE_GPUS=(6 7)
NUM_GPUS=${#VISIBLE_GPUS[@]}
MAX_TASKS_PER_GPU=2
MAX_CONCURRENT=$((NUM_GPUS * MAX_TASKS_PER_GPU))

current_jobs() {
    jobs -rp | wc -l
}

task_id=0

echo "Starting flow experiments..."
echo "VISIBLE_GPUS=${VISIBLE_GPUS[*]}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "MAX_TASKS_PER_GPU=${MAX_TASKS_PER_GPU}"
echo "MAX_CONCURRENT=${MAX_CONCURRENT}"

for seed in "${SEEDS[@]}"; do
    for task in "${DM_CONTROL_TASKS[@]}"; do
        gpu_index=$(( (task_id / MAX_TASKS_PER_GPU) % NUM_GPUS ))
        gpu=${VISIBLE_GPUS[$gpu_index]}

        while [ "$(current_jobs)" -ge "${MAX_CONCURRENT}" ]; do
            sleep 1
        done

        exp_name="dmc_${task}_seed${seed}_${EXTRA}"
        timestamp=$(date "+%Y%m%d_%H%M%S")
        log_file="${log_dir}/${timestamp}_${exp_name}.log"

        echo "Launching: task=${task}, seed=${seed}, update_flow=${UPDATE_FLOW}, extra=${EXTRA} on GPU ${gpu}"

        CUDA_VISIBLE_DEVICES=${gpu} \
        python mfp/train.py \
            task=${task} \
            env_type=${ENV_TYPE} \
            update_flow=${UPDATE_FLOW} \
            seed=${seed} \
            exp_name=${exp_name} \
            extra=${EXTRA} \
            > "${log_file}" 2>&1 &

        ((task_id++))
    done
done

wait
echo "All flow experiments completed!"