#!/bin/bash

set -u

log_dir="training_logs_h1"
mkdir -p "${log_dir}"

# =========================
# Experiment config
# =========================

HUMANOID_TASKS=(
    "humanoid_h1hand-stand-v0"
    "humanoid_h1hand-walk-v0"
    "humanoid_h1hand-run-v0"
    "humanoid_h1hand-sit_simple-v0"
    "humanoid_h1hand-slide-v0"
    "humanoid_h1hand-pole-v0"
    "humanoid_h1hand-hurdle-v0"
)
ENV_TYPE="humanoid"

SEEDS=(11 12 13)
# SEEDS=(11 12 13)

UPDATE_FLOW=true
EXTRA="mfp"

# =========================
# Resource config
# =========================

VISIBLE_GPUS=(2 3 4 5)
NUM_GPUS=${#VISIBLE_GPUS[@]}
MAX_TASKS_PER_GPU=3
MAX_CONCURRENT=$((NUM_GPUS * MAX_TASKS_PER_GPU))

current_jobs() {
    jobs -rp | wc -l
}

task_id=0

echo "Starting humanoid flow experiments..."
echo "VISIBLE_GPUS=${VISIBLE_GPUS[*]}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "MAX_TASKS_PER_GPU=${MAX_TASKS_PER_GPU}"
echo "MAX_CONCURRENT=${MAX_CONCURRENT}"

for seed in "${SEEDS[@]}"; do
    for task in "${HUMANOID_TASKS[@]}"; do
        gpu_index=$(( (task_id / MAX_TASKS_PER_GPU) % NUM_GPUS ))
        gpu=${VISIBLE_GPUS[$gpu_index]}

        while [ "$(current_jobs)" -ge "${MAX_CONCURRENT}" ]; do
            sleep 1
        done

        exp_name="humanoid_${task}_seed${seed}_${EXTRA}"
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
echo "All humanoid flow experiments completed!"
