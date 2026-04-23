#!/bin/bash

set -u

log_dir="training_logs"
mkdir -p "${log_dir}"

# =========================
# Experiment config
# =========================

TASK="humanoid-run"
NUM_FLOW_TRAJS=(24 72 96 120)
ENV_TYPE="dm_control"

SEEDS=(11)
# SEEDS=(11 12 13)

UPDATE_FLOW=true
EXTRA="traj"

# =========================
# Resource config
# =========================

VISIBLE_GPUS=(5)
NUM_GPUS=${#VISIBLE_GPUS[@]}
MAX_TASKS_PER_GPU=4
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

for num_flow_trajs in "${NUM_FLOW_TRAJS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu_index=$(( (task_id / MAX_TASKS_PER_GPU) % NUM_GPUS ))
        gpu=${VISIBLE_GPUS[$gpu_index]}

        while [ "$(current_jobs)" -ge "${MAX_CONCURRENT}" ]; do
            sleep 1
        done

        exp_name="dmc_${TASK}_seed${seed}_${EXTRA}"
        timestamp=$(date "+%Y%m%d_%H%M%S")
        log_file="${log_dir}/${timestamp}_${exp_name}.log"

        echo "Launching: task=${TASK}, seed=${seed}, update_flow=${UPDATE_FLOW}, extra=${EXTRA}, num_flow_trajs=${num_flow_trajs} on GPU ${gpu}"

        CUDA_VISIBLE_DEVICES=${gpu} \
        python mfp/train.py \
            task=${TASK} \
            env_type=${ENV_TYPE} \
            update_flow=${UPDATE_FLOW} \
            seed=${seed} \
            exp_name=${exp_name} \
            num_flow_trajs=${num_flow_trajs} \
            extra="${EXTRA}-${num_flow_trajs}" \
            > "${log_file}" 2>&1 &

        ((task_id++))
    done
done

wait
echo "All flow experiments completed!"