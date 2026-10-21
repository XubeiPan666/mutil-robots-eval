#!/bin/bash
# Train Pairwise Diffusion Policy with multiple H5 files
#
# Usage Option 1 - Clean file + Directory (RECOMMENDED):
#   bash train_pairwise.sh <task_name> <agent_id> <seed> <gpu_id> --clean <clean.h5> --dir <perturbed_dir>
#
# Usage Option 2 - Directory only:
#   bash train_pairwise.sh <task_name> <agent_id> <seed> <gpu_id> --dir <h5_directory>
#
# Usage Option 3 - File list:
#   bash train_pairwise.sh <task_name> <agent_id> <seed> <gpu_id> <h5_file1> <h5_file2> ...
#
# Examples:
#   # Clean file + perturbed directory (RECOMMENDED)
#   bash train_pairwise.sh pairwise_pick_car 0 42 0 \
#       --clean /path/to/clean.h5 \
#       --dir /path/to/perturbed_files/
#
#   # Directory only (all files in directory)
#   bash train_pairwise.sh pairwise_pick_car 0 42 0 --dir /path/to/h5_files/
#
#   # File list
#   bash train_pairwise.sh pairwise_pick_car 0 42 0 data/h5/clean.h5 data/h5/perturbed_0.h5

set -e

task_name=${1}
agent_id=${2}
seed=${3}
gpu_id=${4}
shift 4

DEBUG=False
save_ckpt=True

alg_name=pairwise_dp
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-${alg_name}-${addition_info}

echo -e "\033[33m========================================\033[0m"
echo -e "\033[33mPairwise Diffusion Policy Training\033[0m"
echo -e "\033[33m========================================\033[0m"
echo -e "\033[33mTask: ${task_name}\033[0m"
echo -e "\033[33mAgent ID: ${agent_id}\033[0m"
echo -e "\033[33mSeed: ${seed}\033[0m"
echo -e "\033[33mGPU: ${gpu_id}\033[0m"

# Parse arguments
h5_clean=""
h5_dir=""
h5_files=()

while [ $# -gt 0 ]; do
    case $1 in
        --clean)
            h5_clean=$2
            shift 2
            ;;
        --dir)
            h5_dir=$2
            shift 2
            ;;
        *)
            h5_files+=("$1")
            shift
            ;;
    esac
done

# Build hydra arguments
h5_args=()

if [ -n "$h5_clean" ]; then
    h5_args+=("task.dataset.h5_clean=${h5_clean}")
    echo -e "\033[33mClean H5 file: ${h5_clean}\033[0m"
fi

if [ -n "$h5_dir" ]; then
    h5_args+=("task.dataset.h5_dir=${h5_dir}")
    echo -e "\033[33mPerturbed H5 directory: ${h5_dir}\033[0m"
    echo -e "\033[36m  (will auto-discover all .h5 files)\033[0m"
fi

if [ ${#h5_files[@]} -gt 0 ]; then
    if [ ${#h5_files[@]} -lt 2 ]; then
        echo "Error: At least 2 H5 files are required when using file list"
        exit 1
    fi
    echo -e "\033[33mH5 files (${#h5_files[@]} files):\033[0m"
    for f in "${h5_files[@]}"; do
        echo -e "\033[36m  - $f\033[0m"
    done
    h5_files_str=$(IFS=,; echo "${h5_files[*]}")
    h5_args+=("task.dataset.h5_files=[${h5_files_str}]")
fi

if [ ${#h5_args[@]} -eq 0 ]; then
    echo "Error: No H5 files specified"
    echo "Usage: bash train_pairwise.sh <task_name> <agent_id> <seed> <gpu_id> --clean <clean.h5> --dir <perturbed_dir>"
    exit 1
fi

echo -e "\033[33m========================================\033[0m"

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

# Build command
cmd="python ./policy/Diffusion-Policy/train.py --config-name=${config_name}.yaml \
    task.name=${task_name} \
    task.dataset.agent_id=${agent_id} \
    training.debug=$DEBUG \
    training.seed=${seed} \
    training.device=\"cuda:0\" \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode}"

# Add H5 arguments
for arg in "${h5_args[@]}"; do
    cmd="$cmd $arg"
done

# Execute
eval $cmd
