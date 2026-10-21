#!/bin/bash
# Train Diffusion Policy with Dual Camera
#
# This script trains a dual-camera model (wrist + global head camera).
# Optionally can use pairwise pretrained weights for initialization.
# The dual camera zarr data must have img1 and img2.
#
# Usage:
#   bash train_dual_cam_from_pretrained.sh <task_name> <load_num> <agent_id> <seed> <gpu_id> [pretrained_ckpt]
#
# Examples:
#   # Train from scratch
#   bash train_dual_cam_from_pretrained.sh pick_car_clean 1 0 42 0
#
#   # Train from pretrained checkpoint
#   bash train_dual_cam_from_pretrained.sh pick_car_clean 1 0 42 0 \
#       checkpoints/one_arm_place_object_aug_0129_stage1/300.ckpt

task_name=${1}
load_num=${2}
agent_id=${3}
seed=${4}
gpu_id=${5}
pretrained_ckpt=${6:-""}  # Optional, defaults to empty string

DEBUG=False
save_ckpt=True

alg_name=robot_dual_cam_dp
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-dual_cam-${addition_info}

echo -e "\033[33m========================================\033[0m"
if [ -n "$pretrained_ckpt" ]; then
    echo -e "\033[33mTraining Dual Camera Model from Pretrained Checkpoint\033[0m"
else
    echo -e "\033[33mTraining Dual Camera Model from Scratch\033[0m"
fi
echo -e "\033[33m========================================\033[0m"
echo -e "\033[33mTask: ${task_name}\033[0m"
echo -e "\033[33mAgent ID: ${agent_id}\033[0m"
echo -e "\033[33mSeed: ${seed}\033[0m"
echo -e "\033[33mGPU: ${gpu_id}\033[0m"
if [ -n "$pretrained_ckpt" ]; then
    echo -e "\033[33mPretrained Checkpoint: ${pretrained_ckpt}\033[0m"
else
    echo -e "\033[33mPretrained Checkpoint: None (training from scratch)\033[0m"
fi
echo -e "\033[33mData Format: img1=wrist_camera, img2=head_camera_global\033[0m"
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

# Build training command based on whether pretrained checkpoint is provided
if [ -n "$pretrained_ckpt" ]; then
    # Training with pretrained checkpoint
    echo -e "\033[32mLoading pretrained weights from: ${pretrained_ckpt}\033[0m"
    python ./policy/Diffusion-Policy/train.py --config-name=${config_name}.yaml \
                                                task.name=${task_name} \
                                                task.dataset.zarr_path="data/zarr_data/${task_name}_dual_cam_Agent${agent_id}_${load_num}.zarr" \
                                                training.debug=$DEBUG \
                                                training.seed=${seed} \
                                                training.device="cuda:0" \
                                                training.pretrained_checkpoint="${pretrained_ckpt}" \
                                                training.reset_training_state=True \
                                                training.resume=False \
                                                exp_name=${exp_name} \
                                                logging.mode=${wandb_mode}
else
    # Training from scratch
    echo -e "\033[32mTraining from scratch (no pretrained weights)\033[0m"
    python ./policy/Diffusion-Policy/train.py --config-name=${config_name}.yaml \
                                                task.name=${task_name} \
                                                task.dataset.zarr_path="data/zarr_data/${task_name}_dual_cam_Agent${agent_id}_${load_num}.zarr" \
                                                training.debug=$DEBUG \
                                                training.seed=${seed} \
                                                training.device="cuda:0" \
                                                training.resume=False \
                                                exp_name=${exp_name} \
                                                logging.mode=${wandb_mode}
fi
