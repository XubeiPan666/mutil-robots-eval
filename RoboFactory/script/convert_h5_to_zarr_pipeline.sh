#!/bin/bash
#
# 一键将 H5 数据转换为 Zarr 格式（双相机模式）
#
# 流程：
#   1. H5 -> PKL (parse_h5_to_pkl_dual_cam.py)
#   2. PKL -> Zarr (parse_pkl_to_zarr_dual_cam.py) x agent_num 次
#
# 使用示例：
#   bash script/convert_h5_to_zarr_pipeline.sh --task_name LiftBarrier-rf --load_num 150 --agent_num 2
#
#   # 或者使用简写
#   bash script/convert_h5_to_zarr_pipeline.sh -t LiftBarrier-rf -n 150 -a 2

set -e  # 遇到错误立即退出

# 默认值
TASK_NAME=""
LOAD_NUM=""
AGENT_NUM=1

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --task_name|-t)
            TASK_NAME="$2"
            shift 2
            ;;
        --load_num|-n)
            LOAD_NUM="$2"
            shift 2
            ;;
        --agent_num|-a)
            AGENT_NUM="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --task_name <name> --load_num <num> [--agent_num <num>]"
            echo ""
            echo "Options:"
            echo "  --task_name, -t    Task name (e.g., LiftBarrier-rf)"
            echo "  --load_num, -n     Number of episodes to load"
            echo "  --agent_num, -a    Number of agents (default: 1)"
            echo ""
            echo "Example:"
            echo "  $0 --task_name LiftBarrier-rf --load_num 150 --agent_num 2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [[ -z "$TASK_NAME" ]]; then
    echo "Error: --task_name is required"
    exit 1
fi

if [[ -z "$LOAD_NUM" ]]; then
    echo "Error: --load_num is required"
    exit 1
fi

# 获取脚本所在目录（RoboFactory/script）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# RoboFactory 根目录
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "============================================================"
echo "H5 to Zarr Conversion Pipeline (Dual Camera)"
echo "============================================================"
echo "  Task Name:  $TASK_NAME"
echo "  Load Num:   $LOAD_NUM"
echo "  Agent Num:  $AGENT_NUM"
echo "  Root Dir:   $ROOT_DIR"
echo "============================================================"
echo ""

# 切换到 RoboFactory 根目录
cd "$ROOT_DIR"

# Step 1: H5 -> PKL
echo "[Step 1/$((AGENT_NUM + 1))] Converting H5 to PKL..."
echo "  Command: python script/parse_h5_to_pkl_dual_cam.py --task_name $TASK_NAME --load_num $LOAD_NUM --agent_num $AGENT_NUM"
echo ""

python script/parse_h5_to_pkl_dual_cam.py \
    --task_name "$TASK_NAME" \
    --load_num "$LOAD_NUM" \
    --agent_num "$AGENT_NUM"

echo ""
echo "✓ H5 to PKL conversion complete!"
echo ""

# Step 2: PKL -> Zarr (for each agent)
for ((agent_id=0; agent_id<AGENT_NUM; agent_id++)); do
    step_num=$((agent_id + 2))
    echo "[Step $step_num/$((AGENT_NUM + 1))] Converting PKL to Zarr for Agent $agent_id..."
    echo "  Command: python script/parse_pkl_to_zarr_dual_cam.py --task_name $TASK_NAME --load_num $LOAD_NUM --agent_id $agent_id"
    echo ""
    
    python script/parse_pkl_to_zarr_dual_cam.py \
        --task_name "$TASK_NAME" \
        --load_num "$LOAD_NUM" \
        --agent_id "$agent_id"
    
    echo ""
    echo "✓ Agent $agent_id Zarr conversion complete!"
    echo ""
done

echo "============================================================"
echo "✓ All conversions complete!"
echo ""
echo "Output files:"
for ((agent_id=0; agent_id<AGENT_NUM; agent_id++)); do
    echo "  - data/zarr_data/${TASK_NAME}_dual_cam_Agent${agent_id}_${LOAD_NUM}.zarr"
done
echo "============================================================"
