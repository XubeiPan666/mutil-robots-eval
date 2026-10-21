#!/bin/bash

# 评测 place_factory_object 任务
# 用法: bash script/MACG/eval_placefactory_object.sh <phase_config> <data_num> <checkpoint_num> [task_name] [debug_mode]
# 示例: bash script/MACG/eval_placefactory_object.sh script/MACG/my_phase.yaml 100 300
#       bash script/MACG/eval_placefactory_object.sh script/MACG/my_phase.yaml 100 300 place_factory_object_132

if [ -z "$1" ]; then
    echo "Usage: $0 <phase_config> <data_num> <checkpoint_num> [task_name] [debug_mode]"
    echo "  phase_config:   phase 配置文件路径"
    echo "  data_num:       训练数据量"
    echo "  checkpoint_num: checkpoint epoch 数"
    echo "  task_name:      可选，任务状态名（传给 --status），默认从 phase_config 文件名推导"
    echo "  debug_mode:     可选，默认 false（静默模式）。设为 true 则输出详细信息"
    exit 1
fi
if [ -z "$2" ]; then
    echo "Error: 缺少 data_num 参数"
    echo "Usage: $0 <phase_config> <data_num> <checkpoint_num> [task_name] [debug_mode]"
    exit 1
fi
if [ -z "$3" ]; then
    echo "Error: 缺少 checkpoint_num 参数"
    echo "Usage: $0 <phase_config> <data_num> <checkpoint_num> [task_name] [debug_mode]"
    exit 1
fi

PHASE_CONFIG="$1"
DATA_NUM="$2"
CHECKPOINT_NUM="$3"
DEBUG_MODE="${5:-false}"

# 检查 phase config 文件是否存在
if [ ! -f "$PHASE_CONFIG" ]; then
    echo "Error: Phase config 文件不存在: $PHASE_CONFIG"
    exit 1
fi

# 场景配置固定为 place_factory_object
CONFIG_NAME="configs/table/place_factory_object.yaml"

# task_name: 如果用户传了就用，否则从 phase_config 文件名推导
if [ -n "$4" ]; then
    TASK_NAME="$4"
else
    # 从文件名推导，例如 phase_config_placefactory_object_132.yaml -> place_factory_object_132
    BASENAME=$(basename "$PHASE_CONFIG" .yaml)
    TASK_NAME="$BASENAME"
fi

# 生成带时间戳的日志文件
LOG_FILE="eval_results_${TASK_NAME}_${DATA_NUM}_${CHECKPOINT_NUM}_$(date +"%Y%m%d_%H%M%S").log"

echo "========================================="
echo "评测任务: $TASK_NAME"
echo "配置文件: $CONFIG_NAME"
echo "Phase配置: $PHASE_CONFIG"
echo "数据量: $DATA_NUM, Checkpoint: $CHECKPOINT_NUM"
echo "调试模式: $DEBUG_MODE"
echo "日志文件: $LOG_FILE"
echo "========================================="
echo "Evaluating task: $TASK_NAME"  >> "$LOG_FILE"
echo "Config: $CONFIG_NAME"  >> "$LOG_FILE"
echo "Phase config: $PHASE_CONFIG"  >> "$LOG_FILE"
echo "Data num: $DATA_NUM, Checkpoint num: $CHECKPOINT_NUM"  >> "$LOG_FILE"
echo "========================================="  >> "$LOG_FILE"

TOTAL=0
SUCCESS=0
SUCCESS_RATE=0

for SEED in {1000..1099}
do
    echo "Running evaluation with seed $SEED for task $TASK_NAME..."
    OUTPUT=""
    OUTPUT=$(python script/MACG/eval_placefactory.py \
            --config="$CONFIG_NAME" \
            --phase_config="$PHASE_CONFIG" \
            --data-num=$DATA_NUM \
            --checkpoint-num=$CHECKPOINT_NUM \
            --render-mode="sensors" \
            -o="rgb" \
            -b="cpu" \
            -n 1 \
            -s $SEED \
            --status="$TASK_NAME" \
            --quiet)

    echo "$OUTPUT"
    LAST_LINE=$(echo "$OUTPUT" | tail -n 1)
    FINE=0
    if [[ $LAST_LINE == *"success"* ]]; then
        FINE=1
        SUCCESS=$((SUCCESS + 1))
    fi
    TOTAL=$((TOTAL + 1))
    SUCCESS_RATE=$(echo "scale=4; $SUCCESS / $TOTAL * 100" | bc)
    echo "$SEED, $FINE, $SUCCESS_RATE%" >> "$LOG_FILE"
    echo "Seed $SEED done. Success Rate: $SUCCESS_RATE%"
done

echo "=========================================" >> "$LOG_FILE"
echo "Total: $TOTAL, Success: $SUCCESS, Success Rate: $SUCCESS_RATE%" >> "$LOG_FILE"

echo ""
echo "========================================="
echo "评测完成！"
echo "Total: $TOTAL, Success: $SUCCESS, Success Rate: $SUCCESS_RATE%"
echo "结果已保存至: $LOG_FILE"
echo "========================================="
