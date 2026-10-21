#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <config_path> <data_num> <checkpoint_num> [debug_mode] [task_name] [phase_config] [perturbation_config]"
    exit 1
fi
if [ -z "$2" ]; then
    echo "Usage: $0 <config_path> <data_num> <checkpoint_num> [debug_mode] [task_name] [phase_config] [perturbation_config]"
    exit 1
fi
if [ -z "$3" ]; then
    echo "Usage: $0 <config_path> <data_num> <checkpoint_num> [debug_mode] [task_name] [phase_config] [perturbation_config]"
    exit 1
fi

CONFIG_PATH="$1"
DATA_NUM="$2"
CHECKPOINT_NUM="$3"
DEBUG_MODE="${4:-1}"
TASK_NAME="${5:-three_arms_place_objects_123_123}"
PHASE_CONFIG="$6"
PERTURBATION_CONFIG="$7"

CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
PHASE_NAME=""
if [ -n "$PHASE_CONFIG" ]; then
    PHASE_NAME="_$(basename "$PHASE_CONFIG" .yaml)"
fi

if [ -n "$PERTURBATION_CONFIG" ]; then
    EVAL_MODE="random"
else
    EVAL_MODE="clean"
fi

LOG_FILE="eval_results_${CONFIG_NAME}${PHASE_NAME}_${EVAL_MODE}_$(date +"%Y%m%d_%H%M%S").log"

echo "Evaluating task: $TASK_NAME"
echo "Config: $CONFIG_PATH"
echo "Data num: $DATA_NUM, Checkpoint: $CHECKPOINT_NUM"
echo "Eval mode: $EVAL_MODE"
if [ -n "$PERTURBATION_CONFIG" ]; then
    echo "Perturbation config: $PERTURBATION_CONFIG"
fi
echo "Evaluating task: $TASK_NAME" >> "$LOG_FILE"
echo "Config: $CONFIG_PATH" >> "$LOG_FILE"
echo "Eval mode: $EVAL_MODE" >> "$LOG_FILE"
TOTAL=0
SUCCESS=0
SUCCESS_RATE=0

QUIET_FLAG=""
if [[ "$DEBUG_MODE" == "0" || "$DEBUG_MODE" == "false" ]]; then
    QUIET_FLAG="--quiet"
fi

for SEED in {1000..1099}
do
    echo "Running evaluation with seed $SEED for task $CONFIG_PATH..."
    OUTPUT=""

    CMD="python script/MACG/three_arms_place_objects/eval_place_objects.py \
            --config=\"$CONFIG_PATH\" \
            --data-num=$DATA_NUM \
            --checkpoint-num=$CHECKPOINT_NUM \
            --render-mode=\"rgb_array\" \
            -o=\"rgb\" \
            -b=\"cpu\" \
            -n 1 \
            -s $SEED \
            --quiet"

    if [ -n "$TASK_NAME" ]; then
        CMD="$CMD --status=\"$TASK_NAME\""
    fi

    if [ -n "$PHASE_CONFIG" ]; then
        CMD="$CMD --phase_config=\"$PHASE_CONFIG\""
    fi

    if [ -n "$PERTURBATION_CONFIG" ]; then
        CMD="$CMD --perturbation_config=\"$PERTURBATION_CONFIG\""
    fi

    if [ -n "$REPLAY_MANIFEST_FILE" ]; then
        CMD="$CMD --replay_manifest=\"$REPLAY_MANIFEST_FILE\""
    fi

    OUTPUT=$(eval $CMD)

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
echo "Total: $TOTAL, Success: $SUCCESS, Success Rate: $SUCCESS_RATE%" >> "$LOG_FILE"

echo "Evaluation completed. Results saved in $LOG_FILE."
