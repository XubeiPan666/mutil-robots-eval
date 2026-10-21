#!/bin/bash

# Evaluation script for PlaceFactory cube stacking task
# Usage:
#   bash script/MACG/eval_stackcube.sh <data_num> <checkpoint_num> [phase_config] [debug_mode]
#
# Examples:
#   bash script/MACG/eval_stackcube.sh 1 300 script/MACG/phase_config_stackcube.yaml
#   bash script/MACG/eval_stackcube.sh 1 300 script/MACG/phase_config_stackcube.yaml true

if [ -z "$1" ]; then
    echo "Usage: $0 <data_num> <checkpoint_num> [phase_config] [debug_mode]"
    echo "  data_num:       training data count (metadata)"
    echo "  checkpoint_num: checkpoint epoch number (metadata)"
    echo "  phase_config:   optional, path to phase config YAML file"
    echo "  debug_mode:     optional, default false (quiet mode). Set to true for verbose output"
    exit 1
fi
if [ -z "$2" ]; then
    echo "Error: missing checkpoint_num argument"
    echo "Usage: $0 <data_num> <checkpoint_num> [phase_config] [debug_mode]"
    exit 1
fi

DATA_NUM="$1"
CHECKPOINT_NUM="$2"
PHASE_CONFIG="${3:-script/MACG/phase_config_stackcube.yaml}"
DEBUG_MODE="${4:-false}"

# Fixed scene config for PlaceFactory cube stacking
CONFIG_NAME="configs/table/place_factory_cube.yaml"
TASK_NAME="place_factory_cube_123_123"

# Check that phase config file exists (if provided)
if [ -n "$PHASE_CONFIG" ] && [ ! -f "$PHASE_CONFIG" ]; then
    echo "Error: Phase config file not found: $PHASE_CONFIG"
    exit 1
fi

# Generate timestamped log file
LOG_FILE="eval_results_${TASK_NAME}_${DATA_NUM}_${CHECKPOINT_NUM}_$(date +"%Y%m%d_%H%M%S").log"

echo "========================================="
echo "Evaluating task: $TASK_NAME"
echo "Config file: $CONFIG_NAME"
echo "Phase config: $PHASE_CONFIG"
echo "Data num: $DATA_NUM, Checkpoint: $CHECKPOINT_NUM"
echo "Debug mode: $DEBUG_MODE"
echo "Log file: $LOG_FILE"
echo "========================================="
echo "Evaluating task: $TASK_NAME"  >> "$LOG_FILE"
echo "Config: $CONFIG_NAME"  >> "$LOG_FILE"
echo "Phase config: $PHASE_CONFIG"  >> "$LOG_FILE"
echo "Data num: $DATA_NUM, Checkpoint num: $CHECKPOINT_NUM"  >> "$LOG_FILE"
echo "========================================="  >> "$LOG_FILE"

TOTAL=0
SUCCESS=0
SUCCESS_RATE=0

# If DEBUG_MODE is "0" or "false", enable --quiet flag
QUIET_FLAG=""
if [[ "$DEBUG_MODE" == "0" || "$DEBUG_MODE" == "false" ]]; then
    QUIET_FLAG="--quiet"
fi

for SEED in {1000..1099}
do
    echo "Running evaluation with seed $SEED for task $TASK_NAME..."
    OUTPUT=""

    # Build command with optional phase_config
    CMD="python script/MACG/eval_stackcube.py \
            --config=$CONFIG_NAME \
            --status=$TASK_NAME \
            --data-num=$DATA_NUM \
            --checkpoint-num=$CHECKPOINT_NUM \
            --render-mode=sensors \
            -o=rgb \
            -b=cpu \
            -n 1 \
            -s $SEED \
            $QUIET_FLAG"

    if [ -n "$PHASE_CONFIG" ]; then
        CMD="$CMD --phase_config=$PHASE_CONFIG"
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

echo "=========================================" >> "$LOG_FILE"
echo "Total: $TOTAL, Success: $SUCCESS, Success Rate: $SUCCESS_RATE%" >> "$LOG_FILE"

echo ""
echo "========================================="
echo "Evaluation completed!"
echo "Total: $TOTAL, Success: $SUCCESS, Success Rate: $SUCCESS_RATE%"
echo "Results saved to: $LOG_FILE"
echo "========================================="
