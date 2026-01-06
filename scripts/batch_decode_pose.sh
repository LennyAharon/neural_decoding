#!/bin/bash
# Batch decode pose targets for multiple models
#
# This script decodes pose targets across multiple models for comparison.
#
# Usage:
#   ./scripts/batch_decode_pose.sh
#
# Before running:
# 1. Download common data with download_common_data.py
# 2. Process pose data for all models with batch_process_pose.sh or process_pose_data.py

# Configuration
BASE_PATH="/media/lenny-aharon/T7/ibl-mouse/ibl-mouse_neural-activity"
N_WORKERS=4
METHOD="linear"  # Options: linear, reduced_rank, mlp, lstm
REGION="all"

# List of EIDs to decode
EIDS=(
    "5c0c560e-9e1f-45e9-b66e-e4ee7855be84"
    # Add more EIDs here
)

# List of pose models to decode
MODEL_NAMES=(
    "multiview_transformer_200_0"
    # Add more model names here
)

# List of pose targets to decode
POSE_TARGETS=(
    "lightning-pose-right-pawR-speed"
    "lightning-pose-left-pawR-speed"
    "lightning-pose-right-pawL-speed"
    "lightning-pose-left-pawL-speed"
)

echo "========================================"
echo "Batch Decoding Pose Targets"
echo "========================================"
echo "Base path: $BASE_PATH"
echo "Method: $METHOD"
echo "Region: $REGION"
echo "Number of EIDs: ${#EIDS[@]}"
echo "Number of models: ${#MODEL_NAMES[@]}"
echo "Number of targets: ${#POSE_TARGETS[@]}"
echo ""

cd "$(dirname "$0")/.."  # Navigate to project root

# Track results
declare -a results

for eid in "${EIDS[@]}"; do
    echo "========================================"
    echo "EID: $eid"
    echo "========================================"
    
    for model_name in "${MODEL_NAMES[@]}"; do
        echo ""
        echo "  Model: $model_name"
        
        for target in "${POSE_TARGETS[@]}"; do
            echo "    Target: $target"
            
            python src/decode_single_session.py \
                --eid "$eid" \
                --target "$target" \
                --method "$METHOD" \
                --base_path "$BASE_PATH" \
                --model_name "$model_name" \
                --region "$REGION" \
                --n_workers "$N_WORKERS"
            
            if [ $? -eq 0 ]; then
                echo "      ✓ Success"
                results+=("$eid,$model_name,$target,success")
            else
                echo "      ✗ Failed"
                results+=("$eid,$model_name,$target,failed")
            fi
        done
    done
done

echo ""
echo "========================================"
echo "Batch decoding complete!"
echo "========================================"
echo ""
echo "Results saved to: $BASE_PATH/results/{eid}/{target}/{model_name}/{method}/{region}/"
echo ""
echo "Summary:"
echo "--------"
for result in "${results[@]}"; do
    echo "  $result"
done

