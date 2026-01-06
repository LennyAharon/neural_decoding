#!/bin/bash
# Batch process pose data for multiple models
#
# This script processes pose data for multiple models and/or EIDs.
#
# Usage:
#   ./scripts/batch_process_pose.sh
#
# Before running:
# 1. First download common data with download_common_data.py
# 2. Set the variables below to match your setup

# Configuration
BASE_PATH="/media/lenny-aharon/T7/ibl-mouse/ibl-mouse_neural-activity"
N_WORKERS=4

# List of EIDs to process (or use a file)
EIDS=(
    "5c0c560e-9e1f-45e9-b66e-e4ee7855be84"
    # Add more EIDs here
)

# List of pose models to process
# Each entry: "model_name|pose_model_path"
POSE_MODELS=(
    "multiview_transformer_200_0|/media/lenny-aharon/T7/ibl-mouse/ibl-mouse_pose/test_200_MVT_dlc_patch_masking/multiview_transformer_200_0/videos_new"
    # Add more models here, e.g.:
    # "dlc_baseline|/path/to/dlc/predictions"
    # "another_model|/path/to/another/model/videos_new"
)

echo "========================================"
echo "Batch Processing Pose Data"
echo "========================================"
echo "Base path: $BASE_PATH"
echo "Number of EIDs: ${#EIDS[@]}"
echo "Number of pose models: ${#POSE_MODELS[@]}"
echo ""

cd "$(dirname "$0")/.."  # Navigate to project root

for eid in "${EIDS[@]}"; do
    echo "========================================"
    echo "Processing EID: $eid"
    echo "========================================"
    
    for model_entry in "${POSE_MODELS[@]}"; do
        # Parse model_name and pose_model_path from entry
        IFS='|' read -r model_name pose_model_path <<< "$model_entry"
        
        echo ""
        echo "  Model: $model_name"
        echo "  Path: $pose_model_path"
        echo ""
        
        python src/process_pose_data.py \
            --eid "$eid" \
            --base_path "$BASE_PATH" \
            --pose_model_path "$pose_model_path" \
            --model_name "$model_name" \
            --n_workers "$N_WORKERS"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully processed $model_name for $eid"
        else
            echo "  ✗ Failed to process $model_name for $eid"
        fi
    done
done

echo ""
echo "========================================"
echo "Batch processing complete!"
echo "========================================"
echo ""
echo "Pose data saved to: $BASE_PATH/pose_aligned/{model_name}/{eid}/"
echo ""
echo "To decode, use:"
echo "  python src/decode_single_session.py --eid <eid> --target lightning-pose-right-pawL-speed --model_name <model_name> ..."

