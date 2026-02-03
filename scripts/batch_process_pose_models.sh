#!/bin/bash
# Batch process pose data for multiple EIDs and pose models
#
# Usage:
#   ./scripts/batch_process_pose_models.sh

set -e  # Exit on error

# =============================================================================
# CONFIGURATION - Edit these values as needed
# =============================================================================

BASE_PATH="/media/lenny-aharon/T7/ibl-mouse/ibl-mouse_neural-activity"
POSE_BASE_PATH="/media/lenny-aharon/T7/ibl-mouse/ibl-mouse_pose"
N_WORKERS=4

# List of EIDs to process
EIDS=(
    # "aad23144-0e52-4eac-80c5-c4ee2decb198"
    # "15b69921-d471-4ded-8814-2adad954bcd8"
    # "15763234-d21e-491f-a01b-1238eb96d389"
    # "9b528ad0-4599-4a55-9148-96cc1d93fb24"
    # "4b00df29-3769-43be-bb40-128b1cba6d35"
    # "0841d188-8ef2-4f20-9828-76a94d5343a4"
    # "f312aaec-3b6f-44b3-86b4-3a0c119c0438"
    # "5c0c560e-9e1f-45e9-b66e-e4ee7855be84"

    "0f77ca5d-73c2-45bd-aa4c-4c5ed275dbde"
    "1b715600-0cbc-442c-bd00-5b0ac2865de1"
    "3bcb81b4-d9ca-4fc9-a1cd-353a966239ca"
    "03d9a098-07bf-4765-88b7-85f8d8f620cc"
    "3e6a97d3-3991-49e2-b346-6948cb4580fb"
    "3f859b5c-e73a-4044-b49e-34bb81e96715"
    "4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b"
    "5ae68c54-2897-4d3a-8120-426150704385"
    "5dcee0eb-b34d-4652-acc3-d10afc6eae68"
    "6c6b0d06-6039-4525-a74b-58cfaa1d3a60"
    "7cb81727-2097-4b52-b480-c89867b5b34c"
    "8a3a0197-b40a-449f-be55-c00b23253bbf"
    "30c4e2ab-dffc-499d-aae4-e51d6b3218c2"
    "51e53aff-1d5d-4182-a684-aba783d50ae5"
    "61e11a11-ab65-48fb-ae08-3cb80662e5d6"
    "746d1902-fa59-4cab-b0aa-013be36060d5"
    "824cf03d-4012-4ab1-b499-c83a92c5589e"
    
    "3638d102-e8b6-4230-8742-e548cd87a949"
    "6899a67d-2e53-4215-a52a-c7021b5da5d4"
    "8928f98a-b411-497e-aa4b-aa752434686d"

    # "71e55bfe-5a3a-4cba-bdc7-f085140d798e"
    # "754b74d5-7a06-4004-ae0c-72a10b6ed2e6"
)

# Model directories and names: "model_dir|model_name"
POSE_MODELS=(
    # "test_200_multiview_resnet50/supervised_200_0|resnet50_0"
    # "test_200_multiview_resnet50/supervised_200_1|resnet50_1"
    # "test_200_multiview_resnet50/supervised_200_2|resnet50_2"
    # "test_200_multiview_resnet50/ensemble_median|resnet50_median"
    # "test_200_multiview_resnet50/eks_multiview_linear|resnet50_eks_linear"
    # "test_200_multiview_resnet50_new/ensemble_median|resnet50_median_new"
    # "test_200_MVT_3d_aug_patch_masking/multiview_transformer_200_0|MVT_3d_aug_patch_0"
    # "test_200_MVT_3d_aug_patch_masking/multiview_transformer_200_1|MVT_3d_aug_patch_1"
    # "test_200_MVT_3d_aug_patch_masking/multiview_transformer_200_2|MVT_3d_aug_patch_2"
    # "test_200_MVT_3d_aug_patch_masking/ensemble_median|MVT_3d_aug_patch_median"
    # "test_200_MVT_3d_aug_patch_masking_new/ensemble_median|MVT_3d_aug_patch_median_new"
    "test_200_MVT_3d_aug_patch_masking_new/eks_multiview_linear|MVT_3d_aug_patch_eks_linear_new"
    # "test_200_MVT_3d_aug_patch_masking/eks_multiview|MVT_3d_aug_patch_eks"
    # "test_200_MVT_3d_aug_patch_masking/eks_multiview_linear|MVT_3d_aug_patch_eks_linear"
    # "test_400_MVT_3d_aug_patch_masking/ensemble_median|MVT_3d_aug_patch_median_400"
)

# =============================================================================
# Script logic
# =============================================================================

cd "$(dirname "$0")/.."  # Navigate to project root

echo "========================================"
echo "Batch Processing Pose Data"
echo "========================================"
echo "EIDs: ${#EIDS[@]}"
echo "Models: ${#POSE_MODELS[@]}"
echo ""

for eid in "${EIDS[@]}"; do
    for model_entry in "${POSE_MODELS[@]}"; do
        IFS='|' read -r model_dir model_name <<< "$model_entry"
        pose_model_path="${POSE_BASE_PATH}/${model_dir}/videos_new"
        
        echo "----------------------------------------"
        echo "EID: $eid"
        echo "Model: $model_name"
        echo "----------------------------------------"
        
        python src/process_pose_data.py \
            --eid "$eid" \
            --base_path "$BASE_PATH" \
            --pose_model_path "$pose_model_path" \
            --model_name "$model_name" \
            --n_workers "$N_WORKERS"
        
        echo "✓ Done: $model_name / $eid"
        echo ""
    done
done

echo "========================================"
echo "Complete!"
echo "========================================"


# To make things run we need the following
# 1. chmod +x scripts/batch_process_pose_models.sh
# 2. ./scripts/batch_process_pose_models.sh


