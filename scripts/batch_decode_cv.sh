#!/bin/bash
# Batch decode single sessions with cross-validation
#
# Usage:
#   ./scripts/batch_decode_cv.sh

set -e  # Exit on error

# =============================================================================
# CONFIGURATION - Edit these values as needed
# =============================================================================

BASE_PATH="/media/lenny-aharon/T7/ibl-mouse/ibl-mouse_neural-activity"

# List of EIDs to process
EIDS=(
    "aad23144-0e52-4eac-80c5-c4ee2decb198"
    "15b69921-d471-4ded-8814-2adad954bcd8"
    "15763234-d21e-491f-a01b-1238eb96d389"
    "9b528ad0-4599-4a55-9148-96cc1d93fb24"
    "4b00df29-3769-43be-bb40-128b1cba6d35"
    "0841d188-8ef2-4f20-9828-76a94d5343a4"
    "f312aaec-3b6f-44b3-86b4-3a0c119c0438"
    "5c0c560e-9e1f-45e9-b66e-e4ee7855be84"

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
    # "71e55bfe-5a3a-4cba-bdc7-f085140d798e"
    "746d1902-fa59-4cab-b0aa-013be36060d5"
    # "754b74d5-7a06-4004-ae0c-72a10b6ed2e6"
    "824cf03d-4012-4ab1-b499-c83a92c5589e"
    "0f77ca5d-73c2-45bd-aa4c-4c5ed275dbde"
    "1b715600-0cbc-442c-bd00-5b0ac2865de1"
    "3bcb81b4-d9ca-4fc9-a1cd-353a966239ca"
    "03d9a098-07bf-4765-88b7-85f8d8f620cc"

    "3638d102-e8b6-4230-8742-e548cd87a949"
    "6899a67d-2e53-4215-a52a-c7021b5da5d4"
    "8928f98a-b411-497e-aa4b-aa752434686d"
)

# Targets to decode
TARGETS=(
    # "lightning-pose-pawL-3d-speed"
    "lightning-pose-pawR-3d-speed"
    # "lightning-pose-left-pawL-speed"
    # "lightning-pose-right-pawL-speed"
    # "lightning-pose-right-pawR-speed"
    # "lightning-pose-left-pawR-speed"
    # "wheel-speed"
)

# Model names
MODEL_NAMES=(
    # "resnet50_0"
    # "resnet50_1"
    # "resnet50_2"
    # "resnet50_median"
    # "resnet50_median_new"
    # "resnet50_eks_linear"
    # "MVT_3d_aug_patch_0"
    # "MVT_3d_aug_patch_1"
    # "MVT_3d_aug_patch_2"
    # "MVT_3d_aug_patch_median"
    # "MVT_3d_aug_patch_median_new"
    # "MVT_3d_aug_patch_eks"
    # "MVT_3d_aug_patch_eks_linear"
    # "MVT_3d_aug_patch_eks_linear_new"
    # "MVT_3d_loss_patch_median_new"
    # "MVT_3d_aug_patch_median_400"
    "resnet50_anipose_new"
)

# Decoding options
METHOD="linear"
REGION="all"
SEARCH="--search"  # Set to "" to disable

# =============================================================================
# Script logic
# =============================================================================

cd "$(dirname "$0")/.."  # Navigate to project root

echo "========================================"
echo "Batch Decoding with Cross-Validation"
echo "========================================"
echo "EIDs: ${#EIDS[@]}"
echo "Targets: ${#TARGETS[@]}"
echo "Models: ${#MODEL_NAMES[@]}"
echo "Method: $METHOD"
echo ""

for eid in "${EIDS[@]}"; do
    for target in "${TARGETS[@]}"; do
        for model_name in "${MODEL_NAMES[@]}"; do
            echo "----------------------------------------"
            echo "EID: $eid"
            echo "Target: $target"
            echo "Model: $model_name"
            echo "----------------------------------------"
            
            python src/decode_single_session_cv.py \
                --eid "$eid" \
                --target "$target" \
                --method "$METHOD" \
                --base_path "$BASE_PATH" \
                --model_name "$model_name" \
                --region "$REGION" \
                $SEARCH
            
            echo "✓ Done: $eid / $target / $model_name"
            echo ""
        done
    done
done

echo "========================================"
echo "Complete!"
echo "========================================"


# To make things run we need the following
# 1. chmod +x scripts/batch_decode_cv.sh
# 2. ./scripts/batch_decode_cv.sh