#!/usr/bin/env python3
import os
import sys
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

base_path = "/home/lenny-aharon/IBL results decoding/ibl-mouse/ibl-mouse_neural-activity/results"

eids = [
    "15b69921-d471-4ded-8814-2adad954bcd8",
    "15763234-d21e-491f-a01b-1238eb96d389",
    "aad23144-0e52-4eac-80c5-c4ee2decb198",
    "9b528ad0-4599-4a55-9148-96cc1d93fb24",
    "4b00df29-3769-43be-bb40-128b1cba6d35",
    "f312aaec-3b6f-44b3-86b4-3a0c119c0438",
    "0f77ca5d-73c2-45bd-aa4c-4c5ed275dbde",
    "1b715600-0cbc-442c-bd00-5b0ac2865de1",
    "3e6a97d3-3991-49e2-b346-6948cb4580fb",
    "3f859b5c-e73a-4044-b49e-34bb81e96715",
    "4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b",
    "5ae68c54-2897-4d3a-8120-426150704385",
    "5dcee0eb-b34d-4652-acc3-d10afc6eae68",
    "7cb81727-2097-4b52-b480-c89867b5b34c",
    "8a3a0197-b40a-449f-be55-c00b23253bbf",
    "30c4e2ab-dffc-499d-aae4-e51d6b3218c2",
    "51e53aff-1d5d-4182-a684-aba783d50ae5",
    "746d1902-fa59-4cab-b0aa-013be36060d5",
    "824cf03d-4012-4ab1-b499-c83a92c5589e",
    "3638d102-e8b6-4230-8742-e548cd87a949",
    "6899a67d-2e53-4215-a52a-c7021b5da5d4",
    "8928f98a-b411-497e-aa4b-aa752434686d",
    "46794e05-3f6a-4d35-afb3-9165091a5a74",
    "493170a6-fd94-4ee4-884f-cc018c17eeb9",
    "54238fd6-d2d0-4408-b1a9-d19d24fd29ce",
    "73918ae1-e4fd-4c18-b132-00cb555b1ad2",
    "b03fbc44-3d8e-4a6c-8a50-5ea3498568e0",
    "b22f694e-4a34-4142-ab9d-2556c3487086",
    "c7248e09-8c0d-40f2-9eb4-700a8973d8c8",
    "c7bf2d49-4937-4597-b307-9f39cb1c7b16",
    "ca4ecb4c-4b60-4723-9b9e-2c54a6290a53",
    "d0ea3148-948d-4817-94f8-dcaf2342bbbe",
    "d23a44ef-1402-4ed7-97f5-47e9a7a504d9",
    "dc962048-89bb-4e6a-96a9-b062a2be1426",
    "e535fb62-e245-4a48-b119-88ce62a6fe67",
    "f115196e-8dfe-4d2a-8af3-8206d93c1729",
    "f140a2ec-fd49-4814-994a-fe3476f14e66",
    "f3ce3197-d534-4618-bf81-b687555d1883",
    "ff96bfe1-d925-4553-94b5-bf8297adf259",
]

pose_models = [
    "resnet50_median_new",
    "MVT_3d_aug_patch_median_new",
    "MVT_3d_aug_patch_eks_linear_new",
]
targets = [
    "lightning-pose-pawL-3d-speed",
    "lightning-pose-pawR-3d-speed",
]


def compute_trial_consistency(test_y):
    n_trials = test_y.shape[0]
    corr_matrix = np.corrcoef(test_y)
    upper_tri = np.triu_indices(n_trials, k=1)
    pairwise_corrs = corr_matrix[upper_tri]
    return float(np.nanmean(pairwise_corrs))


def main():
    output_lines = []

    for target in targets:
        output_lines.append(f"TARGET: {target}")

        r2_by_model = {}
        rmse_by_model = {}
        mae_by_model = {}
        tc_by_model = {}

        for eid in eids:
            for pose_model in pose_models:
                results_path = os.path.join(
                    base_path, eid, target, pose_model, "linear", "all", f"{eid}_cv.npy"
                )
                if not os.path.exists(results_path):
                    continue

                data = np.load(results_path, allow_pickle=True).item()
                test_y = data["all_test_y"]
                test_pred = data["all_test_pred"]
                y_flat = test_y.flatten()
                pred_flat = test_pred.flatten()

                r2 = r2_score(y_flat, pred_flat)
                rmse = np.sqrt(mean_squared_error(y_flat, pred_flat))
                mae = np.mean(np.abs(y_flat - pred_flat))
                tc = compute_trial_consistency(test_y)

                if pose_model not in r2_by_model:
                    r2_by_model[pose_model] = []
                    rmse_by_model[pose_model] = []
                    mae_by_model[pose_model] = []
                    tc_by_model[pose_model] = []

                r2_by_model[pose_model].append(r2)
                rmse_by_model[pose_model].append(rmse)
                mae_by_model[pose_model].append(mae)
                tc_by_model[pose_model].append(tc)

        for pose_model in pose_models:
            if pose_model in r2_by_model:
                r2_vals = np.array(r2_by_model[pose_model])
                rmse_vals = np.array(rmse_by_model[pose_model])
                mae_vals = np.array(mae_by_model[pose_model])
                tc_vals = np.array(tc_by_model[pose_model])
                n = len(r2_vals)
                line = (
                    f"  {pose_model}: "
                    f"TC={np.mean(tc_vals):.3f}+/-{np.std(tc_vals)/np.sqrt(n):.3f}  "
                    f"R2={np.mean(r2_vals):.3f}+/-{np.std(r2_vals)/np.sqrt(n):.3f}  "
                    f"MAE={np.mean(mae_vals):.2f}+/-{np.std(mae_vals)/np.sqrt(n):.2f}  "
                    f"RMSE={np.mean(rmse_vals):.2f}+/-{np.std(rmse_vals)/np.sqrt(n):.2f}  "
                    f"(n={n})"
                )
                output_lines.append(line)
        output_lines.append("")

    for line in output_lines:
        print(line)


if __name__ == "__main__":
    main()
