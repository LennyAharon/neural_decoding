"""Process pose data for a specific model and save it aligned with common data.

This script:
1. Loads pose data from a specific model's base_path
2. Applies the same trial filtering and alignment as the common data
3. Saves the aligned pose data separately for each model

Usage:
    python src/process_pose_data.py \
        --eid 5c0c560e-9e1f-45e9-b66e-e4ee7855be84 \
        --base_path /media/lenny-aharon/T7/ibl-mouse/ibl-mouse_neural-activity \
        --pose_model_path /path/to/pose/model/videos_new \
        --model_name multiview_transformer_200_0

Directory structure created:
    base_path/
    ├── ibl_aligned/{eid}/          # Common data (already exists)
    └── pose_aligned/{model_name}/{eid}/
        ├── train_pose.npy
        ├── val_pose.npy
        └── test_pose.npy
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
from one.api import ONE
from utils.ibl_data_utils import (
    get_behavior_per_interval,
    load_pose_data_for_model,
)

logging.basicConfig(level=logging.INFO)

ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, required=True, help="Session ID")
ap.add_argument("--base_path", type=str, required=True, help="Base path for neural data (where ibl_aligned is stored)")
ap.add_argument("--pose_model_path", type=str, required=True, 
                help="Path to pose model predictions (e.g., /path/to/videos_new)")
ap.add_argument("--model_name", type=str, required=True, 
                help="Name of the pose model (used for directory organization)")
ap.add_argument("--n_workers", type=int, default=1)
args = ap.parse_args()

SEED = 42
np.random.seed(SEED)

# Pose behavior names (speed targets)
POSE_BEH_NAMES = [
    'lightning-pose-right-pawR-speed', 
    'lightning-pose-left-pawR-speed',
    'lightning-pose-right-pawL-speed', 
    'lightning-pose-left-pawL-speed',
    'lightning-pose-pawR-3d-speed',
    'lightning-pose-pawL-3d-speed'
]

DYNAMIC_VARS = POSE_BEH_NAMES  # All pose behaviors are dynamic

one = ONE(
    base_url='https://openalyx.internationalbrainlab.org', 
    password='international', silent=True, 
    cache_dir=args.base_path
)

print('==========================')
print(f'Process pose data for session {args.eid}')
print(f'Model: {args.model_name}')
print(f'Pose model path: {args.pose_model_path}')

# Load alignment info from common data
common_data_path = Path(args.base_path) / "ibl_aligned" / args.eid
if not common_data_path.exists():
    raise FileNotFoundError(
        f"Common data not found at {common_data_path}. "
        "Please run download_common_data.py first."
    )

# Load saved trial information
train_idxs = np.load(common_data_path / "train_trial_idxs.npy")
val_idxs = np.load(common_data_path / "val_trial_idxs.npy")
test_idxs = np.load(common_data_path / "test_trial_idxs.npy")
alignment_info = np.load(common_data_path / "alignment_info.npy", allow_pickle=True).item()

params = alignment_info["params"]
bad_trial_idxs = alignment_info["bad_trial_idxs"]
trial_intervals = alignment_info["trial_intervals"]  # Shape: (n_trials, 2)

print(f"Loaded alignment info: {alignment_info['num_aligned_trials']} aligned trials")

# Load frame timestamps to calculate original video indices
# Both left and right cameras share the same timestamps file structure
timestamp_file = Path(args.base_path) / "timestamps" / f"_ibl_leftCamera.times.{args.eid}.npy"
if not timestamp_file.exists():
    # Try alternative location if not in timestamps/
    timestamp_file = Path(args.base_path) / f"_ibl_leftCamera.times.{args.eid}.npy"

if timestamp_file.exists():
    print(f"Loading frame timestamps from {timestamp_file}")
    frame_times = np.load(timestamp_file)
    # Calculate frame indices for each trial
    idxs_beg = np.searchsorted(frame_times, trial_intervals[:, 0], side="right")
    idxs_end = np.searchsorted(frame_times, trial_intervals[:, 1], side="left")
    all_trial_frame_indices = np.c_[idxs_beg, idx_end]
else:
    print(f"WARNING: Frame timestamps not found at {timestamp_file}. Frame indices will not be available.")
    all_trial_frame_indices = None

# Original trial indices (before removing bad trials)
all_original_trial_indices = np.arange(len(trial_intervals))

# Load and bin pose data for this specific model
pose_binned = {}
pose_masks = {}

for pose_beh in POSE_BEH_NAMES:
    print(f"Processing {pose_beh}...")
    
    # Load pose data from the model-specific path
    pose_dict = load_pose_data_for_model(
        one, args.eid, pose_beh, args.pose_model_path
    )
    
    if pose_dict["skip"]:
        print(f"  Skipping {pose_beh} - data not available")
        continue
    
    target_times = pose_dict["times"]
    target_vals = pose_dict["values"]
    
    # Apply same binning as common data (using saved trial intervals)
    target_times_list, target_vals_list, target_mask, skip_reasons = get_behavior_per_interval(
        target_times, 
        target_vals, 
        intervals=trial_intervals,  # Use saved trial intervals instead of trials_df
        trials_df=None, 
        allow_nans=True, 
        n_workers=args.n_workers, 
        **params
    )
    
    pose_binned[pose_beh] = np.array(target_vals_list, dtype=object)
    pose_masks[pose_beh] = target_mask
    
    # Also process ensemble variances if they exist
    for var_name in ["x_ens_var", "y_ens_var", "x_coords", "y_coords"]:
        if pose_dict.get(var_name) is not None:
            _, var_vals_list, _, _ = get_behavior_per_interval(
                target_times, 
                pose_dict[var_name], 
                intervals=trial_intervals, 
                trials_df=None, 
                allow_nans=True, 
                n_workers=args.n_workers, 
                **params
            )
            pose_binned[f"{pose_beh}_{var_name}"] = np.array(var_vals_list, dtype=object)
    
    valid_count = sum(1 for x in target_vals_list if x is not None)
    print(f"  {pose_beh}: {valid_count}/{len(target_vals_list)} valid trials")

# Process 3D Triangulation for each paw
for paw in ["pawR", "pawL"]:
    print(f"Processing 3D triangulation for {paw}...")
    pose_3d_dict = load_pose_data_for_model(
        one, args.eid, f"lightning-pose-{paw}-3d", args.pose_model_path
    )
    
    if pose_3d_dict["skip"]:
        print(f"  Skipping 3D for {paw} - data or aniposelib not available")
        continue
    
    target_times = pose_3d_dict["times"]
    for coord in ["x_3d", "y_3d", "z_3d"]:
        target_vals = pose_3d_dict[coord]
        
        # Apply same binning as common data
        _, var_vals_list, _, _ = get_behavior_per_interval(
            target_times, 
            target_vals, 
            intervals=trial_intervals, 
            trials_df=None, 
            allow_nans=True, 
            n_workers=args.n_workers, 
            **params
        )
        pose_binned[f"lightning-pose-{paw}-{coord}"] = np.array(var_vals_list, dtype=object)
    print(f"  {paw} 3D coordinates binned")

if len(pose_binned) == 0:
    print(f"ERROR: No pose data available for session {args.eid}")
    sys.exit(1)

# Apply same alignment (remove bad trials) as common data
num_trials_orig = len(trial_intervals)
aligned_pose_binned = {}

for beh in pose_binned.keys():
    aligned_pose_binned[beh] = np.delete(pose_binned[beh], bad_trial_idxs, axis=0)
    
    # Convert to proper format
    num_trials = len(aligned_pose_binned[beh])
    aligned_pose_binned[beh] = np.array(
        [y if y is not None else np.nan * np.ones(int(params["interval_len"] / params["binsize"])) 
         for y in aligned_pose_binned[beh]], 
        dtype=float
    ).reshape((num_trials, -1))
    
    # Normalize (same as align_data does for dynamic vars)
    # Only normalize original behaviors (speed), not ensemble variances
    # Note: We are DISABLING normalization for POSE_BEH_NAMES to allow physical comparison
    if beh in POSE_BEH_NAMES:
        pass # Keep original pixels/sec units
    
    # if beh in POSE_BEH_NAMES:
    #     data = aligned_pose_binned[beh]
    #     valid_mask = ~np.isnan(data)
    #     if valid_mask.any():
    #         data_min = np.nanmin(data)
    #         data_max = np.nanmax(data)
    #         if data_max != data_min:
    #             aligned_pose_binned[beh] = (data - data_min) / (data_max - data_min)
    #         else:
    #             aligned_pose_binned[beh] = np.zeros_like(data)

print(f"Aligned pose data: {num_trials} trials")

# Split into train/val/test using the same indices as common data
train_pose, val_pose, test_pose = {}, {}, {}

# Add trial mapping info
aligned_original_trial_indices = np.delete(all_original_trial_indices, bad_trial_idxs)
train_pose["original_trial_indices"] = aligned_original_trial_indices[train_idxs]
val_pose["original_trial_indices"] = aligned_original_trial_indices[val_idxs]
test_pose["original_trial_indices"] = aligned_original_trial_indices[test_idxs]

if all_trial_frame_indices is not None:
    aligned_trial_frame_indices = np.delete(all_trial_frame_indices, bad_trial_idxs, axis=0)
    train_pose["trial_frame_indices"] = aligned_trial_frame_indices[train_idxs]
    val_pose["trial_frame_indices"] = aligned_trial_frame_indices[val_idxs]
    test_pose["trial_frame_indices"] = aligned_trial_frame_indices[test_idxs]

for beh in aligned_pose_binned.keys():
    train_pose[beh] = aligned_pose_binned[beh][train_idxs]
    val_pose[beh] = aligned_pose_binned[beh][val_idxs]
    test_pose[beh] = aligned_pose_binned[beh][test_idxs]

# Save pose data for this model
save_path = Path(args.base_path) / "pose_aligned" / args.model_name / args.eid
os.makedirs(save_path, exist_ok=True)

np.save(save_path / "train_pose.npy", train_pose, allow_pickle=True)
np.save(save_path / "val_pose.npy", val_pose, allow_pickle=True)
np.save(save_path / "test_pose.npy", test_pose, allow_pickle=True)

# Save metadata
metadata = {
    "model_name": args.model_name,
    "pose_model_path": args.pose_model_path,
    "pose_behaviors": list(aligned_pose_binned.keys()),
    "num_train": len(train_idxs),
    "num_val": len(val_idxs),
    "num_test": len(test_idxs),
}
np.save(save_path / "metadata.npy", metadata, allow_pickle=True)

print(f"Saved pose data for model '{args.model_name}' at: {save_path}")
print(f"  Train: {len(train_idxs)} trials")
print(f"  Val: {len(val_idxs)} trials")
print(f"  Test: {len(test_idxs)} trials")
print(f"  Pose behaviors: {list(aligned_pose_binned.keys())}")


# How to run it 
'''
python src/process_pose_data.py \
    --eid 5c0c560e-9e1f-45e9-b66e-e4ee7855be84 \
    --base_path /media/lenny-aharon/T7/ibl-mouse/ibl-mouse_neural-activity \
    --pose_model_path /media/lenny-aharon/T7/ibl-mouse/ibl-mouse_pose/test_200_MVT_dlc_patch_masking/multiview_transformer_200_0/videos_new \
    --model_name MVT_patch_masking_0
'''