"""Download common data (neural + non-pose behaviors) that is shared across all pose models.

This script downloads:
- Neural spiking data
- Non-pose behavioral data (choice, reward, block, wheel-speed, whisker-motion-energy)

Pose data is processed separately for each model using process_pose_data.py
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from one.api import ONE
from datasets import DatasetDict, DatasetInfo
from utils.ibl_data_utils import (
    prepare_data_without_pose,
    select_brain_regions, 
    list_brain_regions, 
    bin_spiking_data,
    bin_behaviors,
    align_data_common,
    load_pose_time_window,
)
from utils.dataset_utils import create_dataset

logging.basicConfig(level=logging.INFO)

ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default=None)
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--n_workers", type=int, default=1)
ap.add_argument("--pose_model_path", type=str, required=True,
                help="Path to pose model predictions (used to filter trials by pose time window)")
args = ap.parse_args()

SEED = 42
np.random.seed(SEED)

one = ONE(
    base_url='https://openalyx.internationalbrainlab.org', 
    password='international', silent=True, 
    cache_dir=args.base_path
)

params = {
    "interval_len": 2, 
    "binsize": 0.02, 
    "single_region": False,
    "align_time": 'stimOn_times', 
    "time_window": (-.5, 1.5), 
    "fr_thresh": 0.5
}

# Common behaviors (non-pose) - these are the same across all pose models
COMMON_BEH_NAMES = ['choice', 'reward', 'block', 'wheel-speed', 'whisker-motion-energy']
DYNAMIC_VARS = list(filter(lambda x: x not in ["choice", "reward", "block"], COMMON_BEH_NAMES))

if args.eid is not None:
    include_eids = [args.eid]
else:
    with open("../data/repro_ephys_release.txt") as file:
        include_eids = [line.rstrip().replace("'", "") for line in file]

print(f"Preprocess a total of {len(include_eids)} EIDs.")
print(f"Using pose model path for trial filtering: {args.pose_model_path}")

for eid_idx, eid in enumerate(include_eids):

    print('==========================')
    print(f'Preprocess session {eid} (common data):')

    # Load and preprocess data WITHOUT pose data
    neural_dict, behave_dict, meta_dict, trials_dict, _ = prepare_data_without_pose(
        one, eid, params, n_workers=args.n_workers
    )
    
    if neural_dict is None:
        print(f"Skip EID {eid} due to missing data.")
        continue
    
    # Filter trials to pose data time window (same as original workflow)
    pose_time_start, pose_time_end = load_pose_time_window(one, eid, args.pose_model_path)
    
    if pose_time_start is not None and pose_time_end is not None:
        print(f"Pose data time window: {pose_time_start:.3f} to {pose_time_end:.3f} seconds")
        
        trials_df = trials_dict["trials_df"]
        trials_mask = trials_dict["trials_mask"]
        
        # Calculate trial interval times
        trial_start_times = trials_df[params["align_time"]].values + params["time_window"][0]
        trial_end_times = trials_df[params["align_time"]].values + params["time_window"][1]
        
        # Keep trials where the entire interval falls within pose data window
        valid_trials = (trial_start_times >= pose_time_start) & (trial_end_times <= pose_time_end)
        
        n_trials_before = len(trials_df)
        trials_df = trials_df[valid_trials].reset_index(drop=True)
        trials_mask = trials_mask[valid_trials] if len(trials_mask) == n_trials_before else trials_mask
        
        trials_dict["trials_df"] = trials_df
        trials_dict["trials_mask"] = trials_mask
        
        n_trials_after = len(trials_df)
        print(f"Filtered trials: {n_trials_before} -> {n_trials_after} trials within pose data window")
    else:
        print("Warning: Could not load pose time window, using all trials")
        
    regions, beryl_reg = list_brain_regions(neural_dict, **params)
    region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)

    bin_spikes, clusters_used_in_bins = bin_spiking_data(
        region_cluster_ids, 
        neural_dict, 
        trials_df=trials_dict["trials_df"], 
        n_workers=args.n_workers, 
        **params
    )
    print(f"Binned Spike Data: {bin_spikes.shape}")

    bin_beh, beh_mask = bin_behaviors(
        one, 
        eid, 
        DYNAMIC_VARS, 
        trials_df=trials_dict["trials_df"], 
        allow_nans=True, 
        n_workers=args.n_workers, 
        **params,
    )

    try:
        bin_beh["prior"] = np.load(Path(args.base_path)/"prior_localization"/eid/"priors.npy", allow_pickle=True)
        assert len(bin_beh["prior"]) == len(bin_spikes)
    except:
        bin_beh["prior"] = bin_beh["block"]

    # Align common data only (without pose)
    try:
        align_bin_spikes, align_bin_beh, _, _, bad_trial_idxs = align_data_common(
            bin_spikes, 
            bin_beh, 
            list(bin_beh.keys()), 
            trials_dict["trials_mask"], 
        )
    except ValueError as e:
        print(f"Skip EID {eid} due to error: {e}")
        continue

    if "whisker-motion-energy" not in align_bin_beh:
        logging.info(f"Skip EID {eid} due to missing whisker data.")
        continue

    print("Spike Data Shape: ", align_bin_spikes.shape)

    # Partition dataset (train: 0.7 val: 0.1 test: 0.2)
    num_trials = len(align_bin_spikes)
    trial_idxs = np.random.choice(np.arange(num_trials), num_trials, replace=False)
    train_idxs = trial_idxs[:int(0.7*num_trials)]
    val_idxs = trial_idxs[int(0.7*num_trials):int(0.8*num_trials)]
    test_idxs = trial_idxs[int(0.8*num_trials):]

    train_beh, val_beh, test_beh = {}, {}, {}
    for beh in align_bin_beh.keys():
        train_beh.update({beh: align_bin_beh[beh][train_idxs]})
        val_beh.update({beh: align_bin_beh[beh][val_idxs]})
        test_beh.update({beh: align_bin_beh[beh][test_idxs]})
    
    train_dataset = create_dataset(
        align_bin_spikes[train_idxs], 
        eid, 
        params,
        meta_data=meta_dict,
        binned_behaviors=train_beh, 
        binned_lfp=None
    )
    val_dataset = create_dataset(
        align_bin_spikes[val_idxs], 
        eid, 
        params,
        meta_data=meta_dict,
        binned_behaviors=val_beh, 
        binned_lfp=None
    )
    test_dataset = create_dataset(
        align_bin_spikes[test_idxs], 
        eid, 
        params,
        meta_data=meta_dict,
        binned_behaviors=test_beh, 
        binned_lfp=None
    )

    # Create dataset
    partitioned_dataset = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset}
    )
    print(partitioned_dataset)

    # Cache dataset (common data)
    save_path = Path(args.base_path)/"ibl_aligned"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    partitioned_dataset.save_to_disk(f'{save_path}/{eid}')

    # Save trial indices for later pose alignment
    np.save(save_path/eid/"train_trial_idxs.npy", train_idxs)
    np.save(save_path/eid/"val_trial_idxs.npy", val_idxs)
    np.save(save_path/eid/"test_trial_idxs.npy", test_idxs)
    
    # Save only the trial intervals needed for pose alignment (much smaller than full trials_df)
    # This is all we need: the start and end times for each trial
    align_times = trials_dict["trials_df"][params["align_time"]].values
    trial_intervals = np.vstack([
        align_times + params["time_window"][0],
        align_times + params["time_window"][1]
    ]).T  # Shape: (n_trials, 2)
    
    # Save alignment info for pose processing
    alignment_info = {
        "bad_trial_idxs": bad_trial_idxs,
        "params": params,
        "num_aligned_trials": num_trials,
        "trial_intervals": trial_intervals,  # Just the time windows, ~16 bytes per trial
        "pose_time_window": (pose_time_start, pose_time_end),  # Time window used for filtering
    }
    np.save(save_path/eid/"alignment_info.npy", alignment_info, allow_pickle=True)

    print(f'Downloaded common data for session {eid}.')
    print(f'Progress: {eid_idx+1} / {len(include_eids)} sessions downloaded.')

