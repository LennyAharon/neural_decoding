"""Data loaders for single/multi-session models."""
import numpy as np
from pathlib import Path
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch import LightningDataModule
import datasets
from utils.dataset_utils import get_binned_spikes_from_sparse

from scipy.ndimage import gaussian_filter1d

seed = 42

# Pose behavior names that can be loaded from model-specific directories
POSE_BEH_NAMES = [
    "lightning-pose-left-pawL-speed",
    "lightning-pose-right-pawL-speed",
    "lightning-pose-left-pawR-speed",
    "lightning-pose-right-pawR-speed",
    "lightning-pose-pawL-3d-speed",
    "lightning-pose-pawR-3d-speed",
]

# ---------
# Helpers
# ---------

def to_tensor(x, device):
    return torch.tensor(x).to(device)

def standardize_spike_data(spike_data, means=None, stds=None):
    
    K, T, N = spike_data.shape
    if (means is None) and (stds == None):
        means, stds = np.empty((T, N)), np.empty((T, N))

    std_spike_data = spike_data.reshape((K, -1))
    std_spike_data[np.isnan(std_spike_data)] = 0
    for t in range(T):
        mean = np.mean(std_spike_data[:, t*N:(t+1)*N])
        std = np.std(std_spike_data[:, t*N:(t+1)*N])
        std_spike_data[:, t*N:(t+1)*N] -= mean
        if std != 0:
            std_spike_data[:, t*N:(t+1)*N] /= std
        means[t], stds[t] = mean, std
    std_spike_data = std_spike_data.reshape(K, T, N)
    return std_spike_data, means, stds

def get_binned_spikes(dataset):
    spikes_sparse_data_list = dataset['spikes_sparse_data']
    spikes_sparse_indices_list = dataset['spikes_sparse_indices']
    spikes_sparse_indptr_list = dataset['spikes_sparse_indptr']
    spikes_sparse_shape_list = dataset['spikes_sparse_shape']
    
    binned_spikes  = get_binned_spikes_from_sparse(
        spikes_sparse_data_list, spikes_sparse_indices_list, spikes_sparse_indptr_list, spikes_sparse_shape_list
    )
    return binned_spikes.astype(float)

# ----------------------------
# Single-session data loaders
# ----------------------------

class SingleSessionDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        eid, 
        beh_name, 
        target, 
        device, 
        split="train", 
        region="all",
        load_local=True,
        huggingface_org="ibl-repro-ephys",
        standardize=False,
        use_nlb=False,
        bin_size=5,
        fold_idx=0,
        pose_model_name=None,  # NEW: Optional model name for pose data
    ):
        """Load and preprocess single-session datasets.
            
        Args:
            data_dir: data path.
            eid: session ID.
            beh_name: behavior name to be loaded, e.g., 'choice', 'wheel-speed'.
            target:
                'cls': classification for discrete behavior.
                'reg': regression for continuous behavior.
            split: data partition; options = ['train', 'val', 'test'].
            region: region name to be loaded, e.g., 'LP', 'CA1'.
            load_local: whether load cached data locally or remotely from Hugging Face.
            pose_model_name: (Optional) Name of the pose model for loading pose-specific data.
                             If provided and beh_name is a pose behavior, loads from 
                             pose_aligned/{pose_model_name}/{eid}/ instead of the main dataset.
        """
        if not use_nlb:
            if load_local:
                dataset = datasets.load_from_disk(Path(data_dir)/eid)
            else:
                dataset = datasets.load_dataset(f"{huggingface_org}/{eid}_aligned", cache_dir=data_dir)
            
            if split == "val":
                try:
                    # if val exists, load pre-partitioned validation set
                    self.spike_data = get_binned_spikes(dataset[split])
                    self.behavior = self._load_behavior(
                        dataset, split, beh_name, data_dir, eid, pose_model_name
                    )
                except:
                    # if not, partition training data into train and val
                    tmp = dataset[split].train_test_split(test_size=0.1, seed=seed)
                    self.spike_data = get_binned_spikes(tmp["test"])
                    self.behavior = np.array(tmp["test"][beh_name])
            else:
                self.spike_data = get_binned_spikes(dataset[split])
                self.behavior = self._load_behavior(
                    dataset, split, beh_name, data_dir, eid, pose_model_name
                )
            
            # Load ensemble variances if available
            self.ens_vars = {}
            self.trial_frame_indices = None
            self.original_trial_indices = None
            
            if pose_model_name is not None and beh_name in POSE_BEH_NAMES:
                pose_dir = Path(data_dir).parent / "pose_aligned" / pose_model_name / eid
                pose_file = pose_dir / f"{split}_pose.npy"
                if pose_file.exists():
                    pose_data = np.load(pose_file, allow_pickle=True).item()
                    
                    # Load trial and frame mapping
                    if "trial_frame_indices" in pose_data:
                        self.trial_frame_indices = pose_data["trial_frame_indices"]
                    if "original_trial_indices" in pose_data:
                        self.original_trial_indices = pose_data["original_trial_indices"]
                    
                    # 2D Variances and Coords (camera-specific)
                    if "3d-speed" in beh_name:
                        # For 3D targets, load both camera views (left and right)
                        paw_name = "pawR" if "pawR" in beh_name else "pawL"
                        for view in ["left", "right"]:
                            # Construct the 2D behavior name that was used during processing
                            ref_beh_name = f"lightning-pose-{view}-{paw_name}-speed"
                            for var_type in ["x_ens_var", "y_ens_var", "x_coords", "y_coords"]:
                                key = f"{ref_beh_name}_{var_type}"
                                if key in pose_data:
                                    self.ens_vars[f"{view}_{var_type}"] = pose_data[key]
                    else:
                        # For 2D targets, load just that specific camera view
                        for var_type in ["x_ens_var", "y_ens_var", "x_coords", "y_coords"]:
                            key = f"{beh_name}_{var_type}"
                            if key in pose_data:
                                self.ens_vars[var_type] = pose_data[key]
                            
                    # 3D Coordinates (paw-specific, triangulated from both cameras)
                    paw_name = "pawR" if "pawR" in beh_name else "pawL" if "pawL" in beh_name else None
                    if paw_name:
                        for coord in ["x_3d", "y_3d", "z_3d"]:
                            key = f"lightning-pose-{paw_name}-{coord}"
                            if key in pose_data:
                                self.ens_vars[coord] = pose_data[key]

            _, means, stds = standardize_spike_data(get_binned_spikes(dataset["train"]))
            self.spike_data, _, _ = standardize_spike_data(self.spike_data, means, stds)
            
            # Filter out trials with NaN behavior data (especially important for pose targets)
            if beh_name in POSE_BEH_NAMES and pose_model_name is not None:
                # Check for NaN rows (trials where all or any values are NaN)
                if len(self.behavior.shape) == 1:
                    valid_mask = ~np.isnan(self.behavior)
                else:
                    # For time-series behavior, check if any timepoint is NaN
                    valid_mask = ~np.any(np.isnan(self.behavior), axis=1)
                
                n_total = len(valid_mask)
                n_valid = np.sum(valid_mask)
                n_filtered = n_total - n_valid
                
                if n_filtered > 0:
                    if split == "train":  # Only print once
                        print(f"Filtering {n_filtered}/{n_total} trials with NaN pose data "
                              f"({n_valid} valid trials remain)")
                    
                    self.spike_data = self.spike_data[valid_mask]
                    self.behavior = self.behavior[valid_mask]
                    for k in self.ens_vars:
                        self.ens_vars[k] = self.ens_vars[k][valid_mask]
                    if self.trial_frame_indices is not None:
                        self.trial_frame_indices = self.trial_frame_indices[valid_mask]
                    if self.original_trial_indices is not None:
                        self.original_trial_indices = self.original_trial_indices[valid_mask]
            
            self.sessions = np.array([eid] * len(self.spike_data))
            self.neuron_regions = np.array(dataset[split]["cluster_regions"])[0]

            for re_idx, re_name in enumerate(self.neuron_regions):
                if "DG" in re_name:
                    self.neuron_regions[re_idx] = "DG"
                elif ("VISa" in re_name) or ("VISam" in re_name):
                    self.neuron_regions[re_idx] = "VISa"

            if region and region != "all":
                neuron_idxs = np.argwhere(self.neuron_regions == region).flatten()
                self.spike_data = self.spike_data[..., neuron_idxs]
                self.regions = np.array([region] * len(self.spike_data))
            else:
                self.regions = np.array(["all"] * len(self.spike_data))

            if target == "clf":
                enc = OneHotEncoder(handle_unknown="ignore")
                self.behavior = enc.fit_transform(self.behavior).toarray()#.argmax(axis=1)
            elif target == "reg":
                pass

            # Handle any remaining NaNs (for non-pose targets or edge cases)
            if np.isnan(self.behavior).sum() != 0:
                self.behavior[np.isnan(self.behavior)] = np.nanmean(self.behavior)
                # Only warn if this is unexpected (not a pose target that should have been filtered)
                if beh_name not in POSE_BEH_NAMES:
                    print(f"{beh_name} in session {eid} contains NaNs; interpolate with trial-average.")

            if target == "reg" and beh_name == "prior":
                self.behavior = self.behavior.reshape(-1,1)

            self.n_trials, self.n_t_steps, self.n_units = self.spike_data.shape
            self.spike_data = to_tensor(self.spike_data, device).double()
            self.behavior = to_tensor(self.behavior, device)
            # self.behavior = self.behavior.long() if target == "clf" else self.behavior.double() 
            self.behavior = self.behavior.double() 

        else:
            dataset = np.load(
                f"/burg/stats/users/yz4123/Downloads/nlb-rtt/xval/fold_{fold_idx}/{split}_dataset_{bin_size}.npy", 
                allow_pickle=True
            ).item()
            # smooth_w = 1
            X = dataset["spikes"].astype(np.float32)
            # X = gaussian_filter1d(X, smooth_w, axis=1)
            self.n_trials, self.n_t_steps, self.n_units = X.shape
            X = X.reshape((-1, self.n_units))
            X_mu = np.mean(X, axis=0)
            X_sigma = np.std(X, axis=0)
            X = (X - X_mu) / X_sigma
            self.spike_data = X.reshape(self.n_trials, self.n_t_steps, self.n_units)
            self.spike_data = to_tensor(self.spike_data, device).double()
            self.behavior = to_tensor(dataset["finger_vel"], device).double()
            if beh_name == "finger_vel_dim_0":
                self.behavior = self.behavior[...,0]
            else:
                self.behavior = self.behavior[...,1]
            self.sessions = np.array([eid] * self.n_trials)
            self.regions = np.array(["all"] * self.n_trials)
            
    def _load_behavior(self, dataset, split, beh_name, data_dir, eid, pose_model_name):
        """Load behavior data, optionally from pose-model-specific directory.
        
        If pose_model_name is provided and beh_name is a pose behavior,
        loads from pose_aligned/{pose_model_name}/{eid}/ directory.
        Otherwise, loads from the main dataset.
        """
        # Check if this is a pose behavior and we have a model name
        if pose_model_name is not None and beh_name in POSE_BEH_NAMES:
            pose_dir = Path(data_dir).parent / "pose_aligned" / pose_model_name / eid
            pose_file = pose_dir / f"{split}_pose.npy"
            
            if pose_file.exists():
                pose_data = np.load(pose_file, allow_pickle=True).item()
                if beh_name in pose_data:
                    # Only print once per session (on first split)
                    if split == "train":
                        print(f"Loading {beh_name} from pose model: {pose_model_name}")
                    return pose_data[beh_name]
                else:
                    print(f"Warning: {beh_name} not found in pose data for model {pose_model_name}")
            else:
                print(f"Warning: Pose file not found: {pose_file}")
        
        # Default: load from main dataset
        return np.array(dataset[split][beh_name])

    def __len__(self):
        return self.n_trials

    def __getitem__(self, trial_idx):
        return (
            self.spike_data[trial_idx], self.behavior[trial_idx], 
            self.regions[trial_idx], self.sessions[trial_idx]
        )

    
class SingleSessionDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config["dirs"]["data_dir"]
        self.eid = config["eid"]
        self.beh_name = config["target"]
        self.target = config["model"]["target"]
        self.region = config["region"]
        self.device = config["training"]["device"]
        self.load_local = config["training"]["load_local"]
        self.batch_size = config["training"]["batch_size"]
        self.n_workers = config["data"]["num_workers"]
        self.use_nlb = config["data"]["use_nlb"]
        self.bin_size = config["data"]["bin_size"]
        self.fold_idx = config["data"]["fold_idx"]
        # NEW: Optional pose model name for loading pose-specific data
        self.pose_model_name = config.get("pose_model_name", None)

    def update_config(self):
        self.val = SingleSessionDataset(
            self.data_dir, self.eid, self.beh_name, self.target, 
            self.device, "val", self.region, self.load_local, 
            use_nlb=self.use_nlb, bin_size=self.bin_size, fold_idx=self.fold_idx,
            pose_model_name=self.pose_model_name
        )
        self.config.update({
            "n_units": self.val.n_units, 
            "n_t_steps": self.val.n_t_steps,
            "eid": self.eid, 
            "region": self.region
        })

    def setup(self, stage=None):
        """Call this function to load and preprocess data."""
        self.train = SingleSessionDataset(
            self.data_dir, self.eid, self.beh_name, self.target, 
            self.device, "train", self.region, self.load_local, 
            use_nlb=self.use_nlb, bin_size=self.bin_size, fold_idx=self.fold_idx,
            pose_model_name=self.pose_model_name
        )
        self.val = SingleSessionDataset(
            self.data_dir, self.eid, self.beh_name, self.target, 
            self.device, "val", self.region, self.load_local, 
            use_nlb=self.use_nlb, bin_size=self.bin_size, fold_idx=self.fold_idx,
            pose_model_name=self.pose_model_name
        )
        self.test = SingleSessionDataset(
            self.data_dir, self.eid, self.beh_name, self.target, 
            self.device, "test", self.region, self.load_local, 
            use_nlb=self.use_nlb, bin_size=self.bin_size, fold_idx=self.fold_idx,
            pose_model_name=self.pose_model_name
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        # return DataLoader(self.train, batch_size=self.batch_size, shuffle=False)
        
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, drop_last=False)


# ---------------------------
# Multi-session data loaders
# ---------------------------

class MultiSessionDataModule(LightningDataModule):
    def __init__(self, eids, configs):
        """Load and preprocess multi-session datasets.
            
        Args:
            eids: a list of session IDs.
            configs: a list of data configs for each session.
        """
        super().__init__()
        self.eids = eids
        self.configs = configs
        self.batch_size = configs[0]['training']['batch_size']

    def update_config(self):
        for config in self.configs:
            dm = SingleSessionDataModule(config)
            dm.update_config()

    def setup(self, stage=None):
        """Call this function to load and preprocess data."""
        self.train, self.val, self.test = [], [], []
        for config in self.configs:
            dm = SingleSessionDataModule(config)
            dm.setup()
            self.train.append(
                DataLoader(dm.train, batch_size = self.batch_size, shuffle=True)
            )
            self.val.append(
                DataLoader(dm.val, batch_size = self.batch_size, shuffle=False, drop_last=False)
            )
            self.test.append(
                DataLoader(dm.test, batch_size = self.batch_size, shuffle=False, drop_last=False)
            )

    def train_dataloader(self):
        data_loader = CombinedLoader(self.train, mode = "max_size_cycle")
        return data_loader

    def val_dataloader(self):
        data_loader = CombinedLoader(self.val)
        return data_loader

    def test_dataloader(self):
        data_loader = CombinedLoader(self.test)
        return data_loader


class MultiRegionDataModule(LightningDataModule):
    def __init__(self, eids, configs):
        """Load and preprocess multi-session datasets.
            
        Args:
            eids: a list of session IDs.
            configs: a list of data configs for each session-region combination.
            query_region: a list of brain regions to decode from.
        """
        super().__init__()
        self.eids = eids
        self.configs = configs
        self.batch_size = configs[0]['training']['batch_size']
        self.query_region = configs[0]['query_region']

    def list_regions(self):
        """Call this function to list all available brain regions from the input sessions."""
        self.all_regions, self.regions_dict = [], {}
        for idx, eid in enumerate(self.eids):
            dm = SingleSessionDataModule(self.configs[idx])
            dm.setup()
            unique_regions = [roi for roi in np.unique(dm.train.neuron_regions) if roi not in ['root', 'void']]
            self.regions_dict[eid] = unique_regions
            self.all_regions.extend(unique_regions)
        self.all_regions = list(np.unique(self.all_regions))

    def update_config(self):
        for config in self.configs:
            dm = SingleSessionDataModule(config)
            dm.update_config()
        
    def setup(self, stage=None):
        """Call this function to load and preprocess data."""
        self.train, self.val, self.test = [], [], []
        for config in self.configs:
            dm = SingleSessionDataModule(config)
            dm.setup()
            self.train.append(
                DataLoader(dm.train, batch_size = self.batch_size, shuffle=True)
            )
            self.val.append(
                DataLoader(dm.val, batch_size = self.batch_size, shuffle=False, drop_last=False)
            )
            self.test.append(
                DataLoader(dm.test, batch_size = self.batch_size, shuffle=False, drop_last=False)
            )

    def train_dataloader(self):
        data_loader = CombinedLoader(self.train, mode = "max_size_cycle")
        return data_loader

    def val_dataloader(self):
        data_loader = CombinedLoader(self.val)
        return data_loader

    def test_dataloader(self):
        data_loader = CombinedLoader(self.test)
        return data_loader
        
    