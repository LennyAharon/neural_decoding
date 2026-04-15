"""5-fold cross-validation with hyperparameter search for single-session decoding.

This script implements proper 5-fold CV where:
- Each trial appears in test set exactly once
- Within each fold, 80% is used for train+val (with hyperparameter search)
- 20% is held out as test
- Results are saved per-fold and aggregated across all folds
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
# Prevent Ray from auto-initializing and disable dashboard
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
os.environ["RAY_AUTO_INIT"] = "0"
os.environ["RAY_DASHBOARD_ENABLED"] = "0"
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge, LogisticRegression
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn

# Conditional imports for non-linear models
try:
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch import Trainer
    from ray import tune
    from ray.train.lightning import (
        RayDDPStrategy,
        RayLightningEnvironment,
        RayTrainReportCallback,
        prepare_trainer,
    )
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    ModelCheckpoint = None
    Trainer = None
    RayDDPStrategy = None
    RayLightningEnvironment = None
    RayTrainReportCallback = None
    prepare_trainer = None
    tune = None

from utils.data_loader_utils import SingleSessionDataModule, SingleSessionDataset
from models.decoders import ReducedRankDecoder, MLPDecoder, LSTMDecoder, SeanMLPDecoder
from utils.eval_utils import eval_model
from utils.sweep_utils import tune_decoder
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config

BINSIZE = 0.02
LENGTH = 2.
CLASSIFICATION = ["choice"]
REGRESSION = ["wheel-speed", "whisker-motion-energy", "prior", "finger_vel_dim_0", "finger_vel_dim_1", 
              "lightning-pose-left-pawL-speed", "lightning-pose-right-pawL-speed",
              "lightning-pose-left-pawR-speed", "lightning-pose-right-pawR-speed",
              "lightning-pose-pawL-3d-speed", "lightning-pose-pawR-3d-speed"]

"""
-----------
USER INPUTS
-----------
"""
ap = argparse.ArgumentParser()
ap.add_argument("--base_path", type=str, default="/scratch/bcxj/hlyu/RRR/")
ap.add_argument("--eid", type=str)
ap.add_argument("--target", type=str, default="choice", choices=CLASSIFICATION+REGRESSION)
ap.add_argument("--region", type=str, default="all")
ap.add_argument("--method", type=str, default="linear", choices=["linear", "reduced_rank", "mlp", "lstm"])
ap.add_argument("--n_workers", type=int, default=1)
ap.add_argument("--search", action="store_true", help="Perform hyperparameter search within each fold")
ap.add_argument("--use_nlb", action="store_true")
ap.add_argument("--bin_size", type=int, default=5)
ap.add_argument("--fold_idx", type=int, default=0)
ap.add_argument("--model_name", type=str, default=None,
                help="Name of the pose model for loading pose-specific data. "
                     "If provided, loads pose data from pose_aligned/{model_name}/{eid}/ "
                     "and saves results to results/{eid}/{target}/{model_name}/{method}/")

args = ap.parse_args()

OUTPUT_SIZE_LOOKUP = {
    "choice": 2, 
    "prior": 1, 
    "wheel-speed": int(LENGTH/BINSIZE), 
    "whisker-motion-energy": int(LENGTH/BINSIZE),
    "pupil-diameter": int(LENGTH/BINSIZE),
    "finger_vel_dim_0": int(0.6/(args.bin_size/1000)),
    "finger_vel_dim_1": int(0.6/(args.bin_size/1000)),
    "lightning-pose-left-pawL-speed": int(LENGTH/BINSIZE),
    "lightning-pose-right-pawL-speed": int(LENGTH/BINSIZE),
    "lightning-pose-left-pawR-speed": int(LENGTH/BINSIZE),
    "lightning-pose-right-pawR-speed": int(LENGTH/BINSIZE),
    "lightning-pose-pawL-3d-speed": int(LENGTH/BINSIZE),
    "lightning-pose-pawR-3d-speed": int(LENGTH/BINSIZE),
}


"""
-------
CONFIGS
-------
"""
kwargs = {"model": "include:src/configs/decoder.yaml"}
config = config_from_kwargs(kwargs)
config = update_config("src/configs/decoder.yaml", config)

if args.target in REGRESSION:
    config = update_config("src/configs/reg_trainer.yaml", config)
elif args.target in CLASSIFICATION:
    config = update_config("src/configs/clf_trainer.yaml", config)
else:
    raise NotImplementedError

set_seed(config["seed"])

config["dirs"]["data_dir"] = Path(args.base_path)/config["dirs"]["data_dir"]

# Determine if this is a pose target that needs model_name
POSE_TARGETS = [
    "lightning-pose-left-pawL-speed", "lightning-pose-right-pawL-speed",
    "lightning-pose-left-pawR-speed", "lightning-pose-right-pawR-speed",
    "lightning-pose-pawL-3d-speed", "lightning-pose-pawR-3d-speed"
]
is_pose_target = args.target in POSE_TARGETS

# Create save paths - include model_name for pose targets
if is_pose_target and args.model_name:
    save_path = Path(args.base_path)/config["dirs"]["output_dir"]/args.eid/args.target/args.model_name/args.method/args.region
    ckpt_path = Path(args.base_path)/config["dirs"]["checkpoint_dir"]/args.eid/args.target/args.model_name/args.method/args.region
elif is_pose_target and not args.model_name:
    print(f"WARNING: Decoding pose target '{args.target}' without --model_name. "
          "Consider providing --model_name to organize results by pose model.")
    save_path = Path(args.base_path)/config["dirs"]["output_dir"]/args.eid/args.target/args.method/args.region
    ckpt_path = Path(args.base_path)/config["dirs"]["checkpoint_dir"]/args.eid/args.target/args.method/args.region
else:
    save_path = Path(args.base_path)/config["dirs"]["output_dir"]/args.eid/args.target/args.method/args.region
    ckpt_path = Path(args.base_path)/config["dirs"]["checkpoint_dir"]/args.eid/args.target/args.method/args.region

os.makedirs(save_path, exist_ok=True)
os.makedirs(ckpt_path, exist_ok=True)

model_class = args.method

# Check if lightning is needed
if model_class != "linear" and not LIGHTNING_AVAILABLE:
    raise ImportError(
        f"PyTorch Lightning is required for '{model_class}' models. "
        f"Please install it with: pip install lightning\n"
        f"Or use --method linear for sklearn-based models."
    )

print(f"5-fold Cross-Validation for {args.target} from session: {args.eid}")
print(f"Model: {model_class}")
if args.search:
    print("Hyperparameter search: ENABLED")
else:
    print("Hyperparameter search: DISABLED (using default hyperparameters)")


"""
--------
HELPER FUNCTIONS
--------
"""

class FoldDataset(Dataset):
    """Custom dataset for a specific fold split."""
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


def load_all_data(config):
    """Load all data (train + val + test) and concatenate."""
    dm = SingleSessionDataModule(config)
    dm.update_config()
    dm.setup()
    
    # Collect all data
    all_x, all_y = [], []
    all_regions, all_eids = [], []
    all_ens_vars = {
        "x_ens_var": [], "y_ens_var": [], 
        "x_coords": [], "y_coords": [],
        "x_3d": [], "y_3d": [], "z_3d": [],
        "left_x_ens_var": [], "left_y_ens_var": [], "left_x_coords": [], "left_y_coords": [],
        "right_x_ens_var": [], "right_y_ens_var": [], "right_x_coords": [], "right_y_coords": []
    }
    all_trial_frame_indices = []
    all_original_trial_indices = []
    has_ens_vars = False
    
    for dataset in [dm.train, dm.val, dm.test]:
        # Collect trial mapping info
        if hasattr(dataset, "trial_frame_indices") and dataset.trial_frame_indices is not None:
            all_trial_frame_indices.append(dataset.trial_frame_indices)
        if hasattr(dataset, "original_trial_indices") and dataset.original_trial_indices is not None:
            all_original_trial_indices.append(dataset.original_trial_indices)

        # Collect ens_vars directly from dataset object
        if hasattr(dataset, "ens_vars") and dataset.ens_vars:
            has_ens_vars = True
            for k in all_ens_vars.keys():
                if k in dataset.ens_vars:
                    all_ens_vars[k].append(dataset.ens_vars[k])
        
        for (x, y, region, eid) in dataset:
            all_x.append(x.cpu())
            all_y.append(y.cpu())
            all_regions.append(region)
            all_eids.append(eid)
    
    all_x = torch.stack(all_x)
    all_y = torch.stack(all_y) if isinstance(all_y[0], torch.Tensor) else torch.tensor(np.stack(all_y))
    
    if has_ens_vars:
        for k in all_ens_vars:
            if all_ens_vars[k]:
                all_ens_vars[k] = np.concatenate(all_ens_vars[k], axis=0)
            else:
                all_ens_vars[k] = None
    else:
        all_ens_vars = None
    
    # Concatenate mapping info
    all_trial_frame_indices = np.concatenate(all_trial_frame_indices, axis=0) if all_trial_frame_indices else None
    all_original_trial_indices = np.concatenate(all_original_trial_indices, axis=0) if all_original_trial_indices else None
    
    # Store metadata for later use
    metadata = {
        "n_units": dm.train.n_units,
        "n_t_steps": dm.train.n_t_steps,
        "regions": np.array(all_regions),
        "eids": np.array(all_eids),
        "ens_vars": all_ens_vars,
        "trial_frame_indices": all_trial_frame_indices,
        "original_trial_indices": all_original_trial_indices,
    }
    
    return all_x, all_y, metadata, dm


def create_fold_datasets(base_dataset, train_val_indices, test_indices):
    """Create train/val/test datasets for a specific fold."""
    # Split train_val into train (70% of 80% = 56%) and val (10% of 80% = 8%)
    n_train_val = len(train_val_indices)
    n_train = int(0.875 * n_train_val)  # 87.5% of train_val = 70% of total
    n_val = n_train_val - n_train  # 12.5% of train_val = 10% of total
    
    # Shuffle train_val_indices for random split
    shuffled_indices = np.random.permutation(train_val_indices)
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:]
    
    train_dataset = FoldDataset(base_dataset, train_indices)
    val_dataset = FoldDataset(base_dataset, val_indices)
    test_dataset = FoldDataset(base_dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset, train_indices, val_indices


def do_hyperparameter_search_linear(model_class, train_x, train_y, val_x, val_y, target):
    """Do hyperparameter search for linear models using GridSearchCV."""
    from sklearn.model_selection import GridSearchCV
    
    # Reshape data for sklearn
    train_x_flat = train_x.reshape((train_x.shape[0], -1))
    val_x_flat = val_x.reshape((val_x.shape[0], -1))
    
    # Combine train and val for grid search
    all_train_x = np.concatenate([train_x_flat, val_x_flat], axis=0)
    all_train_y = np.concatenate([train_y, val_y], axis=0)
    
    if target in CLASSIFICATION:
        param_grid = {"C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4]}
        base_model = LogisticRegression(max_iter=1000, random_state=config["seed"])
    elif target in REGRESSION:
        param_grid = {"alpha": np.logspace(-4, 4, 9)}
        base_model = Ridge()
    else:
        raise NotImplementedError
    
    # Use 5-fold CV within train+val for grid search
    grid_search = GridSearchCV(
        base_model, param_grid, cv=5, 
        scoring='accuracy' if target in CLASSIFICATION else 'r2',
        n_jobs=1
    )
    grid_search.fit(all_train_x, all_train_y)
    
    return grid_search.best_estimator_


def do_hyperparameter_search_nonlinear(
    model_class, config, train_dataset, val_dataset, 
    ckpt_path_fold, device, target
):
    """Do hyperparameter search for non-linear models using Ray Tune."""
    if not LIGHTNING_AVAILABLE:
        raise ImportError("PyTorch Lightning required for non-linear models")
    
    import ray
    import time
    import subprocess
    
    # Initialize Ray if needed
    if not ray.is_initialized():
        if config["tuner"]["use_gpu"] and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            ray.init(
                num_gpus=num_gpus,
                ignore_reinit_error=True,
                include_dashboard=False,
            )
        else:
            ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    # Set up search space
    search_space = config.copy()
    search_space["model"]["output_size"] = OUTPUT_SIZE_LOOKUP[args.target]
    search_space["training"]["device"] = device
    search_space["data"]["use_nlb"] = args.use_nlb
    search_space["data"]["bin_size"] = args.bin_size
    
    search_space["optimizer"]["lr"] = tune.loguniform(1e-4, 1e-2)
    search_space["optimizer"]["weight_decay"] = tune.loguniform(1e-3, 1.)
    
    from itertools import combinations
    def generate_mlp_hyperparams(possible_sizes=[256, 128, 64, 32, 16]):
        hyperparams = []
        for length in range(2, len(possible_sizes)):
            for combo in combinations(possible_sizes, length):
                if all(combo[i] > combo[i+1] for i in range(len(combo)-1)):
                    hyperparams.append(f"({', '.join(map(str, combo))})")
        return hyperparams
    
    if model_class == "reduced_rank":
        search_space["optimizer"]["lr"] = 1e-2
        search_space["optimizer"]["weight_decay"] = 1
        search_space["reduced_rank"]["temporal_rank"] = tune.grid_search(
            list(range(2, min(OUTPUT_SIZE_LOOKUP[args.target]+1, 20)))
        )
        search_space["tuner"]["num_epochs"] = config["training"]["num_epochs"]
        search_space["training"]["num_epochs"] = config["training"]["num_epochs"]
    elif model_class == "lstm":
        search_space["lstm"]["lstm_n_layers"] = tune.randint(1, 3)
        search_space["lstm"]["lstm_hidden_size"] = tune.choice([32, 64, 128, 256, 512])
        search_space["lstm"]["mlp_hidden_size"] = tune.choice(
            generate_mlp_hyperparams(possible_sizes=[256, 128, 64, 32, 16])
        )
        search_space["lstm"]["drop_out"] = tune.uniform(0.1, 0.3)
        search_space["tuner"]["num_epochs"] = config["training"]["num_epochs"]
        search_space["training"]["num_epochs"] = config["training"]["num_epochs"]
    elif model_class == "mlp":
        if not args.use_nlb:
            search_space["mlp"]["mlp_hidden_size"] = tune.choice(
                generate_mlp_hyperparams(possible_sizes=[512, 256, 128, 64, 32, 16])
            )
            search_space["mlp"]["drop_out"] = tune.uniform(0.1, 0.3)
        else:
            search_space["mlp"]["n_layers"] = tune.choice(list(range(1, 11)))
            search_space["mlp"]["hidden_dim"] = tune.choice(list(range(50, 601)))
            search_space["mlp"]["drop_out"] = tune.uniform(0.1, 0.5)
        search_space["tuner"]["num_epochs"] = config["training"]["num_epochs"]
        search_space["training"]["num_epochs"] = config["training"]["num_epochs"]
    else:
        raise NotImplementedError
    
    # Create a custom data module for this fold
    class FoldDataModule:
        def __init__(self, train_dataset, val_dataset, config):
            self.train = train_dataset
            self.val = val_dataset
            self.config = config
        
        def update_config(self):
            # Config already updated
            pass
        
        def setup(self):
            pass
    
    fold_dm = FoldDataModule(train_dataset, val_dataset, search_space)
    
    def train_func(config):
        # Ray Train 2.x+ passes the config directly to the train_func
        # Check if it's nested or not
        if "train_loop_config" in config:
            final_cfg = config["train_loop_config"]
        else:
            final_cfg = config
            
        fold_dm.config = final_cfg
        fold_dm.config["training"]["total_steps"] = (
            fold_dm.config["training"]["num_epochs"] * len(fold_dm.train)
        )
        
        if model_class == "reduced_rank":
            model = ReducedRankDecoder(fold_dm.config)
        elif model_class == "lstm":
            model = LSTMDecoder(fold_dm.config)
        elif model_class == "mlp":
            model = MLPDecoder(fold_dm.config) if not args.use_nlb else SeanMLPDecoder(fold_dm.config)
        else:
            raise NotImplementedError
        
        trainer = Trainer(
            max_epochs=final_cfg["tuner"]["num_epochs"],
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=final_cfg["tuner"]["enable_progress_bar"],
            check_val_every_n_epoch=1,
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(model, train_dataloaders=DataLoader(fold_dm.train, batch_size=fold_dm.config["training"]["batch_size"], shuffle=True),
                    val_dataloaders=DataLoader(fold_dm.val, batch_size=fold_dm.config["training"]["batch_size"], shuffle=False))
    
    # Run hyperparameter search
    results = tune_decoder(
        train_func,
        search_space,
        save_dir=ckpt_path_fold,
        use_gpu=config["tuner"]["use_gpu"],
        max_epochs=config["tuner"]["num_epochs"],
        num_samples=config["tuner"]["num_samples"] if model_class != "reduced_rank" else 1,
        num_workers=args.n_workers,
        metric=config["tuner"]["metric"],
        mode=config["tuner"]["mode"],
    )
    
    best_result = results.get_best_result(metric=config["tuner"]["metric"], mode=config["tuner"]["mode"])
    best_config = best_result.config["train_loop_config"]
    
    return best_config


"""
--------
5-FOLD CROSS-VALIDATION
--------
"""

# Set up base config
base_config = config.copy()
base_config["eid"] = args.eid
base_config["target"] = args.target
base_config["region"] = args.region if args.region != "all" else "all"
base_config["model"]["output_size"] = OUTPUT_SIZE_LOOKUP[args.target]
base_config["training"]["device"] = torch.device(
    "cuda" if np.logical_and(torch.cuda.is_available(), config["training"]["device"] == "gpu") else "cpu"
)
base_config["data"]["use_nlb"] = args.use_nlb
base_config["data"]["bin_size"] = args.bin_size
base_config["data"]["fold_idx"] = args.fold_idx  
base_config["pose_model_name"] = args.model_name if is_pose_target else None

# Load all data
print("Loading all data (train + val + test)...")
all_x, all_y, metadata, original_dm = load_all_data(base_config)
n_trials = len(all_x)
print(f"Total trials: {n_trials}")


# Create base dataset from all data
class AllDataDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], metadata["regions"][idx], metadata["eids"][idx]

base_dataset = AllDataDataset(all_x, all_y)

# Create 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=config["seed"])
fold_splits = list(kf.split(np.arange(n_trials)))

# Initialize arrays to store results
all_test_preds = np.zeros_like(all_y.numpy())
all_test_ys = np.zeros_like(all_y.numpy())
all_test_probs = np.zeros((n_trials, 2)) if args.target in CLASSIFICATION else None
has_metadata_ens = metadata.get("ens_vars") is not None
all_test_ens_vars = {
    "x_ens_var": np.zeros_like(metadata["ens_vars"]["x_ens_var"]) if has_metadata_ens and metadata["ens_vars"].get("x_ens_var") is not None else None,
    "y_ens_var": np.zeros_like(metadata["ens_vars"]["y_ens_var"]) if has_metadata_ens and metadata["ens_vars"].get("y_ens_var") is not None else None,
    "x_coords": np.zeros_like(metadata["ens_vars"]["x_coords"]) if has_metadata_ens and metadata["ens_vars"].get("x_coords") is not None else None,
    "y_coords": np.zeros_like(metadata["ens_vars"]["y_coords"]) if has_metadata_ens and metadata["ens_vars"].get("y_coords") is not None else None,
    "x_3d": np.zeros_like(metadata["ens_vars"]["x_3d"]) if has_metadata_ens and metadata["ens_vars"].get("x_3d") is not None else None,
    "y_3d": np.zeros_like(metadata["ens_vars"]["y_3d"]) if has_metadata_ens and metadata["ens_vars"].get("y_3d") is not None else None,
    "z_3d": np.zeros_like(metadata["ens_vars"]["z_3d"]) if has_metadata_ens and metadata["ens_vars"].get("z_3d") is not None else None,
    "left_x_ens_var": np.zeros_like(metadata["ens_vars"]["left_x_ens_var"]) if has_metadata_ens and metadata["ens_vars"].get("left_x_ens_var") is not None else None,
    "left_y_ens_var": np.zeros_like(metadata["ens_vars"]["left_y_ens_var"]) if has_metadata_ens and metadata["ens_vars"].get("left_y_ens_var") is not None else None,
    "left_x_coords": np.zeros_like(metadata["ens_vars"]["left_x_coords"]) if has_metadata_ens and metadata["ens_vars"].get("left_x_coords") is not None else None,
    "left_y_coords": np.zeros_like(metadata["ens_vars"]["left_y_coords"]) if has_metadata_ens and metadata["ens_vars"].get("left_y_coords") is not None else None,
    "right_x_ens_var": np.zeros_like(metadata["ens_vars"]["right_x_ens_var"]) if has_metadata_ens and metadata["ens_vars"].get("right_x_ens_var") is not None else None,
    "right_y_ens_var": np.zeros_like(metadata["ens_vars"]["right_y_ens_var"]) if has_metadata_ens and metadata["ens_vars"].get("right_y_ens_var") is not None else None,
    "right_x_coords": np.zeros_like(metadata["ens_vars"]["right_x_coords"]) if has_metadata_ens and metadata["ens_vars"].get("right_x_coords") is not None else None,
    "right_y_coords": np.zeros_like(metadata["ens_vars"]["right_y_coords"]) if has_metadata_ens and metadata["ens_vars"].get("right_y_coords") is not None else None,
}
all_test_frame_indices = np.zeros_like(metadata["trial_frame_indices"]) if metadata["trial_frame_indices"] is not None else None
all_test_original_trial_indices = np.zeros_like(metadata["original_trial_indices"]) if metadata["original_trial_indices"] is not None else None
fold_results = []

# Process each fold
for fold_idx, (train_val_indices, test_indices) in enumerate(fold_splits):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/5")
    print(f"{'='*60}")
    print(f"Train+Val: {len(train_val_indices)} trials ({len(train_val_indices)/n_trials*100:.1f}%)")
    print(f"Test: {len(test_indices)} trials ({len(test_indices)/n_trials*100:.1f}%)")
    
    # Create fold-specific checkpoint path
    ckpt_path_fold = ckpt_path / f"fold_{fold_idx}"
    os.makedirs(ckpt_path_fold, exist_ok=True)
    
    # Create train/val/test datasets for this fold
    train_dataset, val_dataset, test_dataset, train_indices, val_indices = create_fold_datasets(
        base_dataset, train_val_indices, test_indices
    )
    
    print(f"  Train: {len(train_dataset)} trials")
    print(f"  Val: {len(val_dataset)} trials (for hyperparameter search)")
    print(f"  Test: {len(test_dataset)} trials")
    
    # HYPERPARAMETER SEARCH
    if args.search:
        print(f"\n  Performing hyperparameter search...")
        
        if model_class == "linear":
            # For linear models, use GridSearchCV
            train_x = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
            train_y = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
            val_x = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
            val_y = torch.stack([val_dataset[i][1] for i in range(len(val_dataset))])
            
            best_model = do_hyperparameter_search_linear(
                model_class, train_x.numpy(), train_y.numpy(),
                val_x.numpy(), val_y.numpy(), args.target
            )
            
            # Retrain on full train+val with best hyperparameters
            all_train_val_x = torch.cat([train_x, val_x], dim=0).numpy()
            all_train_val_y = torch.cat([train_y, val_y], dim=0).numpy()
            all_train_val_x_flat = all_train_val_x.reshape((all_train_val_x.shape[0], -1))
            
            best_model.fit(all_train_val_x_flat, all_train_val_y)
            
        else:
            # For non-linear models, use Ray Tune
            best_config = do_hyperparameter_search_nonlinear(
                model_class, base_config, train_dataset, val_dataset,
                ckpt_path_fold, base_config["training"]["device"], args.target
            )
            torch.save(best_config, ckpt_path_fold / "best_config.pth")
    else:
        # No hyperparameter search - use default config
        if model_class == "linear":
            if args.target in REGRESSION:
                best_model = Ridge(alpha=1.0)
            elif args.target in CLASSIFICATION:
                best_model = LogisticRegression(max_iter=1000, random_state=config["seed"])
        else:
            best_config = base_config.copy()
    
    # TRAIN FINAL MODEL ON FULL TRAIN+VAL
    print(f"\n  Training final model on full train+val set...")
    
    if model_class == "linear":
        # Already trained above
        pass
    else:
        # Create full train+val dataset
        full_train_val_indices = np.concatenate([train_indices, val_indices])
        full_train_val_dataset = FoldDataset(base_dataset, full_train_val_indices)
        
        # Update config with total steps
        if args.search:
            final_config = best_config.copy()
        else:
            final_config = base_config.copy()
        
        final_config["training"]["total_steps"] = (
            final_config["training"]["num_epochs"] * len(full_train_val_dataset)
        )
        
        # Initialize model
        if model_class == "reduced_rank":
            model = ReducedRankDecoder(final_config)
        elif model_class == "lstm":
            model = LSTMDecoder(final_config)
        elif model_class == "mlp":
            model = MLPDecoder(final_config) if not args.use_nlb else SeanMLPDecoder(final_config)
        else:
            raise NotImplementedError
        
        model.to(base_config["training"]["device"])
        
        # Train model
        checkpoint_callback = ModelCheckpoint(
            monitor=config["training"]["metric"],
            mode=config["training"]["mode"],
            dirpath=ckpt_path_fold,
            filename="best_model",
            save_top_k=1,
            save_last=True
        )
        trainer = Trainer(
            max_epochs=final_config["training"]["num_epochs"],
            callbacks=[checkpoint_callback],
            enable_progress_bar=config["training"]["enable_progress_bar"],
            check_val_every_n_epoch=10,
            devices=1,
            strategy="auto",
        )
        
        train_loader = DataLoader(
            full_train_val_dataset,
            batch_size=final_config["training"]["batch_size"],
            shuffle=True
        )
        trainer.fit(model, train_dataloaders=train_loader)
        
        # Load best model if available, otherwise use last model
        if model_class == "reduced_rank":
            MODEL_CLASS = ReducedRankDecoder
        elif model_class == "lstm":
            MODEL_CLASS = LSTMDecoder
        elif model_class == "mlp":
            MODEL_CLASS = MLPDecoder if not args.use_nlb else SeanMLPDecoder
        else:
            raise NotImplementedError
        
        # Check if we have a best model path and it's a file
        best_path = checkpoint_callback.best_model_path
        if best_path and os.path.isfile(best_path):
            print(f"  Loading best model from {best_path}")
            model = MODEL_CLASS.load_from_checkpoint(
                best_path,
                config=final_config
            )
        else:
            last_path = os.path.join(ckpt_path_fold, "last.ckpt")
            if os.path.isfile(last_path):
                print(f"  No best model found (no validation set). Loading last model from {last_path}")
                model = MODEL_CLASS.load_from_checkpoint(
                    last_path,
                    config=final_config
                )
            else:
                print("  No checkpoints found. Using model from memory.")
    
    # EVALUATE ON TEST SET
    print(f"\n  Evaluating on test set...")
    
    if model_class == "linear":
        test_x = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).numpy()
        test_y = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))]).numpy()
        
        test_x_flat = test_x.reshape((test_x.shape[0], -1))
        
        if args.target in CLASSIFICATION:
            test_pred = best_model.predict(test_x_flat)
            test_prob = best_model.predict_proba(test_x_flat)
        else:
            test_pred = best_model.predict(test_x_flat)
            test_prob = None
        
        # Calculate metric
        if args.target in REGRESSION:
            if test_y.shape[-1] == 1:
                from scipy.stats import pearsonr
                metric = pearsonr(test_y.flatten(), test_pred.flatten())[0]
            else:
                from sklearn.metrics import r2_score
                metric = []
                for dim in range(test_y.shape[-1]):
                    metric.append(r2_score(test_y[..., dim].flatten(), test_pred[..., dim].flatten()))
                metric = np.nanmean(metric)
        else:
            from sklearn.metrics import accuracy_score
            metric = accuracy_score(test_y.argmax(1), test_pred)
        
    else:
        # Use eval_model for non-linear models
        # Create dummy train dataset (not used for evaluation)
        dummy_train = FoldDataset(base_dataset, train_indices[:min(10, len(train_indices))])
        
        metric, test_pred, test_y, test_prob = eval_model(
            dummy_train,
            test_dataset,
            model.cpu(),
            target=config["model"]["target"],
            model_class=model_class,
            use_nlb=args.use_nlb,
            bin_size=args.bin_size,
            beh_name=args.target
        )
        
        test_pred = test_pred if isinstance(test_pred, np.ndarray) else test_pred.numpy()
        test_y = test_y if isinstance(test_y, np.ndarray) else test_y.numpy()
    
    print(f"  Test metric: {metric:.4f}")
    
    # Store results in original order
    if OUTPUT_SIZE_LOOKUP[args.target] == 1:
        all_test_preds[test_indices] = test_pred.reshape(-1, 1) if len(test_pred.shape) == 1 else test_pred
        all_test_ys[test_indices] = test_y.reshape(-1, 1) if len(test_y.shape) == 1 else test_y
    else:
        all_test_preds[test_indices] = test_pred
        all_test_ys[test_indices] = test_y
    
    if test_prob is not None:
        all_test_probs[test_indices] = test_prob
    
    # Store ensemble variances for test trials
    if metadata["ens_vars"]:
        for k in all_test_ens_vars.keys():
            if metadata["ens_vars"][k] is not None:
                all_test_ens_vars[k][test_indices] = metadata["ens_vars"][k][test_indices]
    
    # Store trial mapping info
    if all_test_frame_indices is not None:
        all_test_frame_indices[test_indices] = metadata["trial_frame_indices"][test_indices]
    if all_test_original_trial_indices is not None:
        all_test_original_trial_indices[test_indices] = metadata["original_trial_indices"][test_indices]
    
    # Store fold-specific results
    fold_results.append({
        "fold": fold_idx,
        "metric": metric,
        "n_test": len(test_indices),
        "n_train_val": len(train_val_indices),
    })
    
    print(f"  Fold {fold_idx + 1} complete!")

# Calculate overall metrics
print(f"\n{'='*60}")
print("AGGREGATED RESULTS ACROSS ALL FOLDS")
print(f"{'='*60}")

if args.target in REGRESSION:
    if all_test_ys.shape[-1] == 1:
        from scipy.stats import pearsonr
        overall_metric = pearsonr(all_test_ys.flatten(), all_test_preds.flatten())[0]
    else:
        from sklearn.metrics import r2_score
        overall_metric = []
        for dim in range(all_test_ys.shape[-1]):
            overall_metric.append(r2_score(all_test_ys[..., dim].flatten(), all_test_preds[..., dim].flatten()))
        overall_metric = np.nanmean(overall_metric)
    print(f"Overall Pearson R / R²: {overall_metric:.4f}")
else:
    from sklearn.metrics import accuracy_score
    overall_metric = accuracy_score(all_test_ys.argmax(1), all_test_preds)
    print(f"Overall Accuracy: {overall_metric:.4f}")

print(f"Mean fold metric: {np.mean([r['metric'] for r in fold_results]):.4f} ± {np.std([r['metric'] for r in fold_results]):.4f}")

# Save results
res_dict = {
    "all_test_pred": all_test_preds,
    "all_test_y": all_test_ys,
    "all_test_prob": all_test_probs,
    "all_test_ens_vars": all_test_ens_vars,
    "all_test_frame_indices": all_test_frame_indices,
    "all_test_original_trial_indices": all_test_original_trial_indices,
    "overall_metric": overall_metric,
    "fold_results": fold_results,
    "n_trials": n_trials,
    "n_units": metadata["n_units"],
    "config": {
        "target": args.target,
        "method": args.method,
        "region": args.region,
        "model_name": args.model_name,
        "search": args.search,
    }
}

if not args.use_nlb:
    np.save(save_path / f'{args.eid}_cv.npy', res_dict)
else:
    np.save(save_path / f'{args.eid}_cv_binSize{args.bin_size}_fold{args.fold_idx}.npy', res_dict)

print(f"\nResults saved to: {save_path}")
print(f"  - All test predictions: {all_test_preds.shape}")
print(f"  - All test ground truth: {all_test_ys.shape}")
if all_test_ens_vars["x_ens_var"] is not None:
    print(f"  - Ensemble variances included ✓")
print(f"  - Each trial appears in test exactly once ✓")

