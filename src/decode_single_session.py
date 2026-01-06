"""Example script for running single-session reduced-rank model with hyperparameter sweep.
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge, LogisticRegression
import torch

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
    # These will only be used if model_class != "linear"
    ModelCheckpoint = None
    Trainer = None
    RayDDPStrategy = None
    RayLightningEnvironment = None
    RayTrainReportCallback = None
    prepare_trainer = None
    tune = None

from utils.data_loader_utils import SingleSessionDataModule
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
              "lightning-pose-left-pawR-speed", "lightning-pose-right-pawR-speed"]

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
ap.add_argument("--search", action="store_true")
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

set_seed(config.seed)

config["dirs"]["data_dir"] = Path(args.base_path)/config.dirs.data_dir

# Determine if this is a pose target that needs model_name
POSE_TARGETS = [
    "lightning-pose-left-pawL-speed", "lightning-pose-right-pawL-speed",
    "lightning-pose-left-pawR-speed", "lightning-pose-right-pawR-speed"
]
is_pose_target = args.target in POSE_TARGETS

# Create save paths - include model_name for pose targets
if is_pose_target and args.model_name:
    # For pose targets: results/{eid}/{target}/{model_name}/{method}/{region}/
    save_path = Path(args.base_path)/config.dirs.output_dir/args.eid/args.target/args.model_name/args.method/args.region
    ckpt_path = Path(args.base_path)/config.dirs.checkpoint_dir/args.eid/args.target/args.model_name/args.method/args.region
elif is_pose_target and not args.model_name:
    print(f"WARNING: Decoding pose target '{args.target}' without --model_name. "
          "Consider providing --model_name to organize results by pose model.")
    save_path = Path(args.base_path)/config.dirs.output_dir/args.eid/args.target/args.method/args.region
    ckpt_path = Path(args.base_path)/config.dirs.checkpoint_dir/args.eid/args.target/args.method/args.region
else:
    # For non-pose targets: standard path
    save_path = Path(args.base_path)/config.dirs.output_dir/args.eid/args.target/args.method/args.region
    ckpt_path = Path(args.base_path)/config.dirs.checkpoint_dir/args.eid/args.target/args.method/args.region

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

print(f"Decode {args.target} from session: {args.eid}")


"""
--------
DECODING
--------
"""
print(f'Launch single-session {model_class} decoder:')

# set up model configs
search_space = config.copy()
search_space["eid"] = args.eid
search_space["target"] = args.target
search_space["region"] = args.region if args.region != "all" else "all"
search_space["model"]["output_size"] = OUTPUT_SIZE_LOOKUP[args.target]
search_space["training"]["device"] = torch.device(
    "cuda" if np.logical_and(torch.cuda.is_available(), config.training.device == "gpu") else "cpu"
)
search_space["data"]["use_nlb"] = True if args.use_nlb else False
search_space["data"]["bin_size"] = args.bin_size
search_space["data"]["fold_idx"] = args.fold_idx
# Add pose_model_name to config for loading pose-specific data
search_space["pose_model_name"] = args.model_name if is_pose_target else None

# set up for hyperparameter sweep    
if args.search:
    # Initialize Ray early to prevent auto-initialization issues
    # Note: Ray may have auto-initialized when 'from ray import tune' was imported
    import ray
    import torch
    import time
    import subprocess
    
    # Always shutdown any existing Ray instance first (may be broken from auto-init)
    if ray.is_initialized():
        try:
            print("Shutting down any existing Ray instance (may be from auto-init)...")
            ray.shutdown()
            time.sleep(3)  # Give it time to fully shutdown
        except Exception as e:
            print(f"Warning during Ray shutdown: {e}")
    
    # Force kill any remaining Ray processes to ensure clean state
    print("Cleaning up any remaining Ray processes...")
    subprocess.run(["pkill", "-9", "ray"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "raylet"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    time.sleep(2)
    
    # Verify Ray is shutdown
    if ray.is_initialized():
        print("Warning: Ray still appears initialized, forcing shutdown...")
        try:
            ray.shutdown()
        except:
            pass
        time.sleep(1)
    
    if config.tuner.use_gpu and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Initializing Ray with {num_gpus} GPU(s)...")
        
        # Try to initialize Ray
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ray.init(
                    num_gpus=num_gpus,
                    ignore_reinit_error=True,
                    include_dashboard=False,
                )
                # Wait and verify resources are available
                time.sleep(2)
                cluster_resources = ray.cluster_resources()
                print(f"Ray initialized with resources: {cluster_resources}")
                
                if "GPU" in cluster_resources and cluster_resources.get("GPU", 0) > 0:
                    # Test that Ray is actually functional
                    try:
                        test_obj = ray.put([1, 2, 3])
                        ray.get(test_obj)
                        print(f"✓ Ray ready with {cluster_resources.get('GPU', 0)} GPU(s) and functional")
                        break
                    except Exception as test_e:
                        print(f"Ray initialized but not functional: {test_e}")
                        if attempt < max_retries - 1:
                            ray.shutdown()
                            time.sleep(2)
                            subprocess.run(["pkill", "-9", "ray"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                            subprocess.run(["pkill", "-9", "raylet"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                            time.sleep(2)
                        else:
                            raise RuntimeError(f"Ray is not functional: {test_e}")
                else:
                    print(f"Attempt {attempt + 1}: GPU resources not available, retrying...")
                    if attempt < max_retries - 1:
                        ray.shutdown()
                        time.sleep(2)
                        subprocess.run(["pkill", "-9", "ray"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                        subprocess.run(["pkill", "-9", "raylet"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                        time.sleep(2)
                    else:
                        raise RuntimeError("Failed to initialize Ray with GPU after multiple attempts")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    try:
                        ray.shutdown()
                    except:
                        pass
                    subprocess.run(["pkill", "-9", "ray"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                    subprocess.run(["pkill", "-9", "raylet"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Failed to initialize Ray with GPU: {e}")
    else:
        print("Initializing Ray for CPU...")
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
        )
        time.sleep(1)
        cluster_resources = ray.cluster_resources()
        print(f"Ray initialized for CPU with resources: {cluster_resources}")

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
        # search_space["reduced_rank"]["temporal_rank"] = tune.grid_search(list(range(2, config.tuner.num_samples)))
        search_space["reduced_rank"]["temporal_rank"] = tune.grid_search(list(range(2,  OUTPUT_SIZE_LOOKUP[args.target]+1)))
        search_space["tuner"]["num_epochs"] = config.training.num_epochs
        search_space["training"]["num_epochs"] = config.training.num_epochs
    elif model_class == "lstm":
        search_space["lstm"]["lstm_n_layers"] = tune.randint(1, 3)
        search_space["lstm"]["lstm_hidden_size"] = tune.choice([32, 64, 128, 256, 512])
        search_space["lstm"]["mlp_hidden_size"] = tune.choice(
            generate_mlp_hyperparams(possible_sizes=[256, 128, 64, 32, 16])
        )
        search_space["lstm"]["drop_out"] = tune.uniform(0.1, 0.3)
        search_space["tuner"]["num_epochs"] = config.training.num_epochs
        search_space["training"]["num_epochs"] = config.training.num_epochs
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
        search_space["tuner"]["num_epochs"] = config.training.num_epochs
        search_space["training"]["num_epochs"] = config.training.num_epochs
    else:
        raise NotImplementedError

    def train_func(config):
        dm = SingleSessionDataModule(config)
        dm.update_config()
        dm.setup()

        dm.config["training"]["total_steps"] = dm.config["training"]["num_epochs"] * len(dm.train)

        if model_class == "reduced_rank":
            model = ReducedRankDecoder(dm.config)
        elif model_class == "lstm":
            model = LSTMDecoder(dm.config)
        elif model_class == "mlp":
            model = MLPDecoder(dm.config)
        else:
            raise NotImplementedError
            
        trainer = Trainer(
            max_epochs=config["tuner"]["num_epochs"],
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=config["tuner"]["enable_progress_bar"],
            check_val_every_n_epoch=1,
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(model, datamodule=dm)
    
    # hyperparameter sweep
    results = tune_decoder(
        train_func, 
        search_space, 
        save_dir=ckpt_path,
        use_gpu=config.tuner.use_gpu, 
        max_epochs=config.tuner.num_epochs, 
        num_samples=config.tuner.num_samples if model_class != "reduced_rank" else 1, 
        num_workers=args.n_workers,
        metric=config.tuner.metric,
        mode=config.tuner.mode,
    )
    best_result = results.get_best_result(metric=config.tuner.metric, mode=config.tuner.mode)
    best_config = best_result.config["train_loop_config"]

    print("Best model config:")
    print(best_config)

    torch.save(best_config, ckpt_path / "best_config.pth")


if not args.search:
    best_config = search_space
else:
    best_config = torch.load(ckpt_path / "best_config.pth")

# set up data loader
dm = SingleSessionDataModule(best_config)
dm.update_config()
dm.setup()

best_config["training"]["total_steps"] = best_config["training"]["num_epochs"] * len(dm.train)

# init and train model
if model_class == "reduced_rank":
    model = ReducedRankDecoder(best_config)
elif model_class == "lstm":
    model = LSTMDecoder(best_config)
elif model_class == "mlp":
    model = MLPDecoder(best_config) if not args.use_nlb else SeanMLPDecoder(best_config)
elif model_class == "linear":
    from scipy.stats import loguniform
    from sklearn.linear_model import RidgeCV, LogisticRegressionCV, Ridge
    from sklearn.model_selection import GridSearchCV
    if args.target in REGRESSION:
        # model = RidgeCV(
        #     alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4]
        # )
        model = GridSearchCV(Ridge(), {"alpha": np.logspace(-4, 4, 9)})
    elif args.target in CLASSIFICATION:
        model = LogisticRegressionCV(
            Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4]
        )
    else:
        raise NotImplementedError
else:
    raise NotImplementedError


if model_class != "linear":
    model.to(best_config["training"]["device"])

    # set up trainer
    checkpoint_callback = ModelCheckpoint(
        monitor=config.training.metric, 
        mode=config.training.mode, 
        dirpath=ckpt_path
    )
    trainer = Trainer(
        max_epochs=config.training.num_epochs, 
        callbacks=[checkpoint_callback], 
        enable_progress_bar=config.training.enable_progress_bar,
        check_val_every_n_epoch=1 if model_class == "reduced_rank" else 10, # Otherwise too slow
        devices=1, # Use only one GPU
        strategy="auto",  
    )

    trainer.fit(model, datamodule=dm)

    train_dataset, test_dataset = dm.train, dm.test

    if model_class == "reduced_rank":
        MODEL_CLASS = ReducedRankDecoder
    elif model_class == "lstm":
        MODEL_CLASS = LSTMDecoder
    elif model_class == "mlp":
        MODEL_CLASS = MLPDecoder if not args.use_nlb else SeanMLPDecoder
    else:
        raise NotImplementedError

    model = MODEL_CLASS.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        config=best_config
    )

"""
----------
EVALUATION
----------
"""
if model_class != "linear":
    model.eval()
    with torch.no_grad():
        metric, test_pred, test_y, test_prob = eval_model(
            train_dataset, 
            test_dataset, 
            model.cpu(), 
            target=config["model"]["target"], 
            model_class=model_class,
            use_nlb=args.use_nlb,
            bin_size=args.bin_size,
            beh_name=args.target
        )
else:
    dm.setup()
    train_dataset, test_dataset = dm.train, dm.test
    metric, test_pred, test_y, test_prob = eval_model(
        train_dataset, 
        test_dataset, 
        model, 
        target=config["model"]["target"], 
        model_class=model_class,
        use_nlb=args.use_nlb,
        bin_size=args.bin_size,
        beh_name=args.target
    )

print(f"Decoding results for {args.eid}: ", metric)
res_dict = {
    "test_metric": metric, 
    "test_pred": test_pred, 
    "test_y": test_y,
    "test_prob": test_prob,
}
if not args.use_nlb:
    np.save(save_path/f'{args.eid}.npy', res_dict)
else:
    np.save(save_path/f'{args.eid}_binSize{args.bin_size}_fold{args.fold_idx}.npy', res_dict)


# command to decode a session:
'''
# For non-pose targets (wheel-speed, whisker-motion-energy, choice, etc.):
python src/decode_single_session.py \
    --eid 5c0c560e-9e1f-45e9-b66e-e4ee7855be84 \
    --target wheel-speed \
    --method linear \
    --base_path /media/lenny-aharon/T7/ibl-mouse/ibl-mouse_neural-activity \
    --region all

# For pose targets (lightning-pose-*-speed), use --model_name to specify the pose model:
python src/decode_single_session.py \
    --eid 5c0c560e-9e1f-45e9-b66e-e4ee7855be84 \
    --target lightning-pose-right-pawL-speed \
    --method linear \
    --base_path /media/lenny-aharon/T7/ibl-mouse/ibl-mouse_neural-activity \
    --model_name MVT_patch_masking_0 \
    --region all

# Results are saved to:
# - Non-pose: {base_path}/results/{eid}/{target}/{method}/{region}/
# - Pose: {base_path}/results/{eid}/{target}/{model_name}/{method}/{region}/
'''


# the method can be linear / recuced rank / mlp 
# eids_test = [
#     '15b69921-d471-4ded-8814-2adad954bcd8',  # vertical bright strip right
#     '15763234-d21e-491f-a01b-1238eb96d389',  # dark
#     'aad23144-0e52-4eac-80c5-c4ee2decb198',  # wire by tongue
#     '9b528ad0-4599-4a55-9148-96cc1d93fb24',  # vertical bright band left
#     '5c0c560e-9e1f-45e9-b66e-e4ee7855be84',  # vertical bright band left, bright spot right, back paws
# ]

