"""Utils for working with Ray Tune."""
import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig, FailureConfig
from ray.train.torch import TorchTrainer

def tune_decoder(
    train_func, search_space, save_dir='EXAMPLE_PATH',
    max_epochs=500, num_samples=10, use_gpu=False, num_workers=1, 
    metric="val_loss", mode="min", 
):
    # Check if Ray is already initialized with the resources we need
    import torch
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if ray.is_initialized():
        try:
            cluster_resources = ray.cluster_resources()
            if use_gpu and num_gpus > 0:
                # Check if GPU resources are available
                if "GPU" in cluster_resources and cluster_resources.get("GPU", 0) > 0:
                    print(f"✓ Ray already initialized with GPU support: {cluster_resources.get('GPU', 0)} GPU(s)")
                    # Ray is already set up correctly, skip initialization
                else:
                    print("Ray is initialized but doesn't have GPU support. Reinitializing...")
                    ray.shutdown()
                    import time
                    time.sleep(2)
                    ray.init(
                        num_gpus=num_gpus,
                        ignore_reinit_error=True,
                        include_dashboard=False,
                    )
                    print(f"Ray reinitialized with {num_gpus} GPU(s)")
            elif not use_gpu:
                print(f"✓ Ray already initialized: {cluster_resources}")
                # Ray is already set up, skip initialization
        except Exception as e:
            print(f"Warning: Error checking Ray resources: {e}, reinitializing...")
            if ray.is_initialized():
                ray.shutdown()
                import time
                time.sleep(2)
    
    # Initialize Ray only if not already initialized
    if not ray.is_initialized():
        if use_gpu and num_gpus > 0:
            try:
                ray.init(
                    num_gpus=num_gpus,
                    ignore_reinit_error=True,
                    include_dashboard=False,  # Disable dashboard to avoid pydantic errors
                )
                print(f"Ray initialized with {num_gpus} GPU(s)")
                
                # Verify GPU resources are available
                cluster_resources = ray.cluster_resources()
                print(f"Ray cluster resources: {cluster_resources}")
                if "GPU" not in cluster_resources or cluster_resources.get("GPU", 0) == 0:
                    print("Warning: Ray initialized but GPUs not detected in cluster resources!")
                    print("Falling back to CPU mode")
                    use_gpu = False
            except Exception as e:
                print(f"Error initializing Ray with GPU: {e}")
                print("Falling back to CPU mode")
                use_gpu = False
                ray.init(
                    ignore_reinit_error=True,
                    include_dashboard=False,
                )
                print("Ray initialized for CPU")
        else:
            # Initialize Ray for CPU-only
            try:
                ray.init(
                    ignore_reinit_error=True,
                    include_dashboard=False,
                )
                print("Ray initialized for CPU")
            except Exception as e:
                print(f"Error initializing Ray: {e}")
                # Try one more time
                ray.init(
                    ignore_reinit_error=True,
                    include_dashboard=False,
                )
                print("Ray initialized for CPU (fallback)")
    
    if use_gpu:
        # For GPU, use 1 worker per trial (each trial gets 1 GPU)
        # This allows Ray Tune to schedule trials sequentially on the single GPU
        resources_per_worker={"CPU": 1, "GPU": 1}
        # Force num_workers to 1 for GPU to ensure each trial uses exactly 1 GPU
        num_workers = 1
        print(f"GPU mode: Using 1 worker per trial, each trial will use 1 GPU")
    else:
        resources_per_worker={"CPU": 1}
        print(f"CPU mode: Using {num_workers} workers per trial")

    scaling_config = ScalingConfig(
        num_workers=num_workers, 
        use_gpu=use_gpu, 
        resources_per_worker=resources_per_worker if use_gpu else None
    )

    run_config = RunConfig(
        storage_path=save_dir,
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute=metric,
            checkpoint_score_order=mode,
        ),
        # stop={"training_iteration": max_epochs},  # seconds
        # progress_reporter removed - deprecated in newer Ray versions (2.x+)
        failure_config=FailureConfig(max_failures=2), 
    )
    
    scheduler = ASHAScheduler(max_t=max_epochs, grace_period=1, reduction_factor=2)

    # Create TorchTrainer - it will be used as a trainable by Tune
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    
    # Verify Ray is still initialized and has GPU resources
    if not ray.is_initialized():
        raise RuntimeError("Ray was not initialized properly! It should have been initialized earlier.")
    
    # Double-check GPU resources are visible
    import time
    time.sleep(1)  # Give Ray a moment to stabilize
    
    try:
        cluster_resources = ray.cluster_resources()
        print(f"Ray resources before Tune: {cluster_resources}")
        
        if use_gpu:
            available_gpus = cluster_resources.get("GPU", 0)
            if available_gpus == 0:
                print("ERROR: GPU resources not visible to Ray!")
                print("This may cause trials to fail. Trying to reinitialize...")
                # Don't reinitialize here - it should have been done earlier
                raise RuntimeError("GPU resources not available in Ray cluster")
            else:
                print(f"✓ {available_gpus} GPU(s) available for Ray Tune")
    except Exception as e:
        print(f"Error checking Ray resources: {e}")
        # If resources check fails, Ray might be in a bad state
        raise RuntimeError(f"Ray cluster is in a bad state: {e}")
    
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    
    # Verify resources are still visible after creating Tuner
    if ray.is_initialized():
        cluster_resources_after = ray.cluster_resources()
        print(f"Ray resources after Tune creation: {cluster_resources_after}")
    
    return tuner.fit()
    