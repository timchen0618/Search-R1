import ray
import torch
import os

# 1. Force a clean, user-owned temp directory
temp_dir = f"/scratch/hc3337/ray_fix_{os.getpid()}"
os.makedirs(temp_dir, exist_ok=True)

print("Starting Ray with constrained resources...", flush=True)

# 2. Initialize Ray with strict limits
ray.init(
    address="local",
    num_cpus=4,   # Don't check how many CPUs the OS has
    num_gpus=1,   # Don't check how many GPUs the OS has
    object_store_memory=2 * 1024 * 1024 * 1024, # Limit Object Store to 2GB (Critical!)
    _temp_dir=temp_dir,
    include_dashboard=False,
    ignore_reinit_error=True,
    _system_config={
        "automatic_object_store_memory_limit": False
    }
)

print("Ray started! Checking GPU access...", flush=True)

@ray.remote(num_gpus=1)
def check_gpu():
    return torch.cuda.get_device_name(0)

# 3. Run the task
try:
    print(ray.get(check_gpu.remote(), timeout=20))
    print("Success! Ray is working.")
except Exception as e:
    print(f"Task failed: {e}")

ray.shutdown()
