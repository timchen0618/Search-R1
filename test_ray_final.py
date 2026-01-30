import ray
import torch
import os
import time

# 1. Define a clean temp dir to avoid file permission conflicts
temp_dir = f"/scratch/hc3337/ray_run_{int(time.time())}"
os.makedirs(temp_dir, exist_ok=True)

print(f"Starting Ray in {temp_dir}...", flush=True)

# 2. Initialize Ray with explicit constraints
ray.init(
    # Do NOT use address="local" (it triggers auto-detection). Use None to start fresh.
    address=None,
    
    # FORCE 127.0.0.1. This is the critical fix for the connection timeout.
    _node_ip_address="127.0.0.1",
    
    # Hardcode resources to prevent Ray from scanning the overloaded OS
    num_cpus=8,
    num_gpus=2,
    
    # Limit memory to 2GB to avoid crashing the node
    object_store_memory=2 * 1024 * 1024 * 1024,
    
    # Use your scratch dir
    _temp_dir=temp_dir,
    
    # Disable dashboard
    include_dashboard=False,
    ignore_reinit_error=True
)

print("Ray started successfully on 127.0.0.1!", flush=True)

@ray.remote(num_gpus=1)
def check_gpu():
    return torch.cuda.get_device_name(0)

print("Waiting for GPU task...", flush=True)
try:
    # 20 second timeout
    print(ray.get(check_gpu.remote(), timeout=20))
    print("Success! Ray is working.")
except Exception as e:
    print(f"Task failed: {e}")

ray.shutdown()
