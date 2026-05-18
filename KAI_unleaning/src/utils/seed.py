"""
시드 고정을 위한 유틸
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False, device: str = "cuda") -> None:
    if seed is None:
        seed = 42

    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # Device-specific seed settings
    device_lower = device.lower()
    if device_lower == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # cuDNN settings (CUDA only)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark

        # Additional CUDA settings
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    elif device_lower == "mps" and torch.backends.mps.is_available():
        torch.use_deterministic_algorithms(deterministic, warn_only=True)

    # Common settings
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def get_worker_init_fn(seed: int, device: str = "cuda"):
    
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

        device_lower = device.lower()
        if device_lower == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

    return worker_init_fn
