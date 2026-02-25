# src/utils/device_utils.py

import torch
import os

def get_device():
    """
    Returns the device to use for computations, falling back to CPU if GPU or MPS is not available.

    :return: A torch.device object indicating which device to use
    """
    if torch.cuda.is_available():  # Check if CUDA (NVIDIA GPUs) is available
        print(f"Using NVIDIA CUDA on {torch.cuda.get_device_name(torch.cuda.current_device())}")
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  # Check if MPS (Apple Silicon Macs) is available
        print("Using Apple MPS")
        return torch.device('mps')
    else:  # Fallback to CPU
        print("Using CPU")
        return torch.device('cpu')