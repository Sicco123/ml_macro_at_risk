"""Seed control utilities for reproducible experiments."""

import random
import numpy as np
import torch
from typing import Optional


def set_seeds(seed: Optional[int] = None) -> int:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility.
    
    Args:
        seed: Random seed to use. If None, uses 42.
        
    Returns:
        The seed that was set.
    """
    if seed is None:
        seed = 42
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return seed


def get_worker_seed(base_seed: int, worker_id: int) -> int:
    """Generate a unique seed for data loader workers.
    
    Args:
        base_seed: Base random seed
        worker_id: Worker ID
        
    Returns:
        Unique seed for the worker
    """
    return base_seed + worker_id
