"""Utility functions for Factor Neural Networks."""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def get_device(device_spec: str = "auto") -> torch.device:
    """Get PyTorch device.
    
    Args:
        device_spec: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.)
        
    Returns:
        PyTorch device
    """
    if device_spec == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            ##logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            ##logger.info("Using CPU device")
    else:
        device = torch.device(device_spec)
        ##logger.info(f"Using specified device: {device}")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of trainable parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(
    model: torch.nn.Module,
    filepath ,
) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state (optional)
        epoch: Current epoch
        loss: Current loss
        filepath: Output file path
        metadata: Additional metadata to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
   
    torch.save(checkpoint, filepath)
    #logger.info(f"Model saved to {filepath}")


def load_model(
    model: torch.nn.Module,
    filepath,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        model: Model to load state into
        filepath: Checkpoint file path
        optimizer: Optimizer to load state into (optional)
        device: Device to load model to
        
    Returns:
        Checkpoint metadata
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    #logger.info(f"Model loaded from {filepath}")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0.0),
        'metadata': checkpoint.get('metadata', {})
    }


def predictions_to_dataframe(
    predictions: np.ndarray,
    targets: np.ndarray,
    dataset: Any,  # CountryTimeSeriesDataset
    quantiles: List[float],
    horizons: List[int],
    model_name: str = "FactorNN"
) -> pd.DataFrame:
    """Convert predictions to tidy DataFrame.
    
    Args:
        predictions: Predictions array of shape (N, Q, H)
        targets: Targets array of shape (N, H)
        dataset: Dataset object with country and time info
        quantiles: List of quantile levels
        horizons: List of forecast horizons
        model_name: Name of the model
        
    Returns:
        Tidy DataFrame with predictions
    """
    records = []
    
    N, Q, H = predictions.shape
    
    for i in range(N):
        sample_info = dataset.get_sample_info(i)
        country = sample_info['country']
        time = sample_info['time']
        
        for q_idx, quantile in enumerate(quantiles):
            for h_idx, horizon in enumerate(horizons):
                records.append({
                    'country': country,
                    'TIME': time,
                    'quantile': quantile,
                    'horizon': horizon,
                    'y_true': targets[i, h_idx] if targets.ndim > 1 else targets[i],
                    'y_pred': predictions[i, q_idx, h_idx],
                    'model': model_name
                })
    
    return pd.DataFrame(records)


def factors_to_dataframe(
    factors_dict: Dict[str, np.ndarray],
    dataset: Any,  # CountryTimeSeriesDataset
    quantiles: List[float],
    horizons: List[int]
) -> pd.DataFrame:
    """Convert factors to tidy DataFrame.
    
    Args:
        factors_dict: Dictionary mapping branch keys to factor arrays
        dataset: Dataset object with country and time info
        quantiles: List of quantile levels
        horizons: List of forecast horizons
        
    Returns:
        Tidy DataFrame with factors
    """
    records = []
    
    # Determine number of factors from first branch
    first_key = next(iter(factors_dict.keys()))
    n_samples, n_factors = factors_dict[first_key].shape
    
    for i in range(n_samples):
        sample_info = dataset.get_sample_info(i)
        country = sample_info['country']
        time = sample_info['time']
        
        for q_idx, quantile in enumerate(quantiles):
            for h_idx, horizon in enumerate(horizons):
                branch_key = f"q{quantile:.3f}_h{horizon}".replace(".", "_")
                
                if branch_key in factors_dict:
                    factors = factors_dict[branch_key][i]
                    
                    record = {
                        'country': country,
                        'TIME': time,
                        'quantile': quantile,
                        'horizon': horizon
                    }
                    
                    # Add factor columns
                    for f_idx in range(n_factors):
                        record[f'factor_{f_idx + 1}'] = factors[f_idx]
                    
                    records.append(record)
    
    return pd.DataFrame(records)


def create_model_filename(
    model_name: str,
    config_hash: str,
    date_str: str
) -> str:
    """Create standardized model filename.
    
    Args:
        model_name: Name of the model
        config_hash: Hash of configuration
        date_str: Date string
        
    Returns:
        Filename string
    """
    return f"{model_name}_{date_str}_{config_hash[:6]}.pt"
