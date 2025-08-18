"""Time splits and cross-validation utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def create_time_split(
    data: Dict[str, pd.DataFrame],
    train_start: Union[str, datetime],
    test_cutoff: Union[str, datetime],
    min_train_points: int = 60,
    time_col: str = "TIME"
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], List[str]]:
    """Create temporal train/test split for each country.
    
    Args:
        data: Dictionary of country DataFrames
        test_cutoff: Last date in training data
        min_train_points: Minimum number of training points required
        time_col: Name of time column
        
    Returns:
        Tuple of (train_data, test_data, dropped_countries)
    """
    if isinstance(train_start, str):
        train_start = pd.to_datetime(train_start)
    if isinstance(test_cutoff, str):
        test_cutoff = pd.to_datetime(test_cutoff)
    
    train_data = {}
    test_data = {}
    dropped_countries = []
    
    for country_code, df in data.items():

        # remove rows before train_start
        df = df[df[time_col] >= train_start]
        # Split by time
        train_mask = df[time_col] <= test_cutoff
        test_mask = df[time_col] > test_cutoff
        
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
        
        # Check minimum training points
        if len(df_train) < min_train_points:
            logger.warning(f"Country {country_code} has only {len(df_train)} training points, dropping")
            dropped_countries.append(country_code)
            continue
        
        # Only include countries with both train and test data
        if len(df_test) == 0:
            logger.warning(f"Country {country_code} has no test data, dropping")
            dropped_countries.append(country_code)
            continue
        
        train_data[country_code] = df_train
        test_data[country_code] = df_test
        
        logger.debug(f"Country {country_code}: {len(df_train)} train, {len(df_test)} test")
    
    logger.info(f"Created time splits for {len(train_data)} countries, dropped {len(dropped_countries)}")
    
    return train_data, test_data, dropped_countries


def create_country_folds(
    countries: List[str],
    k_folds: int,
    seed: Optional[int] = None
) -> List[Tuple[List[str], List[str]]]:
    """Create k-fold cross-validation splits across countries.
    
    Args:
        countries: List of country codes
        k_folds: Number of folds
        seed: Random seed for shuffling
        
    Returns:
        List of (train_countries, val_countries) tuples
    """
    if seed is not None:
        np.random.seed(seed)
    
    countries = countries.copy()
    np.random.shuffle(countries)
    
    # Split countries into k groups
    folds = np.array_split(countries, k_folds)
    
    # Create train/val splits
    cv_splits = []
    for i in range(k_folds):
        val_countries = folds[i].tolist()
        train_countries = [c for c in countries if c not in val_countries]
        cv_splits.append((train_countries, val_countries))
    
    logger.info(f"Created {k_folds} country folds")
    for i, (train_c, val_c) in enumerate(cv_splits):
        logger.debug(f"Fold {i}: {len(train_c)} train countries, {len(val_c)} val countries")
    
    return cv_splits


def create_forecast_targets(
    data: Dict[str, pd.DataFrame],
    target_col: str,
    horizons: List[int],
    time_col: str = "TIME", 
    remove_other_columns: bool = False
) -> Dict[str, pd.DataFrame]:
    """Create multi-step forecast targets.
    
    Args:
        data: Dictionary of country DataFrames
        target_col: Name of target column
        horizons: List of forecast horizons
        time_col: Name of time column
        
    Returns:
        Dictionary of DataFrames with target columns for each horizon
    """
    target_data = {}

    
    for country_code, df in data.items():
        df_targets = df.copy()
        
        # Create target columns for each horizon
        for h in horizons:
            target_h_col = f"{target_col}_h{h}"
            df_targets[target_h_col] = df[target_col].shift(-h)
        
        # Drop rows where any target is missing
        target_cols = [f"{target_col}_h{h}" for h in horizons]
        df_targets = df_targets.dropna(subset=target_cols)
         
        # add time column to target cols
        if remove_other_columns: 
            target_cols.append(time_col)
            target_data[country_code] = df_targets[target_cols]
        else:
            target_data[country_code] = df_targets
        
        logger.debug(f"Country {country_code}: {len(df)} -> {len(df_targets)} rows after creating targets")
    
    
    return target_data


def get_valid_sample_indices(
    data: Dict[str, pd.DataFrame],
    lags: List[int],
    horizons: List[int],
    target_col: str = "GDP"
) -> Dict[str, np.ndarray]:
    """Get indices of valid samples (with all required lags and targets).
    
    Args:
        data: Dictionary of country DataFrames
        lags: List of lag periods
        horizons: List of forecast horizons
        target_col: Name of target column
        
    Returns:
        Dictionary mapping country codes to valid indices
    """
    valid_indices = {}
    
    max_lag = max(lags) if lags else 0
    max_horizon = max(horizons)
    
    for country_code, df in data.items():
        # Valid indices account for max lag and max horizon
        start_idx = max_lag
        end_idx = len(df) - max_horizon
        
        if end_idx > start_idx:
            valid_indices[country_code] = np.arange(start_idx, end_idx)
        else:
            valid_indices[country_code] = np.array([])
            logger.warning(f"Country {country_code} has no valid samples")
    
    return valid_indices
