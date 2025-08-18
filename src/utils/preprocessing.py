"""Preprocessing utilities for scaling and missing value handling."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Literal
import logging

logger = logging.getLogger(__name__)

MissingPolicy = Literal["forward_fill_then_mean", "interpolate_linear", "drop"]
ScalePolicy = Literal["none", "per_country", "global"]


def handle_missing_values(
    data: Dict[str, pd.DataFrame], 
    policy: MissingPolicy = "forward_fill_then_mean",
    time_col: str = "TIME"
) -> Dict[str, pd.DataFrame]:
    """Handle missing values according to the specified policy.
    
    Args:
        data: Dictionary of country DataFrames
        policy: Missing value handling policy
        time_col: Name of time column
        
    Returns:
        Dictionary of processed DataFrames
    """
    processed_data = {}
    
    for country_code, df in data.items():
        df_processed = df.copy()
        
        if policy == "forward_fill_then_mean":
            # Forward fill first
            for col in df_processed.columns:
                if col != time_col:
                    df_processed[col] = df_processed[col].fillna(method='ffill')
            
            # Fill remaining missing values with mean
            for col in df_processed.columns:
                if col != time_col and df_processed[col].isna().any():
                    mean_val = df_processed[col].mean()
                    df_processed[col] = df_processed[col].fillna(mean_val)
                    
        elif policy == "interpolate_linear":
            for col in df_processed.columns:
                if col != time_col:
                    df_processed[col] = df_processed[col].interpolate(method='linear')
            
            # Fill any remaining NaNs with mean
            for col in df_processed.columns:
                if col != time_col and df_processed[col].isna().any():
                    mean_val = df_processed[col].mean()
                    df_processed[col] = df_processed[col].fillna(mean_val)
                    
        elif policy == "drop":
            df_processed = df_processed.dropna()
            
        else:
            raise ValueError(f"Unknown missing policy: {policy}")
        
        processed_data[country_code] = df_processed
        
        logger.debug(f"Country {country_code}: {len(df)} -> {len(df_processed)} rows after missing value handling")
    
    return processed_data


def scale_features(
    train_data: Dict[str, pd.DataFrame],
    test_data: Optional[Dict[str, pd.DataFrame]] = None,
    policy: ScalePolicy = "per_country",
    target_col: str = "GDP",
    scale_target: bool = False,
    time_col: str = "TIME", 
    trimming: bool = False
) -> Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, pd.DataFrame]], Dict[str, StandardScaler]]:
    """Scale features according to the specified policy.
    
    Args:
        train_data: Training data by country
        test_data: Test data by country (optional)
        policy: Scaling policy
        target_col: Name of target column
        scale_target: Whether to scale the target variable
        time_col: Name of time column
        
    Returns:
        Tuple of (scaled_train_data, scaled_test_data, scalers)
    """
    if policy == "none":
        return train_data, test_data, {}
    
    if trimming:
        # clip each column such that every value aboutthe 95th percentile is set to the 95th percentile
        for country_code, df in train_data.items():
            for col in df.columns:
                if col != time_col:
                    upper_bound = df[col].quantile(0.95)
                    df[col] = np.clip(df[col], None, upper_bound)
            train_data[country_code] = df
            
        


    scalers = {}
    scaled_train = {}
    scaled_test = {} if test_data is not None else None
    
    # Determine columns to scale
    feature_cols = []
    sample_df = next(iter(train_data.values()))
    for col in sample_df.columns:
        if col not in [time_col]:
            if col == target_col and not scale_target:
                continue
            feature_cols.append(col)
    
    if policy == "per_country":
        for country_code, df_train in train_data.items():
            scaler = StandardScaler()
            
            # Fit on training data
            df_scaled_train = df_train.copy()
            # check for infs or nans in df_scaled_train[feature_cols]

            print(country_code) 
            df_scaled_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
            scaled_train[country_code] = df_scaled_train
            
            # Transform test data if available
            if test_data is not None and country_code in test_data:
                df_test = test_data[country_code]
                df_scaled_test = df_test.copy()
                df_scaled_test[feature_cols] = scaler.transform(df_test[feature_cols])
                scaled_test[country_code] = df_scaled_test
            
            scalers[country_code] = scaler
            
    elif policy == "global":
        # Concatenate all training data
        all_train_data = pd.concat(train_data.values(), ignore_index=True)
        
        scaler = StandardScaler()
        scaler.fit(all_train_data[feature_cols])
        
        # Scale training data
        for country_code, df_train in train_data.items():
            df_scaled_train = df_train.copy()
            df_scaled_train[feature_cols] = scaler.transform(df_train[feature_cols])
            scaled_train[country_code] = df_scaled_train
        
        # Scale test data if available
        if test_data is not None:
            for country_code, df_test in test_data.items():
                df_scaled_test = df_test.copy()
                df_scaled_test[feature_cols] = scaler.transform(df_test[feature_cols])
                scaled_test[country_code] = df_scaled_test
        
        scalers['global'] = scaler
    
    else:
        raise ValueError(f"Unknown scaling policy: {policy}")
    
    return scaled_train, scaled_test, scalers


def create_lagged_features(
    data: Dict[str, pd.DataFrame],
    lags: List[int],
    time_col: str = "TIME"
) -> Dict[str, pd.DataFrame]:
    """Create lagged features for all variables.
    
    Args:
        data: Dictionary of country DataFrames
        lags: List of lag periods
        time_col: Name of time column
        
    Returns:
        Dictionary of DataFrames with lagged features
    """

    if lags is None or len(lags) == 0:
        #logger.warning("No lags specified, returning original data")
        return data
    
    lagged_data = {}
    
    for country_code, df in data.items():
        df_lagged = df.copy()
        
        # Create lagged features for all non-time columns
        for col in df.columns:
            if col != time_col:
                for lag in lags:
                    lag_col = f"{col}_l{lag}"
                    df_lagged[lag_col] = df[col].shift(lag)
        
        # Drop rows with missing lagged values
        max_lag = max(lags)
        df_lagged = df_lagged.iloc[max_lag:].reset_index(drop=True)
        
        lagged_data[country_code] = df_lagged
        
        logger.debug(f"Country {country_code}: {len(df)} -> {len(df_lagged)} rows after lagging")
    
    return lagged_data

def create_dummy_variables(
    data: Dict[str, pd.DataFrame],
    time_col: str = "TIME",
    add_country_dummies: bool = False
) -> Dict[str, pd.DataFrame]:
    """Create dummy variables for categorical features.
    
    Args:
        data: Dictionary of country DataFrames
        time_col: Name of time column
        add_country_dummies: Whether to add country dummies
        
    Returns:
        Dictionary of DataFrames with dummy variables
    """
    dummy_data = {}
    
    idx = 0

    if add_country_dummies:

        for country_code, df in data.items():
            df_dummy = df.copy()

            df_dummy['country_dummy'] = np.ones(len(df_dummy)) * idx 
            idx += 1
            
            dummy_data[country_code] = df_dummy
    
    return dummy_data

def validate_frequency(
    data: Dict[str, pd.DataFrame],
    time_col: str = "TIME"
) -> None:
    """Validate that all countries have equal frequency time series.
    
    Args:
        data: Dictionary of country DataFrames
        time_col: Name of time column
        
    Raises:
        ValueError: If frequency is not consistent
    """
    for country_code, df in data.items():
        if len(df) < 2:
            continue
            
        time_series = df[time_col]
        diffs = time_series.diff().dropna()
        
        # Check if all differences are the same
        if not diffs.nunique() == 1:
            logger.warning(f"Country {country_code} has irregular time spacing")
            # Could implement repair logic here if needed
