"""I/O utilities for loading and saving data and models."""

import pandas as pd
import numpy as np
import json
import pickle
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def load_country_data(data_path: Union[str, Path], required_columns: List[str]) -> Dict[str, pd.DataFrame]:
    """Load country CSV files and validate required columns.
    
    Args:
        data_path: Path to directory containing country CSV files
        required_columns: List of required column names
        
    Returns:
        Dictionary mapping country codes to DataFrames
        
    Raises:
        ValueError: If required columns are missing or inconsistent across countries
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Find all CSV files
    csv_files = list(data_path.glob("*.csv"))

    # if no CSV files look for parquet files
    if not csv_files:
        csv_files = list(data_path.glob("*.parquet"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_path}")
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    country_data = {}
    all_columns = set()
    df = pd.DataFrame()  # Initialize an empty DataFrame for column checks
    for csv_file in csv_files:
        country_code = csv_file.stem.upper()
        
        try:
            if csv_file.suffix == '.parquet':
                df = pd.read_parquet(csv_file)
                # put the index as first column
                df.reset_index(inplace=True)
                
            else:   
                df = pd.read_csv(csv_file)

            
            # Parse TIME column
            if 'TIME' in df.columns:
                df['TIME'] = pd.to_datetime(df['TIME'])
                df = df.sort_values('TIME').reset_index(drop=True)
            else:
                logger.warning(f"Country {country_code} missing 'TIME' column, skipping")
                continue

            
            
            # Check for required columns
            if not required_columns:
                required_columns = df.columns.tolist()

            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                logger.warning(f"Country {country_code} missing columns: {missing_cols}")
                continue

            df = df[required_columns]
            
            all_columns.update(df.columns)
            country_data[country_code] = df
            logger.debug(f"Loaded {country_code}: {len(df)} rows, {len(df.columns)} columns")
            
        except Exception as e:
            logger.error(f"Failed to load {csv_file}: {e}")
            continue
    
    if not country_data:
        raise ValueError("No valid country data files loaded")
    
    # Check column consistency across countries
    for country_code, df in country_data.items():
        extra_cols = set(df.columns) - all_columns
        if extra_cols:
            logger.warning(f"Country {country_code} has extra columns: {extra_cols}")
    
    logger.info(f"Successfully loaded {len(country_data)} countries")
    return country_data


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        path: Configuration file path
        
    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from JSON file.
    
    Args:
        path: JSON file path
        
    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """Save object to pickle file.
    
    Args:
        obj: Object to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load object from pickle file.
    
    Args:
        path: Pickle file path
        
    Returns:
        Loaded object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
