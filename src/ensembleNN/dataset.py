"""Dataset and data loading utilities for Factor Neural Networks."""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CountryTimeSeriesDataset(Dataset):
    """Dataset for multi-country time series data with lagged features."""
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        target_col: str,
        quantiles: List[float],
        horizons: List[int],
        lags: List[int] = [1],
        time_col: str = "TIME",
        exclude_contemp_target: bool = False, 
        one_country_idx: Optional[int] = 0,
    ):
        """Initialize dataset.
        
        Args:
            data: Dictionary of country DataFrames
            target_col: Name of target column
            horizons: List of forecast horizons
            lags: List of lag periods
            time_col: Name of time column
            exclude_contemp_target: Whether to exclude contemporaneous target from features
        """
        self.data = data
        self.target_col = target_col
        self.quantiles = quantiles
        self.horizons = horizons
        self.lags = lags
        self.time_col = time_col
        self.exclude_contemp_target = exclude_contemp_target
        self.one_country_idx = one_country_idx  # If only one country, use this index
        
        # Build feature matrix and targets
        self._build_dataset()
    
    def _build_dataset(self):
        """Build the dataset tensors."""
        all_features = []
        all_targets = []
        all_countries = []
        all_times = []

        # if len self.data.items() > 1
        if len(self.data.items()) > 1:
            idx = 0
            for country_code, df in self.data.items():
                features, targets, times = self._process_country_data(df, country_code)
                country_code_int = idx
                if len(features) > 0:
                    all_features.append(features)
                    all_targets.append(targets)
                    all_countries.extend([country_code_int] * len(features))
                    all_times.extend(times)
            idx += 1
        else:
            # Only one country, use index 0
            country_code, df = next(iter(self.data.items()))
            features, targets, times = self._process_country_data(df, country_code)
            country_code_int = self.one_country_idx
            if len(features) > 0:
                all_features.append(features)
                all_targets.append(targets)
                all_countries.extend([country_code_int] * len(features))
                all_times.extend(times)

        if not all_features:
            raise ValueError("No valid samples found in dataset")
        
        # Debug: Check dtypes before stacking
        logger.debug(f"About to stack {len(all_features)} feature arrays")
        for i, feat in enumerate(all_features):
            logger.debug(f"Feature array {i} dtype: {feat.dtype}, shape: {feat.shape}")
            # Force all arrays to be float64 before stacking
            if feat.dtype != np.float64:
                logger.warning(f"Converting feature array {i} from {feat.dtype} to float64")
                all_features[i] = feat.astype(np.float64)
        

        # Concatenate all countries
        self.features = np.vstack(all_features)
        self.targets = np.vstack(all_targets)

        del all_features
        del all_targets
        
        # Debug: Check dtype right after stacking
        logger.debug(f"After vstack - features dtype: {self.features.dtype}, targets dtype: {self.targets.dtype}")

        self.countries = np.array(all_countries).reshape(-1, 1, 1)

        self.times = all_times



        # Ensure features and targets are proper numeric arrays
        if self.features.dtype == object:
            logger.error("Features array has object dtype after vstack, forcing conversion")
            try:
                # First attempt: direct conversion
                self.features = self.features.astype(np.float64)
            except (ValueError, TypeError) as e:
                logger.error(f"Direct conversion failed: {e}")
                try:
                    # Second attempt: element-wise conversion
                    logger.warning("Attempting element-wise conversion")
                    self.features = np.array([[float(x) if pd.notna(x) and x is not None else 0.0 for x in row] for row in self.features], dtype=np.float64)
                except (ValueError, TypeError) as e2:
                    logger.error(f"Element-wise conversion failed: {e2}")
                    # Last resort: create zeros array with correct shape
                    logger.error("Creating zeros array as last resort")
                    self.features = np.zeros(self.features.shape, dtype=np.float64)
        
        if self.targets.dtype == object:
            logger.error("Targets array has object dtype after vstack, forcing conversion")
            try:
                # First attempt: direct conversion
                self.targets = self.targets.astype(np.float64)
            except (ValueError, TypeError) as e:
                logger.error(f"Direct conversion failed: {e}")
                try:
                    # Second attempt: element-wise conversion
                    logger.warning("Attempting element-wise conversion")
                    self.targets = np.array([[float(x) if pd.notna(x) and x is not None else 0.0 for x in row] for row in self.targets], dtype=np.float64)
                except (ValueError, TypeError) as e2:
                    logger.error(f"Element-wise conversion failed: {e2}")
                    # Last resort: create zeros array with correct shape
                    logger.error("Creating zeros array as last resort")
                    self.targets = np.zeros(self.targets.shape, dtype=np.float64)

        #logger.info(f"Dataset built: {len(self.features)} samples, {self.features.shape[1]} features")
        #logger.info(f"Features dtype: {self.features.dtype}, Targets dtype: {self.targets.dtype}")
    
    def _process_country_data(self, df: pd.DataFrame, country_code: str) -> Tuple[np.ndarray, np.ndarray, List]:
        """Process data for a single country."""
        # Determine feature columns
        feature_cols = [f"{self.target_col}"] #"{self.target_col}_untransformed"] #
        target_cols = []
        horizon_target_cols = [f"{self.target_col}_h{h}_q{q}" for h in self.horizons for q in self.quantiles]

        for col in df.columns:
            if col == self.time_col or col == f"{self.target_col}_untransformed" or col == self.target_col:
               
                continue
            # if {target_col}_h{horizon} = col continue 
            if col in horizon_target_cols:
          
                continue

            feature_cols.append(col)

        for q in self.quantiles:
            per_quantile = []
            for h in self.horizons:
                per_quantile.append(f"{self.target_col}_h{h}_q{q}")
            target_cols.append(per_quantile)
            
        feature_data_array = df[feature_cols].values
        
        target_data_array = np.zeros((len(df), len(self.quantiles), len(self.horizons) ))
        for i, q in enumerate(self.quantiles):
            for j, h in enumerate(self.horizons):
                target_col_name = f"{self.target_col}_h{h}_q{q}"
                target_data_array[:, i, j] = df[target_col_name].values

        
        
        # Extract times
        times = df[self.time_col].tolist()



        return feature_data_array, target_data_array, times
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.
        
        Returns:
            Tuple of (features, targets) where targets shape is (H,) for H horizons
        """
        try:
            # # Get data
            # features_data = self.features[idx]
            # targets_data = self.targets[idx]
            # country_codes = self.countries[idx]
            
            # # Robust conversion to float64 - multiple fallback strategies
            # def safe_convert_to_float64(data, name, idx):
            #     """Safely convert data to float64 with multiple fallback strategies."""
            #     if isinstance(data, np.ndarray) and data.dtype == np.float64:
            #         return data
                
            #     # Strategy 1: Direct astype conversion
            #     try:
            #         return np.array(data, dtype=np.float64)
            #     except (ValueError, TypeError):
            #         pass
                
            #     # Strategy 2: Element-wise conversion
            #     try:
            #         if data.ndim == 1:
            #             return np.array([float(x) if pd.notna(x) and x is not None else 0.0 for x in data], dtype=np.float64)
            #         else:
            #             return np.array([[float(x) if pd.notna(x) and x is not None else 0.0 for x in row] for row in data], dtype=np.float64)
            #     except (ValueError, TypeError):
            #         pass
                
            #     # Strategy 3: Last resort - create zeros with same shape
            #     logger.error(f"All conversion strategies failed for {name} at index {idx}, creating zeros")
            #     if hasattr(data, 'shape'):
            #         return np.zeros(data.shape, dtype=np.float64)
            #     else:
            #         return np.zeros(len(data) if hasattr(data, '__len__') else 1, dtype=np.float64)
            
            # features_data = safe_convert_to_float64(features_data, "features", idx)
            # targets_data = safe_convert_to_float64(targets_data, "targets", idx)
            
            # # Convert to tensors
            # features = torch.FloatTensor(features_data)
            # targets = torch.FloatTensor(targets_data)
            # country_codes = torch.IntTensor(country_codes)
            # return features, targets, country_codes
            # Avoid intermediate numpy arrays
            features = torch.from_numpy(self.features[idx]).float()
            targets = torch.from_numpy(self.targets[idx]).float()
            country_codes = torch.from_numpy(self.countries[idx]).int()
            
            return features, targets, country_codes

        except Exception as e:
            logger.error(f"Complete failure converting sample {idx} to tensor: {e}")
            logger.error(f"Sample country: {self.countries[idx] if idx < len(self.countries) else 'Unknown'}")
            logger.error(f"Sample time: {self.times[idx] if idx < len(self.times) else 'Unknown'}")
            
            # Last resort: return zero tensors with correct shapes
            feature_dim = self.features.shape[1] if hasattr(self.features, 'shape') and len(self.features.shape) > 1 else 10
            target_dim = self.targets.shape[1] if hasattr(self.targets, 'shape') and len(self.targets.shape) > 1 else len(self.horizons)
            
            logger.error(f"Returning zero tensors: features({feature_dim},), targets({target_dim},)")
            return torch.zeros(feature_dim), torch.zeros(target_dim)
    
    def get_feature_dim(self) -> int:
        """Get the feature dimension."""
        return self.features.shape[1]
    
    def get_sample_info(self, idx: int) -> Dict[str, any]:
        """Get metadata for a sample."""
        return {
            'country': self.countries[idx],
            'time': self.times[idx],
            'features': self.features[idx],
            'targets': self.targets[idx]
        }


def create_data_loaders(
    data: Dict[str, pd.DataFrame],
    ensemble_size: int,
    target_col: str,
    quantiles: List[float],
    horizons: List[int],
    lags: List[int] = [1],
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None,
    val_size: Optional[float] = None,
) -> Tuple[List[DataLoader], Optional[List[DataLoader]]]:
    """Create train and validation data loaders.
    
    Args:
        data: Training data by country
        ensemble_size: Number of ensemble members
        target_col: Name of target column
        quantiles: List of quantile levels
        horizons: List of forecast horizons
        lags: List of lag periods
        batch_size: Batch size
        shuffle: Whether to shuffle training data
        num_workers: Number of data loader workers
        seed: Random seed for data loading
        val_size: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_loaders, val_loaders)
    """
    train_loaders = []
    val_loaders = []
    
    for idx in range(ensemble_size):
        train_data = {}
        val_data = None
        
        if val_size is not None:
            # Split data into train and validation sets
            for country, df in data.items():
                n_val = int(len(df) * val_size)
                if shuffle:
                    df = df.sample(frac=1, random_state=seed+idx if seed else None).reset_index(drop=True)
                if n_val > 0:
                    train_data[country] = df[:-n_val]
                    if val_data is None:
                        val_data = {}
                    val_data[country] = df[-n_val:]
                else:
                    train_data[country] = df
            
            if val_data is None:
                logger.warning("Validation size is too small, no validation set created")
        else:
            train_data = data
        

        train_dataset = CountryTimeSeriesDataset(
            train_data, target_col, quantiles, horizons, lags
        )


        val_dataset = None
        if val_data is not None:
            val_dataset = CountryTimeSeriesDataset(
                val_data, target_col, quantiles, horizons, lags
            )

    
        # Create data loaders
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed + idx)  # Ensure different seed for each ensemble member
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=generator,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    
    return train_loaders, val_loaders


def get_country_sample_indices(
    dataset: CountryTimeSeriesDataset,
    countries: List[str]
) -> Dict[str, List[int]]:
    """Get sample indices for specific countries.
    
    Args:
        dataset: Dataset object
        countries: List of country codes
        
    Returns:
        Dictionary mapping country codes to sample indices
    """
    indices = {country: [] for country in countries}
    
    for idx, country in enumerate(dataset.countries):
        if country in indices:
            indices[country].append(idx)
    
    return indices



