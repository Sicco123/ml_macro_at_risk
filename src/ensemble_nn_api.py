import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Optional, Literal, Union
from pathlib import Path
import logging
import hashlib
from datetime import datetime
import statsmodels.api as sm

from .ensembleNN.utils import get_device
from .ensembleNN.dataset import CountryTimeSeriesDataset, create_data_loaders
from .ensembleNN.model import EnsembleNN
from .ensembleNN.training import EnsembleNNTrainer

from .utils import (
    create_lagged_features,
    create_forecast_targets,
    set_seeds
)

logger = logging.getLogger(__name__)


class EnsembleNNAPI:  # Changed class name to avoid conflict
    def __init__(self,
        data_list: List[pd.DataFrame],
        target: str,
        quantiles: List[float],
        forecast_horizons: List[int],
        units_per_layer: List[int],
        lags: Optional[List[int]] = None,
        activation: str = "relu",
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = 42,
        transform: Optional[bool] = True,
        prefit_AR: Optional[bool] = True, 
        country_ids: Optional[List[str]] = None, 
        time_col: str = "TIME",
        verbose: int = 1  # Added missing verbose parameter
    ) -> None:
        
        # Set random seed
        if seed is not None:
            set_seeds(seed)
        
        # Set parameters
        self.target = target
        self.quantiles = quantiles
        self.forecast_horizons = forecast_horizons
        self.units_per_layer = units_per_layer
        self.lags = lags
        self.activation = activation
        self.seed = seed
        self.time_col = time_col
        self.verbose = verbose  # Store verbose parameter
        
        if transform:
            self.transformations = {}
        if prefit_AR:
            self.ar_models = {}
        if country_ids is None:
            self.country_ids = [f"COUNTRY_{i:03d}" for i in range(len(data_list))]
        else:
            self.country_ids = country_ids  # Fixed: actually use the provided country_ids
            
        self.country_data = {country_id: data for country_id, data in zip(self.country_ids, data_list)}

        # Setup device
        if isinstance(device, str):
            self.device = get_device(device)
        else:
            self.device = device or get_device("auto")

        # Create lagged features and targets
        self._prepare_data()

        # Initialize model (will be created during fit)
        self.model = None
        self.trainer = None
        self.is_fitted = False
        
        logger.info(f"EnsembleNNAPI initialized with {len(self.country_data)} countries")
        
    def predict(self, data_list: List[pd.DataFrame]) -> Dict[str, np.ndarray]:
        predictions = []
        targets = []

        for i, df in enumerate(data_list):
            pred, target = self.predict_per_country(df, self.country_ids[i])  # Fixed method name
            predictions.append(pred)
            targets.append(target)
        # stack in the first dimension
        predictions = np.stack(predictions, axis=0)  # Shape (N, Q, H)
        targets = np.stack(targets, axis=0)  # Shape (N, H)
        return predictions, targets
    
    def predict_per_country(self, df, country_id):
        # Convert to country dictionary
        pred_country_data = {}
  
        pred_country_data[country_id] = df.copy()
    
        # Prepare data
        pred_lagged_data = create_lagged_features(
            pred_country_data, self.lags, time_col="TIME"
        )
        pred_target_data = create_forecast_targets(
            pred_lagged_data, self.target, self.forecast_horizons, time_col="TIME"
        )

        target_cols = [f"{self.target}_h{h}" for h in self.forecast_horizons]

        targets = pred_target_data[country_id][target_cols].values
        pred_target_data_raw = pred_target_data[country_id].copy()
        # Apply stored transformations to prediction data
        if hasattr(self, 'transformations'):
            for country, df in pred_target_data.items():
                df = self._remove_AR_part(df, country)

                if country in self.transformations:
                    # Apply normalization using stored lambda functions
                    for col, transform_dict in self.transformations[country].items():
                        if col in df.columns:
                            pred_target_data[country][col] = transform_dict['normalize'](df[col])
                else:
                    logger.warning(f"No transformations found for country {country}. Using raw data.")
        
        # Create dataset and data loader
        pred_dataset = CountryTimeSeriesDataset(
            pred_target_data, self.target, self.quantiles, self.forecast_horizons, self.lags
        )
        
        pred_loader = torch.utils.data.DataLoader(
            pred_dataset, batch_size=256, shuffle=False
        )
        
        # Make predictions
        predictions, _ = self.trainer.predict(pred_loader)
        
        
        # Add AR part back to predictions
        predictions = self._add_AR_part(predictions, country_id, pred_target_data_raw)  

        return predictions, targets

    def fit(self,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        validation_size: float = 0.2,
        patience: int = 10,
        verbose: int = 1,
        optimizer: str = "adam",
        parallel_models: int = 1,
        l2: float = 0.0,
        return_validation_loss: bool = False,
        return_train_loss: bool = False,
        shuffle: bool = True
        ) -> Dict:

        if self.seed is not None:
            set_seeds(self.seed)

        self._prefit()
        self._pretransform()
        
        
        train_loaders, val_loaders = create_data_loaders(
            self.features_and_targets,
            parallel_models,  # This should be the ensemble size
            self.target,
            self.quantiles,
            self.forecast_horizons,
            self.lags,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=self.seed, 
            val_size=validation_size,  # Fixed parameter name
        )

        # Initialize model ensemble
        self.model = EnsembleNN(
            input_dim=self.input_dim,
            quantiles=self.quantiles,
            forecast_horizons=self.forecast_horizons,
            units_per_layer=self.units_per_layer,
            n_models=parallel_models,
            activation=self.activation
        )
        
        self.trainer = EnsembleNNTrainer(self.model, self.quantiles, self.device)

        if verbose >= 1:
            logger.info(f"Training ensemble with {parallel_models} models")
            logger.info(f"Train samples: {len(train_loaders[0].dataset)}, Val samples: {len(val_loaders[0].dataset) if val_loaders[0] else 0}")
    
        print("Network part...")
        # Train the ensemble
        history = self.trainer.fit(
            train_loaders=train_loaders,  # Fixed parameter name
            val_loader=val_loaders,
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer_type=optimizer,
            patience=patience,
            verbose=verbose,
            l2=l2
        )

        # history is a dict with train_losses and val_losses append them 
        if 'train_losses' not in history:
            history['train_losses'] = []
        if 'val_losses' not in history:
            history['val_losses'] = []
        

        # Prepare results
        results = {
            'history': history,
            'n_parameters': sum(p.numel() for p in self.model.parameters()),
            'validation_size': validation_size,
            'n_train_samples': len(train_loaders[0].dataset),
            'n_val_samples': len(val_loaders[0].dataset) if val_loaders else 0,
        }
        
        if return_train_loss and 'train_losses' in history:
            results['train_losses'] = history['train_losses']
        
        if return_validation_loss and 'val_losses' in history:
            results['val_losses'] = history['val_losses']
            results['final_val_loss'] = np.min(history['val_losses']) if history['val_losses'] else None
        
        self.is_fitted = True
        
        if verbose >= 1:
            final_val_loss = results.get('final_val_loss')
            if final_val_loss is not None:
                logger.info(f"Training completed. Best validation loss: {final_val_loss:.6f}")
            else:
                logger.info("Training completed.")
        
        return results
    
    def _prepare_data(self) -> None:

        # create lagged features
        self.lagged_data = create_lagged_features(
            self.country_data, self.lags, time_col=self.time_col,
        )
        
        # Create forecast targets
        self.features_and_targets = create_forecast_targets(
            self.lagged_data, self.target, self.forecast_horizons, time_col= self.time_col,
        )
       
        # Determine input dimension
        sample_df = next(iter(self.features_and_targets.values()))
        feature_cols = [col for col in sample_df.columns 
                       if col not in ["TIME"] + [f"{self.target}_h{h}" for h in self.forecast_horizons]]
        
        self.input_dim = len(feature_cols)
        logger.info(f"Input dimension: {self.input_dim} features")

       

        return
    
    def _pretransform(self) -> None:
         # Second pass: calculate global normalization parameters across all countries
        if hasattr(self, 'transformations'):
            # Get columns to normalize (exclude TIME and target columns)
            target_cols = [f"{self.target}_h{h}_q{q}" for h in self.forecast_horizons for q in self.quantiles]
            exclude_cols = ["TIME"] + target_cols
            all_cols = self.features_and_targets[next(iter(self.features_and_targets))].columns.tolist()
            cols_to_normalize = [col for col in all_cols if col not in exclude_cols]


            
            
            # Calculate global normalization parameters for each column
            global_transformations = {}
            for col in cols_to_normalize:
                # Collect all values for this column across all countries
                all_values = []
                for country, df in self.features_and_targets.items():
                    if col in df.columns:
                        # Remove NaN values before concatenating
                        col_values = df[col].dropna()
                        if len(col_values) > 0:
                            all_values.extend(col_values.tolist())
                
                if len(all_values) > 0:
                    # Calculate global mean and std
                    all_values_array = np.array(all_values)
                    col_mean = np.mean(all_values_array)
                    col_std = np.std(all_values_array)
                    
                    # Handle case where std is 0 (constant column)
                    if col_std == 0:
                        col_std = 1.0
                        logger.warning(f"Column {col} has zero variance across all countries. Using std=1 for normalization.")
                    
                    # Store global transformation as lambda functions
                    global_transformations[col] = {
                        'normalize': lambda x, mean=col_mean, std=col_std: (x - mean) / std,
                        'denormalize': lambda x, mean=col_mean, std=col_std: (x * std) + mean,
                        'mean': col_mean,
                        'std': col_std
                    }
                    
                    if self.verbose >= 2:
                        logger.info(f"Global normalization for {col}: mean={col_mean:.4f}, std={col_std:.4f}")
            
            # Store the same global transformations for all countries
            for country in self.features_and_targets.keys():
                self.transformations[country] = global_transformations.copy()
            

            for country, df in self.features_and_targets.items():
    
                for col in cols_to_normalize:
                    if col in df.columns and col in global_transformations:
                        self.features_and_targets[country][col] = global_transformations[col]['normalize'](df[col])
                
                if self.verbose >= 2:
                    normalized_cols = [col for col in cols_to_normalize if col in df.columns]
                    logger.info(f"Country {country}: Applied global normalization to {len(normalized_cols)} columns")
            
            logger.info(f"Global normalization applied to {len(cols_to_normalize)} columns across all countries")

    def _prefit(self) -> None:
         # First pass: split data for all countries
        for country, df in self.features_and_targets.items():
            self._fit_AR_models(df, country)  # Fixed method name
            df = self._remove_AR_part(df, country)  # Fixed method name

    def _fit_AR_models(self, df: pd.DataFrame, country) -> Dict[str, Dict[float, Dict[str, sm.regression.quantile_regression.QuantRegResults]]]:
        
         # get ar data before transformation

        horizon_targets = [f"{self.target}_h{h}" for h in self.forecast_horizons] 
        q_models = {}
        for q in self.quantiles:
            hor_models = {}
            for col in horizon_targets:
                X = df[self.target]
                y = df[col]
                X = sm.add_constant(X)  # Add constant term for intercept
                # linear quantile regression from X on y 
                model = sm.QuantReg(y, X)
                fitted_model = model.fit(q=q)
                hor_models[col] = fitted_model

            q_models[q] = hor_models
        self.ar_models[country] = q_models
        return self.ar_models

    def _remove_AR_part(self, df, country):  # Fixed return type annotation
        horizon_targets = [f"{self.target}_h{h}" for h in self.forecast_horizons]

        for q in self.quantiles:
            for col in horizon_targets:
                X = sm.add_constant(df[self.target])
                df[f'{col}_q{q}'] = df[col] - self.ar_models[country][q][col].predict(X)
        # remove horizon targets from df
        for col in horizon_targets:
            if f"{col}_q{q}" in df.columns:
                df.drop(columns=[f"{col}"], inplace=True)
        return df

    def _add_AR_part(self, predictions, country, pred_target_data):  # Fixed return type annotation
        """Add AR part to predictions."""
        for qdx, q in enumerate(self.quantiles):
            for hdx, h in enumerate(self.forecast_horizons):
                model_name = f"{self.target}_h{h}"
                
                X = sm.add_constant(pred_target_data[self.target])

                predictions[:, qdx, hdx] += self.ar_models[country][q][model_name].predict(X)

        return predictions
