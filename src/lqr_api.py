"""High-level Linear Quantile Regression API."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Literal
import logging

from .lqr.model import MultiQuantileRegressor, cross_validate_alpha
from .lqr.cv import country_validation_split, select_best_alpha
from .utils import (
    create_lagged_features,
    create_forecast_targets,
    set_seeds
)

logger = logging.getLogger(__name__)


class LQR:
    """High-level Linear Quantile Regression interface."""
    
    def __init__(
        self,
        data_list: List[pd.DataFrame],
        target: str,
        quantiles: List[float],
        forecast_horizons: List[int],
        lags: List[int] = [1],
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: str = "huberized",
        seed: Optional[int] = 42,
    ) -> None:
        """Initialize Linear Quantile Regression.
        
        Args:
            data_list: List of country DataFrames
            target: Target column name
            quantiles: List of quantile levels
            forecast_horizons: List of forecast horizons
            lags: List of lag periods
            alpha: L2 regularization strength
            fit_intercept: Whether to fit intercept
            solver: Solver type ("huberized")
            seed: Random seed
        """
        # Set random seed
        if seed is not None:
            set_seeds(seed)
        
        self.target = target
        self.quantiles = quantiles
        self.forecast_horizons = forecast_horizons
        self.lags = lags
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.seed = seed
        
        # Convert data_list to country dictionary
        self.country_data = {}
        for i, df in enumerate(data_list):
            country_code = f"COUNTRY_{i:03d}"  # Default naming
            if 'country' in df.columns:
                country_code = df['country'].iloc[0]
            self.country_data[country_code] = df.copy()
        
        # Create lagged features and targets
        self._prepare_data()
        
        # Initialize model
        self.model = None
        self.is_fitted = False
        self.best_alpha = alpha
        
        logger.info(f"LQR initialized with {len(self.country_data)} countries")
    
    def _prepare_data(self) -> None:
        """Prepare data with lags and forecast targets."""
        # Create lagged features
        self.lagged_data = create_lagged_features(
            self.country_data, self.lags, time_col="TIME"
        )
        
        # Create forecast targets
        self.target_data = create_forecast_targets(
            self.lagged_data, self.target, self.forecast_horizons, time_col="TIME"
        )
        
        # Build feature matrix and targets
        self._build_matrices()
    
    def _build_matrices(self) -> None:
        """Build feature matrix and target matrix from country data."""
        all_features = []
        all_targets = []
        all_countries = []
        
        for country_code, df in self.target_data.items():
            # Feature columns (exclude TIME and target horizons)
            feature_cols = [col for col in df.columns 
                           if col not in ["TIME"] ]
            
            # # Exclude contemporaneous target if present
            # if self.target in feature_cols:
            #     feature_cols.remove(self.target)

            
            # Target columns
            target_cols = [f"{self.target}_h{h}" for h in self.forecast_horizons]
            
            # remove target columns from features
            feature_cols = [col for col in feature_cols if col not in target_cols]
    
            # Extract data
            features = df[feature_cols].values
            targets = df[target_cols].values
            countries = [country_code] * len(df)
            
            all_features.append(features)
            all_targets.append(targets)
            all_countries.extend(countries)
        
        # Combine all countries
        self.X = np.vstack(all_features)

        self.y = np.vstack(all_targets)

        self.countries = np.array(all_countries)
        
        logger.info(f"Data matrices built: {self.X.shape[0]} samples, {self.X.shape[1]} features")
    
    def k_fold_validation(
        self,
        alphas: List[float],
        n_splits: int = 5
    ) -> pd.DataFrame:
        """Perform k-fold cross-validation to select best alpha."""
        if self.seed is not None:
            set_seeds(self.seed)
        
        
        # use. cross_validate_alpha
        # cross_validate_alpha(
        #     X: np.ndarray,
        #     y: np.ndarray,
        #     quantiles: List[float],
        #     horizons: List[int],
        #     alphas: List[float],
        #     cv_folds: int = 5,
        #     scoring: str 
        alpha_scores = cross_validate_alpha(
            X=self.X,
            y=self.y,
            quantiles=self.quantiles,
            horizons=self.forecast_horizons,
            alphas=alphas,  # Use alphas from config
            cv_folds=n_splits,
            seed=self.seed
        )

        # Select best alpha based on validation results
        self.best_alpha = select_best_alpha(alpha_scores)
        
        logger.info(f"Best alpha selected: {self.best_alpha}")
        
        return self.best_alpha

    def validate_alpha(self, alphas: List[float], validation_size: float = 0.2) -> float:
        """Validate alpha parameter using train/validation split.
        
        Args:
            alphas: List of alpha values to try
            validation_size: Fraction of data to use for validation
            
        Returns:
            Best alpha value
        """
        if self.seed is not None:
            set_seeds(self.seed)
        
        logger.info(f"Validating {len(alphas)} alpha values with validation_size={validation_size}")
        
        # Perform country-level train/validation split  
        



        alpha_scores = country_validation_split(
            X=self.X,
            y=self.y,
            countries=self.countries,
            quantiles=self.quantiles,
            horizons=self.forecast_horizons,
            alphas=alphas,
            validation_size=validation_size,
            random_state=self.seed
        )
        
        # Select best alpha
        self.best_alpha = select_best_alpha(alpha_scores)
        
        # Store validation results
        self.validation_results = alpha_scores
        
        return self.best_alpha
    
    def fit(self) -> pd.DataFrame:
        """Fit the Linear Quantile Regression model.
        
        Returns:
            DataFrame with fitted coefficients
        """
        if self.seed is not None:
            set_seeds(self.seed)
        
        # Initialize model with best alpha
        self.model = MultiQuantileRegressor(
            quantiles=self.quantiles,
            horizons=self.forecast_horizons,
            alpha=self.best_alpha,
            fit_intercept=self.fit_intercept,
            solver=self.solver
        )
        
        # Fit model
        logger.info(f"Fitting LQR with alpha={self.best_alpha}")
        self.model.fit(self.X, self.y)
        
        # Get coefficients
        coef_df = self.model.get_coefficients()
        
        self.is_fitted = True
        
        logger.info("LQR model fitted successfully")
        
        return coef_df
    
    def predict(self, data_list: List[pd.DataFrame]) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            data_list: List of country DataFrames
            
        Returns:
            Predictions array of shape (N, Q, H)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to country dictionary
        pred_country_data = {}
        for i, df in enumerate(data_list):
            country_code = f"COUNTRY_{i:03d}"
            if 'country' in df.columns:
                country_code = df['country'].iloc[0]
            pred_country_data[country_code] = df.copy()
        
        # Prepare data
        pred_lagged_data = create_lagged_features(
            pred_country_data, self.lags, time_col="TIME"
        )
        pred_target_data = create_forecast_targets(
            pred_lagged_data, self.target, self.forecast_horizons, time_col="TIME"
        )
        
        # Build feature matrix
        all_features = []
        all_targets = []
        for country_code, df in pred_target_data.items():
            feature_cols = [col for col in df.columns 
                           if col not in ["TIME"] ]
            target_cols = [f"{self.target}_h{h}" for h in self.forecast_horizons]
            
            # remove target columns from features
            feature_cols = [col for col in feature_cols if col not in target_cols]
            
            features = df[feature_cols].values
            targets = df[target_cols].values
            all_features.append(features)
            all_targets.append(targets)
        
        X_pred = np.vstack(all_features)

        y_pred = np.vstack(all_targets) if all_targets else None

        # Make predictions
        predictions = self.model.predict(X_pred)

        return predictions, y_pred
    
    def get_model_summary(self) -> Dict:
        """Get summary of fitted model.
        
        Returns:
            Dictionary with model summary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get summary")
        
        summary = {
            'n_samples': self.X.shape[0],
            'n_features': self.X.shape[1],
            'n_countries': len(np.unique(self.countries)),
            'quantiles': self.quantiles,
            'horizons': self.forecast_horizons,
            'alpha': self.best_alpha,
            'fit_intercept': self.fit_intercept,
            'solver': self.solver
        }
        
        if hasattr(self, 'cv_results'):
            summary['cv_results'] = self.cv_results
        
        return summary


# Alias for compatibility
LQRModel = LQR
