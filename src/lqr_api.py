"""High-level Linear Quantile Regression API."""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Literal
import logging
import pickle

from .lqr.model import MultiQuantileRegressor, cross_validate_alpha
from .lqr.cv import country_validation_split, select_best_alpha
from .utils import (
    create_lagged_features,
    create_forecast_targets,
    set_seeds
)

# Optional R interface imports
# To install: pip install rpy2
# Also requires R with hqreg package: install.packages("hqreg") in R
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False

logger = logging.getLogger(__name__)

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
        solver: Literal["pinball", "hqreg"] = "pinball",
        seed: Optional[int] = 42,
    ) -> None:
        """Initialize Linear Quantile Regression.
        
        Args:
            data_list: List of country DataFrames
            target: Target column name
            quantiles: List of quantile levels
            forecast_horizons: List of forecast horizons
            lags: List of lag periods
            alpha: L2 regularization strength (elastic-net mixing parameter for hqreg)
            fit_intercept: Whether to fit intercept
            solver: Solver type ("pinball" for Python, "hqreg" for R-based)
            seed: Random seed
            gamma: Huber loss tuning parameter (R only, default: IQR(y)/10)
            nlambda: Number of lambda values for regularization path (R only)
            lambda_min: Smallest lambda value as fraction of lambda.max (R only)
            preprocess: Preprocessing technique ("standardize" or "rescale", R only)
            screen: Screening rule ("ASR", "SR", or "none", R only)
            max_iter: Maximum iterations (R only)
            eps: Convergence threshold (R only)
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

        
        # Check if R is available when using hqreg solver
        if solver == "hqreg" and not R_AVAILABLE:
            raise ImportError(
                "R interface (rpy2) is required for hqreg solver. "
                "Please install rpy2: pip install rpy2\n"
                "Also install R hqreg package: install.packages('hqreg') in R\n"
                "See README_R_SETUP.md for detailed installation instructions."
            )
        
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
        
        #logger.info(f"LQR initialized with {len(self.country_data)} countries")
    
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
            target_cols = [f"{self.target}_h{h}" for h in self.forecast_horizons] + [self.target]
            
            # remove target columns from features
            feature_cols = [col for col in feature_cols if col not in target_cols]
    
            # Extract data
            feature_cols = [self.target] + feature_cols
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
        
        #logger.info(f"Data matrices built: {self.X.shape[0]} samples, {self.X.shape[1]} features")
    
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
        
        #logger.info(f"Best alpha selected: {self.best_alpha}")
        
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
        
        #logger.info(f"Validating {len(alphas)} alpha values with validation_size={validation_size}")
        
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
        
        if self.solver == "hqreg":
            # Use R-based hqreg solver
            return self._fit_hqreg_all()
        else:
            # Use Python-based solver
            # Initialize model with best alpha
            self.model = MultiQuantileRegressor(
                quantiles=self.quantiles,
                horizons=self.forecast_horizons,
                alpha=self.best_alpha,
                fit_intercept=self.fit_intercept,
                solver=self.solver
            )
            
            # Fit model
            #logger.info(f"Fitting LQR with alpha={self.best_alpha}")
            self.model.fit(self.X, self.y)
         
            # Get coefficients
            coef_df = self.model.get_coefficients()
            
            self.is_fitted = True
            
            #logger.info("LQR model fitted successfully")
            
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
        if self.solver == "hqreg":
            predictions = self._predict_hqreg(X_pred)
        else:
            predictions = self.model.predict(X_pred)

        return predictions, y_pred
    
    def _fit_hqreg(self, X: np.ndarray, y: np.ndarray, quantile: float, horizon: int) -> Dict:
        """Fit hqreg model for a single quantile and horizon using R.
        
        Args:
            X: Feature matrix
            y: Target vector for specific horizon
            quantile: Quantile level
            horizon: Forecast horizon
            
        Returns:
            Dictionary with coefficients and model info
        """
        if not R_AVAILABLE:
            raise ImportError("R interface (rpy2) is required for hqreg solver")
        
        # Import R packages
        try:
            hqreg = importr('hqreg')
            base = importr('base')
        except Exception as e:
            raise ImportError(f"Failed to import R hqreg package: {e}")
        
        # Convert data to R format - ensure proper conversion
        # Convert numpy arrays to basic Python types first, then to R
        X_list = X.astype(float).tolist()
        y_list = y.astype(float).tolist()
        
        # Create R objects
        r_X = ro.r.matrix(ro.FloatVector([item for sublist in X_list for item in sublist]), 
                          nrow=X.shape[0], ncol=X.shape[1])
        r_y = ro.FloatVector(y_list)
        
        # Set up parameters
        r_alpha = 0
        r_tau = quantile
        r_nlambda = 100
        r_lambda_min = 0.05
        r_preprocess = "standardize"
        r_screen = "ASR"
        r_max_iter = 1000
        r_eps = 1e-7
        penalty_factor = ro.FloatVector([1.0]*X.shape[1])
        penalty_factor[0] = 0.0 

        # Fit hqreg model
        fit = hqreg.hqreg(
            X=r_X,
            y=r_y,
            method="quantile",
            tau=r_tau,
            alpha=r_alpha,
            nlambda=r_nlambda,
            lambda_min=r_lambda_min,
            preprocess=r_preprocess,
            screen=r_screen,
            max_iter=r_max_iter,
            eps=r_eps,
            message=True, 
            penalty_factor= penalty_factor 
        )
        
        # Extract coefficients for the regularization level closest to our alpha
        # For simplicity, use the last lambda value (most regularized)
        r_lambda_seq = fit.rx2('lambda')
        lambda_values = np.array(r_lambda_seq)
        
        # Find lambda closest to our target (alpha is used as lambda here)
        target_lambda = self.alpha
        closest_idx = np.argmin(np.abs(lambda_values - target_lambda))
        
        # Extract coefficients
        r_beta = fit.rx2('beta')
        beta_matrix = np.array(r_beta)
        coefficients = beta_matrix[:, closest_idx]
        
        return {
            'coefficients': coefficients,
            'lambda_used': lambda_values[closest_idx],
            'quantile': quantile,
            'horizon': horizon,
            'r_fit': fit
        }
    
    def _fit_hqreg_all(self) -> pd.DataFrame:
        """Fit hqreg models for all quantiles and horizons.
        
        Returns:
            DataFrame with fitted coefficients
        """
        all_results = []
        
        for h_idx, h in enumerate(self.forecast_horizons):
            y_h = self.y[:, h_idx]  # Target for this horizon
            
            for q in self.quantiles:
                result = self._fit_hqreg(self.X, y_h, q, h)
                
                # Format coefficients into DataFrame rows
                coeffs = result['coefficients']
                for i, coeff in enumerate(coeffs):
                    all_results.append({
                        'quantile': q,
                        'horizon': h,
                        'feature': f'feature_{i}' if i < len(coeffs) - 1 else 'intercept',
                        'coefficient': coeff
                    })
        
        coef_df = pd.DataFrame(all_results)
        
        # Store results for compatibility
        self.hqreg_results = coef_df
        self.is_fitted = True
        
        return coef_df
    
    def _predict_hqreg(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using fitted hqreg models.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predictions array of shape (N, Q, H)
        """
        if not hasattr(self, 'hqreg_results'):
            raise ValueError("hqreg model not fitted")
        
        n_samples = X.shape[0]
        n_quantiles = len(self.quantiles)
        n_horizons = len(self.forecast_horizons)
        
        predictions = np.zeros((n_samples, n_quantiles, n_horizons))
        
        for h_idx, h in enumerate(self.forecast_horizons):
            for q_idx, q in enumerate(self.quantiles):
                # Get coefficients for this quantile and horizon
                mask = (self.hqreg_results['quantile'] == q) & (self.hqreg_results['horizon'] == h)
                coeffs = self.hqreg_results[mask]['coefficient'].values
                
                # Make prediction
                if self.fit_intercept:
                    pred = X @ coeffs[:-1] + coeffs[-1]  # Features * coeffs + intercept
                else:
                    pred = X @ coeffs
                
                predictions[:, q_idx, h_idx] = pred
        
        return predictions
    
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

    def load_model(self, load_dir):
        
        self.model = MultiQuantileRegressor(
            quantiles=self.quantiles,
            horizons=self.forecast_horizons,
            alpha=self.best_alpha,
            fit_intercept=self.fit_intercept,
            solver=self.solver
        )

        # unpickle dataframe
        df = pd.read_pickle(load_dir + "model.pkl")

        # Set coefficients in model
        self.model.set_coefficients(df)

        # read alpha
        with open(load_dir + "alpha.pkl", "rb") as f:
            self.model.alpha = pickle.load(f)

        self.is_fitted = True

    def store_model(self, store_dir):
        # mkdir if needed
        os.makedirs(store_dir, exist_ok=True)

        # pickle dataframe
        df = self.model.get_coefficients()
        df.to_pickle(store_dir + "model.pkl")

        # store alpha
        with open(store_dir + "alpha.pkl", "wb") as f:
            pickle.dump(self.model.alpha, f)


# Alias for compatibility
LQRModel = LQR
