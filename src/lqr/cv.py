"""Cross-validation utilities for LQR."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import logging

from .model import MultiQuantileRegressor
from ..utils.splits import create_country_folds

logger = logging.getLogger(__name__)


def pinball_score(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Compute pinball loss (negative for scikit-learn compatibility).
    
    Args:
        y_true: True values
        y_pred: Predicted values  
        quantile: Quantile level
        
    Returns:
        Negative pinball loss (higher is better)
    """
    error = y_true - y_pred
    loss = np.where(error >= 0, quantile * error, (quantile - 1) * error)
    return -np.mean(loss)


class CountryGroupKFold:
    """K-Fold cross-validation that keeps countries together."""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation splits keeping countries together.
        
        Args:
            X: Feature matrix
            y: Target matrix
            groups: Country group labels for each sample
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        unique_groups = np.unique(groups)
        
        # Create country folds
        country_folds = create_country_folds(
            unique_groups.tolist(), self.n_splits, seed=self.random_state
        )
        
        splits = []
        for train_countries, val_countries in country_folds:
            train_mask = np.isin(groups, train_countries)
            val_mask = np.isin(groups, val_countries)
            
            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]
            
            splits.append((train_indices, val_indices))
        
        return splits
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def country_cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    countries: np.ndarray,
    quantiles: List[float],
    horizons: List[int],
    alphas: List[float],
    cv_folds: int = 5,
    random_state: Optional[int] = None
) -> Dict[float, float]:
    """Cross-validate alpha parameter using country-level folds.
    
    Args:
        X: Feature matrix (N, p)
        y: Target matrix (N, H)
        countries: Country labels for each sample (N,)
        quantiles: List of quantile levels
        horizons: List of horizons
        alphas: List of regularization parameters to try
        cv_folds: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        Dictionary mapping alpha to mean CV score
    """
    cv_splitter = CountryGroupKFold(n_splits=cv_folds, random_state=random_state)
    alpha_scores = {}
    
    for alpha in alphas:
        fold_scores = []
        
        # Cross-validation splits
        for train_idx, val_idx in cv_splitter.split(X, y, countries):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit model
            regressor = MultiQuantileRegressor(
                quantiles=quantiles,
                horizons=horizons,
                alpha=alpha
            )
            regressor.fit(X_train, y_train)
            
            # Predict and score
            y_pred = regressor.predict(X_val)
            
            # Compute scores for all (quantile, horizon) combinations
            scores = []
            for q_idx, q in enumerate(quantiles):
                for h_idx, h in enumerate(horizons):
                    y_true_qh = y_val[:, h_idx] if y_val.ndim > 1 else y_val
                    y_pred_qh = y_pred[:, q_idx, h_idx]
                    score = pinball_score(y_true_qh, y_pred_qh, q)
                    scores.append(score)
            
            fold_scores.append(np.mean(scores))
        
        alpha_scores[alpha] = np.mean(fold_scores)
        
        logger.debug(f"Alpha {alpha}: CV score = {alpha_scores[alpha]:.6f}")
    
    return alpha_scores


def select_best_alpha(
    alpha_scores: Dict[float, float],
    method: str = "max"
) -> float:
    """Select best alpha from cross-validation results.
    
    Args:
        alpha_scores: Dictionary mapping alpha to CV score
        method: Selection method ("max" for highest score)
        
    Returns:
        Best alpha value
    """
    if method == "max":
        best_alpha = max(alpha_scores.keys(), key=lambda a: alpha_scores[a])
    else:
        raise ValueError(f"Unknown selection method: {method}")
    
    #logger.info(f"Selected alpha = {best_alpha} with score = {alpha_scores[best_alpha]:.6f}")
    
    return best_alpha


def create_cv_summary(
    alpha_scores: Dict[float, float],
    quantiles: List[float],
    horizons: List[int]
) -> pd.DataFrame:
    """Create summary DataFrame of cross-validation results.
    
    Args:
        alpha_scores: CV scores by alpha
        quantiles: List of quantile levels
        horizons: List of horizons
        
    Returns:
        Summary DataFrame
    """
    records = []
    
    for alpha, score in alpha_scores.items():
        records.append({
            'alpha': alpha,
            'cv_score': score,
            'n_quantiles': len(quantiles),
            'n_horizons': len(horizons),
            'n_models': len(quantiles) * len(horizons)
        })
    
    df = pd.DataFrame(records).sort_values('cv_score', ascending=False)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def country_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    countries: np.ndarray,
    quantiles: List[float],
    horizons: List[int],
    alphas: List[float],
    validation_size: float = 0.2,
    random_state: Optional[int] = None
) -> Dict[float, float]:
    """Validate alpha parameter using country-level train/validation split.
    
    Args:
        X: Feature matrix (N, p)
        y: Target matrix (N, H)
        countries: Country labels for each sample (N,)
        quantiles: List of quantile levels
        horizons: List of horizons
        alphas: List of regularization parameters to try
        validation_size: Fraction of each country's data to use for validation
        random_state: Random seed
        
    Returns:
        Dictionary mapping alpha to validation score
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    alpha_scores = {}
    unique_countries = np.unique(countries)
    
    # Create train/validation splits for each country
    train_indices = []
    val_indices = []
    
    for country in unique_countries:
        country_mask = (countries == country)
        country_indices = np.where(country_mask)[0]
        
        # Calculate split point
        n_samples = len(country_indices)
        val_samples = int(n_samples * validation_size)
        train_samples = n_samples - val_samples
        
        if train_samples < 1 or val_samples < 1:
            logger.warning(f"Country {country} has insufficient data for split. Skipping.")
            continue
        
        # Use temporal split: last validation_size fraction for validation
        train_indices.extend(country_indices[:train_samples])
        val_indices.extend(country_indices[train_samples:])
    
    if not train_indices or not val_indices:
        raise ValueError("No valid train/validation split could be created")
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    
    # Evaluate each alpha
    for alpha in alphas:
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # Fit model
        regressor = MultiQuantileRegressor(
            quantiles=quantiles,
            horizons=horizons,
            alpha=alpha
        )
        regressor.fit(X_train, y_train)
        
        # Predict and score
        y_pred = regressor.predict(X_val)
        
        # Calculate score (average pinball loss across all quantiles and horizons)
        total_score = 0.0
        total_outputs = 0


        
        for h_idx, horizon in enumerate(horizons):
            for q_idx, quantile in enumerate(quantiles):
                output_idx = h_idx * len(quantiles) + q_idx
                score = pinball_score(y_val[:, h_idx], y_pred[:, output_idx], quantile)
                total_score += score
                total_outputs += 1
        
        alpha_scores[alpha] = total_score / total_outputs if total_outputs > 0 else -np.inf
        
        logger.debug(f"Alpha {alpha}: validation score = {alpha_scores[alpha]:.4f}")
    
    return alpha_scores
