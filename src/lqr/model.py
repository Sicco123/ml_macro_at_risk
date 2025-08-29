"""Linear Quantile Regression model implementation."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from typing import List, Dict, Optional, Tuple, Literal
import logging

logger = logging.getLogger(__name__)

SolverType = Literal["huberized", "linear_programming"]


def huberized_pinball_loss(params: np.ndarray, X: np.ndarray, y: np.ndarray, 
                          quantile: float, alpha: float) -> float:
    """Huberized pinball loss with L2 regularization.
    
    Args:
        params: Model parameters [intercept, coefficients...]
        X: Feature matrix (N, p)
        y: Target vector (N,)
        quantile: Quantile level
        alpha: L2 regularization strength
        
    Returns:
        Loss value
    """
    if len(params) == X.shape[1] + 1:
        # With intercept
        intercept = params[0]
        coef = params[1:]
        y_pred = X @ coef + intercept
    else:
        # Without intercept
        coef = params
        y_pred = X @ coef
        intercept = 0.0
    
    # Pinball loss
    error = y - y_pred
    pinball = np.where(error >= 0, quantile * error, (quantile - 1) * error)
    
    # L2 regularization (don't regularize intercept)
    reg_term = alpha * np.sum(coef[1:]**2)
    
    return np.mean(pinball) + reg_term


def huberized_pinball_gradient(params: np.ndarray, X: np.ndarray, y: np.ndarray, 
                              quantile: float, alpha: float) -> np.ndarray:
    """Gradient of huberized pinball loss.
    
    Args:
        params: Model parameters
        X: Feature matrix
        y: Target vector
        quantile: Quantile level
        alpha: L2 regularization strength
        
    Returns:
        Gradient vector
    """
    if len(params) == X.shape[1] + 1:
        intercept = params[0]
        coef = params[1:]
        y_pred = X @ coef + intercept
        fit_intercept = True
    else:
        coef = params
        y_pred = X @ coef
        fit_intercept = False
    
    error = y - y_pred
    
    # Gradient of pinball loss w.r.t. predictions
    dpinball_dpred = np.where(error >= 0, -quantile, -(quantile - 1))
    
    
    # Gradient w.r.t. coefficients
    grad_coef = (X.T @ dpinball_dpred) / len(y) + 2 * alpha * coef
    
    grad_coef[0] -= 2*alpha*coef[0]

    if fit_intercept:
        # Gradient w.r.t. intercept (no regularization)
        grad_intercept = np.mean(dpinball_dpred)
        return np.concatenate([[grad_intercept],  grad_coef])
    else:
        return grad_coef


class QuantileRegressor(BaseEstimator, RegressorMixin):
    """Single quantile regressor with L2 regularization."""
    
    def __init__(
        self,
        quantile: float,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: SolverType = "huberized",
        max_iter: int = 1000,
        tol: float = 1e-6
    ):
        """Initialize quantile regressor.
        
        Args:
            quantile: Quantile level in (0, 1)
            alpha: L2 regularization strength
            fit_intercept: Whether to fit intercept
            solver: Optimization solver
            max_iter: Maximum iterations
            tol: Convergence tolerance
        """
        self.quantile = quantile
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        
        # Fitted parameters
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileRegressor':
        """Fit the quantile regressor.
        
        Args:
            X: Feature matrix (N, p)
            y: Target vector (N,)
            
        Returns:
            Fitted regressor
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.n_features_in_ = X.shape[1]
        
        if self.solver == "huberized":
            self._fit_huberized(X, y)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        return self
    
    def _fit_huberized(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using huberized pinball loss."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)

        p = X.shape[1]
        if self.fit_intercept:
            init = np.zeros(p + 1, dtype=np.float64)
            init[0] = np.median(y)
        else:
            init = np.zeros(p, dtype=np.float64)

        result = minimize(
            fun=huberized_pinball_loss,      # keep your existing function
            x0=init,
            args=(X, y, self.quantile, self.alpha),
            jac=huberized_pinball_gradient,  # keep your existing gradient
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol}
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        # Extract parameters
        if self.fit_intercept:
            self.intercept_ = result.x[0]
            self.coef_ = result.x[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = result.x
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute negative pinball loss (higher is better).
        
        Args:
            X: Feature matrix
            y: True targets
            
        Returns:
            Negative pinball loss
        """
        y_pred = self.predict(X)
        error = y - y_pred
        pinball = np.where(error >= 0, self.quantile * error, (self.quantile - 1) * error)
        return -np.mean(pinball)


class MultiQuantileRegressor:
    """Multi-quantile regressor for multiple horizons."""
    
    def __init__(
        self,
        quantiles: List[float],
        horizons: List[int],
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: SolverType = "huberized"
    ):
        """Initialize multi-quantile regressor.
        
        Args:
            quantiles: List of quantile levels
            horizons: List of forecast horizons
            alpha: L2 regularization strength
            fit_intercept: Whether to fit intercept
            solver: Optimization solver
        """
        self.quantiles = quantiles
        self.horizons = horizons
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        
        # Create regressors for each (quantile, horizon) pair
        self.regressors = {}
        for q in quantiles:
            for h in horizons:
                key = f"q{q:.3f}_h{h}"
                self.regressors[key] = QuantileRegressor(
                    quantile=q,
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    solver=solver
                )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiQuantileRegressor':
        """Fit all quantile regressors.
        
        Args:
            X: Feature matrix (N, p)
            y: Target matrix (N, H) where H is number of horizons
            
        Returns:
            Fitted regressor
        """
        for h_idx, h in enumerate(self.horizons):
            y_h = y[:, h_idx] if y.ndim > 1 else y
            
            for q in self.quantiles:
                key = f"q{q:.3f}_h{h}"
                self.regressors[key].fit(X, y_h)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for all quantiles and horizons.
        
        Args:
            X: Feature matrix (N, p)
            
        Returns:
            Predictions array (N, Q, H)
        """
        N = X.shape[0]
        Q = len(self.quantiles)
        H = len(self.horizons)
        
        predictions = np.zeros((N, Q, H))
        
        for q_idx, q in enumerate(self.quantiles):
            for h_idx, h in enumerate(self.horizons):
                key = f"q{q:.3f}_h{h}"
                predictions[:, q_idx, h_idx] = self.regressors[key].predict(X)
        
        return predictions
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get coefficients in tidy format.
        
        Returns:
            DataFrame with coefficients
        """
        records = []
        
        for q in self.quantiles:
            for h in self.horizons:
                key = f"q{q:.3f}_h{h}"
                regressor = self.regressors[key]
                
                # Add intercept if fitted
                if self.fit_intercept:
                    records.append({
                        'quantile': q,
                        'horizon': h,
                        'feature': 'intercept',
                        'coef': regressor.intercept_
                    })
                
                # Add feature coefficients
                for i, coef in enumerate(regressor.coef_):
                    records.append({
                        'quantile': q,
                        'horizon': h,
                        'feature': f'feature_{i}',
                        'coef': coef
                    })
        
        return pd.DataFrame(records)


def cross_validate_alpha(
    X: np.ndarray,
    y: np.ndarray,
    quantiles: List[float],
    horizons: List[int],
    alphas: List[float],
    cv_folds: int = 5,
    seed: int = 42
) -> Dict[float, float]:
    """Cross-validate regularization parameter.
    
    Args:
        X: Feature matrix
        y: Target matrix
        quantiles: List of quantile levels
        horizons: List of horizons
        alphas: List of alpha values to try
        cv_folds: Number of CV folds
        seed: Random seed for reproducible splits
        
    Returns:
        Dictionary mapping alpha to CV score
    """
    alpha_scores = {}
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # shuffle X and y and set seed
    np.random.seed(seed)
    mask = np.random.permutation(X.shape[0])
    X = X[mask]
    y = y[mask]

    for alpha in alphas:
        all_scores = []
        
        for q in quantiles:
            for h_idx, h in enumerate(horizons):
                y_h = y[:, h_idx] if y.ndim > 1 else y
                
                fold_scores = []
                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y_h[train_idx], y_h[val_idx]
                    
              
                    # Fit regressor on training fold
                    regressor = QuantileRegressor(quantile=q, alpha=alpha)
                    regressor.fit(X_train, y_train)
                    
                    # Predict on validation fold
                    y_pred = regressor.predict(X_val)
                    
                    # Calculate pinball loss (negative since we want to minimize)
                    error = y_val - y_pred
                    pinball_loss = np.where(error >= 0, q * error, (q - 1) * error)
                    fold_scores.append(-np.mean(pinball_loss))
                
                all_scores.extend(fold_scores)
        
        alpha_scores[alpha] = np.mean(all_scores)
    
    return alpha_scores
