"""Linear Quantile Regression model implementation."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import spdiags
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from typing import List, Dict, Optional, Tuple, Literal
import logging

logger = logging.getLogger(__name__)

SolverType = Literal["linear_programming", "pinball"]


def pinball_loss(params: np.ndarray, X: np.ndarray, y: np.ndarray, 
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
    
    # L2 regularization (don't regularize intercept and first ar term)
    reg_term = alpha * np.sum(coef[1:]**2)
    
    return np.mean(pinball) + reg_term


def pinball_gradient(params: np.ndarray, X: np.ndarray, y: np.ndarray, 
                              quantile: float, alpha: float) -> np.ndarray:
    """Gradient of pinball loss.
    
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

def bound(x: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Allowed step lengths for interior-point method (componentwise)."""
    b = np.full_like(x, 1e20, dtype=float)
    neg = dx < 0
    b[neg] = -x[neg] / dx[neg]
    return b


def lp_fnm(A: np.ndarray, c: np.ndarray, b: np.ndarray, u: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Solve LP by interior-point method (Morillo & Koenker 'lp_fnm').
    min c^T x   s.t.  A x = b,   0 < x < u
    Returns the dual vector y associated with A x = b.
    """
    # constants
    beta  = 0.9995
    small = 1e-5
    max_it = 50

    # shapes
    m, n = A.shape  # A: (m rows) x (n cols); x, c, u, z, w, s are length n; y length m

    # ensure float and 1-D shapes
    A = np.asarray(A, float)
    c = np.asarray(c, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    u = np.asarray(u, float).reshape(-1)
    x = np.asarray(x, float).reshape(-1)

    # initial feasible point
    s = u - x                                      # slack > 0
    # MATLAB: y = (A' \ c')'  == least squares solve A^T y ≈ c
    # Solve min ||A^T y - c|| -> (using lstsq on A^T)
    y = np.linalg.lstsq(A.T, c, rcond=None)[0]     # y in R^m
    r = c - A.T @ y                                # residual in R^n
    r = r + 0.001 * (r == 0)                       # avoid exact zeros (PE 2004)
    z = np.maximum(r, 0.0)                         # z >= 0
    w = z - r                                      # w >= 0, and r = z - w
    gap = float(c @ x - y @ b + w @ u)             # primal-dual gap

    it = 0
    while (gap > small) and (it < max_it):
        it += 1

        # --- Affine step ---
        q = 1.0 / (z / x + w / s)                  # n-vector
        rzw = z - w                                # equals `r` in the MATLAB loop

        # Q = diag(sqrt(q)); AQ = A @ Q; rhs = Q @ rzw
        Q_sqrt = np.sqrt(q)
        AQ  = A * Q_sqrt                           # column-scale A by sqrt(q)
        rhs = Q_sqrt * rzw

        # MATLAB: dy = (AQ' \ rhs)'  -> solve AQ.T * dy = rhs (least squares)
        dy = np.linalg.lstsq(AQ.T, rhs, rcond=None)[0]   # m-vector

        dx = q * (A.T @ dy - rzw)                  # n-vector
        ds = -dx
        dz = -z * (1.0 + dx / x)
        dw = -w * (1.0 + ds / s)

        # step lengths
        fx = bound(x, dx)
        fs = bound(s, ds)
        fw = bound(w, dw)
        fz = bound(z, dz)
        fp = min(float(fx.min()), float(fs.min()))
        fd = min(float(fw.min()), float(fz.min()))
        fp = min(beta * fp, 1.0)
        fd = min(beta * fd, 1.0)

        # --- Centering / correction if not full step ---
        if min(fp, fd) < 1.0:
            mu = float(z @ x + w @ s)
            g  = float((z + fd*dz) @ (x + fp*dx) + (w + fd*dw) @ (s + fp*ds))
            mu = mu * (g / mu)**3 / (2.0 * n)

            dxdz = dx * dz
            dsdw = ds * dw
            xinv = 1.0 / x
            sinv = 1.0 / s
            xi   = mu * (xinv - sinv)             # n-vector

            # rhs <- rhs + Q * (dxdz - dsdw - xi)
            rhs2 = rhs + Q_sqrt * (dxdz - dsdw - xi)

            # dy from AQ.T dy = rhs2
            dy = np.linalg.lstsq(AQ.T, rhs2, rcond=None)[0]

            dx = q * (A.T @ dy + xi - rzw - dxdz + dsdw)
            ds = -dx
            dz = mu * xinv - z - xinv * z * dx - dxdz
            dw = mu * sinv - w - sinv * w * ds - dsdw

            # recompute step lengths
            fx = bound(x, dx)
            fs = bound(s, ds)
            fw = bound(w, dw)
            fz = bound(z, dz)
            fp = min(float(fx.min()), float(fs.min()))
            fd = min(float(fw.min()), float(fz.min()))
            fp = min(beta * fp, 1.0)
            fd = min(beta * fd, 1.0)

        # take the step
        x = x + fp * dx
        s = s + fp * ds
        y = y + fd * dy
        w = w + fd * dw
        z = z + fd * dz

        # update gap
        gap = float(c @ x - y @ b + w @ u)

    return y


def rq_fnm(X: np.ndarray, y: np.ndarray, p: float) -> np.ndarray:
    """
    Quantile regression via dual LP (Koenker). Returns beta-hat.
    X: (m x n), y: (m,), p in (0,1)
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    m, n = X.shape

    u = np.ones(m, dtype=float)              # bounds for x (length m = #obs)
    a = (1.0 - p) * u                        # feasible initial x, 0 < a < u

    # Solve dual LP to get beta: b = -lp_fnm(X', -y', X' a, u, a)'
    beta = -lp_fnm(X.T, -y, X.T @ a, u, a)
    return beta

class QuantileRegressor(BaseEstimator, RegressorMixin):
    """Single quantile regressor with L2 regularization."""
    
    def __init__(
        self,
        quantile: float,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: SolverType = "pinball",
        max_iter: int = 10000,
        tol: float = 1e-8
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
        self.prefit_ar = True

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
        
        if self.solver == "pinball":
            self._fit(X, y)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        return self
    
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using  pinball loss."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)

        # If no regularization, use linear programming approach
        if self.alpha == 0.0:
            if self.fit_intercept:
                # Add intercept column
                X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
                coef_with_intercept = rq_fnm(X_with_intercept, y, self.quantile)
                self.intercept_ = coef_with_intercept[0]
                self.coef_ = coef_with_intercept[1:]
            else:
                self.intercept_ = 0.0
                self.coef_ = rq_fnm(X, y, self.quantile)
            return

        p = X.shape[1]
        if self.fit_intercept:
            init = np.zeros(p + 1, dtype=np.float64)
            init[0] = np.median(y)
        else:
            init = np.zeros(p, dtype=np.float64)

        if self.prefit_ar:
            if self.fit_intercept:
                beta = rq_fnm(X[:,:2], y, self.quantile)          
                init[:2] = beta
            else:
                beta = rq_fnm(X[:,:1], y, self.quantile)
                init[0] = beta[0]
   
        result = minimize(
            fun=pinball_loss,      # keep your existing function
            x0=init,
            args=(X, y, self.quantile, self.alpha),
            jac=pinball_gradient,  # keep your existing gradient
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
        solver: SolverType = "pinball"
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
        
    def set_coefficients(self, df):
        for q in self.quantiles:
            for h in self.horizons:
                key = f"q{q:.3f}_h{h}"
                if key in self.regressors:
                    self.regressors[key].intercept_ = df.loc[(df['quantile'] == q) & (df['horizon'] == h), 'coef'].values[0]
                    self.regressors[key].coef_ = df.loc[(df['quantile'] == q) & (df['horizon'] == h), 'coef'].values[1:]
        return 


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


