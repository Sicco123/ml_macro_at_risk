"""Diebold-Mariano test for comparing forecast accuracy."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acovf
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def newey_west_variance(
    series: np.ndarray,
    lags: Optional[int] = None,
    use_bartlett: bool = True
) -> float:
    """Compute Newey-West HAC variance estimator.
    
    Args:
        series: Time series data
        lags: Number of lags (if None, uses default rule)
        use_bartlett: Whether to use Bartlett weights
        
    Returns:
        HAC variance estimate
    """
    n = len(series)
    
    if lags is None:
        # Default rule: lags = floor(4 * (T/100)^(2/9))
        lags = int(np.floor(4 * (n / 100) ** (2/9)))
    
    # Compute autocovariances
    gamma = acovf(series, nobs=n, fft=True)
    
    # HAC variance estimator
    variance = gamma[0]  # gamma_0
    
    for j in range(1, min(lags + 1, len(gamma))):
        weight = 1.0
        if use_bartlett:
            weight = 1 - j / (lags + 1)
        variance += 2 * weight * gamma[j]
    
    return variance


def diebold_mariano_test(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    h: int = 1,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """Perform Diebold-Mariano test for equal forecast accuracy.
    
    Args:
        losses_a: Loss series for model A
        losses_b: Loss series for model B
        h: Forecast horizon (for HAC adjustment)
        alternative: Type of test ("two-sided", "less", "greater")
        
    Returns:
        Tuple of (DM statistic, p-value)
    """
    # Loss differential
    d = losses_a - losses_b
    n = len(d)
    
    if n == 0:
        return np.nan, np.nan
    
    # Mean differential
    d_mean = np.mean(d)
    
    if np.abs(d_mean) < 1e-10:
        return 0.0, 1.0
    
    # HAC variance (using h-1 lags for h-step ahead forecasts)
    lags = max(h - 1, 0)
    d_var = newey_west_variance(d, lags=lags)
    
    if d_var <= 0:
        return np.nan, np.nan
    
    # DM statistic
    dm_stat = d_mean / np.sqrt(d_var / n)
    
    # P-value (using t-distribution with n-1 degrees of freedom)
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n-1))
    elif alternative == "greater":
        p_value = 1 - stats.t.cdf(dm_stat, df=n-1)
    elif alternative == "less":
        p_value = stats.t.cdf(dm_stat, df=n-1)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    
    return dm_stat, p_value


def dm_test_by_groups(
    loss_df_a: pd.DataFrame,
    loss_df_b: pd.DataFrame,
    model_a_name: str,
    model_b_name: str,
    group_cols: List[str] = ['country', 'quantile', 'horizon'],
    alternative: str = "two-sided"
) -> pd.DataFrame:
    """Perform DM tests by groups (country, quantile, horizon).
    
    Args:
        loss_df_a: Loss DataFrame for model A (with columns: country, quantile, horizon, time, loss)
        loss_df_b: Loss DataFrame for model B
        model_a_name: Name of model A
        model_b_name: Name of model B
        group_cols: Columns to group by
        alternative: Type of test
        
    Returns:
        DataFrame with DM test results
    """
    # Merge loss dataframes
    merged = loss_df_a.merge(
        loss_df_b,
        on=['country', 'quantile', 'horizon', 'time'],
        suffixes=('_a', '_b')
    )
    
    results = []
    
    # Group by specified columns
    for group_values, group_df in merged.groupby(group_cols):
        if len(group_df) < 2:
            continue
        
        # Get horizon for HAC adjustment
        horizon = group_df['horizon'].iloc[0] if 'horizon' in group_cols else 1
        
        # Perform DM test
        dm_stat, p_value = diebold_mariano_test(
            group_df['loss_a'].values,
            group_df['loss_b'].values,
            h=horizon,
            alternative=alternative
        )
        
        # Create result record
        result = dict(zip(group_cols, group_values))
        result.update({
            'model_a': model_a_name,
            'model_b': model_b_name,
            'n_obs': len(group_df),
            'mean_loss_a': group_df['loss_a'].mean(),
            'mean_loss_b': group_df['loss_b'].mean(),
            'mean_diff': group_df['loss_a'].mean() - group_df['loss_b'].mean(),
            'dm_stat': dm_stat,
            'p_value': p_value
        })
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Add significance indicators
    if len(results_df) > 0:
        results_df['significant_05'] = results_df['p_value'] < 0.05
        results_df['significant_01'] = results_df['p_value'] < 0.01
    
    return results_df


def aggregate_dm_tests(
    dm_results: pd.DataFrame,
    aggregation_method: str = "stouffer"
) -> pd.DataFrame:
    """Aggregate DM test results across countries or other dimensions.
    
    Args:
        dm_results: DataFrame with DM test results
        aggregation_method: Method for aggregation ("stouffer", "fisher")
        
    Returns:
        Aggregated test results
    """
    if aggregation_method == "stouffer":
        # Stouffer's method for combining z-scores
        group_cols = [col for col in ['quantile', 'horizon'] if col in dm_results.columns]
        
        if not group_cols:
            return dm_results
        
        aggregated = []
        
        for group_values, group_df in dm_results.groupby(group_cols):
            # Convert t-statistics to z-scores (approximate)
            z_scores = group_df['dm_stat'].values
            weights = np.sqrt(group_df['n_obs'].values)
            
            # Stouffer's statistic
            combined_z = np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2))
            combined_p = 2 * (1 - stats.norm.cdf(np.abs(combined_z)))
            
            result = dict(zip(group_cols, group_values))
            result.update({
                'n_tests': len(group_df),
                'total_obs': group_df['n_obs'].sum(),
                'mean_diff': np.average(group_df['mean_diff'], weights=group_df['n_obs']),
                'combined_z': combined_z,
                'combined_p': combined_p,
                'significant_05': combined_p < 0.05,
                'significant_01': combined_p < 0.01
            })
            
            aggregated.append(result)
        
        return pd.DataFrame(aggregated)
    
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")


def multiple_testing_correction(
    dm_results: pd.DataFrame,
    method: str = "bonferroni",
    alpha: float = 0.05
) -> pd.DataFrame:
    """Apply multiple testing correction to DM test results.
    
    Args:
        dm_results: DataFrame with DM test results
        method: Correction method ("bonferroni", "holm", "fdr_bh")
        alpha: Family-wise error rate
        
    Returns:
        DataFrame with corrected p-values
    """
    from statsmodels.stats.multitest import multipletests
    
    results = dm_results.copy()
    
    if len(results) == 0:
        return results
    
    # Apply correction
    reject, p_corrected, _, _ = multipletests(
        results['p_value'].values,
        alpha=alpha,
        method=method
    )
    
    results[f'p_value_{method}'] = p_corrected
    results[f'significant_{method}'] = reject
    
    return results
