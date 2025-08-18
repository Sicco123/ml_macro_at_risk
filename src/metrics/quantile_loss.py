"""Quantile (pinball) loss functions and aggregations."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> np.ndarray:
    """Compute pinball (quantile) loss.
    
    Args:
        y_true: True values, shape (N,)
        y_pred: Predicted values, shape (N,)
        quantile: Quantile level in (0, 1)
        
    Returns:
        Pinball losses, shape (N,)
    """
    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1) * error)
    return loss


def compute_quantile_losses(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: List[float],
    horizons: List[int],
    countries: List[str],
    indices: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """Compute quantile losses for all combinations.
    
    Args:
        y_true: True values, shape (N, H) where N is total samples, H is horizons
        y_pred: Predictions, shape (N, Q, H) where Q is quantiles
        quantiles: List of quantile levels
        horizons: List of forecast horizons
        countries: List of country codes
        indices: Mapping from country to sample indices
        
    Returns:
        DataFrame with columns: country, quantile, horizon, loss
    """
    results = []
    
    # Map sample indices back to countries
    sample_to_country = {}
    for country, country_indices in indices.items():
        for idx in country_indices:
            sample_to_country[idx] = country
    
    N, Q, H = y_pred.shape
    
    for q_idx, quantile in enumerate(quantiles):
        for h_idx, horizon in enumerate(horizons):
            # Extract predictions and targets for this (q, h)
            pred_qh = y_pred[:, q_idx, h_idx]
            true_qh = y_true[:, h_idx] if y_true.ndim > 1 else y_true
            
            # Compute losses
            losses = pinball_loss(true_qh, pred_qh, quantile)
            
            # Group by country
            for country in countries:
                if country in indices:
                    country_mask = np.array([sample_to_country.get(i) == country for i in range(N)])
                    if country_mask.sum() > 0:
                        country_loss = losses[country_mask].mean()
                        results.append({
                            'country': country,
                            'quantile': quantile,
                            'horizon': horizon,
                            'loss': country_loss,
                            'n_samples': country_mask.sum()
                        })
    
    return pd.DataFrame(results)


def aggregate_losses(
    loss_df: pd.DataFrame,
    groupby_cols: List[str] = ['quantile', 'horizon']
) -> pd.DataFrame:
    """Aggregate losses across specified dimensions.
    
    Args:
        loss_df: DataFrame with loss results
        groupby_cols: Columns to group by for aggregation
        
    Returns:
        Aggregated loss DataFrame
    """
    # Weighted average by number of samples
    agg_df = loss_df.groupby(groupby_cols).apply(
        lambda x: np.average(x['loss'], weights=x['n_samples'])
    ).reset_index(name='mean_loss')
    
    # Add sample count
    sample_counts = loss_df.groupby(groupby_cols)['n_samples'].sum().reset_index()
    agg_df = agg_df.merge(sample_counts, on=groupby_cols)
    
    return agg_df


def create_loss_summary_table(
    loss_df: pd.DataFrame,
    model_name: str = "model"
) -> Dict[str, pd.DataFrame]:
    """Create comprehensive loss summary tables.
    
    Args:
        loss_df: DataFrame with loss results
        model_name: Name of the model
        
    Returns:
        Dictionary of summary tables
    """
    summaries = {}
    
    # By country
    summaries['by_country'] = aggregate_losses(loss_df, ['country'])
    
    # By quantile
    summaries['by_quantile'] = aggregate_losses(loss_df, ['quantile'])
    
    # By horizon
    summaries['by_horizon'] = aggregate_losses(loss_df, ['horizon'])
    
    # By quantile and horizon
    summaries['by_quantile_horizon'] = aggregate_losses(loss_df, ['quantile', 'horizon'])
    
    # Overall
    overall_loss = np.average(loss_df['loss'], weights=loss_df['n_samples'])
    summaries['overall'] = pd.DataFrame({
        'model': [model_name],
        'mean_loss': [overall_loss],
        'n_samples': [loss_df['n_samples'].sum()]
    })
    
    # Add model name to all tables
    for key, df in summaries.items():
        if 'model' not in df.columns:
            df['model'] = model_name
    
    return summaries


def compare_model_losses(
    loss_dfs: Dict[str, pd.DataFrame],
    comparison_cols: List[str] = ['quantile', 'horizon']
) -> pd.DataFrame:
    """Compare losses between models.
    
    Args:
        loss_dfs: Dictionary mapping model names to loss DataFrames
        comparison_cols: Columns to use for comparison
        
    Returns:
        Comparison DataFrame with relative performance
    """
    # Aggregate losses for each model
    model_summaries = []
    for model_name, loss_df in loss_dfs.items():
        summary = aggregate_losses(loss_df, comparison_cols)
        summary['model'] = model_name
        model_summaries.append(summary)
    
    # Combine all summaries
    combined = pd.concat(model_summaries, ignore_index=True)
    
    # Pivot to get models as columns
    pivot_df = combined.pivot_table(
        index=comparison_cols,
        columns='model',
        values='mean_loss'
    ).reset_index()
    
    # Compute relative performance (as percentage difference from first model)
    model_cols = [col for col in pivot_df.columns if col not in comparison_cols]
    if len(model_cols) >= 2:
        baseline_model = model_cols[0]
        for model in model_cols[1:]:
            rel_col = f'{model}_vs_{baseline_model}_pct'
            pivot_df[rel_col] = ((pivot_df[model] - pivot_df[baseline_model]) / 
                                pivot_df[baseline_model] * 100)
    
    return pivot_df
