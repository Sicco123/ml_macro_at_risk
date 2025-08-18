"""Aggregation utilities for evaluation results."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def aggregate_loss_by_dimension(
    loss_df: pd.DataFrame,
    dimension: str,
    weight_col: str = 'n_samples'
) -> pd.DataFrame:
    """Aggregate losses by a specific dimension.
    
    Args:
        loss_df: DataFrame with loss results
        dimension: Dimension to aggregate by ('country', 'quantile', 'horizon', 'model')
        weight_col: Column to use as weights for averaging
        
    Returns:
        Aggregated DataFrame
    """
    if dimension not in loss_df.columns:
        raise ValueError(f"Dimension '{dimension}' not found in loss DataFrame")
    
    # Weighted average
    agg_df = loss_df.groupby(dimension).apply(
        lambda x: pd.Series({
            'mean_loss': np.average(x['loss'], weights=x[weight_col]),
            'total_samples': x[weight_col].sum(),
            'n_groups': len(x)
        })
    ).reset_index()
    
    return agg_df


def create_performance_ranking(
    loss_df: pd.DataFrame,
    group_by: List[str] = ['quantile', 'horizon'],
    metric: str = 'loss'
) -> pd.DataFrame:
    """Create performance ranking of models.
    
    Args:
        loss_df: DataFrame with loss results
        group_by: Columns to group by
        metric: Metric to rank by
        
    Returns:
        DataFrame with rankings
    """
    # Aggregate by model and specified dimensions
    agg_df = loss_df.groupby(['model'] + group_by).agg({
        metric: 'mean',
        'n_samples': 'sum'
    }).reset_index()
    
    # Rank within each group
    ranking_df = agg_df.groupby(group_by).apply(
        lambda x: x.assign(rank=x[metric].rank(method='min'))
    ).reset_index(drop=True)
    
    # Add relative performance
    for group_vals, group_df in ranking_df.groupby(group_by):
        best_score = group_df[metric].min()
        ranking_df.loc[group_df.index, 'relative_performance'] = (
            (group_df[metric] - best_score) / best_score * 100
        )
    
    return ranking_df.sort_values(['rank'] + group_by)


def create_summary_statistics(
    predictions_df: pd.DataFrame,
    actual_col: str = 'y_true',
    predicted_col: str = 'y_pred'
) -> pd.DataFrame:
    """Create summary statistics for predictions.
    
    Args:
        predictions_df: DataFrame with predictions
        actual_col: Column name for actual values
        predicted_col: Column name for predicted values
        
    Returns:
        DataFrame with summary statistics
    """
    # Compute errors
    predictions_df = predictions_df.copy()
    predictions_df['error'] = predictions_df[actual_col] - predictions_df[predicted_col]
    predictions_df['abs_error'] = np.abs(predictions_df['error'])
    predictions_df['squared_error'] = predictions_df['error'] ** 2
    
    # Group by relevant dimensions
    group_cols = [col for col in ['model', 'country', 'quantile', 'horizon'] 
                  if col in predictions_df.columns]
    
    summary = predictions_df.groupby(group_cols).agg({
        actual_col: ['count', 'mean', 'std'],
        predicted_col: ['mean', 'std'],
        'error': ['mean', 'std'],
        'abs_error': 'mean',
        'squared_error': 'mean'
    }).round(6)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    # Add derived metrics
    summary['rmse'] = np.sqrt(summary['squared_error_mean'])
    summary['mae'] = summary['abs_error_mean']
    summary['bias'] = summary['error_mean']
    
    return summary


def create_model_comparison_table(
    loss_dfs: Dict[str, pd.DataFrame],
    comparison_dims: List[str] = ['quantile', 'horizon']
) -> pd.DataFrame:
    """Create a comprehensive model comparison table.
    
    Args:
        loss_dfs: Dictionary mapping model names to loss DataFrames
        comparison_dims: Dimensions to compare across
        
    Returns:
        Comparison table
    """
    # Combine all loss dataframes
    combined_losses = []
    for model_name, loss_df in loss_dfs.items():
        df = loss_df.copy()
        df['model'] = model_name
        combined_losses.append(df)
    
    combined_df = pd.concat(combined_losses, ignore_index=True)
    
    # Aggregate by comparison dimensions and model
    agg_df = combined_df.groupby(['model'] + comparison_dims).agg({
        'loss': lambda x: np.average(x, weights=combined_df.loc[x.index, 'n_samples']),
        'n_samples': 'sum'
    }).reset_index()
    
    # Pivot to get models as columns
    pivot_df = agg_df.pivot_table(
        index=comparison_dims,
        columns='model',
        values='loss'
    ).reset_index()
    
    # Add rankings and relative performance
    model_cols = [col for col in pivot_df.columns if col not in comparison_dims]
    
    # Best model for each row
    pivot_df['best_model'] = pivot_df[model_cols].idxmin(axis=1)
    pivot_df['best_loss'] = pivot_df[model_cols].min(axis=1)
    
    # Relative performance (percentage worse than best)
    for model in model_cols:
        rel_col = f'{model}_rel_pct'
        pivot_df[rel_col] = ((pivot_df[model] - pivot_df['best_loss']) / 
                            pivot_df['best_loss'] * 100)
    
    return pivot_df


def compute_coverage_statistics(
    predictions_df: pd.DataFrame,
    actual_col: str = 'y_true',
    predicted_col: str = 'y_pred',
    quantile_col: str = 'quantile'
) -> pd.DataFrame:
    """Compute empirical coverage statistics for quantile predictions.
    
    Args:
        predictions_df: DataFrame with quantile predictions
        actual_col: Column name for actual values
        predicted_col: Column name for predicted values
        quantile_col: Column name for quantile levels
        
    Returns:
        DataFrame with coverage statistics
    """
    coverage_stats = []
    
    # Group by relevant dimensions (excluding quantile for now)
    group_cols = [col for col in ['model', 'country', 'horizon'] 
                  if col in predictions_df.columns]
    
    for group_vals, group_df in predictions_df.groupby(group_cols):
        group_dict = dict(zip(group_cols, group_vals)) if len(group_cols) > 1 else {group_cols[0]: group_vals}
        
        # Compute coverage for each quantile
        for quantile in group_df[quantile_col].unique():
            quantile_df = group_df[group_df[quantile_col] == quantile]
            
            if len(quantile_df) == 0:
                continue
            
            # Empirical coverage
            below_forecast = (quantile_df[actual_col] <= quantile_df[predicted_col]).mean()
            
            # Expected coverage is the quantile level
            expected_coverage = quantile
            coverage_error = below_forecast - expected_coverage
            
            stat_record = group_dict.copy()
            stat_record.update({
                'quantile': quantile,
                'n_obs': len(quantile_df),
                'empirical_coverage': below_forecast,
                'expected_coverage': expected_coverage,
                'coverage_error': coverage_error,
                'abs_coverage_error': abs(coverage_error)
            })
            
            coverage_stats.append(stat_record)
    
    return pd.DataFrame(coverage_stats)


def create_comprehensive_report(
    predictions_dfs: Dict[str, pd.DataFrame],
    loss_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """Create a comprehensive evaluation report.
    
    Args:
        predictions_dfs: Dictionary mapping model names to prediction DataFrames
        loss_dfs: Dictionary mapping model names to loss DataFrames
        
    Returns:
        Dictionary of report tables
    """
    report = {}
    
    # 1. Model comparison by loss
    report['loss_comparison'] = create_model_comparison_table(loss_dfs)
    
    # 2. Performance ranking
    combined_losses = []
    for model_name, loss_df in loss_dfs.items():
        df = loss_df.copy()
        df['model'] = model_name
        combined_losses.append(df)
    
    if combined_losses:
        combined_loss_df = pd.concat(combined_losses, ignore_index=True)
        report['performance_ranking'] = create_performance_ranking(combined_loss_df)
    
    # 3. Summary statistics
    for model_name, pred_df in predictions_dfs.items():
        summary_stats = create_summary_statistics(pred_df)
        report[f'summary_stats_{model_name}'] = summary_stats
    
    # 4. Coverage statistics (for quantile predictions)
    for model_name, pred_df in predictions_dfs.items():
        if 'quantile' in pred_df.columns:
            coverage_stats = compute_coverage_statistics(pred_df)
            report[f'coverage_stats_{model_name}'] = coverage_stats
    
    # 5. Overall summary
    overall_summary = []
    for model_name, loss_df in loss_dfs.items():
        weighted_loss = np.average(loss_df['loss'], weights=loss_df['n_samples'])
        overall_summary.append({
            'model': model_name,
            'overall_loss': weighted_loss,
            'total_samples': loss_df['n_samples'].sum(),
            'n_countries': loss_df['country'].nunique() if 'country' in loss_df.columns else None
        })
    
    report['overall_summary'] = pd.DataFrame(overall_summary).sort_values('overall_loss')
    
    return report
