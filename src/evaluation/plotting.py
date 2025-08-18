"""Plotting utilities for evaluation and visualization."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Curves",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot training and validation loss curves.
    
    Args:
        train_losses: Training losses over epochs
        val_losses: Validation losses over epochs (optional)
        title: Plot title
        save_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses is not None:
        val_epochs = range(1, len(val_losses) + 1)
        ax.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    
    return fig


def plot_forecast_paths(
    predictions_df: pd.DataFrame,
    countries: Optional[List[str]] = None,
    quantiles: Optional[List[float]] = None,
    horizons: Optional[List[int]] = None,
    n_countries: int = 4,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot forecast paths for selected countries and quantiles.
    
    Args:
        predictions_df: DataFrame with predictions
        countries: List of countries to plot (if None, selects first n_countries)
        quantiles: List of quantiles to plot (if None, plots all)
        horizons: List of horizons to plot (if None, plots all)
        n_countries: Number of countries to plot if countries not specified
        save_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure
    """
    df = predictions_df.copy()
    
    # Select countries
    if countries is None:
        available_countries = df['country'].unique()
        countries = available_countries[:min(n_countries, len(available_countries))]
    
    # Filter data
    df = df[df['country'].isin(countries)]
    
    if quantiles is not None:
        df = df[df['quantile'].isin(quantiles)]
    
    if horizons is not None:
        df = df[df['horizon'].isin(horizons)]
    
    # Create subplots
    n_cols = 2
    n_rows = int(np.ceil(len(countries) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, country in enumerate(countries):
        ax = axes[i]
        country_df = df[df['country'] == country]
        
        # Plot by horizon
        for horizon in country_df['horizon'].unique():
            horizon_df = country_df[country_df['horizon'] == horizon]
            
            # Sort by time
            horizon_df = horizon_df.sort_values('TIME')
            
            # Plot quantiles
            quantile_values = sorted(horizon_df['quantile'].unique())
            
            if len(quantile_values) >= 3:
                # Plot prediction intervals
                lower_q = quantile_values[0]
                median_q = quantile_values[len(quantile_values) // 2]
                upper_q = quantile_values[-1]
                
                lower_df = horizon_df[horizon_df['quantile'] == lower_q]
                median_df = horizon_df[horizon_df['quantile'] == median_q]
                upper_df = horizon_df[horizon_df['quantile'] == upper_q]
                
                # Plot actual values
                ax.plot(median_df['TIME'], median_df['y_true'], 'k-', 
                       label='Actual' if horizon == country_df['horizon'].unique()[0] else "",
                       linewidth=2)
                
                # Plot median prediction
                ax.plot(median_df['TIME'], median_df['y_pred'], '--', 
                       label=f'Median (h={horizon})', linewidth=2)
                
                # Plot prediction interval
                ax.fill_between(lower_df['TIME'], lower_df['y_pred'], upper_df['y_pred'],
                               alpha=0.3, label=f'{lower_q:.0%}-{upper_q:.0%} PI (h={horizon})')
            
            else:
                # Just plot individual quantiles
                for q in quantile_values:
                    q_df = horizon_df[horizon_df['quantile'] == q]
                    ax.plot(q_df['TIME'], q_df['y_pred'], '--', 
                           label=f'q={q:.2f}, h={horizon}', linewidth=2)
        
        ax.set_title(f'{country}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(len(countries), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Forecast paths saved to {save_path}")
    
    return fig


def plot_loss_comparison(
    loss_comparison_df: pd.DataFrame,
    comparison_dims: List[str] = ['quantile', 'horizon'],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot model loss comparison.
    
    Args:
        loss_comparison_df: DataFrame from create_model_comparison_table
        comparison_dims: Dimensions to compare across
        save_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure
    """
    # Get model columns
    model_cols = [col for col in loss_comparison_df.columns 
                  if col not in comparison_dims + ['best_model', 'best_loss'] and not col.endswith('_rel_pct')]
    
    # Create combination labels
    if len(comparison_dims) == 1:
        loss_comparison_df['combination'] = loss_comparison_df[comparison_dims[0]].astype(str)
    else:
        loss_comparison_df['combination'] = loss_comparison_df[comparison_dims].apply(
            lambda x: ' | '.join(f'{dim}={val}' for dim, val in zip(comparison_dims, x)), axis=1
        )
    
    # Melt for plotting
    plot_df = loss_comparison_df.melt(
        id_vars=['combination'],
        value_vars=model_cols,
        var_name='model',
        value_name='loss'
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.barplot(data=plot_df, x='combination', y='loss', hue='model', ax=ax)
    
    ax.set_title('Model Loss Comparison')
    ax.set_xlabel(' | '.join(comparison_dims))
    ax.set_ylabel('Average Loss')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Rotate x labels if needed
    if len(loss_comparison_df) > 5:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Loss comparison saved to {save_path}")
    
    return fig


def plot_calibration(
    predictions_df: pd.DataFrame,
    models: Optional[List[str]] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot calibration (reliability) diagram for quantile predictions.
    
    Args:
        predictions_df: DataFrame with quantile predictions
        models: List of models to plot (if None, plots all)
        save_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure
    """
    if 'quantile' not in predictions_df.columns:
        raise ValueError("DataFrame must contain 'quantile' column for calibration plot")
    
    df = predictions_df.copy()
    
    if models is not None:
        df = df[df['model'].isin(models)]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # Plot calibration for each model
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        quantile_levels = []
        empirical_coverage = []
        
        for q in sorted(model_df['quantile'].unique()):
            q_df = model_df[model_df['quantile'] == q]
            coverage = (q_df['y_true'] <= q_df['y_pred']).mean()
            
            quantile_levels.append(q)
            empirical_coverage.append(coverage)
        
        ax.plot(quantile_levels, empirical_coverage, 'o-', 
               label=model, linewidth=2, markersize=6)
    
    ax.set_xlabel('Nominal Coverage (Quantile Level)')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title('Calibration Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration plot saved to {save_path}")
    
    return fig


def plot_factor_correlations(
    factors_df: pd.DataFrame,
    country: str,
    quantile: float,
    horizon: int,
    input_features: Optional[List[str]] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot correlations between factors and input features.
    
    Args:
        factors_df: DataFrame with factor values
        country: Country to analyze
        quantile: Quantile level to analyze
        horizon: Horizon to analyze
        input_features: List of input feature names (optional)
        save_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure
    """
    # Filter data
    subset_df = factors_df[
        (factors_df['country'] == country) & 
        (factors_df['quantile'] == quantile) & 
        (factors_df['horizon'] == horizon)
    ].copy()
    
    if len(subset_df) == 0:
        raise ValueError(f"No data found for country={country}, quantile={quantile}, horizon={horizon}")
    
    # Get factor columns
    factor_cols = [col for col in subset_df.columns if col.startswith('factor_')]
    
    if not factor_cols:
        raise ValueError("No factor columns found in DataFrame")
    
    # If input features provided, compute correlations
    if input_features is not None:
        # Assume input features are available in the same DataFrame or provided separately
        corr_data = subset_df[factor_cols + input_features].corr()
        
        # Extract factor-feature correlations
        factor_feature_corr = corr_data.loc[factor_cols, input_features]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(factor_feature_corr, annot=True, cmap='RdBu_r', center=0,
                   fmt='.2f', ax=ax)
        ax.set_title(f'Factor-Feature Correlations\n{country}, q={quantile}, h={horizon}')
        
    else:
        # Just plot factor correlations
        factor_corr = subset_df[factor_cols].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(factor_corr, annot=True, cmap='RdBu_r', center=0,
                   fmt='.2f', ax=ax)
        ax.set_title(f'Factor Correlations\n{country}, q={quantile}, h={horizon}')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Factor correlations saved to {save_path}")
    
    return fig


def create_evaluation_dashboard(
    predictions_dfs: Dict[str, pd.DataFrame],
    loss_comparison_df: pd.DataFrame,
    save_dir: Optional[Path] = None
) -> Dict[str, plt.Figure]:
    """Create a comprehensive evaluation dashboard.
    
    Args:
        predictions_dfs: Dictionary mapping model names to prediction DataFrames
        loss_comparison_df: Model comparison DataFrame
        save_dir: Directory to save plots (optional)
        
    Returns:
        Dictionary of matplotlib figures
    """
    figures = {}
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Loss comparison
    fig1 = plot_loss_comparison(loss_comparison_df)
    figures['loss_comparison'] = fig1
    if save_dir:
        fig1.savefig(save_dir / 'loss_comparison.png', dpi=300, bbox_inches='tight')
    
    # 2. Forecast paths for each model
    for model_name, pred_df in predictions_dfs.items():
        fig2 = plot_forecast_paths(pred_df, n_countries=4)
        figures[f'forecast_paths_{model_name}'] = fig2
        if save_dir:
            fig2.savefig(save_dir / f'forecast_paths_{model_name}.png', dpi=300, bbox_inches='tight')
    
    # 3. Calibration plot (if quantile predictions available)
    combined_predictions = []
    for model_name, pred_df in predictions_dfs.items():
        if 'quantile' in pred_df.columns:
            df = pred_df.copy()
            df['model'] = model_name
            combined_predictions.append(df)
    
    if combined_predictions:
        combined_pred_df = pd.concat(combined_predictions, ignore_index=True)
        fig3 = plot_calibration(combined_pred_df)
        figures['calibration'] = fig3
        if save_dir:
            fig3.savefig(save_dir / 'calibration.png', dpi=300, bbox_inches='tight')
    
    logger.info(f"Created {len(figures)} evaluation plots")
    
    return figures
