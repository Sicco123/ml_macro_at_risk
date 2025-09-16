#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantile Regression Results Analysis Tool

This script analyzes the output from the quantile regression runner and produces:
1. Pinball loss tables (readable and LaTeX formats)
2. Forecast visualizations per country
3. Diebold-Mariano tests
4. Model confidence sets
"""

import os
import sys
import yaml
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# from scipy.stats import rankdata
# import statsmodels.api as sm
# from statsmodels.tsa.stattools import acf

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class QuantileAnalyzer:
    """Main class for analyzing quantile regression results."""
    
    def __init__(self, config_path: str):
        """Initialize analyzer with configuration."""
        self.config = self._load_config(config_path)
        self.data = {}
        self.covid_start = pd.Timestamp('2020-03-01')
        self.covid_end = pd.Timestamp('2021-12-31')
        self.test_start = pd.Timestamp(self.config['test_period']['start_date'])
        self.test_end = pd.Timestamp(self.config['test_period']['end_date'])
        
        # Set up paths
        self.output_dir = Path(self.config['output']['base_dir'])
        self.forecasts_dir = Path(self.config['input']['forecasts_dir'])
        self.setup_directories()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_directories(self):
        """Create output directories."""
        dirs = ['tables', 'figures', 'latex', 'tests', 'confidence_sets']
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load all forecast data from the SQLite database."""
        print("Loading forecast data from database...")
        
        # Get database path from config
        if 'database' in self.config['input']:
            db_path = Path(self.config['input']['database'])
        else:
            # Default to the standard location
            db_path = self.forecasts_dir.parent / "task_coordination.db"
        
        if not db_path.exists():
            raise ValueError(f"Database file not found: {db_path}")
        
        # Connect to database and load all forecast data
        import sqlite3
        
        with sqlite3.connect(str(db_path)) as conn:
            # Read all forecasts from database
            query = """
                SELECT time, country, true_data, forecast, horizon, quantile, model, version, window_start, window_end
                FROM forecasts
                ORDER BY time, country, horizon, quantile, model, version
            """
            
            self.data = pd.read_sql_query(query, conn)
        
        if len(self.data) == 0:
            raise ValueError("No forecast data found in database!")
        
        # Convert time columns to datetime
        self.data['TIME'] = pd.to_datetime(self.data['time'])
        self.data = self.data.drop('time', axis=1)  # Remove the original time column
        
        # Rename columns to match expected format (uppercase)
        self.data = self.data.rename(columns={
            'country': 'COUNTRY',
            'true_data': 'TRUE_DATA', 
            'forecast': 'FORECAST',
            'horizon': 'HORIZON',
            'quantile': 'QUANTILE',
            'model': 'MODEL',
            'version': 'VERSION',
            'window_start': 'WINDOW_START',
            'window_end': 'WINDOW_END'
        })
        
        # Create a combined model identifier that includes version information
        if 'VERSION' in self.data.columns:
            self.data['MODEL_FULL'] = self.data['MODEL'].astype(str) + '_' + self.data['VERSION'].fillna('default').astype(str)
        else:
            self.data['MODEL_FULL'] = self.data['MODEL'].astype(str)
        
        self.data = self.data.sort_values(['TIME', 'COUNTRY', 'HORIZON', 'QUANTILE', 'MODEL_FULL'])

        # write to a file the number of rows per model, quantile, horizon
        summary = self.data.groupby(['MODEL_FULL', 'QUANTILE', 'HORIZON']).size().reset_index(name='N_ROWS')
        summary.to_csv(self.output_dir / 'data_summary.csv', index=False)



        # check which dates are missing per model, country, quantile, horizon between test_start and test_end
        all_dates = pd.date_range(self.test_start, self.test_end, freq='MS')
        missing_summary = []
        for (model, country, quantile, horizon), group in self.data.groupby(['MODEL_FULL', 'COUNTRY', 'QUANTILE', 'HORIZON']):
            group_dates = pd.to_datetime(group['WINDOW_END'].unique())
            missing_dates = set(all_dates) - set(group_dates)
            if missing_dates:
                missing_summary.append({
                    'MODEL_FULL': model[-4:],
                    'QUANTILE': quantile,
                    'HORIZON': horizon,
                    'MISSING_DATES': ','.join(sorted([d.strftime('%Y-%m-%d') for d in missing_dates]))
            })
        missing_df = pd.DataFrame(missing_summary)
        missing_df.to_csv(self.output_dir / 'missing_dates_summary.csv', index=False, sep= ",")
        
        print(f"Loaded {len(self.data)} forecasts from database")
        print(f"Models: {sorted(self.data['MODEL_FULL'].unique())}")
        print(f"Countries: {sorted(self.data['COUNTRY'].unique())}")
        print(f"Quantiles: {sorted(self.data['QUANTILE'].unique())}")
        print(f"Horizons: {sorted(self.data['HORIZON'].unique())}")
    
    def pinball_loss(self, y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        """Calculate pinball loss for a given quantile."""
        error = y_true - y_pred
        return np.mean(np.maximum(quantile * error, (quantile - 1) * error))
    
    def calculate_pinball_losses(self, exclude_covid: bool = False) -> pd.DataFrame:
        """Calculate pinball losses for all model/country/quantile/horizon combinations."""
        print("Calculating pinball losses...")
        
        data = self.data.copy()
        
        # Filter COVID period if requested
        if exclude_covid:
            data = data[~((data['TIME'] >= self.covid_start) & (data['TIME'] <= self.covid_end))]
            print(f"Excluded COVID period ({self.covid_start.date()} to {self.covid_end.date()})")
        
        results = []
        
        for (model, country, quantile, horizon), group in data.groupby(['MODEL_FULL', 'COUNTRY', 'QUANTILE', 'HORIZON']):
            if len(group) > 0:
                pb_loss = self.pinball_loss(
                    group['TRUE_DATA'].values,
                    group['FORECAST'].values,
                    quantile
                )
                results.append({
                    'MODEL': model,
                    'COUNTRY': country,
                    'QUANTILE': quantile,
                    'HORIZON': horizon,
                    'PINBALL_LOSS': pb_loss,
                    'N_FORECASTS': len(group)
                })
        
        return pd.DataFrame(results)
    
    def create_pinball_tables(self):
        """Create pinball loss tables for each quantile/horizon combination."""
        print("Creating pinball loss tables...")
        
        # Get DM test results for significance stars
        dm_results = self._get_dm_results_for_tables()
        
        for exclude_covid in [False, True]:
            covid_suffix = "_excl_covid" if exclude_covid else "_incl_covid"
            
            losses = self.calculate_pinball_losses(exclude_covid=exclude_covid)
            
            for quantile in sorted(losses['QUANTILE'].unique()):
                for horizon in sorted(losses['HORIZON'].unique()):
                    
                    # Filter data
                    subset = losses[
                        (losses['QUANTILE'] == quantile) & 
                        (losses['HORIZON'] == horizon)
                    ].copy()
                    
                    if len(subset) == 0:
                        continue
                    
                    # Pivot table
                    table = subset.pivot(index='COUNTRY', columns='MODEL', values='PINBALL_LOSS')
                    
                    # Add average row
                    avg_row = table.mean(axis=0)
                    avg_row.name = 'AVERAGE'
                    table = pd.concat([table, avg_row.to_frame().T])
                    
                    # Round for display
                    table_display = table.round(4)
                    
                    # Add significance stars
                    table_with_stars = self._add_dm_stars_to_table(
                        table_display, dm_results, quantile, horizon, exclude_covid
                    )
                    
                    # Save readable version as formatted text
                    filename = f"pinball_q{quantile}_h{horizon}{covid_suffix}"
                    self._save_formatted_table(table_with_stars, self.output_dir / 'tables' / f"{filename}.txt")
                    
                    # Create LaTeX version
                    latex_table = self._create_latex_table(
                        table_with_stars, 
                        f"Pinball Loss: Quantile {quantile}, Horizon {horizon}{' (Excl. COVID)' if exclude_covid else ' (Incl. COVID)'}",
                        f"pinball-q{quantile}-h{horizon}{covid_suffix}"
                    )
                    
                    with open(self.output_dir / 'latex' / f"{filename}.tex", 'w') as f:
                        f.write(latex_table)
                    
                    print(f"Created table for q={quantile}, h={horizon}{covid_suffix}")
    
    def _get_dm_results_for_tables(self) -> pd.DataFrame:
        """Get Diebold-Mariano test results for adding stars to pinball tables using quantile losses."""
        print("Running Diebold-Mariano tests for significance stars...")
        
        benchmark = self.config['analysis']['benchmark_model']
        
        # Check if we have multiple models
        models = self.data['MODEL_FULL'].unique()
        if len(models) < 2:
            print(f"No DM tests possible: Only {len(models)} model(s) found")
            return pd.DataFrame()
        
        # Find benchmark model (may include version)
        benchmark_candidates = [m for m in models if m.startswith(benchmark)]
        if len(benchmark_candidates) == 0:
            print(f"Warning: Benchmark model '{benchmark}' not found in data. Using first model as benchmark.")
            benchmark_model = models[0]
        elif len(benchmark_candidates) == 1:
            benchmark_model = benchmark_candidates[0]
        else:
            print(f"Warning: Multiple benchmark variants found: {benchmark_candidates}. Using first one.")
            benchmark_model = benchmark_candidates[0]
        
        results = []
        
        for exclude_covid in [False, True]:
            data = self.data.copy()
            if exclude_covid:
                data = data[~((data['TIME'] >= self.covid_start) & (data['TIME'] <= self.covid_end))]
            
            for (country, quantile, horizon), group in data.groupby(['COUNTRY', 'QUANTILE', 'HORIZON']):
                
                # Get benchmark errors
                benchmark_data = group[group['MODEL_FULL'] == benchmark_model]
                if len(benchmark_data) == 0:
                    continue
                
                # Compare all other models to benchmark
                models = group['MODEL_FULL'].unique()
                for model in models:
                    if model == benchmark_model:
                        # Benchmark gets no stars (reference model)
                        results.append({
                            'COUNTRY': country,
                            'QUANTILE': quantile,
                            'HORIZON': horizon,
                            'MODEL': model,
                            'P_VALUE': 1.0,  # No test against itself
                            'COVID_EXCLUDED': exclude_covid
                        })
                        continue
                    
                    model_data = group[group['MODEL_FULL'] == model]
                    if len(model_data) == 0:
                        continue
                    
                    # Align data by time
                    common_times = pd.Index(benchmark_data['TIME']).intersection(pd.Index(model_data['TIME']))
                    if len(common_times) < 10:  # Need sufficient observations
                        continue
                    
                    bench_aligned = benchmark_data[benchmark_data['TIME'].isin(common_times)].sort_values('TIME')
                    model_aligned = model_data[model_data['TIME'].isin(common_times)].sort_values('TIME')
                    
                    # Calculate pinball losses for each model
                    bench_losses = np.array([
                        self.pinball_loss(np.array([true]), np.array([pred]), quantile)
                        for true, pred in zip(bench_aligned['TRUE_DATA'].values, bench_aligned['FORECAST'].values)
                    ])
                    model_losses = np.array([
                        self.pinball_loss(np.array([true]), np.array([pred]), quantile)
                        for true, pred in zip(model_aligned['TRUE_DATA'].values, model_aligned['FORECAST'].values)
                    ])
                    
                    # Run DM test: model vs benchmark (negative stat means model is better)
                    dm_stat, p_value = self.diebold_mariano_test(model_losses, bench_losses, horizon)
                    
                    results.append({
                        'COUNTRY': country,
                        'QUANTILE': quantile,
                        'HORIZON': horizon,
                        'MODEL': model,
                        'DM_STAT': dm_stat,
                        'P_VALUE': p_value,
                        'COVID_EXCLUDED': exclude_covid
                    })
        
        return pd.DataFrame(results)
    
    def _add_dm_stars_to_table(self, table: pd.DataFrame, dm_results: pd.DataFrame, 
                              quantile: float, horizon: int, exclude_covid: bool) -> pd.DataFrame:
        """Add Diebold-Mariano significance stars to pinball loss table."""
        
        if len(dm_results) == 0:
            return table
        
        # Filter DM results for this specific combination
        dm_subset = dm_results[
            (dm_results['QUANTILE'] == quantile) & 
            (dm_results['HORIZON'] == horizon) &
            (dm_results['COVID_EXCLUDED'] == exclude_covid)
        ]
        
        if len(dm_subset) == 0:
            return table
        
        # Create a copy to modify
        table_with_stars = table.copy()
        
        # Function to get significance stars - only for models with significantly LOWER loss
        def get_stars(p_value, dm_stat):
            if pd.isna(p_value) or pd.isna(dm_stat):
                return ""
            # Only add stars if model has significantly lower loss (negative DM stat means better)
            if dm_stat < 0:  # Model is better than benchmark
                if p_value < 0.01:
                    return "***"
                elif p_value < 0.05:
                    return "**"
                elif p_value < 0.10:
                    return "*"
            return ""
        
        # Add stars to each cell
        for country in table_with_stars.index:
            for model in table_with_stars.columns:
                if country == 'AVERAGE':
                    # For average row, use overall significance
                    model_dm = dm_subset[dm_subset['MODEL'] == model]
                    if len(model_dm) > 0:
                        avg_p_value = model_dm['P_VALUE'].mean()
                        avg_dm_stat = model_dm['DM_STAT'].mean()
                        stars = get_stars(avg_p_value, avg_dm_stat)
                    else:
                        stars = ""
                else:
                    # For individual countries
                    model_dm = dm_subset[
                        (dm_subset['MODEL'] == model) & 
                        (dm_subset['COUNTRY'] == country)
                    ]
                    if len(model_dm) > 0:
                        p_value = model_dm['P_VALUE'].iloc[0]
                        dm_stat = model_dm['DM_STAT'].iloc[0]
                        stars = get_stars(p_value, dm_stat)
                    else:
                        stars = ""
                
                # Add stars to the value
                current_value = table_with_stars.loc[country, model]
                if pd.notna(current_value):
                    if isinstance(current_value, str):
                        table_with_stars.loc[country, model] = f"{current_value}{stars}"
                    else:
                        table_with_stars.loc[country, model] = f"{current_value:.4f}{stars}"
        
        return table_with_stars
    
    def _create_latex_table(self, df: pd.DataFrame, caption: str, label: str) -> str:
        """Create a LaTeX table from a DataFrame."""
        
        # Prepare column specification
        n_cols = len(df.columns)
        col_spec = 'l' + 'c' * n_cols  # left-aligned for index, centered for data
        
        # Check if this is a pinball loss table with stars (contains numeric values with asterisks)
        has_stars = any('*' in str(val) for val in df.values.flatten() if pd.notna(val) and isinstance(val, str))
        
        latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{tab:{label}}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
"""
        
        # Header
        latex += "Country & " + " & ".join([f"\\textbf{{{col}}}" for col in df.columns]) + " \\\\\n"
        latex += "\\midrule\n"
        
        # Data rows
        for idx, row in df.iterrows():
            if idx == 'AVERAGE':
                latex += "\\midrule\n"
                latex += f"\\textbf{{{idx}}} & "
            else:
                latex += f"{idx} & "
            
            # Handle both numeric and string values
            values = []
            for val in row:
                if pd.isna(val):
                    values.append("---")
                elif isinstance(val, str):
                    values.append(val)
                else:
                    values.append(f"{val:.4f}")
            latex += " & ".join(values) + " \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        
        # Add note about significance stars if present
        if has_stars:
            latex += "\\begin{tablenotes}\n"
            latex += "\\small\n"
            latex += "\\item Notes: Significance stars indicate models with significantly lower pinball loss than benchmark (DM test with Harvey correction): "
            latex += "*** p<0.01, ** p<0.05, * p<0.10\n"
            latex += "\\end{tablenotes}\n"
        
        latex += "\\end{table}\n"
        
        return latex
    
    def _save_formatted_table(self, df: pd.DataFrame, filepath: str):
        """Save a DataFrame as a nicely formatted text file with proper column spacing."""
        with open(filepath, 'w') as f:
            # Convert all values to strings and handle NaN values
            df_str = df.copy()
            for col in df_str.columns:
                df_str[col] = df_str[col].astype(str).replace('nan', '---')
            
            # Calculate column widths
            col_widths = {}
            
            # Check index width
            index_width = max(len(str(idx)) for idx in df_str.index)
            index_width = max(index_width, len('Country'))  # Header width
            
            # Check column widths
            for col in df_str.columns:
                col_width = max(len(str(col)), max(len(str(val)) for val in df_str[col]))
                col_widths[col] = col_width
            
            # Write header
            header = f"{'Country':<{index_width}}"
            for col in df_str.columns:
                header += f" | {col:>{col_widths[col]}}"
            f.write(header + '\n')
            
            # Write separator
            separator = '-' * index_width
            for col in df_str.columns:
                separator += '-+-' + '-' * col_widths[col]
            f.write(separator + '\n')
            
            # Write data rows
            for idx, row in df_str.iterrows():
                line = f"{str(idx):<{index_width}}"
                for col in df_str.columns:
                    value = str(row[col])
                    line += f" | {value:>{col_widths[col]}}"
                f.write(line + '\n')
    
    def plot_forecasts_by_country(self):
        """Create forecast plots for each country."""
        print("Creating forecast plots...")
        
        countries = sorted(self.data['COUNTRY'].unique())
        models = sorted(self.data['MODEL_FULL'].unique())
        quantiles = sorted(self.data['QUANTILE'].unique())
        horizons = sorted(self.data['HORIZON'].unique())
        
        # Configuration options
        separate_quantiles = self.config['plots'].get('separate_quantiles', True)
        high_quality = self.config['plots'].get('high_quality', False)
        
        # Set DPI and format
        dpi = 300 if high_quality else 100
        fmt = 'pdf' if high_quality else 'png'
        
        for country in countries:
            country_data = self.data[self.data['COUNTRY'] == country]
            
            for horizon in horizons:
                horizon_data = country_data[country_data['HORIZON'] == horizon]
                
                if len(horizon_data) == 0:
                    continue
                
                if separate_quantiles:
                    # One plot per quantile
                    for quantile in quantiles:
                        quant_data = horizon_data[horizon_data['QUANTILE'] == quantile]
                        if len(quant_data) == 0:
                            continue
                        
                        self._create_forecast_plot(
                            quant_data, country, horizon, [quantile], models, 
                            dpi=dpi, fmt=fmt
                        )
                else:
                    # All quantiles in one plot
                    self._create_forecast_plot(
                        horizon_data, country, horizon, quantiles, models, 
                        dpi=dpi, fmt=fmt
                    )
    
    def _create_forecast_plot(self, data: pd.DataFrame, country: str, horizon: int, 
                            quantiles: List[float], models: List[str], 
                            dpi: int = 100, fmt: str = 'png'):
        """Create a single forecast plot."""
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 4 * len(models)), 
                                sharex=True, sharey=True, dpi=dpi)
        if len(models) == 1:
            axes = [axes]
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(quantiles)))
        
        for i, model in enumerate(models[:4]):
            ax = axes[i]
            model_data = data[data['MODEL_FULL'] == model].copy()
            
            if len(model_data) == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=14)
                ax.set_title(f"{model}")
                continue
            
            # Plot true data (same for all quantiles)
            true_data = model_data.groupby('TIME')['TRUE_DATA'].first()
            ax.plot(true_data.index, true_data.values, 'k-', 
                   linewidth=2, label='True Data', alpha=0.8)
            
            # Plot forecasts for each quantile
            for j, quantile in enumerate(quantiles):
                quant_data = model_data[model_data['QUANTILE'] == quantile]
                if len(quant_data) > 0:
                    forecasts = quant_data.groupby('TIME')['FORECAST'].first()
                    ax.plot(forecasts.index, forecasts.values, 
                           color=colors[j], linewidth=2, 
                           label=f'Q{quantile}', alpha=0.7, marker='o', markersize=2)
            
            ax.set_title(f"{model}", fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Highlight COVID period
            ax.axvspan(self.covid_start, self.covid_end, alpha=0.2, color='red', 
                      label='COVID-19' if i == 0 else '')
        
        plt.suptitle(f"Forecasts: {country}, Horizon {horizon}", 
                    fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        quant_str = f"q{'_'.join(map(str, quantiles))}"
        filename = f"forecasts_{country}_h{horizon}_{quant_str}.{fmt}"
        plt.savefig(self.output_dir / 'figures' / filename, 
                   dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def plot_cumulative_losses(self):
        """Create cumulative loss plots for each quantile/horizon combination."""
        print("Creating cumulative loss plots...")
        
        quantiles = sorted(self.data['QUANTILE'].unique())
        horizons = sorted(self.data['HORIZON'].unique())
        models = sorted(self.data['MODEL_FULL'].unique())
        countries = sorted(self.data['COUNTRY'].unique())
        
        # Configuration options
        high_quality = self.config['plots'].get('high_quality', False)
        dpi = 300 if high_quality else 100
        fmt = 'pdf' if high_quality else 'png'
        
        for quantile in quantiles:
            for horizon in horizons:
                self._create_cumulative_loss_plot(
                    quantile, horizon, models, countries, dpi=dpi, fmt=fmt
                )
    
    def _create_cumulative_loss_plot(self, quantile: float, horizon: int, 
                                   models: List[str], countries: List[str],
                                   dpi: int = 100, fmt: str = 'png'):
        """Create a single cumulative loss plot for all countries."""
        
        # Filter data for this quantile/horizon
        data = self.data[
            (self.data['QUANTILE'] == quantile) & 
            (self.data['HORIZON'] == horizon)
        ].copy()
        
        if len(data) == 0:
            print(f"No data for q={quantile}, h={horizon}")
            return
        
        # Calculate grid size for subplots
        n_countries = len(countries)
        n_cols = min(4, n_countries)  # Max 4 columns
        n_rows = int(np.ceil(n_countries / n_cols))
        
        # Create figure with extra space at bottom for legend
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows + 1), 
                                sharex=False, sharey=False, dpi=dpi)
        
        # Handle single subplot case
        if n_countries == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten() if n_countries > 1 else axes
        
        # Colors for models
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        model_colors = dict(zip(models, colors))
        
        # Store legend handles and labels from first subplot with data
        legend_handles = []
        legend_labels = []
        
        for i, country in enumerate(countries):
            ax = axes_flat[i]
            
            country_data = data[data['COUNTRY'] == country].copy()
            
            if len(country_data) == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(country, fontsize=12, fontweight='bold')
                continue
            
            # For each model, calculate cumulative loss
            for model in models:
                model_data = country_data[country_data['MODEL_FULL'] == model].copy()
                
                if len(model_data) == 0:
                    continue
                
                # Sort by time
                model_data = model_data.sort_values('TIME')
                
                # Calculate pinball losses for each time point
                losses = []
                times = []
                for _, row in model_data.iterrows():
                    loss = self.pinball_loss(
                        np.array([row['TRUE_DATA']]),
                        np.array([row['FORECAST']]),
                        quantile
                    )
                    losses.append(loss)
                    times.append(row['TIME'])
                
                if len(losses) == 0:
                    continue
                
                # Calculate cumulative sum
                cumulative_losses = np.cumsum(losses)
                
                # Plot
                line = ax.plot(times, cumulative_losses, 
                             color=model_colors[model], linewidth=2, 
                             label=model, alpha=0.8)
                
                # Collect legend info from first subplot with data
                if i == 0 and len(legend_handles) < len(models):
                    legend_handles.append(line[0])
                    legend_labels.append(model)
            
            ax.set_title(country, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Highlight COVID period
            ax.axvspan(self.covid_start, self.covid_end, alpha=0.2, color='red')
        
        # Hide empty subplots
        for j in range(n_countries, len(axes_flat)):
            axes_flat[j].set_visible(False)
        
        # Set common labels
        fig.text(0.5, 0.08, 'Time', ha='center', fontsize=14)
        fig.text(0.02, 0.5, 'Cumulative Pinball Loss', va='center', rotation='vertical', fontsize=14)
        
        # Main title
        plt.suptitle(f"Cumulative Pinball Loss: Quantile {quantile}, Horizon {horizon}", 
                    fontsize=16, fontweight='bold', y=0.96)
        
        # Create legend underneath all subplots
        if legend_handles:
            fig.legend(legend_handles, legend_labels, 
                      loc='lower center', 
                      bbox_to_anchor=(0.5, 0.02),
                      ncol=min(len(legend_labels), 4),
                      fontsize=12,
                      frameon=True,
                      fancybox=True,
                      shadow=True)
        
        # Tight layout with minimal spacing and extra room for legend
        plt.tight_layout(rect=[0.02, 0.15, 0.98, 0.94])
        
        # Save plot
        filename = f"cumulative_loss_q{quantile}_h{horizon}.{fmt}"
        plt.savefig(self.output_dir / 'figures' / filename, 
                   dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Created cumulative loss plot for q={quantile}, h={horizon}")
    
    def diebold_mariano_test(self, losses1: np.ndarray, losses2: np.ndarray, 
                           h: int = 1) -> Tuple[float, float]:
        """
        Perform Diebold-Mariano test for forecast accuracy using quantile losses.
        
        Args:
            losses1: Pinball losses for model 1
            losses2: Pinball losses for model 2  
            h: Forecast horizon (for HAC adjustment and Harvey correction)
            
        Returns:
            DM statistic and p-value
        """
        # Calculate loss differential
        d = losses1 - losses2
        
        # Mean of differences
        d_mean = np.mean(d)
        
        # Standard error with HAC adjustment
        n = len(d)
        
        if h == 1:
            # No autocorrelation adjustment needed
            d_var = np.var(d, ddof=1)
        else:
            # HAC adjustment for multi-step forecasts using Newey-West
            gamma_0 = np.var(d, ddof=1)
            gamma_sum = 0
            
            # Use Newey-West with bandwidth h-1
            for lag in range(1, h):
                if lag < n:
                    gamma_lag = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
                    weight = 1 - lag / h
                    gamma_sum += 2 * weight * gamma_lag
            
            d_var = gamma_0 + gamma_sum
        
        # DM statistic
        if d_var <= 0:
            return np.nan, np.nan
        
        dm_stat = d_mean / np.sqrt(d_var / n)
        
        # Apply Harvey correction for multi-step forecasts
        if h > 1:
            # Harvey, Leybourne, and Newbold (1997) correction
            # Adjust the test statistic to account for finite sample bias
            correction = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
            dm_stat = dm_stat * correction
        
        # Two-sided p-value
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        
        return dm_stat, p_value
    
    def run_diebold_mariano_tests(self):
        """Run Diebold-Mariano tests comparing all models to benchmark using quantile losses."""
        print("Running Diebold-Mariano tests...")
        
        benchmark = self.config['analysis']['benchmark_model']
        
        # Check if we have multiple models
        models = self.data['MODEL_FULL'].unique()
        if len(models) < 2:
            print(f"Skipping DM tests: Only {len(models)} model(s) found, need at least 2")
            return pd.DataFrame()
        
        # Find benchmark model (may include version)
        benchmark_candidates = [m for m in models if m.startswith(benchmark)]
        if len(benchmark_candidates) == 0:
            print(f"Warning: Benchmark model '{benchmark}' not found in data. Using first model as benchmark.")
            benchmark_model = models[0]
        elif len(benchmark_candidates) == 1:
            benchmark_model = benchmark_candidates[0]
        else:
            print(f"Warning: Multiple benchmark variants found: {benchmark_candidates}. Using first one.")
            benchmark_model = benchmark_candidates[0]
        
        results = []
        
        for exclude_covid in [False, True]:
            covid_suffix = "_excl_covid" if exclude_covid else "_incl_covid"
            
            data = self.data.copy()
            if exclude_covid:
                data = data[~((data['TIME'] >= self.covid_start) & (data['TIME'] <= self.covid_end))]
            
            for (country, quantile, horizon), group in data.groupby(['COUNTRY', 'QUANTILE', 'HORIZON']):
                
                # Get benchmark errors
                benchmark_data = group[group['MODEL_FULL'] == benchmark_model]
                if len(benchmark_data) == 0:
                    continue
                
                benchmark_errors = benchmark_data['TRUE_DATA'].values - benchmark_data['FORECAST'].values
                
                # Compare all other models to benchmark
                models = group['MODEL_FULL'].unique()
                for model in models:
                    if model == benchmark_model:
                        continue
                    
                    model_data = group[group['MODEL_FULL'] == model]
                    if len(model_data) == 0:
                        continue
                    
                    # Align data by time
                    common_times = pd.Index(benchmark_data['TIME']).intersection(pd.Index(model_data['TIME']))
                    if len(common_times) < 10:  # Need sufficient observations
                        continue
                    
                    bench_aligned = benchmark_data[benchmark_data['TIME'].isin(common_times)].sort_values('TIME')
                    model_aligned = model_data[model_data['TIME'].isin(common_times)].sort_values('TIME')
                    
                    # Calculate pinball losses for each model
                    bench_losses = np.array([
                        self.pinball_loss(np.array([true]), np.array([pred]), quantile)
                        for true, pred in zip(bench_aligned['TRUE_DATA'].values, bench_aligned['FORECAST'].values)
                    ])
                    model_losses = np.array([
                        self.pinball_loss(np.array([true]), np.array([pred]), quantile)
                        for true, pred in zip(model_aligned['TRUE_DATA'].values, model_aligned['FORECAST'].values)
                    ])
                    
                    # Run DM test: model vs benchmark (negative stat means model is better)
                    dm_stat, p_value = self.diebold_mariano_test(model_losses, bench_losses, horizon)
                    
                    results.append({
                        'COUNTRY': country,
                        'QUANTILE': quantile,
                        'HORIZON': horizon,
                        'MODEL': model,
                        'BENCHMARK': benchmark_model,
                        'DM_STAT': dm_stat,
                        'P_VALUE': p_value,
                        'SIGNIFICANT_5PCT': p_value < 0.05 if pd.notna(p_value) else False,
                        'SIGNIFICANT_10PCT': p_value < 0.10 if pd.notna(p_value) else False,
                        'COVID_EXCLUDED': exclude_covid,
                        'N_OBS': len(common_times)
                    })
        
        dm_results = pd.DataFrame(results)
        
        if len(dm_results) == 0:
            print("No DM test results to save")
            return dm_results
        
        # Save results
        dm_results.to_csv(self.output_dir / 'tests' / 'diebold_mariano_results.txt', index=False, sep='\t')
        
        # Create summary tables
        self._create_dm_summary_tables(dm_results)
        
        return dm_results
    
    def _create_dm_summary_tables(self, dm_results: pd.DataFrame):
        """Create summary tables for Diebold-Mariano test results."""
        
        for exclude_covid in [False, True]:
            covid_suffix = "_excl_covid" if exclude_covid else "_incl_covid"
            subset = dm_results[dm_results['COVID_EXCLUDED'] == exclude_covid]
            
            for quantile in sorted(subset['QUANTILE'].unique()):
                for horizon in sorted(subset['HORIZON'].unique()):
                    
                    data = subset[
                        (subset['QUANTILE'] == quantile) & 
                        (subset['HORIZON'] == horizon)
                    ]
                    
                    if len(data) == 0:
                        continue
                    
                    # Create DM statistic table
                    dm_stat_table = data.pivot(index='COUNTRY', columns='MODEL', values='DM_STAT')
                    
                    # Create p-value table
                    pval_table = data.pivot(index='COUNTRY', columns='MODEL', values='P_VALUE')
                    
                    # Create significance stars table
                    sig_table = data.pivot(index='COUNTRY', columns='MODEL', values='SIGNIFICANT_5PCT')
                    sig_table = sig_table.astype(str).replace({'True': '***', 'False': '', 'nan': ''})
                    
                    # Combine DM stat with p-value in parentheses and add significance stars
                    combined = dm_stat_table.copy()
                    for country in combined.index:
                        for model in combined.columns:
                            dm_stat = dm_stat_table.loc[country, model]
                            p_val = pval_table.loc[country, model]
                            stars = sig_table.loc[country, model] if model in sig_table.columns else ''
                            
                            if pd.notna(dm_stat) and pd.notna(p_val):
                                combined.loc[country, model] = f"{dm_stat:.3f} ({p_val:.3f}){stars}"
                            else:
                                combined.loc[country, model] = "---"
                    
                    # Save
                    filename = f"dm_test_q{quantile}_h{horizon}{covid_suffix}"
                    self._save_formatted_table(combined, self.output_dir / 'tests' / f"{filename}.txt")
                    
                    # LaTeX version
                    latex_table = self._create_latex_table(
                        combined,
                        f"Diebold-Mariano Test Results: Quantile {quantile}, Horizon {horizon}{' (Excl. COVID)' if exclude_covid else ' (Incl. COVID)'}",
                        f"dm-test-q{quantile}-h{horizon}{covid_suffix}"
                    )
                    
                    with open(self.output_dir / 'latex' / f"{filename}.tex", 'w') as f:
                        f.write(latex_table)
    
    def model_confidence_set(self, losses: np.ndarray, alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Model Confidence Set using the procedure of Hansen, Lunde, and Nason (2011).
        
        Args:
            losses: Array of shape (n_observations, n_models) with loss values
            alpha: Significance level
            
        Returns:
            Tuple of (included_models_mask, p_values)
        """
        n_obs, n_models = losses.shape
        
        if n_models <= 1:
            return np.ones(n_models, dtype=bool), np.ones(n_models)
        
        # Step 1: Compute relative performance
        avg_losses = np.mean(losses, axis=0)
        best_model = np.argmin(avg_losses)
        
        # Loss differentials relative to best model
        d = losses - losses[:, [best_model]]
        d_mean = np.mean(d, axis=0)
        
        # Step 2: Compute test statistics and p-values
        included = np.ones(n_models, dtype=bool)
        p_values = np.ones(n_models)
        
        # Bootstrap p-values (simplified version)
        n_boot = 1000
        t_stats = []
        
        for j in range(n_models):
            if j == best_model:
                continue
            
            # Test statistic for model j vs best
            d_j = d[:, j]
            t_j = np.sqrt(n_obs) * np.mean(d_j) / (np.std(d_j) + 1e-8)
            t_stats.append(abs(t_j))
            
            # Bootstrap p-value
            boot_stats = []
            for _ in range(n_boot):
                boot_indices = np.random.choice(n_obs, n_obs, replace=True)
                d_boot = d_j[boot_indices] - np.mean(d_j)
                t_boot = np.sqrt(n_obs) * np.mean(d_boot) / (np.std(d_boot) + 1e-8)
                boot_stats.append(abs(t_boot))
            
            p_values[j] = np.mean(np.array(boot_stats) >= abs(t_j))
        
        # Step 3: Apply sequential testing
        # Simplified: exclude models with p-value < alpha
        included = p_values >= alpha
        included[best_model] = True  # Always include best model
        
        return included, p_values
    
    def run_model_confidence_sets(self):
        """Run Model Confidence Set analysis."""
        print("Running Model Confidence Set analysis...")
        
        # Check if we have multiple models
        models = self.data['MODEL_FULL'].unique()
        if len(models) < 2:
            print(f"Skipping MCS analysis: Only {len(models)} model(s) found, need at least 2")
            return pd.DataFrame()
        
        alpha = self.config['analysis'].get('mcs_alpha', 0.1)
        results = []
        
        for exclude_covid in [False, True]:
            covid_suffix = "_excl_covid" if exclude_covid else "_incl_covid"
            
            losses = self.calculate_pinball_losses(exclude_covid=exclude_covid)
            
            for (country, quantile, horizon), group in losses.groupby(['COUNTRY', 'QUANTILE', 'HORIZON']):
                
                # Get data for all models
                models = sorted(group['MODEL'].unique())
                n_models = len(models)
                
                if n_models <= 1:
                    continue
                
                # Get forecast data for MCS calculation
                forecast_data = self.data[
                    (self.data['COUNTRY'] == country) &
                    (self.data['QUANTILE'] == quantile) &
                    (self.data['HORIZON'] == horizon)
                ].copy()
                
                if exclude_covid:
                    forecast_data = forecast_data[
                        ~((forecast_data['TIME'] >= self.covid_start) & 
                          (forecast_data['TIME'] <= self.covid_end))
                    ]
                
                # Calculate losses for each observation
                loss_matrix = []
                times = sorted(forecast_data['TIME'].unique())
                
                for time in times:
                    time_data = forecast_data[forecast_data['TIME'] == time]
                    time_losses = []
                    
                    for model in models:
                        model_data = time_data[time_data['MODEL_FULL'] == model]
                        if len(model_data) > 0:
                            true_val = model_data['TRUE_DATA'].iloc[0]
                            forecast = model_data['FORECAST'].iloc[0]
                            loss = self.pinball_loss(np.array([true_val]), 
                                                   np.array([forecast]), 
                                                   quantile)
                            time_losses.append(loss)
                        else:
                            time_losses.append(np.nan)
                    
                    if not any(np.isnan(time_losses)):
                        loss_matrix.append(time_losses)
                
                if len(loss_matrix) < 10:  # Need sufficient observations
                    continue
                
                loss_matrix = np.array(loss_matrix)
                
                # Run MCS
                included, p_values = self.model_confidence_set(loss_matrix, alpha)
                
                # Store results
                for i, model in enumerate(models):
                    results.append({
                        'COUNTRY': country,
                        'QUANTILE': quantile,
                        'HORIZON': horizon,
                        'MODEL': model,
                        'INCLUDED': included[i],
                        'P_VALUE': p_values[i],
                        'AVG_LOSS': np.mean(loss_matrix[:, i]),
                        'COVID_EXCLUDED': exclude_covid,
                        'N_OBS': len(loss_matrix),
                        'ALPHA': alpha
                    })
        
        mcs_results = pd.DataFrame(results)
        
        if len(mcs_results) == 0:
            print("No MCS results to save")
            return mcs_results
        
        # Save results
        mcs_results.to_csv(self.output_dir / 'confidence_sets' / 'mcs_results.txt', index=False, sep='\t')
        
        # Create summary tables
        self._create_mcs_summary_tables(mcs_results)
        
        return mcs_results
    
    def _create_mcs_summary_tables(self, mcs_results: pd.DataFrame):
        """Create summary tables for Model Confidence Set results."""
        
        for exclude_covid in [False, True]:
            covid_suffix = "_excl_covid" if exclude_covid else "_incl_covid"
            subset = mcs_results[mcs_results['COVID_EXCLUDED'] == exclude_covid]
            
            for quantile in sorted(subset['QUANTILE'].unique()):
                for horizon in sorted(subset['HORIZON'].unique()):
                    
                    data = subset[
                        (subset['QUANTILE'] == quantile) & 
                        (subset['HORIZON'] == horizon)
                    ]
                    
                    if len(data) == 0:
                        continue
                    
                    # Create inclusion table
                    inclusion_table = data.pivot(index='COUNTRY', columns='MODEL', values='INCLUDED')
                    inclusion_table = inclusion_table.astype(str).replace({'True': '✓', 'False': '✗', 'nan': ''})
                    
                    # Save
                    filename = f"mcs_q{quantile}_h{horizon}{covid_suffix}"
                    self._save_formatted_table(inclusion_table, self.output_dir / 'confidence_sets' / f"{filename}.txt")
                    
                    # LaTeX version
                    latex_table = self._create_latex_table(
                        inclusion_table,
                        f"Model Confidence Set: Quantile {quantile}, Horizon {horizon}{' (Excl. COVID)' if exclude_covid else ' (Incl. COVID)'}",
                        f"mcs-q{quantile}-h{horizon}{covid_suffix}"
                    )
                    
                    with open(self.output_dir / 'latex' / f"{filename}.tex", 'w') as f:
                        f.write(latex_table)
    
    def create_summary_report(self):
        """Create a summary report of all analyses."""
        print("Creating summary report...")
        
        report = f"""
# Quantile Regression Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Overview
- Models: {sorted(self.data['MODEL_FULL'].unique())}
- Countries: {len(self.data['COUNTRY'].unique())} countries
- Quantiles: {sorted(self.data['QUANTILE'].unique())}
- Horizons: {sorted(self.data['HORIZON'].unique())}
- Time period: {self.data['TIME'].min().date()} to {self.data['TIME'].max().date()}
- Total forecasts: {len(self.data)}

## COVID-19 Period
- Start: {self.covid_start.date()}
- End: {self.covid_end.date()}

## Analysis Configuration
- Benchmark model: {self.config['analysis']['benchmark_model']}
- MCS significance level: {self.config['analysis'].get('mcs_alpha', 0.1)}
- High quality plots: {self.config['plots'].get('high_quality', False)}

## Output Files Generated
### Tables
- Pinball loss tables (TXT and LaTeX): `tables/` and `latex/`
- Diebold-Mariano test results: `tests/`
- Model Confidence Set results: `confidence_sets/`

### Figures
- Forecast plots by country: `figures/`
- Cumulative loss plots: `figures/`

### Tests
- Diebold-Mariano test results: `tests/diebold_mariano_results.txt`
- Model Confidence Set results: `confidence_sets/mcs_results.txt`
"""
        
        with open(self.output_dir / 'analysis_report.md', 'w') as f:
            f.write(report)
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting full quantile regression analysis...")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Create pinball loss tables
        self.create_pinball_tables()
        
        # Create forecast plots
        self.plot_forecasts_by_country()
        
        # Create cumulative loss plots
        self.plot_cumulative_losses()
        
        # Run statistical tests
        self.run_diebold_mariano_tests()
        
        # Model confidence sets
        self.run_model_confidence_sets()
        
        # Create summary report
        self.create_summary_report()
        
        print("=" * 60)
        print("Analysis complete!")
        print(f"Results saved to: {self.output_dir}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Analyze quantile regression results")
    parser.add_argument("--config", required=True, help="Path to analysis configuration file")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = QuantileAnalyzer(args.config)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
