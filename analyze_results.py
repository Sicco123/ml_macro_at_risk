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
        """Load all forecast data from the quantile runner output."""
        print("Loading forecast data...")
        
        all_data = []
        
        # Iterate through all quantile/horizon combinations
        for q_dir in self.forecasts_dir.glob("q=*"):
            quantile = float(q_dir.name.split('=')[1])
            
            for h_dir in q_dir.glob("h=*"):
                horizon = int(h_dir.name.split('=')[1])
                
                parquet_file = h_dir / "rolling_window.parquet"
                if parquet_file.exists():
                    df = pd.read_parquet(parquet_file)
                    df['TIME'] = pd.to_datetime(df['TIME'])
                    all_data.append(df)
        
        if not all_data:
            raise ValueError("No forecast data found!")
        
        self.data = pd.concat(all_data, ignore_index=True)
        self.data = self.data.sort_values(['TIME', 'COUNTRY', 'HORIZON', 'QUANTILE', 'MODEL'])
        
        print(f"Loaded {len(self.data)} forecasts")
        print(f"Models: {sorted(self.data['MODEL'].unique())}")
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
        
        for (model, country, quantile, horizon), group in data.groupby(['MODEL', 'COUNTRY', 'QUANTILE', 'HORIZON']):
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
                    
                    # Save readable version
                    filename = f"pinball_q{quantile}_h{horizon}{covid_suffix}"
                    table_display.to_csv(self.output_dir / 'tables' / f"{filename}.csv")
                    
                    # Create LaTeX version
                    latex_table = self._create_latex_table(
                        table_display, 
                        f"Pinball Loss: Quantile {quantile}, Horizon {horizon}{' (Excl. COVID)' if exclude_covid else ' (Incl. COVID)'}",
                        f"pinball-q{quantile}-h{horizon}{covid_suffix}"
                    )
                    
                    with open(self.output_dir / 'latex' / f"{filename}.tex", 'w') as f:
                        f.write(latex_table)
                    
                    print(f"Created table for q={quantile}, h={horizon}{covid_suffix}")
    
    def _create_latex_table(self, df: pd.DataFrame, caption: str, label: str) -> str:
        """Create a LaTeX table from a DataFrame."""
        
        # Prepare column specification
        n_cols = len(df.columns)
        col_spec = 'l' + 'c' * n_cols  # left-aligned for index, centered for data
        
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
            
            values = [f"{val:.4f}" if pd.notna(val) else "---" for val in row]
            latex += " & ".join(values) + " \\\\\n"
        
        latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        return latex
    
    def plot_forecasts_by_country(self):
        """Create forecast plots for each country."""
        print("Creating forecast plots...")
        
        countries = sorted(self.data['COUNTRY'].unique())
        models = sorted(self.data['MODEL'].unique())
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
        
        fig, axes = plt.subplots(len(models), 1, figsize=(14, 4 * len(models)), 
                                sharex=True, sharey=True, dpi=dpi)
        if len(models) == 1:
            axes = [axes]
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(quantiles)))
        
        for i, model in enumerate(models):
            ax = axes[i]
            model_data = data[data['MODEL'] == model].copy()
            
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
    
    def diebold_mariano_test(self, errors1: np.ndarray, errors2: np.ndarray, 
                           h: int = 1) -> Tuple[float, float]:
        """
        Perform Diebold-Mariano test for forecast accuracy.
        
        Args:
            errors1: Forecast errors for model 1
            errors2: Forecast errors for model 2  
            h: Forecast horizon (for HAC adjustment)
            
        Returns:
            DM statistic and p-value
        """
        # Calculate loss differential
        d = errors1**2 - errors2**2
        
        # Mean of differences
        d_mean = np.mean(d)
        
        # Standard error with HAC adjustment
        n = len(d)
        
        if h == 1:
            # No autocorrelation adjustment needed
            d_var = np.var(d, ddof=1)
        else:
            # HAC adjustment for multi-step forecasts
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
        
        # Two-sided p-value
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        
        return dm_stat, p_value
    
    def run_diebold_mariano_tests(self):
        """Run Diebold-Mariano tests comparing all models to benchmark."""
        print("Running Diebold-Mariano tests...")
        
        benchmark = self.config['analysis']['benchmark_model']
        
        # Check if we have multiple models
        models = self.data['MODEL'].unique()
        if len(models) < 2:
            print(f"Skipping DM tests: Only {len(models)} model(s) found, need at least 2")
            return pd.DataFrame()
        
        if benchmark not in models:
            print(f"Warning: Benchmark model '{benchmark}' not found in data. Using first model as benchmark.")
            benchmark = models[0]
        
        results = []
        
        for exclude_covid in [False, True]:
            covid_suffix = "_excl_covid" if exclude_covid else "_incl_covid"
            
            data = self.data.copy()
            if exclude_covid:
                data = data[~((data['TIME'] >= self.covid_start) & (data['TIME'] <= self.covid_end))]
            
            for (country, quantile, horizon), group in data.groupby(['COUNTRY', 'QUANTILE', 'HORIZON']):
                
                # Get benchmark errors
                benchmark_data = group[group['MODEL'] == benchmark]
                if len(benchmark_data) == 0:
                    continue
                
                benchmark_errors = benchmark_data['TRUE_DATA'].values - benchmark_data['FORECAST'].values
                
                # Compare all other models to benchmark
                models = group['MODEL'].unique()
                for model in models:
                    if model == benchmark:
                        continue
                    
                    model_data = group[group['MODEL'] == model]
                    if len(model_data) == 0:
                        continue
                    
                    # Align data by time
                    common_times = pd.Index(benchmark_data['TIME']).intersection(pd.Index(model_data['TIME']))
                    if len(common_times) < 10:  # Need sufficient observations
                        continue
                    
                    bench_aligned = benchmark_data[benchmark_data['TIME'].isin(common_times)].sort_values('TIME')
                    model_aligned = model_data[model_data['TIME'].isin(common_times)].sort_values('TIME')
                    
                    bench_errors = bench_aligned['TRUE_DATA'].values - bench_aligned['FORECAST'].values
                    model_errors = model_aligned['TRUE_DATA'].values - model_aligned['FORECAST'].values
                    
                    # Run DM test
                    dm_stat, p_value = self.diebold_mariano_test(model_errors, bench_errors, horizon)
                    
                    results.append({
                        'COUNTRY': country,
                        'QUANTILE': quantile,
                        'HORIZON': horizon,
                        'MODEL': model,
                        'BENCHMARK': benchmark,
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
        dm_results.to_csv(self.output_dir / 'tests' / 'diebold_mariano_results.csv', index=False)
        
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
                    
                    # Create significance table
                    sig_table = data.pivot(index='COUNTRY', columns='MODEL', values='SIGNIFICANT_5PCT')
                    sig_table = sig_table.astype(str).replace({'True': '***', 'False': '', 'nan': ''})
                    
                    # Create p-value table
                    pval_table = data.pivot(index='COUNTRY', columns='MODEL', values='P_VALUE')
                    pval_table = pval_table.round(3)
                    
                    # Combine tables
                    combined = pval_table.astype(str) + sig_table
                    
                    # Save
                    filename = f"dm_test_q{quantile}_h{horizon}{covid_suffix}"
                    combined.to_csv(self.output_dir / 'tests' / f"{filename}.csv")
                    
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
        models = self.data['MODEL'].unique()
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
                        model_data = time_data[time_data['MODEL'] == model]
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
        mcs_results.to_csv(self.output_dir / 'confidence_sets' / 'mcs_results.csv', index=False)
        
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
                    inclusion_table.to_csv(self.output_dir / 'confidence_sets' / f"{filename}.csv")
                    
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
- Models: {sorted(self.data['MODEL'].unique())}
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
- Pinball loss tables (CSV and LaTeX): `tables/` and `latex/`
- Diebold-Mariano test results: `tests/`
- Model Confidence Set results: `confidence_sets/`

### Figures
- Forecast plots by country: `figures/`

### Tests
- Diebold-Mariano test results: `tests/diebold_mariano_results.csv`
- Model Confidence Set results: `confidence_sets/mcs_results.csv`
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
