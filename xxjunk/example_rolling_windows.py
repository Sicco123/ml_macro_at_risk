#!/usr/bin/env python3
"""
Example script to demonstrate rolling window logic
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_example_windows():
    """Generate example rolling windows to show the concept"""
    
    print("="*60)
    print("ROLLING WINDOW FORECASTING EXAMPLE")
    print("="*60)
    
    # Example configuration
    window_size = 60  # 5 years of monthly data
    step = 1         # Advance 1 month at a time
    start_date = "2018-01-01"
    end_date = "2020-12-01"
    
    print(f"\nConfiguration:")
    print(f"- Window size: {window_size} months")
    print(f"- Step size: {step} month(s)")
    print(f"- Forecast period: {start_date} to {end_date}")
    
    # Generate sample date range (1997-2023)
    all_dates = pd.date_range("1997-01-01", "2023-12-01", freq="MS")
    
    print(f"\nAvailable data: {all_dates[0].strftime('%Y-%m')} to {all_dates[-1].strftime('%Y-%m')} ({len(all_dates)} months)")
    
    # Generate rolling windows
    windows = []
    current_forecast = pd.to_datetime(start_date)
    end_forecast = pd.to_datetime(end_date)
    
    while current_forecast <= end_forecast:
        # Training window ends 1 month before forecast
        window_end = current_forecast - pd.DateOffset(months=1)
        window_start = window_end - pd.DateOffset(months=window_size-1)
        
        # Check if window is valid
        if window_start >= all_dates[0] and window_end <= all_dates[-1]:
            windows.append({
                'window_start': window_start.strftime('%Y-%m-%d'),
                'window_end': window_end.strftime('%Y-%m-%d'),
                'forecast_date': current_forecast.strftime('%Y-%m-%d'),
                'train_months': window_size
            })
        
        current_forecast += pd.DateOffset(months=step)
    
    print(f"\nGenerated {len(windows)} rolling windows:")
    print("-" * 80)
    print(f"{'Window':<8} {'Training Period':<25} {'Forecast Date':<15} {'Months'}")
    print("-" * 80)
    
    for i, window in enumerate(windows[:10]):  # Show first 10
        train_period = f"{window['window_start'][:7]} to {window['window_end'][:7]}"
        print(f"{i+1:<8} {train_period:<25} {window['forecast_date']:<15} {window['train_months']}")
    
    if len(windows) > 10:
        print(f"... and {len(windows) - 10} more windows")
    
    # Example task generation
    countries = ['AT', 'DE', 'FR', 'NL']
    models = ['lqr', 'ensemble_nn']
    quantiles = [0.1, 0.5, 0.9]
    horizons = [1, 3, 6, 12]
    
    total_tasks = len(countries) * len(models) * len(quantiles) * len(horizons) * len(windows)
    
    print(f"\nExample task calculation:")
    print(f"- Countries: {len(countries)} ({', '.join(countries)})")
    print(f"- Models: {len(models)} ({', '.join(models)})")
    print(f"- Quantiles: {len(quantiles)} ({quantiles})")
    print(f"- Horizons: {len(horizons)} ({horizons})")
    print(f"- Windows: {len(windows)}")
    print(f"- Total tasks: {total_tasks:,}")
    
    # Memory estimation example
    print(f"\nMemory estimation example:")
    est_mb_per_task = {
        'lqr': 100,
        'ensemble_nn': 500
    }
    
    max_cores = 8
    max_ram_gb = 16
    max_ram_mb = max_ram_gb * 1024
    
    for model in models:
        mem_per_task = est_mb_per_task[model]
        max_parallel = min(max_cores, max_ram_mb // mem_per_task)
        model_tasks = len(countries) * len(quantiles) * len(horizons) * len(windows)
        
        print(f"- {model}: {mem_per_task}MB per task")
        print(f"  Max parallel: {max_parallel} tasks")
        print(f"  Total {model} tasks: {model_tasks:,}")
        print(f"  Estimated time: {model_tasks // max_parallel} batches")
    
    print(f"\nOutput files example:")
    print("Forecast files (one per model/quantile/horizon):")
    for model in models:
        for quantile in quantiles:
            for horizon in horizons:
                filename = f"{model}_prc_hicp_manr_CP00_q{quantile}_h{horizon}.parquet"
                print(f"  forecasts/{filename}")
    
    print(f"\nModel files (one per task):")
    example_window = windows[0]
    for country in countries[:2]:  # Show 2 countries
        for model in models[:1]:  # Show 1 model
            for quantile in quantiles[:1]:  # Show 1 quantile
                for horizon in horizons[:1]:  # Show 1 horizon
                    task_id = f"{model}_prc_hicp_manr_CP00_{quantile}_{horizon}_{country}_{example_window['window_start']}_{example_window['window_end']}"
                    print(f"  models/{model}/{task_id}.pkl")
    print("  ... and many more model files")

def demonstrate_task_flow():
    """Show how a single task flows through the system"""
    
    print("\n" + "="*60)
    print("SINGLE TASK EXECUTION FLOW")
    print("="*60)
    
    # Example task
    task = {
        'model': 'lqr',
        'target': 'prc_hicp_manr_CP00',
        'quantile': 0.1,
        'horizon': 3,
        'country': 'DE',
        'window_start': '2017-01-01',
        'window_end': '2017-12-01'
    }
    
    print(f"\nExample Task:")
    for key, value in task.items():
        print(f"  {key}: {value}")
    
    print(f"\nExecution steps:")
    print(f"1. Load data for {task['country']}")
    print(f"2. Filter training data: {task['window_start']} to {task['window_end']}")
    print(f"3. Preprocess data (handle missing, create lags, scale)")
    print(f"4. Check if model exists: models/{task['model']}/{task['model']}_...pkl")
    print(f"5. If not exists or retrain=True:")
    print(f"   - Train {task['model']} model for quantile {task['quantile']}")
    print(f"   - Save model to disk")
    print(f"6. Generate forecast for horizon {task['horizon']}")
    print(f"7. Save forecast to: forecasts/{task['model']}_...q{task['quantile']}_h{task['horizon']}.parquet")
    print(f"8. Update progress: status='done'")
    
    print(f"\nForecast output:")
    forecast_date = pd.to_datetime(task['window_end']) + pd.DateOffset(months=task['horizon'])
    print(f"  TIME: {forecast_date.strftime('%Y-%m-%d')}")
    print(f"  COUNTRY: {task['country']}")
    print(f"  TARGET: {task['target']}")
    print(f"  MODEL: {task['model']}")
    print(f"  QUANTILE: {task['quantile']}")
    print(f"  HORIZON: {task['horizon']}")
    print(f"  FORECAST: [predicted value]")

if __name__ == "__main__":
    generate_example_windows()
    demonstrate_task_flow()
    
    print("\n" + "="*60)
    print("To run the actual parallel forecasting:")
    print("python3 parallel_runner.py --config configs/parallel_config.yaml")
    print("="*60)
