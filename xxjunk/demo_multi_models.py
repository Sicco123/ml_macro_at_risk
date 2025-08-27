#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Run Multiple Models for Analysis Demonstration

This script shows how to run the quantile regression with multiple models
to demonstrate the full analysis capabilities.
"""

import subprocess
import sys
from pathlib import Path

def run_with_multiple_models():
    """Run quantile regression with multiple models enabled."""
    
    # Create a config with multiple models enabled
    config_content = """
seed: 123

data:
  path: processed_per_country
  time_col: TIME
  country_col: COUNTRY
  target: prc_hicp_manr_CP00
  lags: [1,2,3,6,12]
  horizons: [1,3]
  quantiles: [0.1, 0.5, 0.9]
  missing: forward_fill_then_mean

rolling_window:
  size: 60
  step: 12  # Fewer windows for faster execution
  start: auto
  end: auto

splits:
  test_cutoff: "2017-12-01"
  min_train_points: 24

runtime:
  allow_reload: true
  retrain_if_exists: false
  max_cores: 2  # Limit cores for example
  max_ram_gb: auto
  safety_ram_fraction: 0.8
  mem_probe_fudge_mb: 200
  retries: 1
  thread_pinning: true
  progress_refresh_sec: 5

io:
  output_root: outputs_demo
  models_dir: models
  forecasts_dir: forecasts
  progress_parquet: progress/progress.parquet
  errors_parquet: progress/errors.parquet
  logs_dir: logs

models:
  - type: ar-qr
    enabled: true
    params:
      solver: huberized
      alphas: [10, 20, 30]  # Fewer alphas for speed
      use_cv: true
      cv_splits: 3
  - type: lqr
    enabled: true
    params:
      solver: huberized
      alphas: [10, 20, 30]
      use_cv: true
      cv_splits: 3
  - type: ar-qr-per-country
    enabled: true
    params:
      solver: huberized
      alphas: [10, 20, 30]
      use_cv: true
      cv_splits: 3
  - type: lqr-per-country
    enabled: true
    params:
      solver: huberized
      alphas: [10, 20, 30]
      use_cv: true
      cv_splits: 3
"""
    
    # Write config file
    config_path = Path("configs/demo_multi_models.yaml")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print("Created demo configuration with multiple models")
    print(f"Config saved to: {config_path}")
    
    # Run quantile regression
    print("\n" + "="*60)
    print("Running quantile regression with multiple models...")
    print("This may take a while depending on your data size")
    print("="*60)
    
    try:
        result = subprocess.run([
            sys.executable, "quant_runner.py", 
            "--config", str(config_path)
        ], check=True, capture_output=True, text=True)
        
        print("Quantile regression completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running quantile regression: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    # Create analysis config pointing to demo outputs
    analysis_config = """
input:
  forecasts_dir: "outputs_demo/forecasts"

output:
  base_dir: "analysis_output_demo"

analysis:
  benchmark_model: "ar-qr"
  mcs_alpha: 0.1

plots:
  separate_quantiles: true
  high_quality: false

covid:
  start_date: "2020-03-01"
  end_date: "2021-12-31"
"""
    
    analysis_config_path = Path("configs/analysis_demo.yaml")
    with open(analysis_config_path, 'w') as f:
        f.write(analysis_config)
    
    print(f"Created analysis config: {analysis_config_path}")
    
    # Run analysis
    print("\n" + "="*60)
    print("Running comprehensive analysis...")
    print("="*60)
    
    try:
        result = subprocess.run([
            sys.executable, "analyze_results.py",
            "--config", str(analysis_config_path)
        ], check=True, capture_output=True, text=True)
        
        print("Analysis completed successfully!")
        print("\nResults available in:")
        print("- Tables: analysis_output_demo/tables/")
        print("- Plots: analysis_output_demo/figures/")
        print("- LaTeX: analysis_output_demo/latex/")
        print("- Tests: analysis_output_demo/tests/")
        print("- Confidence Sets: analysis_output_demo/confidence_sets/")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running analysis: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    return True

def main():
    """Main execution function."""
    print("Multi-Model Quantile Regression Demo")
    print("=" * 40)
    print("This script will:")
    print("1. Create a configuration with multiple models")
    print("2. Run quantile regression")
    print("3. Generate comprehensive analysis")
    print("4. Demonstrate all analysis features")
    print()
    
    response = input("Do you want to proceed? This may take 10-30 minutes (y/n): ")
    if response.lower() != 'y':
        print("Demo cancelled.")
        return 0
    
    success = run_with_multiple_models()
    
    if success:
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nYou can now explore:")
        print("1. Pinball loss tables comparing all models")
        print("2. Forecast plots for each country")
        print("3. Diebold-Mariano test results")
        print("4. Model Confidence Set analysis")
        print("\nTo run analysis with different settings:")
        print("python analyze_results.py --config configs/analysis_demo.yaml")
        return 0
    else:
        print("\nDemo failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
