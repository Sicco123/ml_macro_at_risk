#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Per-Country LQR Model Estimation

This script demonstrates how to estimate LQR models for each country individually,
similar to the code snippet provided in the user request.
"""

import yaml
import pandas as pd
from pathlib import Path
from src.lqr_api import LQRModel

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_country_data(data_dir: Path) -> dict:
    """Load data files for each country."""
    country_files = {}
    for file_path in data_dir.glob("*.parquet"):
        country = file_path.stem.upper()
        country_files[country] = file_path
    return country_files

def estimate_lqr_per_country(config_path: str = "configs/experiment_per_country.yaml"):
    """
    Estimate LQR models for each country individually.
    
    This demonstrates the per-country approach where each country gets its own
    model with country-specific coefficients.
    """
    
    # Load configuration
    config = load_config(config_path)
    
    # Set up paths
    data_dir = Path(config['data']['path'])
    
    # Load country data files
    country_files = load_country_data(data_dir)
    
    print("\n" + "="*50)
    print("LQR Coefficients by Country (Per-Country Models)")
    print("="*50)
    
    lqr_models = {}
    
    for i, (country, file_path) in enumerate(country_files.items()):
        print(f"\nProcessing country: {country}")
        
        # Load country data
        data = pd.read_parquet(file_path)
        data['TIME'] = pd.to_datetime(data['TIME'])
        data = data.sort_values('TIME').reset_index(drop=True)
        
        # Show data head for first country
        if i == 0:
            print("Sample data structure:")
            print(data.head())
        
        # Initialize LQR using high-level API for this country only
        lqr_model = LQRModel(
            data_list=[data],  # Only this country's data
            target=config['data']['target'],
            quantiles=config['data']['quantiles'],
            forecast_horizons=config['data']['horizons'],
            lags=config['data']['lags'],
            alpha=1.0,  # Will be cross-validated
            fit_intercept=True,
            solver=config['models'][2]['params']['solver'],  # lqr-per-country config
            seed=config['seed']
        )
        
        # Cross-validate regularization parameter
        best_alpha = lqr_model.k_fold_validation(
            alphas=config['models'][2]['params']['alphas'],
            n_splits=config['models'][2]['params']['cv_splits']
        )
        
        print(f"Best alpha for {country}: {best_alpha}")
        
        # Fit final model
        lqr_coefficients = lqr_model.fit()
        
        # Store model and results
        lqr_models[country] = {
            'model': lqr_model,
            'coefficients': lqr_coefficients,
            'best_alpha': best_alpha
        }
        
        print(f"Model fitted for {country}")
        print(f"Coefficient shape: {lqr_coefficients.shape}")
    
    return lqr_models

def estimate_ar_per_country(config_path: str = "configs/experiment_per_country.yaml"):
    """
    Estimate AR-QR models for each country individually.
    
    This demonstrates the per-country AR approach using only autoregressive terms.
    """
    
    # Load configuration
    config = load_config(config_path)
    
    # Set up paths
    data_dir = Path(config['data']['path'])
    
    # Load country data files
    country_files = load_country_data(data_dir)
    
    print("\n" + "="*50)
    print("AR-QR Coefficients by Country (Per-Country Models)")
    print("="*50)
    
    ar_models = {}
    
    for i, (country, file_path) in enumerate(country_files.items()):
        print(f"\nProcessing country: {country}")
        
        # Load country data
        data = pd.read_parquet(file_path)
        data['TIME'] = pd.to_datetime(data['TIME'])
        data = data.sort_values('TIME').reset_index(drop=True)
        
        # For AR model, use only TIME and target columns
        target = config['data']['target']
        ar_data = data[['TIME', target]].copy()
        
        # Initialize AR-QR using high-level API for this country only
        ar_model = LQRModel(
            data_list=[ar_data],  # Only this country's AR data
            target=target,
            quantiles=config['data']['quantiles'],
            forecast_horizons=config['data']['horizons'],
            lags=config['data']['lags'],
            alpha=1.0,  # Will be cross-validated
            fit_intercept=True,
            solver=config['models'][3]['params']['solver'],  # ar-qr-per-country config
            seed=config['seed']
        )
        
        # Cross-validate regularization parameter
        best_alpha = ar_model.k_fold_validation(
            alphas=config['models'][3]['params']['alphas'],
            n_splits=config['models'][3]['params']['cv_splits']
        )
        
        print(f"Best alpha for {country} (AR): {best_alpha}")
        
        # Fit final model
        ar_coefficients = ar_model.fit()
        
        # Store model and results
        ar_models[country] = {
            'model': ar_model,
            'coefficients': ar_coefficients,
            'best_alpha': best_alpha
        }
        
        print(f"AR model fitted for {country}")
        print(f"Coefficient shape: {ar_coefficients.shape}")
    
    return ar_models

def main():
    """Main execution function."""
    
    print("Per-Country LQR and AR-QR Model Estimation Example")
    print("=" * 60)
    
    try:
        # Estimate LQR models per country
        lqr_results = estimate_lqr_per_country()
        
        # Estimate AR-QR models per country  
        ar_results = estimate_ar_per_country()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"LQR models estimated for {len(lqr_results)} countries")
        print(f"AR-QR models estimated for {len(ar_results)} countries")
        
        print("\nCountries processed:")
        for country in lqr_results.keys():
            print(f"  - {country}")
        
        print("\n✓ Per-country model estimation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during model estimation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
