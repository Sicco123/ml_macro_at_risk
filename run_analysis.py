#!/usr/bin/env python3
"""
Cross-Country Quantile Forecasting with Factor Neural Networks

This script demonstrates a complete end-to-end workflow for multi-horizon quantile forecasting 
across countries using Factor Neural Networks (FNN) and Linear Quantile Regression (LQR).

Converted from Jupyter notebook to Python script with file output and plot saving.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import yaml
from datetime import datetime
import sys
import os
import traceback
from scipy import stats

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Create output directories
output_dir = Path("analysis_output")
plots_dir = output_dir / "plots"
data_dir = output_dir / "data"
logs_dir = output_dir / "logs"

for dir_path in [output_dir, plots_dir, data_dir, logs_dir]:
    dir_path.mkdir(exist_ok=True)

# Set up logging to file
import logging
log_file = logs_dir / f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def save_output(content, filename, subdir=""):
    """Save text content to file"""
    if subdir:
        save_path = output_dir / subdir / filename
        (output_dir / subdir).mkdir(exist_ok=True)
    else:
        save_path = output_dir / filename
    
    with open(save_path, 'w') as f:
        f.write(content)
    logging.info(f"Saved output to {save_path}")

def save_plot(fig, filename, subdir="plots"):
    """Save matplotlib figure to file"""
    save_path = output_dir / subdir / filename
    (output_dir / subdir).mkdir(exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved plot to {save_path}")

def save_dataframe(df, filename, subdir="data"):
    """Save dataframe to CSV"""
    save_path = output_dir / subdir / filename
    (output_dir / subdir).mkdir(exist_ok=True)
    df.to_csv(save_path, index=False)
    logging.info(f"Saved dataframe to {save_path}")

def main():
    """Main analysis function"""
    

    # chekc avialability of GPU 



    logging.info("Starting Cross-Country Quantile Forecasting Analysis")
    logging.info("="*70)
    
    # Project imports (adjust paths as needed)
    sys.path.append('.')
    
    try:
        from src.utils import (
            load_country_data, 
            load_config,
            set_seeds,
            handle_missing_values,
            scale_features,
            create_lagged_features,
            create_forecast_targets,
            create_time_split
        )
        
        # Import high-level API classes
        from src.ensemble_nn_api import EnsembleNNAPI
        from src.lqr_api import LQRModel
        
        from src.metrics import (
            pinball_loss,
            compute_quantile_losses,
            create_loss_summary_table,
            dm_test_by_groups,
            multiple_testing_correction
        )
        
        from src.evaluation import (
            create_model_comparison_table,
            plot_training_curves,
            plot_forecast_paths,
            plot_loss_comparison,
            plot_calibration,
            create_evaluation_dashboard
        )
        
        import torch 
        torch.set_num_threads(4)
        
        logging.info("Libraries imported successfully!")
        logging.info(f"Numpy version: {np.__version__}")
        logging.info(f"Pandas version: {pd.__version__}")
        
    except ImportError as e:
        logging.error(f"Import error: {e}")
        return
    
    # ===== CONFIGURATION AND SETUP =====
    logging.info("\n" + "="*50)
    logging.info("CONFIGURATION AND SETUP")
    logging.info("="*50)
    
    config_path = "configs/experiment_big_data.yaml"
    
    try:
        config = load_config(config_path)
        logging.info("Configuration loaded successfully!")
        config_summary = f"""
Key parameters:
Target variable: {config['data']['target']}
Quantiles: {config['data']['quantiles']}
Horizons: {config['data']['horizons']}
Lags: {config['data']['lags']}
Validation size: {config['splits']['validation_size']}
"""
        logging.info(config_summary)
        save_output(config_summary, "config_summary.txt")
        
    except FileNotFoundError:
        logging.info(f"Configuration file not found at {config_path}")
        logging.info("Using default configuration...")
        
        # Default configuration
        config = {
            'seed': 123,
            'data': {
                'target': 'GDP',
                'required_columns': ['TIME', 'CISS', 'INFL_OECD', 'GDP', 'CREDIT_TO_GDP'],
                'quantiles': [0.05, 0.95],
                'horizons': [1],
                'lags': [1],
                'missing': 'forward_fill_then_mean',
                'scale': 'per_country',
                'scale_target': False,
                'trimming': None
            },
            'splits': {
                'validation_size': 0.2,
                'train_start': '1990-12-31',
                'test_cutoff': '2018-12-31',
                'min_train_points': 60
            },
            'ensemble_nn': {
                'units_per_layer': [8, 4],
                'activation': 'relu',
                'optimizer': 'adam',
                'learning_rate': 1e-3,
                'epochs': 50,
                'batch_size': 64,
                'patience': 10,
                'parallel_models': 1,
                'device': 'auto',
                'per_country': True,
                'country_dummies': False,
                'l2_penalty': 0.001
            },
            'lqr': {
                'alphas': [0.0, 0.1, 1.0, 10.0],
                'solver': 'huberized'
            }
        }
    
    # Set random seed
    set_seeds(config['seed'])
    logging.info(f"Random seed set to: {config['seed']}")
    
    # ===== LOAD AND EXPLORE DATASET =====
    logging.info("\n" + "="*50)
    logging.info("LOAD AND EXPLORE DATASET")
    logging.info("="*50)
    
    data_path = "processed_per_country"
    
    try:
        # Load all country CSV files
        country_data = load_country_data(
            data_path=data_path,
            required_columns=config['data']['required_columns']
        )
        
        data_summary = f"Successfully loaded data for {len(country_data)} countries:\n"
        for country, df in country_data.items():
            data_summary += f"  {country}: {len(df)} observations, {df['TIME'].min()} to {df['TIME'].max()}\n"
        
        logging.info(data_summary)
        save_output(data_summary, "data_loading_summary.txt")
        
        # Save sample data
        sample_country = list(country_data.keys())[0]
        sample_df = country_data[sample_country].head(10)
        save_dataframe(sample_df, f"sample_data_{sample_country}.csv")
        
        # Save summary statistics
        summary_stats = country_data[sample_country].describe()
        save_dataframe(summary_stats, f"summary_stats_{sample_country}.csv")
        
    except FileNotFoundError:
        logging.info(f"Data directory not found at {data_path}")
        logging.info("Creating synthetic data for demonstration...")
        
        # Create synthetic data for demonstration
        countries = ['USA', 'DEU', 'FRA', 'GBR']
        start_date = '2000-01-01'
        end_date = '2023-12-31'
        freq = 'Q'  # Quarterly data
        
        country_data = {}
        np.random.seed(config['seed'])
        
        for country in countries:
            dates = pd.date_range(start_date, end_date, freq=freq)
            n_obs = len(dates)
            
            # Generate synthetic time series with trends and cycles
            t = np.arange(n_obs)
            trend = 0.01 * t + np.random.normal(0, 0.1, n_obs).cumsum()
            
            df = pd.DataFrame({
                'TIME': dates,
                'GDP': 100 + trend + np.random.normal(0, 2, n_obs),
                'CISS': np.random.beta(2, 5, n_obs),
                'INFL_OECD': np.random.normal(2, 1, n_obs),
                'CREDIT_TO_GDP': 150 + 10 * np.sin(t / 20) + np.random.normal(0, 5, n_obs)
            })
            
            country_data[country] = df
        
        logging.info(f"Created synthetic data for {len(country_data)} countries")
        
        # Save synthetic data
        for country, df in country_data.items():
            save_dataframe(df, f"synthetic_data_{country}.csv")
    
    # Update config with actual columns
    config['data']['required_columns'] = list(country_data[list(country_data.keys())[0]].columns)
    #logging.info(f"Updated required columns in config: {config['data']['required_columns']}")
    
    # Visualize the data
    variables = config['data']['required_columns'][1:]  # Exclude 'TIME'
    colors = plt.cm.Set1(np.linspace(0, 1, len(country_data)))
    
   
    # Correlation analysis
    correlation_summary = "Correlation analysis:\n"
    for country, df in list(country_data.items())[:2]:  # Show first 2 countries
        correlation_summary += f"\n{country} - Correlation matrix:\n"
        corr_matrix = df[variables].corr()
        correlation_summary += corr_matrix.round(3).to_string() + "\n"
        
        # Save correlation matrix
        save_dataframe(corr_matrix.round(3), f"correlation_matrix_{country}.csv")
    
    save_output(correlation_summary, "correlation_analysis.txt")
    
    # ===== DATA PREPROCESSING =====
    logging.info("\n" + "="*50)
    logging.info("DATA PREPROCESSING")
    logging.info("="*50)
    
    # 1. Handle missing values
    logging.info("Handling missing values...")
    processed_data = handle_missing_values(
        country_data, 
        policy=config['data']['missing']
    )
    
    # Check for missing values
    missing_summary = "Missing values after processing:\n"
    for country, df in processed_data.items():
        missing_count = df.isnull().sum().sum()
        missing_summary += f"{country}: {missing_count} missing values\n"
    
    logging.info(missing_summary)
    save_output(missing_summary, "missing_values_summary.txt")
    
    # 2. Create time splits
    logging.info("Creating time splits...")
    train_data, test_data, dropped_countries = create_time_split(
        processed_data,
        train_start=config['splits']['train_start'],
        test_cutoff=config['splits']['test_cutoff'],
        min_train_points=config['splits']['min_train_points']
    )
    
    split_summary = f"""Train/test split created:
  Training countries: {len(train_data)}
  Test countries: {len(test_data)}
  Dropped countries: {dropped_countries}

Split sizes:"""
    
    for country in list(train_data.keys())[:3]:  # First 3 countries
        train_size = len(train_data[country])
        test_size = len(test_data[country])
        split_summary += f"\n  {country}: {train_size} train, {test_size} test"
    
    logging.info(split_summary)
    save_output(split_summary, "train_test_split_summary.txt")
    
    # 3. Scale features
    logging.info("Scaling features...")
    scaled_train, scaled_test, scalers = scale_features(
        train_data=train_data,
        test_data=test_data,
        policy=config['data']['scale'],
        target_col=config['data']['target'],
        scale_target=config['data']['scale_target'],
        trimming=config['data'].get('trimming', None),
    )
    
    logging.info(f"Feature scaling completed using '{config['data']['scale']}' policy")
    
    # ===== FEATURE ENGINEERING =====
    logging.info("\n" + "="*50)
    logging.info("FEATURE ENGINEERING")
    logging.info("="*50)
    
    # 1. Create lagged features
    logging.info("Creating lagged features...")
    lagged_train = create_lagged_features(
        scaled_train, 
        lags=config['data']['lags']
    )
    
    lagged_test = create_lagged_features(
        scaled_test, 
        lags=config['data']['lags']
    )
    
    # Show example of lagged features
    sample_country = list(lagged_train.keys())[0]
    feature_summary = f"""Lagged features for {sample_country}:
Original columns: {list(scaled_train[sample_country].columns)}
With lags: {list(lagged_train[sample_country].columns)}
Data shape: {lagged_train[sample_country].shape}"""
    
    logging.info(feature_summary)
    save_output(feature_summary, "feature_engineering_summary.txt")
    
    # 2. Create forecast targets
    logging.info("Creating forecast targets...")
    target_train = create_forecast_targets(
        lagged_train,
        target_col=config['data']['target'],
        horizons=config['data']['horizons']
    )
    
    target_test = create_forecast_targets(
        lagged_test,
        target_col=config['data']['target'],
        horizons=config['data']['horizons']
    )
    
    # Show target structure
    target_cols = [f"{config['data']['target']}_h{h}" for h in config['data']['horizons']]
    target_summary = f"Target columns: {target_cols}\n"
    
    for country in list(target_train.keys())[:2]:
        target_summary += f"{country} training shape: {target_train[country].shape}\n"
        target_summary += f"{country} test shape: {target_test[country].shape}\n"
    
    logging.info(target_summary)
    save_output(target_summary, "target_creation_summary.txt")
    
    # Save sample of processed data
    sample_processed = target_train[sample_country][['TIME'] + target_cols].head()
    save_dataframe(sample_processed, f"processed_data_sample_{sample_country}.excsv")
    
    # ===== MODEL TRAINING =====
    logging.info("\n" + "="*50)
    logging.info("MODEL TRAINING")
    logging.info("="*50)
    
    def create_dummy_variables(data_list, time_col="TIME", add_country_dummies=False):
        """Create dummy variables for countries if needed"""
        num_of_countries = len(data_list)
        if not add_country_dummies:
            return data_list
        
        new_data_list = []
        for i, df in enumerate(data_list):
            df_copy = df.copy()
            for j in range(num_of_countries):
                dummy_col = f'dummy-{j}'
                if i != j:
                    df_copy[dummy_col] = 0
                else:
                    df_copy[dummy_col] = 1
            new_data_list.append(df_copy)
        
        return new_data_list
    
    # Prepare data for models
    train_data_list = []
    test_data_list = []
    
    for country, df in scaled_train.items():
        df_copy = df.copy()
        train_data_list.append(df_copy)
    
    for country, df in scaled_test.items():
        df_copy = df.copy()
        test_data_list.append(df_copy)
    
    logging.info(f"Prepared data for {len(train_data_list)} countries")
    
    # Ensure all data types are proper
    for i, df in enumerate(train_data_list):
        if 'TIME' in df.columns and df['TIME'].dtype == 'object':
            train_data_list[i]['TIME'] = pd.to_datetime(df['TIME'])
                
    for i, df in enumerate(test_data_list):
        if 'TIME' in df.columns and df['TIME'].dtype == 'object':
            test_data_list[i]['TIME'] = pd.to_datetime(df['TIME'])
    
    if config['ensemble_nn']['country_dummies'] and not config['ensemble_nn']['per_country']:
        global_train_data_list = create_dummy_variables(
            train_data_list,
            time_col="TIME",
            add_country_dummies=config['ensemble_nn']['country_dummies']
        )
        
        global_test_data_list = create_dummy_variables(
            test_data_list,
            time_col="TIME",
            add_country_dummies=config['ensemble_nn']['country_dummies']
        )
    else:
        global_train_data_list = train_data_list
        global_test_data_list = test_data_list
    
    # Train Factor Neural Network
    logging.info("\n" + "="*50)
    logging.info("Training Factor Neural Network")
    logging.info("="*50)
    
    n_countries = len(train_data_list)
    logging.info(f"Available countries: {n_countries}")
    
    country_names = [country for country in list(target_train.keys())]
    
    if config['ensemble_nn']['per_country']:
        fnn_models = {}
        for i, df in enumerate(train_data_list):
            logging.info(f"Training Factor NN for {i}...")
            
            try:
                ensemble_nn = EnsembleNNAPI(
                    data_list=[df],
                    target=config['data']['target'],
                    quantiles=config['data']['quantiles'],
                    forecast_horizons=config['data']['horizons'],
                    units_per_layer=config['ensemble_nn']['units_per_layer'],
                    lags=config['data']['lags'],
                    activation=config['ensemble_nn']['activation'],
                    device=config['ensemble_nn']['device'],
                    seed=config['seed'], 
                    transform=True,
                )
            
                ensemble_results = ensemble_nn.fit(
                    epochs=config['ensemble_nn']['epochs'],
                    learning_rate=config['ensemble_nn']['learning_rate'],
                    batch_size=config['ensemble_nn']['batch_size'],
                    validation_size=config['splits']['validation_size'],
                    patience=config['ensemble_nn']['patience'],
                    verbose=1,
                    optimizer=config['ensemble_nn']['optimizer'],
                    parallel_models=config['ensemble_nn']['parallel_models'],
                    l2=config['ensemble_nn']['l2_penalty'],
                    return_validation_loss=True,
                    return_train_loss=True,
                    shuffle=True
                )
                
                logging.info(f"✓ Factor NN for {i} trained successfully!")
                
                # Store model and results
                fnn_models[i] = {
                    'model': ensemble_nn,
                    'results': ensemble_results
                }
                
              
                
            except Exception as e:
                logging.error(f"Factor NN training failed for country {i}: {e}")
                continue
        
        logging.info(f"Trained Factor NN models for {len(fnn_models)} countries.")
        fnn_training_successful = len(fnn_models) > 0
        ensemble_nn = None
        
    else:
        try:
            # Initialize Ensemble NN using high-level API
            ensemble_nn = EnsembleNNAPI(
                data_list=global_train_data_list,
                target=config['data']['target'],
                quantiles=config['data']['quantiles'],
                forecast_horizons=config['data']['horizons'],
                units_per_layer=config['ensemble_nn']['units_per_layer'],
                lags=config['data']['lags'],
                activation=config['ensemble_nn']['activation'],
                device=config['ensemble_nn']['device'],
                seed=config['seed'], 
                transform=True,
                prefit_AR=True,
                country_ids=country_names,
                time_col="TIME",
                verbose=1
            )
            
            training_summary = f"""✓ Ensemble NN initialized successfully!
  Input dimension: {ensemble_nn.input_dim}
  Target quantiles: {ensemble_nn.quantiles}
  Forecast horizons: {ensemble_nn.forecast_horizons}"""
            
            logging.info(training_summary)
            save_output(training_summary, "fnn_training_summary.txt")
            
            # Train with cross-validation
            logging.info("Starting training with cross-validation...")
            ensemble_results = ensemble_nn.fit(
                epochs=config['ensemble_nn']['epochs'],
                learning_rate=config['ensemble_nn']['learning_rate'],
                batch_size=config['ensemble_nn']['batch_size'],
                validation_size=config['splits']['validation_size'],
                patience=config['ensemble_nn']['patience'],
                verbose=1,
                optimizer=config['ensemble_nn']['optimizer'],
                parallel_models=config['ensemble_nn']['parallel_models'],
                l2=config['ensemble_nn']['l2_penalty'],
                return_validation_loss=True,
                return_train_loss=True,
                shuffle=True
            )
            
            final_summary = f"""✓ Ensemble NN training completed successfully!
  Results keys: {list(ensemble_results.keys())}
  Number of parameters: {ensemble_results.get('n_parameters', 'Unknown')}
  Final validation loss: {ensemble_results.get('final_val_loss', 'Unknown')}"""
            
            logging.info(final_summary)
            save_output(final_summary, "fnn_final_results.txt")
            
          
            fnn_training_successful = True
            
        except Exception as e:
            logging.error(f"Factor NN training failed: {e}")
            logging.error("Traceback:")
            logging.error(traceback.format_exc())
            
            ensemble_nn = None
            fnn_training_successful = False
    
    # Train Linear Quantile Regression per country
    logging.info("\n" + "="*50)
    logging.info("Training Linear Quantile Regression")
    logging.info("="*50)
    
    return

if __name__ == "__main__":
    main()
