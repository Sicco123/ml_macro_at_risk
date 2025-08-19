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
    logging.info(f"Updated required columns in config: {config['data']['required_columns']}")
    
    # Visualize the data
    variables = config['data']['required_columns'][1:]  # Exclude 'TIME'
    colors = plt.cm.Set1(np.linspace(0, 1, len(country_data)))
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(variables[:6]):
        ax = axes[i]
        
        for j, (country, df) in enumerate(country_data.items()):
            ax.plot(df['TIME'], df[var], label=country, color=colors[j], linewidth=2)
        
        ax.set_title(f'{var} Across Countries')
        ax.set_xlabel('Time')
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, "data_visualization.png")
    
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
    save_dataframe(sample_processed, f"processed_data_sample_{sample_country}.csv")
    
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
                
                # Plot training curves if available
                if 'train_losses' in ensemble_results and 'val_losses' in ensemble_results:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    train_losses = ensemble_results['train_losses'][min(config['ensemble_nn']['patience']-1,5):]
                    val_losses = ensemble_results['val_losses'][min(config['ensemble_nn']['patience']-1,5):]
                    
                    ax.plot(train_losses, label='Training Loss', linewidth=2)
                    ax.plot(val_losses, label='Validation Loss', linewidth=2)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title(f'Factor NN Training Curves - Country {i}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    save_plot(fig, f"fnn_training_curves_country_{i}.png")
                
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
            
            # Plot training curves if available
            if 'train_losses' in ensemble_results and 'val_losses' in ensemble_results:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(ensemble_results['train_losses'], label='Training Loss', linewidth=2)
                ax.plot(ensemble_results['val_losses'], label='Validation Loss', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Ensemble NN Training Curves')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                save_plot(fig, "ensemble_nn_training_curves.png")
            
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
    
    lqr_models = {}
    
    for i, data in enumerate(train_data_list):
        try:
            # Initialize LQR using high-level API
            lqr_model = LQRModel(
                data_list=[data],
                target=config['data']['target'],
                quantiles=config['data']['quantiles'],
                forecast_horizons=config['data']['horizons'],
                lags=config['data']['lags'],
                alpha=1.0,
                fit_intercept=True,
                solver=config['lqr']['solver'],
                seed=config['seed']
            )
            
            # Cross-validate regularization parameter
            best_alpha = lqr_model.k_fold_validation(
                alphas=config['lqr']['alphas'],
                n_splits=10
            )
            
            logging.info(f"Best alpha for country {i}: {best_alpha}")
            
            # Fit final model
            lqr_coefficients = lqr_model.fit()
            
            lqr_models[i] = {
                'model': lqr_model,
                'coefficients': lqr_coefficients
            }
            
        except Exception as e:
            logging.error(f"LQR training failed for country {i}: {e}")
            continue
    
    # Train LQR AR models (autoregressive only)
    logging.info("\nTraining LQR AR models...")
    lqr_ar_models = {}
    
    for i, data in enumerate(train_data_list):
        try:
            # Only use the target column and the time column
            data_temp = data[['TIME', config['data']['target']]].copy()
            
            lqr_model = LQRModel(
                data_list=[data_temp],
                target=config['data']['target'],
                quantiles=config['data']['quantiles'],
                forecast_horizons=config['data']['horizons'],
                lags=config['data']['lags'],
                alpha=0.0,
                fit_intercept=True,
                solver=config['lqr']['solver'],
                seed=config['seed']
            )
            
            # Fit final model
            lqr_coefficients = lqr_model.fit()
            
            lqr_ar_models[i] = {
                'model': lqr_model,
                'coefficients': lqr_coefficients
            }
            
        except Exception as e:
            logging.error(f"LQR AR training failed for country {i}: {e}")
            continue
    
    # Status summary
    status_summary = f"""TRAINING STATUS SUMMARY:
✓ Data preprocessing: SUCCESSFUL
✓ Feature engineering: SUCCESSFUL 
✓ Dataset creation: SUCCESSFUL
✓ Data type handling: SUCCESSFUL
{'✓' if fnn_training_successful else '🚨'} Factor NN training: {'SUCCESSFUL' if fnn_training_successful else 'FAILED'}
✓ LQR training: SUCCESSFUL ({len(lqr_models)} models)
✓ LQR AR training: SUCCESSFUL ({len(lqr_ar_models)} models)"""
    
    logging.info(status_summary)
    save_output(status_summary, "training_status_summary.txt")
    
    # ===== MODEL PREDICTIONS =====
    logging.info("\n" + "="*50)
    logging.info("GENERATING PREDICTIONS")
    logging.info("="*50)
    
    # Generate predictions for each model
    lqr_per_country_predictions, lqr_per_country_targets = {}, {}
    country_names = [country for country in list(target_train.keys())]
    
    for idx, test_data in enumerate(test_data_list):
        if idx in lqr_models:
            model = lqr_models[idx]['model']
            predictions, targets = model.predict([test_data])
            lqr_per_country_predictions[country_names[idx]] = predictions
            lqr_per_country_targets[country_names[idx]] = targets
    
    lqr_ar_per_country_predictions, lqr_ar_per_country_targets = {}, {}
    
    for idx, test_data in enumerate(test_data_list):
        if idx in lqr_ar_models:
            model = lqr_ar_models[idx]['model']
            predictions, targets = model.predict([test_data[['TIME', config['data']['target']]]])
            lqr_ar_per_country_predictions[country_names[idx]] = predictions
            lqr_ar_per_country_targets[country_names[idx]] = targets
    
    fnn_per_country_predictions, fnn_per_country_targets = {}, {}
    
    for idx, test_data in enumerate(global_test_data_list):
        if config['ensemble_nn']['per_country']:
            if idx in fnn_models:
                model = fnn_models[idx]['model']
                predictions, targets = model.predict_per_country(test_data, 0)
                fnn_per_country_predictions[country_names[idx]] = predictions
                fnn_per_country_targets[country_names[idx]] = targets
        else:
            if ensemble_nn:
                predictions, targets = ensemble_nn.predict_per_country(test_data, country_names[idx])
                fnn_per_country_predictions[country_names[idx]] = predictions
                fnn_per_country_targets[country_names[idx]] = targets
    
    prediction_summary = f"""Prediction Generation Summary:
FNN predictions: {len(fnn_per_country_predictions)} countries
LQR predictions: {len(lqr_per_country_predictions)} countries  
LQR AR predictions: {len(lqr_ar_per_country_predictions)} countries"""
    
    logging.info(prediction_summary)
    save_output(prediction_summary, "prediction_summary.txt")
    
    # ===== FORECAST VISUALIZATION =====
    logging.info("\n" + "="*50)
    logging.info("FORECAST VISUALIZATION")
    logging.info("="*50)
    
    countries = list(scaled_test.keys())
    n_countries_to_plot = len(countries)
    div_2 = (n_countries_to_plot + 1) // 2
    
    for h, horizon in enumerate(config['data']['horizons']):
        fig, axes = plt.subplots(div_2, 2, figsize=(15, 4*div_2))
        if div_2 == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        logging.info(f"Creating forecast plots for horizon {horizon}...")
        
        for i, country in enumerate(countries[:n_countries_to_plot]):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Check if we have predictions for this country
            if (country in fnn_per_country_targets and 
                country in lqr_per_country_targets and 
                country in lqr_ar_per_country_targets):
                
                # Extract actual values for this country and horizon
                if fnn_per_country_targets[country].ndim == 2:
                    actual = fnn_per_country_targets[country][:, h]
                else:
                    actual = fnn_per_country_targets[country]
                
                # Extract predictions for this country and horizon
                if (country in fnn_per_country_predictions and 
                    fnn_per_country_predictions[country] is not None):
                    fnn_pred_this_horizon = fnn_per_country_predictions[country][:, h]
                else:
                    fnn_pred_this_horizon = np.full((len(actual), len(config['data']['quantiles'])), np.nan)
                
                if (country in lqr_per_country_predictions and 
                    lqr_per_country_predictions[country] is not None):
                    lqr_pred_this_horizon = lqr_per_country_predictions[country][:, h]
                else:
                    lqr_pred_this_horizon = np.full((len(actual), len(config['data']['quantiles'])), np.nan)
                
                if (country in lqr_ar_per_country_predictions and 
                    lqr_ar_per_country_predictions[country] is not None):
                    lqr_ar_pred_this_horizon = lqr_ar_per_country_predictions[country][:, h]
                else:
                    lqr_ar_pred_this_horizon = np.full((len(actual), len(config['data']['quantiles'])), np.nan)
                
                # Extract quantiles (assuming first is lower, second is upper)
                try:
                    fnn_pred_q05 = fnn_pred_this_horizon[:, 0] if fnn_pred_this_horizon.shape[1] > 0 else np.full(len(actual), np.nan)
                    fnn_pred_q95 = fnn_pred_this_horizon[:, 1] if fnn_pred_this_horizon.shape[1] > 1 else np.full(len(actual), np.nan)
                    lqr_pred_q05 = lqr_pred_this_horizon[:, 0] if lqr_pred_this_horizon.shape[1] > 0 else np.full(len(actual), np.nan)
                    lqr_pred_q95 = lqr_pred_this_horizon[:, 1] if lqr_pred_this_horizon.shape[1] > 1 else np.full(len(actual), np.nan)
                    lqr_ar_pred_q05 = lqr_ar_pred_this_horizon[:, 0] if lqr_ar_pred_this_horizon.shape[1] > 0 else np.full(len(actual), np.nan)
                    lqr_ar_pred_q95 = lqr_ar_pred_this_horizon[:, 1] if lqr_ar_pred_this_horizon.shape[1] > 1 else np.full(len(actual), np.nan)
                except (IndexError, AttributeError):
                    fnn_pred_q05 = fnn_pred_q95 = np.full(len(actual), np.nan)
                    lqr_pred_q05 = lqr_pred_q95 = np.full(len(actual), np.nan)
                    lqr_ar_pred_q05 = lqr_ar_pred_q95 = np.full(len(actual), np.nan)
                
                # Time index for plotting
                time_idx = range(len(actual))
                
                # Plot actual values
                ax.plot(time_idx, actual, 'k-', linewidth=2, label='Actual', alpha=0.8)
                
                # Plot predictions with uncertainty bands
                if not np.all(np.isnan(fnn_pred_q05)):
                    ax.fill_between(time_idx, fnn_pred_q05, fnn_pred_q95, 
                                   alpha=0.3, color='blue', label='FNN 90% CI')
                    fnn_median = (fnn_pred_q05 + fnn_pred_q95) / 2
                    ax.plot(time_idx, fnn_median, 'b--', 
                           linewidth=1.5, label='FNN Median', alpha=0.8)
                
                if not np.all(np.isnan(lqr_pred_q05)):
                    ax.fill_between(time_idx, lqr_pred_q05, lqr_pred_q95, 
                                   alpha=0.2, color='red', label='LQR 90% CI')
                    lqr_median = (lqr_pred_q05 + lqr_pred_q95) / 2
                    ax.plot(time_idx, lqr_median, 'r:', 
                           linewidth=1.5, label='LQR Median', alpha=0.8)
                
                if not np.all(np.isnan(lqr_ar_pred_q05)):
                    ax.fill_between(time_idx, lqr_ar_pred_q05, lqr_ar_pred_q95, 
                                   alpha=0.2, color='green', label='LQR AR 90% CI')
                    lqr_ar_median = (lqr_ar_pred_q05 + lqr_ar_pred_q95) / 2
                    ax.plot(time_idx, lqr_ar_median, 'g-.', 
                           linewidth=1.5, label='LQR AR Median', alpha=0.8)
                
                ax.set_title(f'{country} - Horizon {horizon}')
                ax.set_xlabel('Time Period')
                ax.set_ylabel(config['data']['target'])
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No predictions\navailable for\n{country}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{country} - Horizon {horizon}')
        
        # Hide unused subplots
        for i in range(n_countries_to_plot, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Forecast Comparison - {horizon}-Period Ahead', fontsize=16)
        plt.tight_layout()
        save_plot(fig, f"forecast_comparison_horizon_{horizon}.png")
    
    # ===== STATISTICAL SIGNIFICANCE TESTING =====
    logging.info("\n" + "="*50)
    logging.info("STATISTICAL SIGNIFICANCE TESTING")
    logging.info("="*50)
    
    def compare_models_dm_test(model1_predictions, model2_predictions, targets, 
                              model1_name="Model1", model2_name="Model2", countries=None):
        """Perform Diebold-Mariano test for forecast accuracy comparison"""
        
        if countries is None:
            countries = list(model1_predictions.keys())
        
        dm_results_per_country = {}
        
        # Per-Country Diebold-Mariano Tests
        for country in countries:
            if (model1_predictions.get(country) is not None and 
                model2_predictions.get(country) is not None and
                targets.get(country) is not None):
                
                dm_results_per_country[country] = {}
                country_summary = f"\n{country} - Diebold-Mariano Tests ({model1_name} vs {model2_name}):\n"
                
                for h, horizon in enumerate(config['data']['horizons']):
                    dm_results_per_country[country][horizon] = {}
                    
                    for q, quantile in enumerate(config['data']['quantiles']):
                        # Get targets for this country and horizon
                        if targets[country].ndim == 2:
                            actual = targets[country][:, h]
                        else:
                            actual = targets[country]
                        
                        # Get predictions for this country, horizon, and quantile
                        if (model1_predictions[country].ndim >= 2 and 
                            model2_predictions[country].ndim >= 2):
                            model1_pred = model1_predictions[country][:, h][:, q]
                            model2_pred = model2_predictions[country][:, h][:, q]
                            
                            if len(actual) > 5:  # Need minimum samples for meaningful test
                                # Calculate quantile losses
                                model1_qloss = pinball_loss(actual, model1_pred, quantile)
                                model2_qloss = pinball_loss(actual, model2_pred, quantile)
                                
                                # Diebold-Mariano test statistic
                                loss_diff = model1_qloss - model2_qloss
                                
                                # Calculate test statistic
                                mean_diff = np.mean(loss_diff)
                                var_diff = np.var(loss_diff, ddof=1)
                                
                                if var_diff > 0:
                                    dm_stat = mean_diff / np.sqrt(var_diff / len(loss_diff))
                                    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
                                    
                                    dm_results_per_country[country][horizon][quantile] = {
                                        'dm_statistic': dm_stat,
                                        'p_value': p_value,
                                        'significant': p_value < 0.05,
                                        'better_model': model2_name if dm_stat > 0 else model1_name
                                    }
                                    
                                    significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
                                    
                                    country_summary += f"  H{horizon} Q{quantile:.2f}: DM={dm_stat:.3f}, p={p_value:.3f}{significance} -> {dm_results_per_country[country][horizon][quantile]['better_model']}\n"
                                else:
                                    dm_results_per_country[country][horizon][quantile] = {
                                        'dm_statistic': 0,
                                        'p_value': 1.0,
                                        'significant': False,
                                        'better_model': 'Tie'
                                    }
                                    country_summary += f"  H{horizon} Q{quantile:.2f}: No variance in loss differences\n"
                            else:
                                dm_results_per_country[country][horizon][quantile] = {
                                    'dm_statistic': np.nan,
                                    'p_value': np.nan,
                                    'significant': False,
                                    'better_model': 'Insufficient data'
                                }
                                country_summary += f"  H{horizon} Q{quantile:.2f}: Insufficient data for testing\n"
                
                logging.info(country_summary)
                save_output(country_summary, f"dm_test_{country}_{model1_name}_vs_{model2_name}.txt")
        
        return dm_results_per_country
    
    # Perform DM tests for FNN vs LQR
    dm_results_fnn_lqr = compare_models_dm_test(
        fnn_per_country_predictions, 
        lqr_per_country_predictions, 
        fnn_per_country_targets,
        model1_name="FNN",
        model2_name="LQR",
        countries=countries
    )
    
    # ===== PER-COUNTRY QUANTILE LOSS ANALYSIS =====
    logging.info("\n" + "="*50)
    logging.info("PER-COUNTRY QUANTILE LOSS ANALYSIS")
    logging.info("="*50)
    
    def compute_per_country_metrics(model1_predictions, model2_predictions, targets,
                                   model1_name="Model1", model2_name="Model2", countries=None):
        """Calculate Quantile Loss for each country, quantile, and horizon"""
        
        if countries is None:
            countries = list(model1_predictions.keys())
        
        per_country_metrics = {}
        
        for country in countries:
            if (model1_predictions.get(country) is not None and 
                model2_predictions.get(country) is not None and
                targets.get(country) is not None):
                
                per_country_metrics[country] = {}
                
                for h, horizon in enumerate(config['data']['horizons']):
                    per_country_metrics[country][horizon] = {}
                    
                    # Get targets for this country and horizon
                    if targets[country].ndim == 2:
                        actual = targets[country][:, h]
                    else:
                        actual = targets[country]
                    
                    for q, quantile in enumerate(config['data']['quantiles']):
                        per_country_metrics[country][horizon][quantile] = {}
                        
                        # Get predictions for this country, horizon, and quantile
                        if (model1_predictions[country].ndim >= 2 and 
                            model2_predictions[country].ndim >= 2):
                            model1_pred = model1_predictions[country][:, h][:, q]
                            model2_pred = model2_predictions[country][:, h][:, q]
                            
                            # Calculate Quantile Loss for each model
                            for model_name, pred in [(model1_name, model1_pred), (model2_name, model2_pred)]:
                                q_loss = np.mean(pinball_loss(actual, pred, quantile))
                                per_country_metrics[country][horizon][quantile][model_name] = q_loss
        
        return per_country_metrics
    
    # Calculate metrics for FNN vs LQR
    per_country_metrics = compute_per_country_metrics(
        fnn_per_country_predictions,
        lqr_per_country_predictions,
        fnn_per_country_targets,
        model1_name="FNN",
        model2_name="LQR", 
        countries=countries
    )
    
    # Create metrics comparison table
    def create_metrics_comparison_table(per_country_metrics, countries, 
                                       model1_name="Model1", model2_name="Model2"):
        """Create comparison tables and compute aggregate metrics"""
        
        # Create per-country metrics table
        country_metrics_data = []
        
        for country in countries:
            if country in per_country_metrics:
                for horizon in config['data']['horizons']:
                    for quantile in config['data']['quantiles']:
                        if (horizon in per_country_metrics[country] and 
                            quantile in per_country_metrics[country][horizon]):
                            model1_qloss = per_country_metrics[country][horizon][quantile].get(model1_name, np.nan)
                            model2_qloss = per_country_metrics[country][horizon][quantile].get(model2_name, np.nan)
                            
                            if not np.isnan(model1_qloss) and not np.isnan(model2_qloss) and model2_qloss != 0:
                                improvement = ((model2_qloss - model1_qloss) / model2_qloss) * 100
                                better_model = model1_name if model1_qloss < model2_qloss else model2_name
                            else:
                                improvement = np.nan
                                better_model = "N/A"
                            
                            country_metrics_data.append({
                                'Country': country,
                                'Horizon': horizon,
                                'Quantile': quantile,
                                f'{model1_name}_QLoss': model1_qloss,
                                f'{model2_name}_QLoss': model2_qloss,
                                'Improvement_%': improvement,
                                'Better_Model': better_model
                            })
        
        country_metrics_df = pd.DataFrame(country_metrics_data)
        save_dataframe(country_metrics_df.round(4), f"per_country_metrics_{model1_name}_vs_{model2_name}.csv")
        
        # Create aggregate summary
        aggregate_metrics_data = []
        
        for horizon in config['data']['horizons']:
            for quantile in config['data']['quantiles']:
                # Aggregate across all countries
                all_model1_qloss = []
                all_model2_qloss = []
                
                for country in countries:
                    if (country in per_country_metrics and 
                        horizon in per_country_metrics[country] and
                        quantile in per_country_metrics[country][horizon]):
                        model1_val = per_country_metrics[country][horizon][quantile].get(model1_name, np.nan)
                        model2_val = per_country_metrics[country][horizon][quantile].get(model2_name, np.nan)
                        
                        if not np.isnan(model1_val) and not np.isnan(model2_val):
                            all_model1_qloss.append(model1_val)
                            all_model2_qloss.append(model2_val)
                
                if len(all_model1_qloss) > 0:
                    avg_model1_qloss = np.mean(all_model1_qloss)
                    avg_model2_qloss = np.mean(all_model2_qloss)
                    
                    if avg_model2_qloss != 0:
                        improvement = ((avg_model2_qloss - avg_model1_qloss) / avg_model2_qloss) * 100
                        better_model = model1_name if avg_model1_qloss < avg_model2_qloss else model2_name
                    else:
                        improvement = np.nan
                        better_model = "N/A"
                    
                    aggregate_metrics_data.append({
                        'Horizon': horizon,
                        'Quantile': quantile,
                        f'{model1_name}_QLoss_Avg': avg_model1_qloss,
                        f'{model2_name}_QLoss_Avg': avg_model2_qloss,
                        'Improvement_%': improvement,
                        'Better_Model': better_model
                    })
        
        aggregate_df = pd.DataFrame(aggregate_metrics_data)
        save_dataframe(aggregate_df.round(4), f"aggregate_metrics_{model1_name}_vs_{model2_name}.csv")
        
        return country_metrics_df, aggregate_df
    
    # Generate tables for FNN vs LQR comparison
    country_metrics_df, aggregate_df = create_metrics_comparison_table(
        per_country_metrics, countries, 
        model1_name="FNN", model2_name="LQR"
    )
    
    # ===== FINAL SUMMARY =====
    logging.info("\n" + "="*60)
    logging.info("FINAL QUANTILE LOSS EVALUATION SUMMARY")
    logging.info("="*60)
    
    def generate_final_summary(country_metrics_df, per_country_metrics, countries,
                              model1_name="Model1", model2_name="Model2"):
        """Generate final summary and recommendations"""
        
        # Calculate win rates for Quantile Loss
        qloss_win_summary = {
            f'{model1_name}_wins': 0,
            f'{model2_name}_wins': 0,
            'total_comparisons': len(country_metrics_df)
        }
        
        for _, row in country_metrics_df.iterrows():
            if not pd.isna(row[f'{model1_name}_QLoss']) and not pd.isna(row[f'{model2_name}_QLoss']):
                if row[f'{model1_name}_QLoss'] < row[f'{model2_name}_QLoss']:
                    qloss_win_summary[f'{model1_name}_wins'] += 1
                else:
                    qloss_win_summary[f'{model2_name}_wins'] += 1
        
        final_summary = f"""FINAL QUANTILE LOSS EVALUATION SUMMARY ({model1_name} vs {model2_name})
{'='*70}

Quantile Loss Win Rate Summary:
{'-'*50}"""
        
        model1_wins = qloss_win_summary[f'{model1_name}_wins']
        total = qloss_win_summary['total_comparisons']
        if total > 0:
            model1_rate = (model1_wins / total) * 100
            final_summary += f"\nQuantile Loss: {model1_name} wins {model1_wins}/{total} ({model1_rate:.1f}%)"
        
        # Country-by-Country Summary
        final_summary += f"\n\nCountry-by-Country Performance Summary:\n{'-'*60}"
        for country in countries:
            if country in per_country_metrics:
                country_data = country_metrics_df[country_metrics_df['Country'] == country]
                if len(country_data) > 0:
                    country_model1_wins = sum(1 for _, row in country_data.iterrows() 
                                            if row['Better_Model'] == model1_name)
                    country_total = len(country_data)
                    country_model1_rate = (country_model1_wins / country_total) * 100 if country_total > 0 else 0
                    
                    avg_improvement = country_data['Improvement_%'].mean()
                    
                    final_summary += f"\n{country}: {model1_name} wins {country_model1_wins}/{country_total} ({country_model1_rate:.1f}%) | "
                    final_summary += f"Avg improvement: {avg_improvement:+.2f}%"
        
        # Overall recommendation
        if total > 0:
            if model1_rate > 60:
                recommendation = f"{model1_name} shows superior quantile forecasting performance"
            elif model1_rate < 40:
                recommendation = f"{model2_name} shows superior quantile forecasting performance"
            else:
                recommendation = f"Both {model1_name} and {model2_name} show comparable quantile forecasting performance"
        else:
            recommendation = "Insufficient data for meaningful comparison"
        
        final_summary += f"\n\nOverall Recommendation: {recommendation}"
        
        # Performance by Quantile and Horizon
        final_summary += f"\n\nPerformance by Quantile and Horizon:\n{'-'*50}"
        for horizon in config['data']['horizons']:
            final_summary += f"\n\nHorizon {horizon}:"
            for quantile in config['data']['quantiles']:
                horizon_quantile_data = aggregate_df[
                    (aggregate_df['Horizon'] == horizon) & 
                    (aggregate_df['Quantile'] == quantile)
                ]
                if not horizon_quantile_data.empty:
                    row = horizon_quantile_data.iloc[0]
                    final_summary += f"\n  Quantile {quantile:.2f}: {row['Better_Model']} wins | "
                    final_summary += f"Improvement: {row['Improvement_%']:+.2f}%"
        
        final_summary += f"""

Key Insights:
• Quantile Loss is the most relevant metric for quantile forecasting evaluation
• Per-country analysis reveals model performance heterogeneity across countries
• Different quantiles may have varying predictability across forecast horizons
• Country-specific characteristics may favor different modeling approaches

✓ Complete per-country quantile loss evaluation finished ({model1_name} vs {model2_name})!"""
        
        logging.info(final_summary)
        save_output(final_summary, f"final_summary_{model1_name}_vs_{model2_name}.txt")
        
        return final_summary
    
    # Generate summary for FNN vs LQR
    final_summary = generate_final_summary(
        country_metrics_df, per_country_metrics, countries,
        model1_name="FNN", model2_name="LQR"
    )
    
    # ===== COMPLETION =====
    completion_message = f"""
ANALYSIS COMPLETED SUCCESSFULLY!
{'-'*50}

All outputs have been saved to: {output_dir.absolute()}

Generated files:
• Plots: {len(list(plots_dir.glob('*.png')))} visualization files
• Data: {len(list(data_dir.glob('*.csv')))} CSV files  
• Logs: {len(list(logs_dir.glob('*.txt')))} text output files
• Analysis log: {log_file}

Total runtime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    logging.info(completion_message)
    save_output(completion_message, "completion_summary.txt")
    
    print(f"\n✅ Analysis complete! Check the '{output_dir}' directory for all outputs.")

if __name__ == "__main__":
    main()
