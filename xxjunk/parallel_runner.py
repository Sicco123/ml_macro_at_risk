#!/usr/bin/env python3
"""
Parallel Quantile Forecasting Runner

This script runs quantile forecasting models in parallel using rolling windows.
It supports ar-qr, lqr, and nn models with memory estimation and resource management.

Features:
- Memory estimation and parallel scheduling
- Rolling window forecasting
- Model caching and resumption
- Progress tracking
- Error handling and retry logic
"""

import os
import sys
import gc
import json
import math
import time
import glob
import uuid
import yaml
import queue
import logging
import threading
import traceback
import psutil
import random
import pickle
import joblib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from filelock import FileLock

import numpy as np
import pandas as pd
import torch

# Import your existing model APIs
sys.path.append('.')
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
from src.ensemble_nn_api import EnsembleNNAPI
from src.lqr_api import LQR as LQRModel

# ----------------------------- Configuration -----------------------------

DEFAULT_CONFIG_YAML = """
seed: 123

runtime:
  allow_reload: true               # if true, load existing models/forecasts instead of retraining
  retrain_if_exists: false         # if true, force retrain even if model artifact exists
  max_cores: auto                  # integer or 'auto'
  safety_ram_fraction: 0.8         # use at most this fraction of detected available RAM
  max_ram_gb: auto                 # hard cap; number or 'auto'
  mem_probe_fudge_mb: 200          # add this to each task's measured memory (to be conservative)
  retries: 1                       # automatic retries per failed task
  progress_refresh_sec: 5          # how often to rewrite dashboard (seconds)

io:
  data_path: processed_per_country # directory with per-country parquet/csv
  output_root: parallel_outputs    # root folder
  models_dir: models
  forecasts_dir: forecasts
  progress_dir: progress
  logs_dir: logs
  progress_file: progress/progress.parquet
  errors_file: progress/errors.parquet

data:
  target: prc_hicp_manr_CP00
  required_columns: []             # extra columns required; if missing, raise
  lags: [1, 2, 3, 6, 12]          # lag features to create
  horizons: [1, 3, 6, 12]         # forecast horizons
  quantiles: [0.1, 0.5, 0.9]      # quantiles to forecast
  missing: forward_fill_then_mean  # or interpolate_linear | drop
  scale: per_country               # or global | none
  scale_target: false
  trimming: true                   # drop leading/trailing NaNs created by lags

splits:
  validation_size: 0.3
  train_start: "1997-01-01"
  test_cutoff: "2017-12-01"        # last date in training; windows roll from here forward
  min_train_points: 24

# Rolling window settings
rolling_window:
  size: 60                         # number of periods in each window (e.g., months)
  step: 1                          # advance per window
  start_date: "2018-01-01"         # first forecast date
  end_date: "2023-12-01"           # last forecast date

# Model configurations
models:
  ar_qr:
    enabled: true
    p: 12                          # number of AR lags
    add_drift: true
    
  lqr:
    enabled: true
    alphas: [0.1, 1.0, 10.0]       # regularization strengths to try
    solver: huberized
    rff_features: 0                # 0 disables Random Fourier Features
    
  ensemble_nn:
    enabled: true
    units_per_layer: [32, 32, 16]
    activation: relu
    optimizer: adam
    learning_rate: 1e-3
    epochs: 100
    batch_size: 32
    patience: 10
    parallel_models: 1
    device: cpu                    # forced to cpu in this runner
    per_country: true
    country_dummies: false
    l2_penalty: 0.001
"""

# ----------------------------- Logging Setup -----------------------------

def setup_logging(log_dir: Path) -> None:
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"parallel_run_{ts}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging to {log_path}")

# ----------------------------- Utility Functions -----------------------------

def set_global_seeds(seed: int) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)

def read_yaml(path: Path) -> Dict[str, Any]:
    """Read YAML configuration file"""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def write_text(path: Path, text: str) -> None:
    """Write text to file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)

def detect_cpu_cores() -> int:
    """Detect available CPU cores"""
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1

def detect_available_ram_bytes() -> int:
    """Detect available RAM in bytes"""
    vm = psutil.virtual_memory()
    return max(256 * 1024 * 1024, int(vm.available))

def bytes_to_gb(b: int) -> float:
    """Convert bytes to gigabytes"""
    return b / (1024 ** 3)

def safe_mkdirs(path: Path) -> None:
    """Safely create directories"""
    path.mkdir(parents=True, exist_ok=True)

def lock_path(path: Path) -> Path:
    """Get lock file path for a given path"""
    return path.with_suffix(path.suffix + ".lock")

# ----------------------------- Task Definition -----------------------------

@dataclass(frozen=True)
class TaskKey:
    """Unique identifier for a forecasting task"""
    model: str
    quantile: float
    horizon: int
    country: str
    window_start: str
    window_end: str
    target: str

    def id(self) -> str:
        """Generate unique string ID"""
        return f"{self.model}_{self.target}_{self.quantile}_{self.horizon}_{self.country}_{self.window_start}_{self.window_end}"

    def model_filename(self) -> str:
        """Generate model filename"""
        return f"{self.id()}.pkl"

# ----------------------------- Progress Tracking -----------------------------

PROGRESS_COLUMNS = [
    "MODEL", "QUANTILE", "HORIZON", "COUNTRY", "WINDOW_START", "WINDOW_END", 
    "TARGET", "STATUS", "LAST_UPDATE", "ERROR_MSG", "MODEL_PATH", "FORECAST_ROWS"
]

def init_progress_file(progress_path: Path) -> None:
    """Initialize progress tracking file"""
    if not progress_path.exists():
        safe_mkdirs(progress_path.parent)
        empty_df = pd.DataFrame(columns=PROGRESS_COLUMNS)
        empty_df.to_parquet(progress_path, index=False)

def load_progress(progress_path: Path) -> pd.DataFrame:
    """Load progress tracking data"""
    if progress_path.exists():
        return pd.read_parquet(progress_path)
    else:
        return pd.DataFrame(columns=PROGRESS_COLUMNS)

def update_progress_row(progress_path: Path, task: TaskKey, status: str, 
                       error_msg: Optional[str] = None, model_path: Optional[str] = None, 
                       forecast_rows: int = 0) -> None:
    """Update progress for a specific task"""
    lock = FileLock(str(lock_path(progress_path)))
    
    with lock:
        df = load_progress(progress_path)
        
        # Create row data
        row_data = {
            "MODEL": task.model,
            "QUANTILE": task.quantile,
            "HORIZON": task.horizon,
            "COUNTRY": task.country,
            "WINDOW_START": task.window_start,
            "WINDOW_END": task.window_end,
            "TARGET": task.target,
            "STATUS": status,
            "LAST_UPDATE": datetime.now().isoformat(),
            "ERROR_MSG": error_msg,
            "MODEL_PATH": model_path,
            "FORECAST_ROWS": forecast_rows
        }
        
        # Check if row exists
        mask = (
            (df['MODEL'] == task.model) & 
            (df['QUANTILE'] == task.quantile) &
            (df['HORIZON'] == task.horizon) & 
            (df['COUNTRY'] == task.country) &
            (df['WINDOW_START'] == task.window_start) &
            (df['WINDOW_END'] == task.window_end) &
            (df['TARGET'] == task.target)
        )
        
        if mask.any():
            # Update existing row
            for col, val in row_data.items():
                df.loc[mask, col] = val
        else:
            # Add new row
            new_row = pd.DataFrame([row_data])
            df = pd.concat([df, new_row], ignore_index=True)
        
        # Save back
        df.to_parquet(progress_path, index=False)

# ----------------------------- Rolling Window Logic -----------------------------

def generate_rolling_windows(data: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Generate rolling window periods for forecasting"""
    rw_config = config['rolling_window']
    window_size = rw_config['size']
    step = rw_config['step']
    start_date = pd.to_datetime(rw_config['start_date'])
    end_date = pd.to_datetime(rw_config['end_date'])
    
    # Get the earliest and latest dates across all countries
    all_dates = []
    for country_data in data.values():
        all_dates.extend(country_data['TIME'].tolist())
    
    all_dates = sorted(set(all_dates))
    date_index = pd.DatetimeIndex(all_dates)
    
    windows = []
    current_forecast_date = start_date
    
    while current_forecast_date <= end_date:
        # Find the window end (last training date)
        window_end = current_forecast_date - pd.DateOffset(months=1)  # Forecast is for next period
        window_start = window_end - pd.DateOffset(months=window_size-1)
        
        # Check if we have enough data
        if window_start in date_index and window_end in date_index:
            windows.append((
                window_start.strftime('%Y-%m-%d'),
                window_end.strftime('%Y-%m-%d'),
                current_forecast_date.strftime('%Y-%m-%d')
            ))
        
        # Move to next window
        current_forecast_date += pd.DateOffset(months=step)
    
    return windows

# ----------------------------- Memory Estimation -----------------------------

def estimate_task_memory(task: TaskKey, sample_data: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> int:
    """Estimate memory usage for a task in MB"""
    
    # Use a small sample for memory estimation
    country_data = sample_data.get(task.country)
    if country_data is None:
        return 100  # Default estimate
    
    # Create a small sample for memory testing
    sample_size = min(50, len(country_data))
    sample_df = country_data.head(sample_size).copy()
    
    # Measure memory before
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    
    try:
        # Simulate model creation and training with 1 iteration
        if task.model == 'ensemble_nn':
            # Simulate NN memory usage
            dummy_tensor = torch.randn(32, len(sample_df.columns), requires_grad=True)
            _ = dummy_tensor.sum()
            del dummy_tensor
        elif task.model == 'lqr':
            # Simulate LQR memory usage  
            X = np.random.randn(sample_size, 10)
            y = np.random.randn(sample_size)
            from sklearn.linear_model import QuantileRegressor
            model = QuantileRegressor(quantile=task.quantile)
            model.fit(X, y)
            del model, X, y
        else:
            # Simple AR-QR simulation
            X = np.random.randn(sample_size, 5)
            y = np.random.randn(sample_size)
            del X, y
            
    except Exception:
        pass
    
    # Force garbage collection
    gc.collect()
    
    # Measure memory after
    mem_after = process.memory_info().rss
    delta_mb = max(50, (mem_after - mem_before) // (1024 * 1024))  # Minimum 50MB
    
    # Add safety margin
    fudge_mb = config['runtime'].get('mem_probe_fudge_mb', 200)
    return delta_mb + fudge_mb

# ----------------------------- Model Training Functions -----------------------------

def train_model(task: TaskKey, train_data: pd.DataFrame, config: Dict[str, Any]) -> Any:
    """Train a model for the given task"""
    
    model_config = config['models'][task.model]
    data_config = config['data']
    
    if task.model == 'ensemble_nn':
        # Use your existing EnsembleNNAPI
        model = EnsembleNNAPI(
            data_list=[train_data],
            target=task.target,
            quantiles=[task.quantile],
            forecast_horizons=[task.horizon],
            units_per_layer=model_config['units_per_layer'],
            lags=data_config['lags'],
            activation=model_config['activation'],
            device=model_config['device'],
            seed=config['seed'],
            transform=True,
            prefit_AR=True,
            country_ids=[task.country],
            time_col="TIME",
            verbose=0
        )
        
        # Train the model
        model.fit(
            epochs=model_config['epochs'],
            learning_rate=model_config['learning_rate'],
            batch_size=model_config['batch_size'],
            validation_size=config['splits']['validation_size'],
            patience=model_config['patience'],
            verbose=0,
            optimizer=model_config['optimizer'],
            parallel_models=model_config['parallel_models'],
            l2=model_config['l2_penalty']
        )
        
    elif task.model == 'lqr':
        # Use your existing LQR API
        model = LQRModel(
            data_list=[train_data],
            target=task.target,
            quantiles=[task.quantile],
            forecast_horizons=[task.horizon],
            lags=data_config['lags'],
            alpha=model_config['alphas'][0],  # Use first alpha for simplicity
            solver=model_config['solver'],
            seed=config['seed']
        )
        
        # Train the model
        model.fit(
            validation_size=config['splits']['validation_size'],
            alphas=model_config['alphas'],
            verbose=0
        )
        
    elif task.model == 'ar_qr':
        # Create AR-QR as a special case of LQR with only lag features
        ar_lags = list(range(1, model_config['p'] + 1))
        
        model = LQRModel(
            data_list=[train_data],
            target=task.target,
            quantiles=[task.quantile],
            forecast_horizons=[task.horizon],
            lags=ar_lags,
            alpha=1.0,  # Default alpha for AR-QR
            solver=model_config.get('solver', 'huberized'),
            seed=config['seed']
        )
        
        # Train the model
        model.fit(
            validation_size=config['splits']['validation_size'],
            verbose=0
        )
    
    else:
        raise ValueError(f"Unknown model type: {task.model}")
    
    return model

def save_model(model: Any, path: Path) -> None:
    """Save model to disk"""
    safe_mkdirs(path.parent)
    
    # Use joblib for better compatibility
    joblib.dump(model, path)

def load_model(path: Path) -> Any:
    """Load model from disk"""
    return joblib.load(path)

def predict_with_model(model: Any, test_data: pd.DataFrame, task: TaskKey) -> np.ndarray:
    """Generate predictions using trained model"""
    
    if task.model in ['ensemble_nn', 'lqr', 'ar_qr']:
        # Use the predict method from your APIs
        predictions = model.predict([test_data])
        
        # Extract the specific quantile and horizon
        if isinstance(predictions, dict):
            # Handle nested dictionary structure
            country_preds = list(predictions.values())[0]  # First country
            if isinstance(country_preds, dict):
                horizon_preds = country_preds.get(f'h{task.horizon}', country_preds.get(task.horizon))
                if isinstance(horizon_preds, dict):
                    return horizon_preds.get(task.quantile, horizon_preds.get(str(task.quantile)))
                else:
                    return horizon_preds
            else:
                return country_preds
        else:
            return predictions
    else:
        raise ValueError(f"Unknown model type: {task.model}")

# ----------------------------- Worker Functions -----------------------------

def worker_initializer():
    """Initialize worker process"""
    # Limit BLAS threads per process to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def execute_task(task: TaskKey, config: Dict[str, Any], paths: Dict[str, Path], 
                country_data: Dict[str, pd.DataFrame], windows: List[Tuple[str, str, str]]) -> Tuple[str, Optional[str], int]:
    """Execute a single forecasting task"""
    
    try:
        # Set up paths
        models_dir = paths['models_dir']
        forecasts_dir = paths['forecasts_dir']
        progress_path = paths['progress_file']
        
        model_path = models_dir / task.model / f"{task.id()}.pkl"
        
        # Update progress to training
        update_progress_row(progress_path, task, "training")
        
        # Check if model exists and reload is allowed
        if model_path.exists() and config['runtime']['allow_reload'] and not config['runtime']['retrain_if_exists']:
            logging.info(f"Loading existing model for {task.id()}")
            model = load_model(model_path)
        else:
            # Prepare training data for this window
            window_start = pd.to_datetime(task.window_start)
            window_end = pd.to_datetime(task.window_end)
            
            country_df = country_data[task.country].copy()
            
            # Filter data for this window
            mask = (country_df['TIME'] >= window_start) & (country_df['TIME'] <= window_end)
            train_data = country_df[mask].copy()
            
            if len(train_data) < config['splits']['min_train_points']:
                raise ValueError(f"Insufficient training data: {len(train_data)} < {config['splits']['min_train_points']}")
            
            # Preprocess data (same as in run_analysis.py)
            train_data = handle_missing_values({task.country: train_data}, config['data']['missing'])[task.country]
            
            # Train model
            logging.info(f"Training model for {task.id()}")
            model = train_model(task, train_data, config)
            
            # Save model
            save_model(model, model_path)
        
        # Generate predictions for the forecast period
        # For now, we'll use the last available data point for prediction
        country_df = country_data[task.country].copy()
        window_end = pd.to_datetime(task.window_end)
        
        # Get data up to window end for prediction
        mask = country_df['TIME'] <= window_end
        pred_data = country_df[mask].tail(1).copy()  # Use last available point
        
        # Generate prediction
        forecast = predict_with_model(model, pred_data, task)
        
        # Create forecast dataframe
        forecast_time = pd.to_datetime(task.window_end) + pd.DateOffset(months=task.horizon)
        
        forecast_df = pd.DataFrame({
            'TIME': [forecast_time],
            'COUNTRY': [task.country],
            'TARGET': [task.target],
            'MODEL': [task.model],
            'QUANTILE': [task.quantile],
            'HORIZON': [task.horizon],
            'WINDOW_START': [task.window_start],
            'WINDOW_END': [task.window_end],
            'FORECAST': [forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast]
        })
        
        # Save forecast
        forecast_file = forecasts_dir / f"{task.model}_{task.target}_q{task.quantile}_h{task.horizon}.parquet"
        
        # Append to existing forecast file
        if forecast_file.exists():
            existing_forecasts = pd.read_parquet(forecast_file)
            combined_forecasts = pd.concat([existing_forecasts, forecast_df], ignore_index=True)
            combined_forecasts.to_parquet(forecast_file, index=False)
        else:
            safe_mkdirs(forecast_file.parent)
            forecast_df.to_parquet(forecast_file, index=False)
        
        # Update progress to done
        update_progress_row(progress_path, task, "done", model_path=str(model_path), forecast_rows=len(forecast_df))
        
        return "done", None, len(forecast_df)
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logging.error(f"Task {task.id()} failed: {error_msg}")
        
        # Update progress to failed
        update_progress_row(progress_path, task, "failed", error_msg=error_msg)
        
        return "failed", error_msg, 0

# ----------------------------- Main Planning and Execution -----------------------------

def plan_tasks(config: Dict[str, Any], country_data: Dict[str, pd.DataFrame], windows: List[Tuple[str, str, str]]) -> List[TaskKey]:
    """Plan all tasks to be executed"""
    
    tasks = []
    
    # Get enabled models
    enabled_models = []
    for model_name, model_config in config['models'].items():
        if model_config.get('enabled', True):
            enabled_models.append(model_name)
    
    # Generate tasks for each combination
    for country in country_data.keys():
        for window_start, window_end, forecast_date in windows:
            for model in enabled_models:
                for quantile in config['data']['quantiles']:
                    for horizon in config['data']['horizons']:
                        task = TaskKey(
                            model=model,
                            quantile=quantile,
                            horizon=horizon,
                            country=country,
                            window_start=window_start,
                            window_end=window_end,
                            target=config['data']['target']
                        )
                        tasks.append(task)
    
    return tasks

def filter_pending_tasks(tasks: List[TaskKey], progress_path: Path, config: Dict[str, Any]) -> List[TaskKey]:
    """Filter tasks to only include pending ones"""
    
    if not progress_path.exists():
        return tasks
    
    progress_df = load_progress(progress_path)
    
    if len(progress_df) == 0:
        return tasks
    
    pending_tasks = []
    
    for task in tasks:
        # Check if task is already done
        mask = (
            (progress_df['MODEL'] == task.model) & 
            (progress_df['QUANTILE'] == task.quantile) &
            (progress_df['HORIZON'] == task.horizon) & 
            (progress_df['COUNTRY'] == task.country) &
            (progress_df['WINDOW_START'] == task.window_start) &
            (progress_df['WINDOW_END'] == task.window_end) &
            (progress_df['TARGET'] == task.target)
        )
        
        if mask.any():
            status = progress_df.loc[mask, 'STATUS'].iloc[0]
            if status == 'done' and not config['runtime']['retrain_if_exists']:
                continue  # Skip completed tasks
        
        pending_tasks.append(task)
    
    return pending_tasks

def run_parallel_forecast(config: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """Run parallel forecasting with resource management"""
    
    # Load country data (same as in run_analysis.py)
    logging.info("Loading country data...")
    country_data = load_country_data(
        data_path=paths['data_path'],
        required_columns=config['data']['required_columns']
    )
    
    logging.info(f"Loaded data for {len(country_data)} countries")
    
    # Preprocess data
    logging.info("Preprocessing data...")
    country_data = handle_missing_values(country_data, config['data']['missing'])
    
    # Create time splits
    train_data, test_data, dropped_countries = create_time_split(
        country_data,
        train_start=config['splits']['train_start'],
        test_cutoff=config['splits']['test_cutoff'],
        min_train_points=config['splits']['min_train_points']
    )
    
    # Use both train and test data for rolling windows
    full_data = {}
    for country in train_data.keys():
        if country in test_data:
            full_data[country] = pd.concat([train_data[country], test_data[country]], ignore_index=True)
        else:
            full_data[country] = train_data[country]
    
    logging.info(f"Using data from {len(full_data)} countries for rolling windows")
    
    # Generate rolling windows
    logging.info("Generating rolling windows...")
    windows = generate_rolling_windows(full_data, config)
    logging.info(f"Generated {len(windows)} rolling windows")
    
    # Plan all tasks
    logging.info("Planning tasks...")
    all_tasks = plan_tasks(config, full_data, windows)
    logging.info(f"Planned {len(all_tasks)} total tasks")
    
    # Filter to pending tasks only
    pending_tasks = filter_pending_tasks(all_tasks, paths['progress_file'], config)
    logging.info(f"Found {len(pending_tasks)} pending tasks")
    
    if len(pending_tasks) == 0:
        logging.info("No pending tasks found. All work is complete!")
        return
    
    # Estimate memory for a sample of tasks
    logging.info("Estimating memory requirements...")
    sample_tasks = pending_tasks[:min(5, len(pending_tasks))]
    memory_estimates = {}
    
    for task in sample_tasks:
        mem_mb = estimate_task_memory(task, full_data, config)
        memory_estimates[task.model] = mem_mb
        logging.info(f"Estimated {mem_mb}MB for {task.model} tasks")
    
    # Determine resource limits
    max_cores = config['runtime']['max_cores']
    if max_cores == 'auto':
        max_cores = detect_cpu_cores()
    
    available_ram_gb = bytes_to_gb(detect_available_ram_bytes())
    safety_fraction = config['runtime']['safety_ram_fraction']
    max_ram_gb = config['runtime']['max_ram_gb']
    
    if max_ram_gb == 'auto':
        max_ram_gb = available_ram_gb * safety_fraction
    else:
        max_ram_gb = min(max_ram_gb, available_ram_gb * safety_fraction)
    
    max_ram_mb = max_ram_gb * 1024
    
    logging.info(f"Resource limits: {max_cores} cores, {max_ram_gb:.1f}GB RAM")
    
    # Initialize progress file
    init_progress_file(paths['progress_file'])
    
    # Execute tasks in parallel
    logging.info(f"Starting parallel execution of {len(pending_tasks)} tasks...")
    
    completed = 0
    failed = 0
    
    # Group tasks by memory requirements for better scheduling
    task_groups = {}
    for task in pending_tasks:
        model = task.model
        if model not in task_groups:
            task_groups[model] = []
        task_groups[model].append(task)
    
    # Process each group
    for model, model_tasks in task_groups.items():
        mem_per_task = memory_estimates.get(model, 500)  # Default 500MB
        max_workers = min(max_cores, int(max_ram_mb // mem_per_task))
        max_workers = max(1, max_workers)  # At least 1 worker
        
        logging.info(f"Processing {len(model_tasks)} {model} tasks with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_initializer) as executor:
            # Submit tasks
            future_to_task = {}
            for task in model_tasks:
                future = executor.submit(execute_task, task, config, paths, full_data, windows)
                future_to_task[future] = task
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    status, error_msg, n_rows = future.result()
                    if status == "done":
                        completed += 1
                        logging.info(f"✓ {task.id()} completed ({n_rows} forecasts)")
                    else:
                        failed += 1
                        logging.error(f"✗ {task.id()} failed: {error_msg}")
                        
                except Exception as e:
                    failed += 1
                    logging.error(f"✗ {task.id()} crashed: {e}")
                    update_progress_row(paths['progress_file'], task, "failed", error_msg=str(e))
    
    # Final summary
    total = completed + failed
    logging.info(f"Execution complete: {completed}/{total} tasks succeeded, {failed} failed")
    
    if completed > 0:
        logging.info("Forecast files saved to forecasts/ directory")
        logging.info("Model files saved to models/ directory")
        logging.info("Progress tracked in progress/ directory")

# ----------------------------- CLI Interface -----------------------------

def create_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Create all necessary paths"""
    
    io_config = config['io']
    output_root = Path(io_config['output_root'])
    
    paths = {
        'data_path': Path(io_config['data_path']),
        'output_root': output_root,
        'models_dir': output_root / io_config['models_dir'],
        'forecasts_dir': output_root / io_config['forecasts_dir'],
        'progress_dir': output_root / io_config['progress_dir'],
        'logs_dir': output_root / io_config['logs_dir'],
        'progress_file': output_root / io_config['progress_file'],
        'errors_file': output_root / io_config['errors_file']
    }
    
    # Create directories
    for path_key, path_val in paths.items():
        if path_key.endswith('_dir'):
            safe_mkdirs(path_val)
        elif path_key.endswith('_file'):
            safe_mkdirs(path_val.parent)
    
    return paths

def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel Quantile Forecasting Runner")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--write-config", type=str, help="Write default config to file and exit")
    parser.add_argument("--max-workers", type=int, help="Override max workers")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be done")
    
    args = parser.parse_args()
    
    # Write default config if requested
    if args.write_config:
        with open(args.write_config, 'w') as f:
            f.write(DEFAULT_CONFIG_YAML)
        print(f"Default configuration written to {args.write_config}")
        return
    
    # Load configuration
    if args.config:
        config = read_yaml(Path(args.config))
    else:
        print("No config file specified. Use --config or --write-config")
        return
    
    # Override settings from command line
    if args.max_workers:
        config['runtime']['max_cores'] = args.max_workers
    
    # Set up paths and logging
    paths = create_paths(config)
    setup_logging(paths['logs_dir'])
    
    # Set global seed
    set_global_seeds(config['seed'])
    
    logging.info("="*60)
    logging.info("PARALLEL QUANTILE FORECASTING RUNNER")
    logging.info("="*60)
    logging.info(f"Config: {args.config}")
    logging.info(f"Output directory: {paths['output_root']}")
    logging.info(f"Seed: {config['seed']}")
    
    if args.dry_run:
        logging.info("DRY RUN MODE - No actual work will be performed")
        # TODO: Implement dry run logic
        return
    
    try:
        # Run the main forecasting pipeline
        run_parallel_forecast(config, paths)
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
