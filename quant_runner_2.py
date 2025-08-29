#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-Core Rolling-Window Quantile Forecast Runner with File-based Task Coordination

IMPORTANT:
- This runner processes tasks ONE BY ONE on a SINGLE CORE only
- Multiple instances can run simultaneously without conflicts via file locking
- Each instance claims a task, processes it, and moves to the next available task
- Thread/core limits are enforced to prevent parallel execution within the process

THREAD LIMITING:
- Set environment variables to limit threads BEFORE importing numpy/scipy/torch
- Uses file locks to coordinate between multiple script instances
- Progress tracking prevents duplicate work across instances

CONFIG: Same YAML format as quant_runner.py but ignores all parallelism settings
"""

# CRITICAL: Set thread limits BEFORE any imports that might use BLAS/LAPACK
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

# Standard imports
import sys
import gc
import time
import json
import yaml
import psutil
import random
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from filelock import FileLock
from tqdm import tqdm
import pyarrow.parquet as pq

# Global model cache for sharing trained models across countries in the same window
_global_model_cache = {}

# Global data cache for sharing loaded dataframes across tasks
_global_data_cache = {}

# Your model APIs
try:
    from src.ensemble_nn_api import EnsembleNNAPI
    from src.lqr_api import LQRModel
except Exception as e:
    print(f"[IMPORT ERROR] Could not import your model APIs: {e}", file=sys.stderr)
    raise

# ----------------------------- Force Single Core -----------------------------

def enforce_single_core():
    """Enforce single-core operation at runtime"""
    # Set CPU affinity to single core if possible
    try:
        import psutil
        p = psutil.Process()
        if hasattr(p, 'cpu_affinity'):
            current_affinity = p.cpu_affinity()
            if len(current_affinity) > 1:
                # Pin to first available core
                p.cpu_affinity([current_affinity[0]])
                logging.info(f"Pinned process to CPU core {current_affinity[0]}")
    except Exception as e:
        logging.warning(f"Could not set CPU affinity: {e}")

    # Additional thread enforcement
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # Try to limit torch threads if available
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass

# ----------------------------- Logging -----------------------------

def setup_logging(log_dir: Path, instance_id: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"single_core_{instance_id}_{ts}.log"
    
    # Clear any existing handlers to avoid conflicts
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create file handler with explicit encoding and buffering
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        f"[%(asctime)s][{instance_id}][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    # Force immediate write to ensure log file is created
    logging.info(f"Single-core runner {instance_id} logging to {log_path}")
    file_handler.flush()
    
    print(f"[SETUP] Log file created at: {log_path}")  # Debug print

# ----------------------------- Utils -----------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

def read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def safe_mkdirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def lock_path(p: Path) -> Path:
    return p.with_suffix(p.suffix + ".lock")

# ----------------------------- Data IO (with caching) -----------------------------

def load_country_files(data_dir: Path) -> Dict[str, Path]:
    files: List[Path] = []
    for ext in ("*.parquet", "*.pq", "*.feather", "*.csv"):
        files.extend(data_dir.glob(ext))
    if not files:
        raise FileNotFoundError(f"No data files found under {data_dir}")
    mapping: Dict[str, Path] = {}
    for f in files:
        stem = f.stem
        country = stem.split("__")[0].upper()
        mapping[country] = f
    return mapping

def read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".parquet", ".pq", ".feather"]:
        df = pd.read_parquet(path)
        df.reset_index(inplace=True)
        return df
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

def load_and_cache_all_country_data(data_dir: Path, cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Load all country data files into memory and cache them.
    Apply common preprocessing once during loading.
    """
    global _global_data_cache
    
    if _global_data_cache:
        logging.info("Using existing cached country data")
        return _global_data_cache
    
    logging.info("Loading and caching all country data...")
    
    countries = load_country_files(data_dir)
    time_col = cfg["data"].get("time_col", "TIME")
    missing_strategy = cfg["data"].get("missing", "forward_fill_then_mean")
    
    cached_data = {}
    
    for country, country_file in countries.items():
        try:
            #logging.info(f"Loading {country} from {country_file}")
            df = read_df(country_file)
            df = ensure_time_sorted(df, time_col)
            df = handle_missing(df, missing_strategy)
            cached_data[country] = df
            #logging.info(f"Cached {country}: {len(df)} rows, columns: {list(df.columns)}")
        except Exception as e:
            logging.error(f"Failed to load {country} from {country_file}: {e}")
            raise
    
    _global_data_cache = cached_data
    logging.info(f"Successfully cached data for {len(cached_data)} countries")
    
    return cached_data

def get_cached_country_data(country: str) -> pd.DataFrame:
    """
    Get cached country data. Returns a copy to avoid modifying the cache.
    """
    global _global_data_cache
    
    if country not in _global_data_cache:
        raise KeyError(f"Country {country} not found in cache. Available: {list(_global_data_cache.keys())}")
    
    return _global_data_cache[country].copy()

def clear_data_cache():
    """Clear the global data cache to free memory if needed"""
    global _global_data_cache
    _global_data_cache.clear()
    logging.info("Cleared global data cache")

def ensure_time_sorted(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out.sort_values(time_col, inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out

def handle_missing(df: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "forward_fill_then_mean":
        df = df.ffill()
        df = df.fillna(df.mean(numeric_only=True))
    elif how == "interpolate_linear":
        df = df.interpolate(method="linear")
    elif how == "drop":
        df = df.dropna()
    return df

# ----------------------------- Rolling Windows (same as original) -----------------------------

def generate_windows(
    dates: pd.Series,
    size: int,
    step: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    horizon: int
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    idx = pd.Index(pd.to_datetime(dates))
    out: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for i in range(len(idx)):
        fs = idx[i]
        if fs < start_date or fs > end_date:
            continue
        j = i - (size - 1)
        if j < 0:
            continue
        win_start = idx[j]
        win_end = idx[i]
        k = i + horizon
        if k >= len(idx):
            break
        ftime = idx[k]
        out.append((win_start, win_end, ftime))
        i += step - 1
    return out

# ----------------------------- Task Management with File Locking -----------------------------

PROGRESS_COLUMNS = [
    "MODEL", "NN_VERSION", "QUANTILE", "HORIZON", "COUNTRY",
    "WINDOW_START", "WINDOW_END", "STATUS", "LAST_UPDATE",
    "ERROR_MSG", "MODEL_PATH", "FORECAST_ROWS", "IS_GLOBAL", 
    "INSTANCE_ID", "CLAIMED_AT"
]

@dataclass(frozen=True)
class TaskKey:
    model: str
    nn_version: Optional[str]
    quantile: float
    horizon: int
    country: str
    window_start: str
    window_end: str
    is_global: bool = False

    def id(self) -> str:
        nv = self.nn_version or "-"
        global_suffix = "-global" if self.is_global else ""
        return f"{self.model}{global_suffix}|{nv}|q={self.quantile}|h={self.horizon}|{self.country}|{self.window_start}_{self.window_end}"

def load_progress(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=PROGRESS_COLUMNS)

def acquire_lock_with_backoff(lock_path: Path, max_retries: int = 5, base_timeout: float = 2.0) -> FileLock:
    """Acquire lock with exponential backoff"""
    for attempt in range(max_retries):
        try:
            timeout = base_timeout * (2 ** attempt) + random.uniform(0, 1)
            lock = FileLock(str(lock_path), timeout=min(timeout, 30.0))
            lock.acquire()
            return lock
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = random.uniform(0.1, 0.5) * (2 ** attempt)
            time.sleep(wait_time)
    raise RuntimeError("Could not acquire lock after retries")

def filter_completed_tasks(all_tasks: List[TaskKey], progress_path: Path) -> List[TaskKey]:
    """
    Remove tasks that are already completed from the task list using separate files.
    """
    base_path = progress_path.parent
    completed_path = base_path / "completed_tasks.parquet"
    
    if not completed_path.exists():
        return all_tasks
    
    try:
        completed_df = pd.read_parquet(completed_path)
        if completed_df.empty:
            return all_tasks
        
        # Create set of completed task identifiers for fast lookup (vectorized)
        completed_tasks = set()
        if not completed_df.empty:
            for _, row in completed_df.iterrows():
                task_id = (
                    row["MODEL"],
                    row["NN_VERSION"] or "",
                    row["QUANTILE"],
                    row["HORIZON"],
                    row["COUNTRY"],
                    row["WINDOW_START"],
                    row["WINDOW_END"],
                    row.get("IS_GLOBAL", False)
                )
                completed_tasks.add(task_id)
        
        # Filter out completed tasks
        remaining_tasks = []
        for task in all_tasks:
            task_id = (
                task.model,
                task.nn_version or "",
                task.quantile,
                task.horizon,
                task.country,
                task.window_start,
                task.window_end,
                task.is_global
            )
            if task_id not in completed_tasks:
                remaining_tasks.append(task)
        
        return remaining_tasks
    
    except Exception as e:
        logging.warning(f"Failed to filter completed tasks: {e}. Processing all tasks.")
        return all_tasks

def get_worker_chunk(
    all_tasks: List[TaskKey], 
    worker_index: int, 
    num_workers: int, 
    batch_size: int
) -> List[TaskKey]:
    """
    Distribute tasks evenly among workers.
    Each worker gets approximately len(all_tasks) / num_workers tasks.
    """
    if not all_tasks:
        return []
    
    # Calculate tasks per worker for even distribution
    tasks_per_worker = len(all_tasks) // num_workers
    tasks_assigned = max(tasks_per_worker, batch_size)
    remainder = len(all_tasks) % num_workers
    
    # Calculate start and end indices for this worker
    start_idx = worker_index * tasks_per_worker
    
    # Distribute remainder tasks to first few workers
    if worker_index < remainder:
        start_idx += worker_index
        end_idx = start_idx + tasks_assigned + 1
    else:
        start_idx += remainder
        end_idx = start_idx + tasks_assigned    
    
    # Ensure we don't go beyond the task list
    end_idx = min(end_idx, len(all_tasks))
    
    worker_chunk = all_tasks[start_idx:end_idx]
    
    logging.info(f"Worker {worker_index} assigned tasks {start_idx}-{end_idx-1} ({len(worker_chunk)} tasks out of {len(all_tasks)} total)")
    
    return worker_chunk

def claim_task_batch(
    progress_path: Path, 
    all_tasks: List[TaskKey], 
    instance_id: str,
    batch_size: int = 5,
    timeout_minutes: int = 30
) -> List[TaskKey]:
    """
    Atomically claim a batch of available tasks using separate files for better performance.
    Returns list of claimed tasks (may be fewer than batch_size if not enough available).
    """
    # Randomize task order to reduce contention
    shuffled_tasks = all_tasks.copy()
    random.shuffle(shuffled_tasks)
    
    # Load completed and claimed tasks from separate files
    base_path = progress_path.parent
    completed_path = base_path / "completed_tasks.parquet"
    claimed_path = base_path / "claimed_tasks.parquet"
    failed_path = base_path / "failed_tasks.parquet"
    
    # Build exclusion set from separate files (much faster)
    active_task_ids = set()
    current_time = datetime.now()
    timeout_seconds = timeout_minutes * 60
    
    # Load completed tasks (never changes, no lock needed if read-only)
    if completed_path.exists():
        try:
            completed_df = pd.read_parquet(completed_path)
            if not completed_df.empty:
                for _, row in completed_df.iterrows():
                    task_id = (
                        row["MODEL"], row["NN_VERSION"] or "", row["QUANTILE"], 
                        row["HORIZON"], row["COUNTRY"], row["WINDOW_START"], 
                        row["WINDOW_END"], row.get("IS_GLOBAL", False)
                    )
                    active_task_ids.add(task_id)
        except Exception as e:
            logging.warning(f"Failed to load completed tasks: {e}")
    
    # Load failed tasks (rarely changes)
    if failed_path.exists():
        try:
            failed_df = pd.read_parquet(failed_path)
            if not failed_df.empty:
                for _, row in failed_df.iterrows():
                    task_id = (
                        row["MODEL"], row["NN_VERSION"] or "", row["QUANTILE"], 
                        row["HORIZON"], row["COUNTRY"], row["WINDOW_START"], 
                        row["WINDOW_END"], row.get("IS_GLOBAL", False)
                    )
                    active_task_ids.add(task_id)
        except Exception as e:
            logging.warning(f"Failed to load failed tasks: {e}")
    
    # Load claimed tasks with timeout check (requires brief lock)
    if claimed_path.exists():
        lock_file = lock_path(claimed_path)
        try:
            lock = acquire_lock_with_backoff(lock_file, max_retries=1, base_timeout=0.2)
            try:
                with lock:
                    claimed_df = pd.read_parquet(claimed_path)
                    if not claimed_df.empty:
                        # Vectorized timeout check
                        claimed_times = pd.to_datetime(claimed_df["CLAIMED_AT"])
                        time_diffs = (current_time - claimed_times).dt.total_seconds()
                        valid_mask = time_diffs < timeout_seconds
                        
                        valid_claimed = claimed_df[valid_mask]
                        for _, row in valid_claimed.iterrows():
                            task_id = (
                                row["MODEL"], row["NN_VERSION"] or "", row["QUANTILE"],
                                row["HORIZON"], row["COUNTRY"], row["WINDOW_START"],
                                row["WINDOW_END"], row.get("IS_GLOBAL", False)
                            )
                            active_task_ids.add(task_id)
                        
                        # Clean up expired claims while we have the lock
                        if len(valid_claimed) < len(claimed_df):
                            temp_path = claimed_path.with_suffix(f".tmp_{instance_id}_{os.getpid()}")
                            valid_claimed.to_parquet(temp_path, index=False)
                            temp_path.rename(claimed_path)
                            
            finally:
                try:
                    lock.release()
                except:
                    pass
        except Exception as e:
            logging.warning(f"Failed to load claimed tasks: {e}")
    
    # Find available tasks (fast, no locks)
    claimed_tasks = []
    new_rows = []
    now_iso = current_time.isoformat()
    
    for task in shuffled_tasks:
        if len(claimed_tasks) >= batch_size:
            break
        
        task_id = (
            task.model, task.nn_version or "", task.quantile,
            task.horizon, task.country, task.window_start,
            task.window_end, task.is_global
        )
        
        if task_id not in active_task_ids:
            row = {
                "MODEL": task.model,
                "NN_VERSION": task.nn_version,
                "QUANTILE": task.quantile,
                "HORIZON": task.horizon,
                "COUNTRY": task.country,
                "WINDOW_START": task.window_start,
                "WINDOW_END": task.window_end,
                "STATUS": "claimed",
                "LAST_UPDATE": now_iso,
                "ERROR_MSG": None,
                "MODEL_PATH": None,
                "FORECAST_ROWS": 0,
                "IS_GLOBAL": task.is_global,
                "INSTANCE_ID": instance_id,
                "CLAIMED_AT": now_iso
            }
            
            new_rows.append(row)
            claimed_tasks.append(task)
            active_task_ids.add(task_id)  # Prevent duplicates in same batch
    
    # Write claimed tasks to separate file (minimal lock time)
    if claimed_tasks:
        _append_to_separate_file(claimed_path, new_rows, instance_id)
        logging.info(f"Claimed batch of {len(claimed_tasks)} tasks")
    
    return claimed_tasks

def update_task_batch_status(
    progress_path: Path,
    task_updates: List[Tuple[TaskKey, str, Optional[str], Optional[Path], int]],
    instance_id: str
) -> None:
    """
    Update status for a batch of tasks using separate files for completed vs claimed tasks.
    This reduces lock conflicts by avoiding updates to a single monolithic file.
    task_updates: List of (task, status, error_msg, model_path, forecast_rows)
    """
    if not task_updates:
        return
    
    # Separate updates by status type
    completed_tasks = []
    claimed_tasks = []
    failed_tasks = []
    
    current_time = datetime.now().isoformat()
    
    for task, status, error_msg, model_path, forecast_rows in task_updates:
        row = {
            "MODEL": task.model,
            "NN_VERSION": task.nn_version,
            "QUANTILE": task.quantile,
            "HORIZON": task.horizon,
            "COUNTRY": task.country,
            "WINDOW_START": task.window_start,
            "WINDOW_END": task.window_end,
            "STATUS": status,
            "LAST_UPDATE": current_time,
            "ERROR_MSG": error_msg,
            "MODEL_PATH": str(model_path) if model_path else None,
            "FORECAST_ROWS": forecast_rows,
            "IS_GLOBAL": task.is_global,
            "INSTANCE_ID": instance_id,
            "CLAIMED_AT": current_time if status == "claimed" else None
        }
        
        if status == "done":
            completed_tasks.append(row)
        elif status == "claimed":
            claimed_tasks.append(row)
        elif status == "failed":
            failed_tasks.append(row)
    
    # Write to separate files to reduce lock conflicts
    base_path = progress_path.parent
    
    # Append completed tasks (write-only, no conflicts)
    if completed_tasks:
        completed_path = base_path / "completed_tasks.parquet"
        _append_to_separate_file(completed_path, completed_tasks, instance_id)
    
    # Append failed tasks (write-only, no conflicts)  
    if failed_tasks:
        failed_path = base_path / "failed_tasks.parquet"
        _append_to_separate_file(failed_path, failed_tasks, instance_id)
    
    # For claimed tasks, still need some coordination but much less contention
    if claimed_tasks:
        claimed_path = base_path / "claimed_tasks.parquet"
        _append_to_separate_file(claimed_path, claimed_tasks, instance_id)

def _append_to_separate_file(file_path: Path, rows: List[Dict], instance_id: str) -> None:
    """Append rows to a separate file with minimal locking."""
    if not rows:
        return
        
    new_df = pd.DataFrame(rows)
    lock_file = lock_path(file_path)
    
    for retry in range(2):  # Fewer retries since less contention
        try:
            lock = acquire_lock_with_backoff(lock_file, max_retries=1, base_timeout=0.1)
            
            try:
                with lock:
                    if file_path.exists():
                        existing_df = pd.read_parquet(file_path)
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    else:
                        combined_df = new_df
                    
                    # Atomic write
                    temp_path = file_path.with_suffix(f".tmp_{instance_id}_{os.getpid()}")
                    combined_df.to_parquet(temp_path, index=False)
                    temp_path.rename(file_path)
                    return  # Success
                    
            finally:
                try:
                    lock.release()
                except:
                    pass
                    
        except Exception as e:
            if retry == 0:
                time.sleep(random.uniform(0.01, 0.05))  # Very short wait
            else:
                logging.warning(f"Failed to append to {file_path.name}: {e}")
                return

def update_task_status(
    progress_path: Path,
    task: TaskKey,
    status: str,
    instance_id: str,
    error_msg: Optional[str] = None,
    model_path: Optional[Path] = None,
    forecast_rows: int = 0
) -> None:
    """Update single task status - wrapper around batch update for compatibility"""
    update_task_batch_status(
        progress_path, 
        [(task, status, error_msg, model_path, forecast_rows)], 
        instance_id
    )

# ----------------------------- Forecast Output (optimized) -----------------------------

def append_forecasts(out_path: Path, rows: pd.DataFrame) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path(out_path)
    
    for retry in range(3):
        try:
            lock = acquire_lock_with_backoff(lock_file, max_retries=2, base_timeout=0.5)
            
            try:
                with lock:
                    if out_path.exists():
                        existing = pq.ParquetFile(out_path).read().to_pandas()
                        combined = pd.concat([existing, rows], ignore_index=True)
                        combined = combined.drop_duplicates(
                            subset=["TIME", "COUNTRY", "HORIZON", "QUANTILE", "MODEL", "NN_VERSION"], 
                            keep="last"
                        )
                        combined.sort_values(["TIME", "COUNTRY", "HORIZON", "QUANTILE"], inplace=True)
                        
                        # Atomic write
                        temp_path = out_path.with_suffix(out_path.suffix + f".tmp_{os.getpid()}")
                        combined.to_parquet(temp_path, index=False)
                        temp_path.rename(out_path)
                    else:
                        rows.to_parquet(out_path, index=False)
                    
                    return  # Success
                    
            finally:
                try:
                    lock.release()
                except:
                    pass
                    
        except Exception as e:
            logging.warning(f"Forecast append attempt {retry + 1} failed: {e}")
            if retry < 2:
                wait_time = random.uniform(0.05, 0.15) * (2 ** retry)
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to append forecasts after {retry + 1} attempts: {e}")

def append_forecasts_batch(forecast_batch: List[Tuple[float, int, pd.DataFrame]], paths: Dict[str, Path]) -> None:
    """
    Append a batch of forecasts grouped by quantile and horizon to reduce I/O operations.
    forecast_batch: List of (quantile, horizon, forecast_df) tuples
    """
    if not forecast_batch:
        return
    
    # Group forecasts by (quantile, horizon) to minimize file operations
    grouped_forecasts = {}
    for quantile, horizon, forecast_df in forecast_batch:
        key = (quantile, horizon)
        if key not in grouped_forecasts:
            grouped_forecasts[key] = []
        grouped_forecasts[key].append(forecast_df)
    
    # Append each group to its respective file
    for (quantile, horizon), df_list in grouped_forecasts.items():
        if not df_list:
            continue
        
        # Combine all forecasts for this (quantile, horizon) pair
        combined_df = pd.concat(df_list, ignore_index=True)
        
        # Append to the appropriate file
        out_path = paths["forecasts_root"] / f"q={quantile}" / f"h={horizon}" / "rolling_window.parquet"
        append_forecasts(out_path, combined_df)
        
        logging.debug(f"Appended batch of {len(combined_df)} forecasts to q={quantile}, h={horizon}")



# ----------------------------- Model Functions (adapted from original) -----------------------------

def _build_lqr(
    train_df: pd.DataFrame,
    target: str,
    quantile: float,
    horizon: int,
    lags: List[int],
    params: Dict[str, Any],
    seed: int,
    ar_only: bool
) -> LQRModel:
    if ar_only:
        data_list = [train_df[[c for c in train_df.columns if c in ["TIME", target]]].copy()]
    else:
        data_list = [train_df.copy()]

    mdl = LQRModel(
        data_list=data_list,
        target=target,
        quantiles=[quantile],
        forecast_horizons=[horizon],
        lags=lags,
        alpha=float(params.get("alpha", 0.0)),
        fit_intercept=bool(params.get("fit_intercept", True)),
        solver=params.get("solver", "huberized"),
        seed=seed
    )

    if params.get("use_cv", False):
        alphas = params.get("alphas", [])
        if alphas:
            splits = int(params.get("cv_splits", 5))
            try:
                mdl.k_fold_validation(alphas=alphas, n_splits=splits)
            except Exception as e:
                logging.warning(f"[{('AR-QR' if ar_only else 'LQR')}] CV failed: {e}")

    mdl.fit()
    return mdl

def _build_nn(
    train_df: pd.DataFrame,
    target: str,
    quantile: float,
    horizon: int,
    lags: List[int],
    nn_params: Dict[str, Any],
    seed: int
) -> EnsembleNNAPI:
    mdl = EnsembleNNAPI(
        data_list=[train_df.copy()],
        target=target,
        quantiles=[quantile],
        forecast_horizons=[horizon],
        units_per_layer=list(nn_params.get("units_per_layer", [32, 32])),
        lags=lags,
        activation=nn_params.get("activation", "relu"),
        device=nn_params.get("device", "cpu"),
        seed=seed,
        transform=True,
        prefit_AR=bool(nn_params.get("prefit_AR", True)),
        time_col="TIME",
        verbose=0
    )

    epochs = int(nn_params.get("epochs", 100))
    mdl.fit(
        epochs=epochs,
        learning_rate=float(nn_params.get("learning_rate", 1e-3)),
        batch_size=int(nn_params.get("batch_size", 64)),
        validation_size=float(nn_params.get("validation_size", 0.2)),
        patience=int(nn_params.get("patience", 20)),
        verbose=0,
        optimizer=nn_params.get("optimizer", "adam"),
        parallel_models=1,  # Force single model in NN ensemble
        l2=float(nn_params.get("l2_penalty", 0.0)),
        return_validation_loss=False,
        return_train_loss=False,
        shuffle=True
    )
    return mdl

def _build_nn_global(
    train_data_list: List[pd.DataFrame],
    target: str,
    quantile: float,
    horizon: int,
    lags: List[int],
    nn_params: Dict[str, Any],
    seed: int,
    country_names: List[str]
) -> EnsembleNNAPI:
    """
    Construct and fit YOUR EnsembleNNAPI on multiple countries' data (global model).
    """
    # YOUR API expects a list of dataframes (countries). For a global model we pass all countries' data.
    mdl = EnsembleNNAPI(
        data_list=train_data_list,
        target=target,
        quantiles=[quantile],
        forecast_horizons=[horizon],
        units_per_layer=list(nn_params.get("units_per_layer", [32, 32])),
        lags=lags,
        activation=nn_params.get("activation", "relu"),
        device=nn_params.get("device", "cpu"),
        seed=seed,
        transform=True,
        prefit_AR=bool(nn_params.get("prefit_AR", True)),  # Enable for global models
        country_ids=country_names,  # Pass country names for global model
        time_col="TIME",
        verbose=0
    )

    epochs = int(nn_params.get("epochs", 100))
    mdl.fit(
        epochs=epochs,
        learning_rate=float(nn_params.get("learning_rate", 1e-3)),
        batch_size=int(nn_params.get("batch_size", 64)),
        validation_size=float(nn_params.get("validation_size", 0.2)),
        patience=int(nn_params.get("patience", 20)),
        verbose=0,
        optimizer=nn_params.get("optimizer", "adam"),
        parallel_models=1,  # Force single model in NN ensemble
        l2=float(nn_params.get("l2_penalty", 0.0)),
        return_validation_loss=False,
        return_train_loss=False,
        shuffle=True
    )
    return mdl

def _predict_lqr_single(
    mdl: LQRModel,
    full_country_df: pd.DataFrame,
    target: str,
    time_col: str,
    window_end: pd.Timestamp,
    horizon: int,
    quantile: float,
    ar_only: bool = False
) -> float:
    test_df = full_country_df.loc[full_country_df[time_col] >= window_end].copy()
    
    if ar_only:
        test_df = test_df[[c for c in test_df.columns if c in ["TIME", target]]].copy()
   
    preds, _targets = mdl.predict([test_df])
    arr = np.asarray(preds)
    
    if arr.ndim == 3 and arr.shape[1] >= 1 and arr.shape[2] >= 1:
        h_idx = max(0, min(arr.shape[1] - 1, horizon - 1))
        q_idx = 0  # Single quantile
        return float(arr[0, h_idx, q_idx])
    elif arr.ndim == 2 and arr.shape[1] >= 1:
        return float(arr[0, 0])
    
    raise RuntimeError("Could not infer prediction array shape from LQRModel.predict output.")

def _predict_nn_single(
    mdl: EnsembleNNAPI,
    full_country_df: pd.DataFrame,
    country_id: str,
    time_col: str,
    window_end: pd.Timestamp,
    horizon: int,
    quantile: float
) -> float:
    test_df = full_country_df.loc[full_country_df[time_col] >= window_end].copy()
    preds, _targets = mdl.predict_per_country(test_df, country_id)
    arr = np.asarray(preds)
    
    if arr.ndim == 3:
        if arr.shape[1] >= 1 and arr.shape[2] >= 1:
            h_idx = max(0, min(arr.shape[1] - 1, horizon - 1))
            q_idx = 0  # Single quantile
            return float(arr[0, h_idx, q_idx])
    elif arr.ndim == 2 and arr.shape[1] >= 1:
        return float(arr[0, 0])

    raise RuntimeError("Could not infer prediction array shape from EnsembleNNAPI.predict_per_country output.")

# ----------------------------- Config Helpers (same as original) -----------------------------

def cfg_for_lqr(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for m in cfg.get("models", []):
        if m.get("type") == "lqr":
            return m.get("params", {})
    return {}

def cfg_for_arqr(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for m in cfg.get("models", []):
        if m.get("type") == "ar-qr":
            return m.get("params", {})
    return {}

def cfg_for_nn_versions(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    for m in cfg.get("models", []):
        if m.get("type") == "nn" and m.get("enabled", False):
            versions = m.get("versions", [])
            return [(v.get("name"), v.get("params", {})) for v in versions]
    return []

def cfg_for_nn_version(cfg: Dict[str, Any], name: Optional[str]) -> Dict[str, Any]:
    versions = cfg_for_nn_versions(cfg)
    for n, p in versions:
        if n == name:
            return p
    return versions[0][1] if versions else {}

def cfg_for_nn_per_country(cfg: Dict[str, Any]) -> bool:
    """Check if NN models should be trained per country (True) or globally (False)"""
    for m in cfg.get("models", []):
        if m.get("type") == "nn" and m.get("enabled", False):
            return bool(m.get("per_country", True))  # Default to True (per-country)
    return True

# ----------------------------- Task Execution (single task) -----------------------------

def execute_single_task_batch(task: TaskKey, cfg: Dict[str, Any], paths: Dict[str, Path], instance_id: str) -> Tuple[bool, Optional[Tuple[TaskKey, str, Optional[str], Optional[Path], int]], Optional[pd.DataFrame]]:
    """
    Execute a single task and return success status, task update info, and forecast data for batch processing.
    Returns (success, task_update_tuple, forecast_df) where:
    - task_update_tuple is (task, status, error_msg, model_path, forecast_rows)
    - forecast_df is the DataFrame to be batched for forecast output
    """
    try:
        #logging.info(f"Processing task: {task.id()}")
        
        time_col = cfg["data"].get("time_col", "TIME")
        target = cfg["data"]["target"]
        lags = cfg["data"].get("lags")
        
        ws = pd.Timestamp(task.window_start)
        we = pd.Timestamp(task.window_end)
        
        # Handle global models differently
        if task.is_global and task.model == "nn":
            return execute_global_nn_task_batch(task, cfg, paths, instance_id)
        
        # Per-country model logic (existing logic)
        # Model artifact path
        model_dir = paths["models_root"] / task.model / (f"nn_version={task.nn_version}" if task.nn_version else "default") / f"country={task.country}" / f"q={task.quantile}" / f"h={task.horizon}" / f"window_{task.window_start}_{task.window_end}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        
        # Try loading existing model
        model_obj = None
        if cfg["runtime"].get("allow_reload", True) and model_path.exists():
            try:
                model_obj = joblib.load(model_path)
                #logging.info(f"Loaded model from {model_path}")
            except Exception as e:
                logging.warning(f"Failed to load model: {e}")
        
        # Train if needed
        if model_obj is None:
            df = get_cached_country_data(task.country)
            train_df = df.loc[(df[time_col] >= ws) & (df[time_col] <= we)].copy()

            if len(train_df) < int(cfg["splits"].get("min_train_points", 12)):
                raise RuntimeError(f"Insufficient training points: {len(train_df)}")
            
            if target not in df.columns:
                raise RuntimeError(f"Target '{target}' not found in {task.country}")
            
            # Build model based on type
            if task.model == "lqr":
                model_obj = _build_lqr(train_df, target, task.quantile, task.horizon, lags, cfg_for_lqr(cfg), seed=int(cfg.get("seed", 42)), ar_only=False)
            elif task.model == "ar-qr":
                model_obj = _build_lqr(train_df, target, task.quantile, task.horizon, lags, cfg_for_arqr(cfg), seed=int(cfg.get("seed", 42)), ar_only=True)
            elif task.model == "nn":
                nn_params = cfg_for_nn_version(cfg, task.nn_version)
                model_obj = _build_nn(train_df, target, task.quantile, task.horizon, lags, nn_params, seed=int(cfg.get("seed", 42)))
            else:
                raise ValueError(f"Unknown model type: {task.model}")
            
            # Save model
            if cfg["runtime"].get("save_models", True):
                try:
                    joblib.dump(model_obj, model_path)
                    #logging.info(f"Saved model to {model_path}")
                except Exception as e:
                    logging.warning(f"Could not save model: {e}")
            
        # Generate forecast
        df = get_cached_country_data(task.country)
        
        # Find forecast time
        idx_end = df.index[df[time_col] == we]
        if len(idx_end) == 0:
            raise RuntimeError("window_end not found in data")
        
        i = int(idx_end[0])
        k = i + task.horizon
        if k >= len(df):
            raise RuntimeError("Forecast horizon beyond available data")
        
        t_forecast = pd.Timestamp(df.iloc[k][time_col])
        true_val = float(df.iloc[k][target]) if target in df.columns else np.nan
        
        # Predict
        if task.model == "lqr":
            yhat = _predict_lqr_single(model_obj, df, target, time_col, we, task.horizon, task.quantile, ar_only=False)
        elif task.model == "ar-qr":
            yhat = _predict_lqr_single(model_obj, df, target, time_col, we, task.horizon, task.quantile, ar_only=True)
        elif task.model == "nn":
            yhat = _predict_nn_single(model_obj, df, task.country, time_col, we, task.horizon, task.quantile)
        else:
            raise ValueError(f"Unknown model for prediction: {task.model}")
        
        # Create output DataFrame for batching
        out_df = pd.DataFrame({
            "TIME": [t_forecast],
            "COUNTRY": [task.country],
            "TRUE_DATA": [true_val],
            "FORECAST": [float(yhat)],
            "HORIZON": [task.horizon],
            "QUANTILE": [task.quantile],
            "MODEL": [task.model],
            "NN_VERSION": [task.nn_version or ""],
            "WINDOW_START": [task.window_start],
            "WINDOW_END": [task.window_end],
        })
        
        # Return success, task update, and forecast data for batch processing
        task_update = (task, "done", None, model_path, 1)
        #logging.info(f"Completed task: {task.id()}")
        return True, task_update, out_df
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logging.error(f"Task failed: {task.id()} -> {error_msg}")
        logging.error(traceback.format_exc())
        
        # Return failure and task update for batch processing
        task_update = (task, "failed", error_msg, None, 0)
        return False, task_update, None

def execute_global_nn_task_batch(task: TaskKey, cfg: Dict[str, Any], paths: Dict[str, Path], instance_id: str) -> Tuple[bool, Optional[Tuple[TaskKey, str, Optional[str], Optional[Path], int]], Optional[pd.DataFrame]]:
    """
    Execute a global NN task and return success status, task update info, and forecast data for batch processing.
    """
    try:
        time_col = cfg["data"].get("time_col", "TIME")
        target = cfg["data"]["target"]
        lags = cfg["data"].get("lags")
        ws = pd.Timestamp(task.window_start)
        we = pd.Timestamp(task.window_end)
        
        # Global model artifact path
        model_dir = paths["models_root"] / f"{task.model}-global" / (f"nn_version={task.nn_version}" if task.nn_version else "default") / f"q={task.quantile}" / f"h={task.horizon}" / f"window_{task.window_start}_{task.window_end}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        
        # Check cache first
        cache_key = f"{task.model}|{task.nn_version or '-'}|q={task.quantile}|h={task.horizon}|{task.window_start}_{task.window_end}"
        model_obj = None
        
        if cache_key in _global_model_cache:
            model_obj = _global_model_cache[cache_key]
            logging.info(f"[CACHE] Using cached global model: {cache_key}")
        
        # Try loading from disk if not in cache
        if model_obj is None and cfg["runtime"].get("allow_reload", True) and model_path.exists():
            try:
                model_obj = joblib.load(model_path)
                logging.info(f"[CACHE] Loaded model from disk: {model_path}")
                _global_model_cache[cache_key] = model_obj
            except Exception as e:
                logging.warning(f"[CACHE] Failed to load model at {model_path}, will retrain. Reason: {e}")
        
        # Train if needed
        if model_obj is None:
            # Build training data for all eligible countries using cached data
            all_train_data = []
            country_names = []
            
            for country in _global_data_cache.keys():
                dfc = get_cached_country_data(country)
                if target not in dfc.columns:
                    logging.warning(f"[GLOBAL NN] Missing target '{target}' in {country}, skipping")
                    continue
                train_df = dfc.loc[(dfc[time_col] >= ws) & (dfc[time_col] <= we)].copy()
                if len(train_df) < int(cfg["splits"].get("min_train_points", 12)):
                    logging.warning(f"[GLOBAL NN] Too few points in {country} ({len(train_df)}), skipping")
                    continue
                all_train_data.append(train_df)
                country_names.append(country)
            
            if not all_train_data:
                raise RuntimeError("No countries with sufficient data for global NN training.")
            
            logging.info(f"[GLOBAL NN] Training model on {len(all_train_data)} countries -> {model_path}")
            nn_params = cfg_for_nn_version(cfg, task.nn_version)
            model_obj = _build_nn_global(
                all_train_data, target, task.quantile, task.horizon, lags,
                nn_params, seed=int(cfg.get("seed", 42)), country_names=country_names
            )
            
            # Save model
            try:
                joblib.dump(model_obj, model_path)
                logging.info(f"[GLOBAL NN] Saved model to {model_path}")
            except Exception as e:
                logging.warning(f"[SAVE] Could not pickle global NN: {e}")
            
            # Cache the model
            _global_model_cache[cache_key] = model_obj
        
        # Generate forecasts for ALL countries using cached data
        rows = []
        for country in _global_data_cache.keys():
            try:
                dfc = get_cached_country_data(country)
                if target not in dfc.columns:
                    continue
                
                # Find forecast time
                idx_end = dfc.index[dfc[time_col] == we]
                if len(idx_end) == 0:
                    continue
                i = int(idx_end[0])
                k = i + task.horizon
                if k >= len(dfc):
                    continue
                
                t_forecast = pd.Timestamp(dfc.iloc[k][time_col])
                true_val = float(dfc.iloc[k][target]) if target in dfc.columns else np.nan
                
                # Predict for this country
                yhat = _predict_nn_single(model_obj, dfc, country, time_col, we, task.horizon, task.quantile)
                
                rows.append({
                    "TIME": t_forecast,
                    "COUNTRY": country,
                    "TRUE_DATA": true_val,
                    "FORECAST": float(yhat),
                    "HORIZON": task.horizon,
                    "QUANTILE": task.quantile,
                    "MODEL": "nn-global",
                    "NN_VERSION": task.nn_version or "",
                    "WINDOW_START": task.window_start,
                    "WINDOW_END": task.window_end,
                })
            except Exception as e:
                logging.warning(f"[GLOBAL NN] Predict failed for {country}: {e}")
                continue
        
        if not rows:
            raise RuntimeError("Global NN produced no forecasts.")
        
        out_df = pd.DataFrame(rows)
        
        # Return success, task update, and forecast data for batch processing
        task_update = (task, "done", None, model_path, len(out_df))
        logging.info(f"Completed global task: {task.id()} with {len(out_df)} forecasts")
        return True, task_update, out_df
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logging.error(f"Global task failed: {task.id()} -> {error_msg}")
        logging.error(traceback.format_exc())
        
        # Return failure and task update for batch processing
        task_update = (task, "failed", error_msg, None, 0)
        return False, task_update, None

def execute_single_task(task: TaskKey, cfg: Dict[str, Any], paths: Dict[str, Path], instance_id: str) -> bool:
    """
    Execute a single task. Returns True if successful, False if failed.
    This is a wrapper around execute_single_task_batch for backwards compatibility.
    """
    success, task_update, forecast_df = execute_single_task_batch(task, cfg, paths, instance_id)
    
    # Update status immediately for single task execution
    if task_update:
        update_task_batch_status(paths["progress_parquet"], [task_update], instance_id)
    
    # Save forecast immediately for single task execution
    if forecast_df is not None:
        q = task.quantile
        h = task.horizon
        out_path = paths["forecasts_root"] / f"q={q}" / f"h={h}" / "rolling_window.parquet"
        append_forecasts(out_path, forecast_df)
    
    return success

def execute_global_nn_task(task: TaskKey, cfg: Dict[str, Any], paths: Dict[str, Path], instance_id: str) -> bool:
    """
    Execute a global NN task. Wrapper around batch version for backwards compatibility.
    """
    success, task_update, forecast_df = execute_global_nn_task_batch(task, cfg, paths, instance_id)
    
    # Update status immediately for single task execution
    if task_update:
        update_task_batch_status(paths["progress_parquet"], [task_update], instance_id)
    
    # Save forecast immediately for single task execution
    if forecast_df is not None:
        q = task.quantile
        h = task.horizon
        out_path = paths["forecasts_root"] / f"q={q}" / f"h={h}" / "rolling_window.parquet"
        append_forecasts(out_path, forecast_df)
    
    return success

# ----------------------------- Task Planning (adapted from original) -----------------------------

def plan_all_tasks(cfg: Dict[str, Any], paths: Dict[str, Path]) -> List[TaskKey]:
    """Generate all possible tasks. Data will be cached during this process."""
    # Load and cache all country data upfront
    cached_countries = load_and_cache_all_country_data(paths["data_path"], cfg)
    
    # Update paths to include country files for backwards compatibility
    countries = load_country_files(paths["data_path"])
    paths["countries"] = countries
    
    time_col = cfg["data"].get("time_col", "TIME")
    horizons = list(map(int, cfg["data"]["horizons"]))
    quantiles = [float(q) for q in cfg["data"]["quantiles"]]
    
    rw = cfg["rolling_window"]
    size = int(rw["size"])
    step = int(rw.get("step", 1))
    
    # Enabled models
    enabled_models: List[Tuple[str, Optional[str], bool]] = []  # (model_type, nn_version, is_global)
    for m in cfg.get("models", []):
        if not m.get("enabled", False):
            continue
        if m.get("type") == "nn":
            is_global = not bool(m.get("per_country", True))  # Default to per-country unless explicitly set to global
            for vname, _ in cfg_for_nn_versions(cfg):
                enabled_models.append(("nn", vname, is_global))
        else:
            enabled_models.append((m.get("type"), None, False))
    
    # Generate windows using first country from cached data as reference
    ref_country = next(iter(cached_countries.keys()))
    ref_df = cached_countries[ref_country]
    dates = ref_df[time_col]
    
    start_date = pd.Timestamp(cfg["splits"]["test_cutoff"]) if str(rw.get("start", "auto")).lower() == "auto" else pd.Timestamp(rw["start"])
    end_date = dates.iloc[-1] if str(rw.get("end", "auto")).lower() == "auto" else pd.Timestamp(rw["end"])
    
    all_tasks: List[TaskKey] = []
    
    for h in horizons:
        windows = generate_windows(dates, size=size, step=step, start_date=start_date, end_date=end_date, horizon=h)
        for (wstart, wend, _ftime) in windows:
            for q in quantiles:
                for (model_type, nn_ver, is_global) in enabled_models:
                    if is_global and model_type == "nn":
                        # ONE global task per (window, q, h, version)
                        all_tasks.append(TaskKey(
                            model="nn",
                            nn_version=nn_ver,
                            quantile=q,
                            horizon=h,
                            country="__ALL__",          # sentinel for global model
                            window_start=str(wstart.date()),
                            window_end=str(wend.date()),
                            is_global=True
                        ))
                    else:
                        # per-country tasks as before
                        for country in cached_countries.keys():
                            all_tasks.append(TaskKey(
                                model=model_type,
                                nn_version=nn_ver,
                                quantile=q,
                                horizon=h,
                                country=country,
                                window_start=str(wstart.date()),
                                window_end=str(wend.date()),
                                is_global=False
                            ))
    
    logging.info(f"Planned {len(all_tasks)} tasks using cached data for {len(cached_countries)} countries")
    return all_tasks

# ----------------------------- Main Runner -----------------------------

def run_single_core_loop(cfg: Dict[str, Any], paths: Dict[str, Path], instance_id: str, worker_index: int = 0) -> None:
    """Main loop that processes tasks in batches with data caching"""
    # Plan all possible tasks (this will load and cache all data)
    all_tasks = plan_all_tasks(cfg, paths)
    
    if not all_tasks:
        logging.info("No tasks to process")
        return

    # Show memory usage after caching
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logging.info(f"Memory usage after data caching: {memory_mb:.1f} MB")
        logging.info(f"Cached data for {len(_global_data_cache)} countries")
        
        # Show sample of cached data sizes
        for i, (country, df) in enumerate(list(_global_data_cache.items())[:5]):
            size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            logging.info(f"  {country}: {len(df)} rows, {size_mb:.1f} MB")
        if len(_global_data_cache) > 5:
            logging.info(f"  ... and {len(_global_data_cache) - 5} more countries")
            
    except Exception as e:
        logging.warning(f"Could not get memory usage: {e}")
    
    logging.info(f"Total tasks planned: {len(all_tasks)}")
    
    # Filter out completed tasks
    remaining_tasks = filter_completed_tasks(all_tasks, paths["progress_parquet"])
    logging.info(f"Tasks remaining after filtering completed: {len(remaining_tasks)}")
    
    if not remaining_tasks:
        logging.info("All tasks already completed")
        return
    
    # Get worker configuration
    num_workers = int(cfg.get("runtime", {}).get("num_workers", 1))
    batch_size = int(cfg.get("runtime", {}).get("batch_size", 50))  # Increase batch size significantly
    
    # Get worker's chunk of tasks
    worker_tasks = get_worker_chunk(remaining_tasks, worker_index, num_workers, batch_size)
    
    if not worker_tasks:
        logging.info(f"No tasks assigned to worker {worker_index}")
        return
    
    logging.info(f"Worker {worker_index} processing {len(worker_tasks)} out of {len(remaining_tasks)} remaining tasks")
    logging.info(f"Using batch size: {batch_size}")
    
    completed = 0
    failed = 0
    consecutive_no_tasks = 0
    start_time = time.time()
    
    while True:
        # Claim batch of tasks
        task_batch = claim_task_batch(paths["progress_parquet"], worker_tasks, instance_id, batch_size)
        
        if not task_batch:
            consecutive_no_tasks += 1
            if consecutive_no_tasks >= 3:
                logging.info("No more tasks available after 3 attempts")
                break
            
            # Wait with exponential backoff before trying again
            wait_time = min(1.0 * (2 ** (consecutive_no_tasks - 1)), 10.0)
            wait_time += random.uniform(0, wait_time * 0.1)  # Add jitter
            logging.info(f"No tasks found, waiting {wait_time:.1f}s before retry")
            time.sleep(wait_time)
            continue
        
        consecutive_no_tasks = 0  # Reset counter
        
        # Process the batch of tasks
        task_updates = []
        forecast_batch = []
        batch_completed = 0
        batch_failed = 0
        batch_start_time = time.time()
        
        for task in task_batch:
            #logging.info(f"Processing task: {task.id()}")
            success, task_update, forecast_df = execute_single_task_batch(task, cfg, paths, instance_id)
            
            if success:
                batch_completed += 1
            else:
                batch_failed += 1
            
            if task_update:
                task_updates.append(task_update)
            
            # Collect forecast data for batch processing
            if forecast_df is not None:
                forecast_batch.append((task.quantile, task.horizon, forecast_df))
        
        # Update all task statuses in a single batch operation
        if task_updates:
            update_task_batch_status(paths["progress_parquet"], task_updates, instance_id)
        
        # Append all forecasts in a single batch operation
        if forecast_batch:
            append_forecasts_batch(forecast_batch, paths)
        
        completed += batch_completed
        failed += batch_failed
        
        # Calculate timing statistics
        batch_time = time.time() - batch_start_time
        total_time = time.time() - start_time
        total_tasks = completed + failed
        
        avg_time_per_task = total_time / total_tasks if total_tasks > 0 else 0
        batch_avg_time = batch_time / len(task_batch) if len(task_batch) > 0 else 0
        
        logging.info(f"Batch progress: {batch_completed}/{len(task_batch)} completed, {batch_failed}/{len(task_batch)} failed")
        logging.info(f"Total progress: {completed} completed, {failed} failed")
        logging.info(f"Timing: batch took {batch_time:.1f}s (avg {batch_avg_time:.1f}s/task), overall avg {avg_time_per_task:.1f}s/task")
        
        # Show memory usage periodically
        if (completed + failed) % 100 == 0:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                logging.info(f"Current memory usage: {memory_mb:.1f} MB")
            except:
                pass
        
        # Shorter delay to reduce idle time
        #time.sleep(0.2)  # Reduce from 0.05 to 0.2 to reduce file contention
        
        # Garbage collection to keep memory usage low
        if (completed + failed) % 50 == 0:  # Less frequent GC for larger batches
            gc.collect()
    
    final_time = time.time() - start_time
    final_avg_time = final_time / (completed + failed) if (completed + failed) > 0 else 0
    
    logging.info(f"Worker {worker_index} (instance {instance_id}) finished. Completed: {completed}, Failed: {failed}")
    logging.info(f"Final timing: {final_time:.1f}s total, {final_avg_time:.1f}s average per task")
    
    # Final memory usage
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logging.info(f"Final memory usage: {memory_mb:.1f} MB")
    except:
        pass

# ----------------------------- Paths and Setup -----------------------------

def make_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Create path dictionary for home directory operations."""
    root = Path(cfg["io"]["output_root"]).absolute()
    
    paths = {
        "data_path": Path(cfg["data"]["path"]).absolute(),
        "output_root": root,
        "models_root": root / cfg["io"]["models_dir"],
        "forecasts_root": root / cfg["io"]["forecasts_dir"],
        "progress_parquet": root / cfg["io"]["progress_parquet"],
        "errors_parquet": root / cfg["io"]["errors_parquet"],
        "logs_dir": root / cfg["io"]["logs_dir"],
    }
    
    # Ensure directories exist
    for key, path in paths.items():
        if key in ["data_path", "output_root"]:
            continue
        if isinstance(path, Path) and path.suffix == "":  # Directory
            safe_mkdirs(path)
        elif isinstance(path, Path):  # File
            safe_mkdirs(path.parent)
    
    return paths





# ----------------------------- CLI -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import uuid
    
    ap = argparse.ArgumentParser("Single-Core Rolling-Window Quantile Forecast Runner")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--instance-id", type=str, default=None, help="Unique instance identifier")
    ap.add_argument("--worker-index", type=int, default=0, help="Worker index (0-based) for task distribution")
    ap.add_argument("--dry-run", action="store_true", help="Show planned tasks without executing")
    args = ap.parse_args(argv)
    
    # Generate unique instance ID if not provided
    if args.instance_id is None:
        args.instance_id = f"worker_{args.worker_index}_{str(uuid.uuid4().int)[:10]}"
    
    print(f"[MAIN] Starting worker {args.worker_index} with instance ID: {args.instance_id}")
    
    # Force single core BEFORE any heavy imports
    enforce_single_core()
    
    print(f"[MAIN] Loading config from: {args.config}")
    
    # Load config
    cfg = read_yaml(Path(args.config))
    set_seeds(int(cfg.get("seed", 42)))
    
    # Setup paths
    paths = make_paths(cfg)
    
    print(f"[MAIN] Setting up logging to: {paths['logs_dir']}")
    
    # Setup logging EARLY and test it immediately
    setup_logging(paths["logs_dir"], args.instance_id)
    
    # Test logging immediately after setup
    logging.info("=" * 50)
    logging.info("LOGGING TEST - This message should appear in the log file")
    logging.info(f"Starting single-core runner worker {args.worker_index} with instance ID: {args.instance_id}")
    logging.info(f"Process ID: {os.getpid()}")
    logging.info(f"Config file: {args.config}")
    logging.info("=" * 50)
    
    if args.dry_run:
        logging.info("Running in dry-run mode")
        all_tasks = plan_all_tasks(cfg, paths)
        remaining_tasks = filter_completed_tasks(all_tasks, paths["progress_parquet"])
        
        num_workers = int(cfg.get("runtime", {}).get("num_workers", 1))
        worker_tasks = get_worker_chunk(remaining_tasks, args.worker_index, num_workers, 1)
        
        print(f"Total tasks planned: {len(all_tasks)}")
        print(f"Tasks remaining after filtering completed: {len(remaining_tasks)}")
        print(f"Tasks assigned to worker {args.worker_index}: {len(worker_tasks)}")
        
        logging.info(f"Total tasks planned: {len(all_tasks)}")
        logging.info(f"Tasks remaining after filtering completed: {len(remaining_tasks)}")
        logging.info(f"Tasks assigned to worker {args.worker_index}: {len(worker_tasks)}")
        
        for i, task in enumerate(worker_tasks[:10]):  # Show first 10
            task_info = f"{i+1}. {task.id()}"
            print(task_info)
            logging.info(task_info)
        if len(worker_tasks) > 10:
            remaining_msg = f"... and {len(worker_tasks) - 10} more tasks for this worker"
            print(remaining_msg)
            logging.info(remaining_msg)
        logging.info("Dry-run completed")
        return 0
    
    # Run the main loop
    try:
        logging.info("Starting main processing loop")
        run_single_core_loop(cfg, paths, args.instance_id, args.worker_index)
        logging.info("Single-core runner completed successfully")
        return 0
    except KeyboardInterrupt:
        logging.info("Runner interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Runner failed: {e}")
        logging.error(traceback.format_exc())
        return 1
    finally:
        logging.info("Shutting down logging")
        # Ensure all log messages are flushed
        for handler in logging.root.handlers:
            handler.flush()

if __name__ == "__main__":
    sys.exit(main())
