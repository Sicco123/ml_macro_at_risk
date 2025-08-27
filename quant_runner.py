#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel Rolling-Window Quantile Forecast Runner (CPU-only)

IMPORTANT
- This runner orchestrates YOUR existing models. It never re-implements them.
- It imports the same classes you use in run_analysis.py:
    from src.ensemble_nn_api import EnsembleNNAPI
    from src.lqr_api import LQRModel
- "ar-qr" is treated as LQR with AR-only features (target lags only).
- "nn" supports multiple versions; each version is the same class with different hyperparams.
- Per (model, quantile, horizon, country, window) we train a separate model, save it, and append the forecast.
- It performs a 1-iteration/short training memory probe and schedules tasks under CPU+RAM limits.
- It tracks status in a progress parquet, continues on failures, and resumes on reruns.

CONFIG (YAML) — expected keys (example template you’ll provide separately)
---------------------------------------------------------------------------
seed: 123

data:
  path: processed_per_country          # folder with per-country files (.parquet/.csv)
  time_col: TIME
  country_col: COUNTRY                 # optional; if missing we infer from filename
  target: prc_hicp_manr_CP00
  lags: [1,2,3,6,12]                   # if [], AR window size will be used to derive lags 1..p
  horizons: [1,3,6,12]
  quantiles: [0.1, 0.5, 0.9]
  missing: forward_fill_then_mean      # or interpolate_linear | drop

rolling_window:
  size: 60                             # in periods (index steps)
  step: 1                              # in periods
  start: auto                          # or explicit 'YYYY-MM-DD' (forecast_start_date)
  end: auto

splits:
  test_cutoff: "2017-12-01"            # first forecast_start if rolling_window.start=='auto'
  min_train_points: 24

runtime:
  allow_reload: true                   # load model/artifact if exists
  retrain_if_exists: false             # force re-train even if artifact exists
  max_cores: auto                      # int or 'auto'
  max_ram_gb: auto                     # float or 'auto'
  safety_ram_fraction: 0.8             # cap fraction of available RAM for tasks
  mem_probe_fudge_mb: 200              # safety margin per task after probing
  retries: 1                           # automatic retries on failure
  thread_pinning: true                 # pin MKL/OMP/NumExpr to 1 thread in workers
  progress_refresh_sec: 5

io:
  output_root: outputs
  models_dir: models
  forecasts_dir: forecasts
  progress_parquet: progress/progress.parquet
  errors_parquet: progress/errors.parquet
  logs_dir: logs

models:                                 # which models to run
  - type: ar-qr
    enabled: true
    params:
      # AR-QR uses LQRModel under the hood, but passes only target lags
      solver: huberized
      alphas: [7.5, 10, 15, 20, 25, 30, 35]
      use_cv: true
      cv_splits: 5
  - type: lqr
    enabled: true
    params:
      solver: huberized
      alphas: [7.5, 10, 15, 20, 25, 30, 35]
      use_cv: true
      cv_splits: 5
  - type: lqr-per-country
    enabled: true
    params:
      # LQR estimated separately for each country
      solver: huberized
      alphas: [7.5, 10, 15, 20, 25, 30, 35]
      use_cv: true
      cv_splits: 5
  - type: ar-qr-per-country
    enabled: true
    params:
      # AR-QR estimated separately for each country
      solver: huberized
      alphas: [7.5, 10, 15, 20, 25, 30, 35]
      use_cv: true
      cv_splits: 5
  - type: nn
    enabled: true
    versions:
      - name: v1
        params:
          units_per_layer: [32,32,32]
          activation: relu
          optimizer: adam
          learning_rate: 5.0e-4
          epochs: 200
          batch_size: 64
          patience: 30
          parallel_models: 5
          device: cpu
          l2_penalty: 1.0e-4
      - name: v2
        params:
          units_per_layer: [64,64,64,64]
          activation: relu
          optimizer: adam
          learning_rate: 2.0e-4
          epochs: 400
          batch_size: 64
          patience: 50
          parallel_models: 10
          device: cpu
          l2_penalty: 1.0e-4
"""

from __future__ import annotations

import os
import sys
import gc
import math
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
from filelock import FileLock
from tqdm import tqdm

# parquet
import pyarrow.parquet as pq

# Global model cache for sharing trained models across countries in the same window
_global_model_cache = {}
try:
    from src.ensemble_nn_api import EnsembleNNAPI
    from src.lqr_api import LQRModel
except Exception as e:
    # Keep import error explicit to the user
    print(f"[IMPORT ERROR] Could not import your model APIs: {e}", file=sys.stderr)
    raise

from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------- Logging -----------------------------

def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{ts}.log"
    handlers = [logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        handlers=handlers,
    )
    logging.info(f"Logging to {log_path}")

# ----------------------------- Utils -----------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # Torch is optional; only set if available
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

def detect_cpu_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1

def detect_available_ram_bytes() -> int:
    vm = psutil.virtual_memory()
    avail = vm.available
    # cgroup v1/v2
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            if os.path.exists(p):
                s = open(p).read().strip()
                if s.isdigit():
                    lim = int(s)
                    if lim > 0 and lim < (1 << 60):
                        avail = min(avail, lim)
        except Exception:
            pass
    return max(avail, 256 * 1024 * 1024)

# ----------------------------- Data IO -----------------------------

def load_country_files(data_dir: Path) -> Dict[str, Path]:
    files: List[Path] = []
    for ext in ("*.parquet", "*.pq", "*.feather", "*.csv"):
        files.extend(data_dir.glob(ext))
    if not files:
        raise FileNotFoundError(f"No data files found under {data_dir}")
    mapping: Dict[str, Path] = {}
    for f in files:
        # Infer country code from filename stem before '__' if present
        stem = f.stem
        country = stem.split("__")[0].upper()
        mapping[country] = f
    return mapping

def read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".parquet", ".pq", ".feather"]:
        df = pd.read_parquet(path)
        df.reset_index( inplace=True)
        return df
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

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
    else:
        pass
    return df

# ----------------------------- Rolling Windows -----------------------------

def generate_windows(
    dates: pd.Series,
    size: int,
    step: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    horizon: int
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Returns list of (win_start, win_end, forecast_time = win_end + horizon steps)
    Windows are aligned in index space.
    """
    idx = pd.Index(pd.to_datetime(dates))
    out: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    # Iterate by index position; forecast start loop
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
        # move by step (index space)
        i += step - 1
    return out

# ----------------------------- Progress Tracking -----------------------------

PROGRESS_COLUMNS = [
    "MODEL", "NN_VERSION", "QUANTILE", "HORIZON", "COUNTRY",
    "WINDOW_START", "WINDOW_END", "STATUS", "LAST_UPDATE",
    "ERROR_MSG", "MODEL_PATH", "FORECAST_ROWS", "IS_GLOBAL"
]

def load_progress(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=PROGRESS_COLUMNS)

def upsert_progress_row(progress_path: Path, lock: FileLock, row: Dict[str, Any]) -> None:
    with lock:
        df = load_progress(progress_path)
        mask = (
            (df["MODEL"] == row["MODEL"]) &
            (df["NN_VERSION"].fillna("") == (row.get("NN_VERSION") or "")) &
            (df["QUANTILE"] == row["QUANTILE"]) &
            (df["HORIZON"] == row["HORIZON"]) &
            (df["COUNTRY"] == row["COUNTRY"]) &
            (df["WINDOW_START"] == row["WINDOW_START"]) &
            (df["WINDOW_END"] == row["WINDOW_END"]) &
            (df.get("IS_GLOBAL", False) == row.get("IS_GLOBAL", False))
        )
        df = df.loc[~mask]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_parquet(progress_path, index=False)

def append_error(errors_path: Path, lock: FileLock, row: Dict[str, Any]) -> None:
    with lock:
        if errors_path.exists():
            df = pd.read_parquet(errors_path)
        else:
            df = pd.DataFrame(columns=list(row.keys()))
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_parquet(errors_path, index=False)

# ----------------------------- Forecast Sink -----------------------------

def append_forecasts(out_path: Path, rows: pd.DataFrame) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path(out_path)))
    with lock:
        if out_path.exists():
            # merge by keys (TIME, COUNTRY, HORIZON, QUANTILE)
            existing = pq.ParquetFile(out_path).read().to_pandas()
            combined = pd.concat([existing, rows], ignore_index=True)
            combined = combined.drop_duplicates(subset=["TIME", "COUNTRY", "HORIZON", "QUANTILE"], keep="last")
            combined.sort_values(["TIME", "COUNTRY", "HORIZON", "QUANTILE"], inplace=True)
            combined.to_parquet(out_path, index=False)
        else:
            rows.to_parquet(out_path, index=False)

# ----------------------------- Task Key -----------------------------

@dataclass(frozen=True)
class TaskKey:
    model: str                   # 'ar-qr' | 'lqr' | 'nn' | 'nn-global'
    nn_version: Optional[str]    # e.g., 'v1' | 'v2' for NN; None otherwise
    quantile: float
    horizon: int
    country: str                 # For global models, this represents the prediction target country
    window_start: str            # 'YYYY-MM-DD'
    window_end: str              # 'YYYY-MM-DD'
    is_global: bool = False      # True for global models that train on all countries

    def id(self) -> str:
        nv = self.nn_version or "-"
        global_suffix = "-global" if self.is_global else ""
        return f"{self.model}{global_suffix}|{nv}|q={self.quantile}|h={self.horizon}|{self.country}|{self.window_start}_{self.window_end}"

# ----------------------------- Worker -----------------------------

def worker_init(thread_pinning: bool = True):
    # CPU-only; pin threads in BLAS stacks to avoid oversubscription
    if thread_pinning:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def estimate_memory_mb(callable_fn, *args, **kwargs) -> int:
    """Rudimentary RSS delta probe around callable_fn()."""
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    try:
        callable_fn(*args, **kwargs)
    except Exception:
        # swallow probe errors; we just need a rough allocation number
        pass
    gc.collect()
    rss_after = proc.memory_info().rss
    delta = max(0, rss_after - rss_before)
    return int(delta / (1024 * 1024))

# ----------------------------- Model Adapters (using YOUR APIs) -----------------------------

def _build_lqr_per_country(
    train_df: pd.DataFrame,
    target: str,
    quantile: float,
    horizon: int,
    lags: List[int],
    params: Dict[str, Any],
    seed: int,
    ar_only: bool,
    probe: bool
) -> LQRModel:
    """
    Construct and fit YOUR LQRModel for a single country using only that country's data.
    This estimates a separate model for each country individually.
    - ar_only=True => pass only TIME + target into data_list (AR-QR behavior).
    - probe=True => skip CV and keep it as cheap as possible.
    """
    if ar_only:
        data_list = [train_df[[c for c in train_df.columns if c in ["TIME", target]]].copy()]
    else:
        data_list = [train_df.copy()]

    # Create the model - using single country data
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

    # Optional CV for alpha - optimized for single country
    if  params.get("use_cv", False):
        alphas = params.get("alphas", [])
        if alphas:
            splits = int(params.get("cv_splits", 5))
            try:
                # Use k-fold validation for single country
                mdl.k_fold_validation(alphas=alphas, n_splits=splits)
            except Exception as e:
                logging.warning(f"[{('AR-QR-PC' if ar_only else 'LQR-PC')}] CV failed, falling back to default alpha. Reason: {e}")

    # Fit
    mdl.fit()
    return mdl

def _build_lqr(
    train_df: pd.DataFrame,
    target: str,
    quantile: float,
    horizon: int,
    lags: List[int],
    params: Dict[str, Any],
    seed: int,
    ar_only: bool,
    probe: bool
) -> LQRModel:
    """
    Construct and fit YOUR LQRModel on a single-country rolling window.
    - ar_only=True => pass only TIME + target into data_list (AR-QR behavior).
    - probe=True => skip CV and keep it as cheap as possible.
    """
    if ar_only:
        data_list = [train_df[[c for c in train_df.columns if c in ["TIME", target]]].copy()]
    else:
        data_list = [train_df.copy()]

    # Create the model
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

    # Optional CV for alpha
    if params.get("use_cv", False):
        alphas = params.get("alphas", [])
        if alphas:
            splits = int(params.get("cv_splits", 5))
            try:
                mdl.k_fold_validation(alphas=alphas, n_splits=splits)
            except Exception as e:
                logging.warning(f"[{('AR-QR' if ar_only else 'LQR')}] CV failed, falling back to default alpha. Reason: {e}")

    # Fit
    mdl.fit()
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
    """
    Use YOUR LQRModel to predict the forecast at t = window_end + horizon.
    We pass post-window data into predict() and extract the correct position.
    - ar_only=True => pass only TIME + target columns for AR-QR models.
    """
    # Build a small test DF that includes exactly the rows needed to emit h-step ahead.
    # We pass the slice starting at window_end (inclusive) so internal lag logic can resolve.
    test_df = full_country_df.loc[full_country_df[time_col] >= window_end].copy()
    
    # Filter columns for AR-only models (same logic as in build functions)
    if ar_only:
        test_df = test_df[[c for c in test_df.columns if c in ["TIME", target]]].copy()
   
    preds, _targets = mdl.predict([test_df])  # YOUR API returns arrays
    # Shape expectation: preds shape (T_test, H, Q) or (T_test, Q, H). We try to infer.
    arr = np.asarray(preds)
    # Heuristics to extract the first emitted prediction for the given horizon/quantile:
    # Try common shapes:
    # 1) (T, H, Q)
    if arr.ndim == 3 and arr.shape[1] >= 1 and arr.shape[2] >= 1:
        # use the first row (aligned to first predictable time after window_end),
        # pick specific horizon & quantile
        h_idx = horizon_index(horizon, arr.shape[1])
        q_idx = quantile_index(quantile, arr.shape[2], quantile_list=None)
        return float(arr[0, h_idx, q_idx])
    # 2) (T, Q, H)
    if arr.ndim == 3 and arr.shape[2] >= 1 and arr.shape[1] >= 1:
        h_idx = horizon_index(horizon, arr.shape[2])
        q_idx = quantile_index(quantile, arr.shape[1], quantile_list=None)
        return float(arr[0, q_idx, h_idx])
    # 3) (T, Q) for single-horizon models
    if arr.ndim == 2 and arr.shape[1] >= 1:
        q_idx = quantile_index(quantile, arr.shape[1], quantile_list=None)
        return float(arr[0, q_idx])

    raise RuntimeError("Could not infer prediction array shape from LQRModel.predict output.")

def _build_nn(
    train_df: pd.DataFrame,
    target: str,
    quantile: float,
    horizon: int,
    lags: List[int],
    nn_params: Dict[str, Any],
    seed: int,
    probe: bool
) -> EnsembleNNAPI:
    """
    Construct and fit YOUR EnsembleNNAPI on a single-country rolling window.
    For probe=True, we shorten epochs to 1 to keep memory usage minimal.
    """
    # YOUR API expects a list of dataframes (countries). For a per-country window we pass one DF.
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
        prefit_AR=bool(nn_params.get("prefit_AR", False)),
        time_col="TIME",
        verbose=0
    )

    epochs = 1 if probe else int(nn_params.get("epochs", 100))
    mdl.fit(
        epochs=epochs,
        learning_rate=float(nn_params.get("learning_rate", 1e-3)),
        batch_size=int(nn_params.get("batch_size", 64)),
        validation_size=float(nn_params.get("validation_size", 0.2)),
        patience=int(nn_params.get("patience", 20)),
        verbose=0,
        optimizer=nn_params.get("optimizer", "adam"),
        parallel_models=int(nn_params.get("parallel_models", 1)),
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
    probe: bool,
    country_names: List[str]
) -> EnsembleNNAPI:
    """
    Construct and fit YOUR EnsembleNNAPI on multiple countries' data (global model).
    For probe=True, we shorten epochs to 1 to keep memory usage minimal.
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
        verbose=0 if probe else 1
    )

    epochs = 1 if probe else int(nn_params.get("epochs", 100))
    mdl.fit(
        epochs=epochs,
        learning_rate=float(nn_params.get("learning_rate", 1e-3)),
        batch_size=int(nn_params.get("batch_size", 64)),
        validation_size=float(nn_params.get("validation_size", 0.2)),
        patience=int(nn_params.get("patience", 20)),
        verbose=0,
        optimizer=nn_params.get("optimizer", "adam"),
        parallel_models=int(nn_params.get("parallel_models", 1)),
        l2=float(nn_params.get("l2_penalty", 0.0)),
        return_validation_loss=False,
        return_train_loss=False,
        shuffle=True
    )
    return mdl

def _predict_nn_single(
    mdl: EnsembleNNAPI,
    full_country_df: pd.DataFrame,
    country_id: str,
    time_col: str,
    window_end: pd.Timestamp,
    horizon: int,
    quantile: float
) -> float:
    """
    Use YOUR EnsembleNNAPI to predict at t = window_end + horizon.
    We pass the post-window slice and extract the first prediction.
    """
    test_df = full_country_df.loc[full_country_df[time_col] >= window_end].copy()
    preds, _targets = mdl.predict_per_country(test_df, country_id)
    arr = np.asarray(preds)
    # Try common shapes: (T, H, Q) or (T, Q, H) or (T, Q)
    if arr.ndim == 3:
        # Prefer (T, H, Q)
        if arr.shape[1] >= 1 and arr.shape[2] >= 1:
            h_idx = horizon_index(horizon, arr.shape[1])
            q_idx = quantile_index(quantile, arr.shape[2], quantile_list=None)
            return float(arr[0, h_idx, q_idx])
        # (T, Q, H)
        if arr.shape[2] >= 1 and arr.shape[1] >= 1:
            h_idx = horizon_index(horizon, arr.shape[2])
            q_idx = quantile_index(quantile, arr.shape[1], quantile_list=None)
            return float(arr[0, q_idx, h_idx])
    if arr.ndim == 2 and arr.shape[1] >= 1:
        q_idx = quantile_index(quantile, arr.shape[1], quantile_list=None)
        return float(arr[0, q_idx])

    raise RuntimeError("Could not infer prediction array shape from EnsembleNNAPI.predict_per_country output.")

def horizon_index(h: int, H_dim: int) -> int:
    """Map horizon h to index. Assumes horizons are 1..H; maps h -> h-1 within bounds."""
    idx = max(0, min(H_dim - 1, h - 1))
    return idx

def quantile_index(q: float, Q_dim: int, quantile_list: Optional[List[float]]) -> int:
    """Map quantile value to index; if quantile_list provided, use exact match else round by position."""
    if quantile_list:
        try:
            return quantile_list.index(q)
        except ValueError:
            pass
    # fallback: evenly spread
    # if Q_dim==1 -> index 0; if >=3, try mapping common {0.1, 0.5, 0.9}
    if Q_dim == 1:
        return 0
    # crude rank approximation
    # map q in (0,1) to nearest bin among Q_dim
    bins = np.linspace(0, 1, Q_dim)
    return int(np.argmin(np.abs(bins - q)))

# ----------------------------- Execute One Task -----------------------------

def execute_task(task: TaskKey, cfg: Dict[str, Any], paths: Dict[str, Path]) -> Tuple[TaskKey, str, Optional[str], int, Optional[pd.DataFrame]]:
    """
    Train/Load the model for one (model, nn_version, q, h, country, window) and emit a 1-row forecast DF.
    For global models (is_global=True), trains once on all countries' data and caches the result.
    Returns (task, status, err_msg, n_rows, df_or_none)
    """
    progress_lock = FileLock(str(lock_path(paths["progress_parquet"])))
    errors_lock = FileLock(str(lock_path(paths["errors_parquet"])))

    def mark(status: str, err: Optional[str] = None, model_path: Optional[Path] = None, n_rows: int = 0):
        row = {
            "MODEL": task.model,
            "NN_VERSION": task.nn_version,
            "QUANTILE": task.quantile,
            "HORIZON": task.horizon,
            "COUNTRY": task.country,
            "WINDOW_START": task.window_start,
            "WINDOW_END": task.window_end,
            "STATUS": status,
            "LAST_UPDATE": datetime.now().isoformat(),
            "ERROR_MSG": err,
            "MODEL_PATH": str(model_path) if model_path else None,
            "FORECAST_ROWS": n_rows,
            "IS_GLOBAL": task.is_global
        }
        upsert_progress_row(paths["progress_parquet"], progress_lock, row)
        if err:
            append_error(paths["errors_parquet"], errors_lock, row)

    try:
        mark("training")
        
        # Load country data
        time_col = cfg["data"].get("time_col", "TIME")
        target = cfg["data"]["target"]
        
        # Window timestamps
        ws = pd.Timestamp(task.window_start)
        we = pd.Timestamp(task.window_end)
        
        # Model artifact path - different for global vs per-country models
        if task.is_global:
            # For global models, store under a shared path (not country-specific for the model itself)
            model_dir = paths["models_root"] / f"{task.model}-global" / (f"nn_version={task.nn_version}" if task.nn_version else "default") / f"q={task.quantile}" / f"h={task.horizon}" / f"window_{task.window_start}_{task.window_end}"
        else:
            model_dir = paths["models_root"] / task.model / (f"nn_version={task.nn_version}" if task.nn_version else "default") / f"country={task.country}" / f"q={task.quantile}" / f"h={task.horizon}" / f"window_{task.window_start}_{task.window_end}"
        
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        
        # Reload options
        reload_ok = bool(cfg["runtime"].get("allow_reload", True)) and not bool(cfg["runtime"].get("retrain_if_exists", False))
        model_obj = None
        
        # For global models, check cache first, then disk
        if task.is_global:
            cache_key = f"{task.model}|{task.nn_version or '-'}|q={task.quantile}|h={task.horizon}|{task.window_start}_{task.window_end}"
            if cache_key in _global_model_cache:
                model_obj = _global_model_cache[cache_key]
                logging.info(f"[CACHE] Using cached global model: {cache_key}")
        
        # Try loading from disk if not in cache
        if model_obj is None and reload_ok and model_path.exists():
            try:
                import joblib
                model_obj = joblib.load(model_path)
                logging.info(f"[CACHE] Loaded model from disk: {model_path}")
                # Store in cache for future use
                if task.is_global:
                    _global_model_cache[cache_key] = model_obj
            except Exception as e:
                logging.warning(f"[CACHE] Failed to load model at {model_path}, will retrain. Reason: {e}")

        # Build lags
        lags = cfg["data"].get("lags")

        # Train if needed
        if model_obj is None:
            if task.is_global and task.model == "nn":
                # 1) Build training data for all eligible countries
                all_train_data = []
                country_names = []
                time_col = cfg["data"].get("time_col", "TIME")
                target = cfg["data"]["target"]
                ws, we = pd.Timestamp(task.window_start), pd.Timestamp(task.window_end)

                for country, country_file in paths["countries"].items():
                    dfc = read_df(country_file)
                    dfc = ensure_time_sorted(dfc, time_col)
                    dfc = handle_missing(dfc, cfg["data"].get("missing", "forward_fill_then_mean"))
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

                # 2) Train or load ONCE (file-locked)
                nn_params = cfg_for_nn_version(cfg, task.nn_version)
                cache_key = f"{task.model}|{task.nn_version or '-'}|q={task.quantile}|h={task.horizon}|{task.window_start}_{task.window_end}"
                model_dir = paths["models_root"] / f"{task.model}-global" / (f"nn_version={task.nn_version}" if task.nn_version else "default") / f"q={task.quantile}" / f"h={task.horizon}" / f"window_{task.window_start}_{task.window_end}"
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / "model.pkl"

                import joblib
                model_obj = None
                if cache_key in _global_model_cache:
                    model_obj = _global_model_cache[cache_key]

                if model_obj is None:
                    model_lock = FileLock(str(lock_path(model_path)))
                    with model_lock:
                        if bool(cfg["runtime"].get("allow_reload", True)) and model_path.exists():
                            logging.info(f"[GLOBAL NN] Loading cached model: {model_path}")
                            model_obj = joblib.load(model_path)
                        else:
                            logging.info(f"[GLOBAL NN] Training model -> {model_path}")
                            lags = cfg["data"].get("lags")
                            model_obj = _build_nn_global(
                                all_train_data, target, task.quantile, task.horizon, lags,
                                nn_params, seed=int(cfg.get("seed", 42)), probe=False,
                                country_names=country_names
                            )
                            try:
                                joblib.dump(model_obj, model_path)
                            except Exception as e:
                                logging.warning(f"[SAVE] Could not pickle global NN: {e}")
                    _global_model_cache[cache_key] = model_obj

                # 3) Predict for ALL countries and build one output DataFrame
                rows = []
                for country, country_file in paths["countries"].items():
                    dfc = read_df(country_file)
                    dfc = ensure_time_sorted(dfc, time_col)
                    dfc = handle_missing(dfc, cfg["data"].get("missing", "forward_fill_then_mean"))
                    if target not in dfc.columns:
                        continue

                    # locate window_end by position, then t+h
                    idx_end = dfc.index[dfc[time_col] == we]
                    if len(idx_end) == 0:
                        continue
                    i = int(idx_end[0])
                    k = i + task.horizon
                    if k >= len(dfc):
                        continue

                    t_forecast = pd.Timestamp(dfc.iloc[k][time_col])
                    true_val = float(dfc.iloc[k][target]) if target in dfc.columns else np.nan
                    try:
                        yhat = _predict_nn_single(model_obj, dfc, country, time_col, we, task.horizon, task.quantile)
                    except Exception as e:
                        logging.warning(f"[GLOBAL NN] Predict failed for {country}: {e}")
                        continue

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

                if not rows:
                    raise RuntimeError("Global NN produced no forecasts.")

                out_df = pd.DataFrame(rows)

                # 4) Mark + return once with all rows
                mark("done", err=None, model_path=model_path, n_rows=len(out_df))
                return (task, "done", None, len(out_df), out_df)
                
            else:
                # Non-global models (existing logic)
                country_file = paths["countries"][task.country]
                df = read_df(country_file)
                df = ensure_time_sorted(df, time_col)
                df = handle_missing(df, cfg["data"].get("missing", "forward_fill_then_mean"))

                # Train slice
                train_df = df.loc[(df[time_col] >= ws) & (df[time_col] <= we)].copy()

                if len(train_df) < int(cfg["splits"].get("min_train_points", 12)):
                    raise RuntimeError(f"Insufficient training points in window: {len(train_df)}")

                # Ensure the target exists
                if target not in df.columns:
                    raise RuntimeError(f"Target column '{target}' not found in data for {task.country}")

                if task.model == "lqr":
                    model_obj = _build_lqr(train_df, target, task.quantile, task.horizon, lags, cfg_for_lqr(cfg), seed=int(cfg.get("seed", 42)), ar_only=False, probe=False)
                elif task.model == "ar-qr":
                    model_obj = _build_lqr(train_df, target, task.quantile, task.horizon, lags, cfg_for_arqr(cfg), seed=int(cfg.get("seed", 42)), ar_only=True, probe=False)
                elif task.model == "lqr-per-country":
                    model_obj = _build_lqr_per_country(train_df, target, task.quantile, task.horizon, lags, cfg_for_lqr_per_country(cfg), seed=int(cfg.get("seed", 42)), ar_only=False, probe=False)
                elif task.model == "ar-qr-per-country":
                    model_obj = _build_lqr_per_country(train_df, target, task.quantile, task.horizon, lags, cfg_for_arqr_per_country(cfg), seed=int(cfg.get("seed", 42)), ar_only=True, probe=False)
                elif task.model == "nn":
                    nn_params = cfg_for_nn_version(cfg, task.nn_version)
                    model_obj = _build_nn(train_df, target, task.quantile, task.horizon, lags, nn_params, seed=int(cfg.get("seed", 42)), probe=False)
                else:
                    raise ValueError(f"Unknown model type: {task.model}")

            # Save artifact (best-effort)
            try:
                import joblib
                joblib.dump(model_obj, model_path)
            except Exception as e:
                logging.warning(f"[SAVE] Could not pickle model to {model_path}: {e}")

        # Generate forecast for the target country
        country_file = paths["countries"][task.country]
        df = read_df(country_file)
        df = ensure_time_sorted(df, time_col)
        df = handle_missing(df, cfg["data"].get("missing", "forward_fill_then_mean"))
        
        # Forecast for t = window_end + horizon
        # True value (if available)
        full_df = df
        # Locate t+h time
        idx_end = full_df.index[full_df[time_col] == we]
        if len(idx_end) == 0:
            raise RuntimeError("window_end not found in index.")
        i = int(idx_end[0])
        k = i + task.horizon
        if k >= len(full_df):
            raise RuntimeError("Forecast horizon goes beyond available data.")
        t_forecast = pd.Timestamp(full_df.iloc[k][time_col])
        true_val = float(full_df.iloc[k][target]) if target in full_df.columns else np.nan

        # Predict
        if task.model in ("lqr", "lqr-per-country"):
            yhat = _predict_lqr_single(model_obj, full_df, target, time_col, we, task.horizon, task.quantile, ar_only=False)
        elif task.model in ("ar-qr", "ar-qr-per-country"):
            yhat = _predict_lqr_single(model_obj, full_df, target, time_col, we, task.horizon, task.quantile, ar_only=True)
        elif task.model == "nn":
            yhat = _predict_nn_single(model_obj, full_df, task.country, time_col, we, task.horizon, task.quantile)
        else:
            raise ValueError(f"Unknown model type for prediction: {task.model}")

        model_name = f"{task.model}-global" if task.is_global else task.model
        out = pd.DataFrame({
            "TIME": [t_forecast],
            "COUNTRY": [task.country],
            "TRUE_DATA": [true_val],
            "FORECAST": [float(yhat)],
            "HORIZON": [task.horizon],
            "QUANTILE": [task.quantile],
            "MODEL": [model_name],
            "NN_VERSION": [task.nn_version or ""],
            "WINDOW_START": [task.window_start],
            "WINDOW_END": [task.window_end],
        })

        mark("done", err=None, model_path=model_path, n_rows=1)
        return (task, "done", None, 1, out)

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc(limit=3)
        logging.error(f"[TASK ERROR] {task.id()} -> {err}\n{tb}")
        mark("failed", err=err, model_path=None, n_rows=0)
        return (task, "failed", err, 0, None)

# ----------------------------- Config Helpers -----------------------------

def cfg_for_lqr(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for m in cfg.get("models", []):
        if m.get("type") == "lqr":
            return m.get("params", {})
    return {}

def cfg_for_lqr_per_country(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for m in cfg.get("models", []):
        if m.get("type") == "lqr-per-country":
            return m.get("params", {})
    return {}

def cfg_for_arqr(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for m in cfg.get("models", []):
        if m.get("type") == "ar-qr":
            return m.get("params", {})
    return {}

def cfg_for_arqr_per_country(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for m in cfg.get("models", []):
        if m.get("type") == "ar-qr-per-country":
            return m.get("params", {})
    return {}

def cfg_for_nn_versions(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    for m in cfg.get("models", []):
        if m.get("type") == "nn" and m.get("enabled", False):
            versions = m.get("versions", [])
            return [(v.get("name"), v.get("params", {})) for v in versions]
    return []

def cfg_for_nn_per_country(cfg: Dict[str, Any]) -> bool:
    """Check if NN models should be trained per country (True) or globally (False)"""
    for m in cfg.get("models", []):
        if m.get("type") == "nn" and m.get("enabled", False):
            return bool(m.get("per_country", True))  # Default to True (per-country)
    return True

def cfg_for_nn_version(cfg: Dict[str, Any], name: Optional[str]) -> Dict[str, Any]:
    versions = cfg_for_nn_versions(cfg)
    for n, p in versions:
        if n == name:
            return p
    # fallback to first enabled version or empty
    return versions[0][1] if versions else {}

# ----------------------------- Planning & Memory Probe -----------------------------

@dataclass
class PlannedTask:
    key: TaskKey
    mem_est_mb: int = 0

def plan_tasks(cfg: Dict[str, Any], paths: Dict[str, Path]) -> Tuple[List[PlannedTask], Dict[str, Path]]:
    countries = load_country_files(paths["data_path"])
    paths["countries"] = countries

    time_col = cfg["data"].get("time_col", "TIME")
    horizons = list(map(int, cfg["data"]["horizons"]))
    quantiles = [float(q) for q in cfg["data"]["quantiles"]]

    rw = cfg["rolling_window"]
    size = int(rw["size"])
    step = int(rw.get("step", 1))

    planned: List[PlannedTask] = []

    enabled_models: List[Tuple[str, Optional[str], bool]] = []
    for m in cfg.get("models", []):
        if not m.get("enabled", False):
            continue
        if m.get("type") == "nn":
            per_country = bool(m.get("per_country", True))
            is_global = not per_country
            for vname, _ in cfg_for_nn_versions(cfg):
                enabled_models.append(("nn", vname, is_global))
        else:
            enabled_models.append((m.get("type"), None, False))

    # Build windows ONCE (any country’s calendar will do for boundaries; use union via a reference file)
    # Using the first country’s dates for simplicity:
    ref_country, ref_path = next(iter(countries.items()))
    ref_df = ensure_time_sorted(read_df(ref_path), time_col)
    dates = ref_df[time_col]
    start_date = pd.Timestamp(cfg["splits"]["test_cutoff"]) if str(rw.get("start", "auto")).lower() == "auto" else pd.Timestamp(rw["start"])
    end_date = dates.iloc[-1] if str(rw.get("end", "auto")).lower() == "auto" else pd.Timestamp(rw["end"])

    for h in horizons:
        windows = generate_windows(dates, size=size, step=step, start_date=start_date, end_date=end_date, horizon=h)
        for (wstart, wend, _ftime) in windows:
            for q in quantiles:
                for (model_type, nn_ver, is_global) in enabled_models:
                    if is_global and model_type == "nn":
                        # ONE global task per (window, q, h, version)
                        planned.append(PlannedTask(
                            key=TaskKey(
                                model="nn",
                                nn_version=nn_ver,
                                quantile=q,
                                horizon=h,
                                country="__ALL__",          # sentinel
                                window_start=str(wstart.date()),
                                window_end=str(wend.date()),
                                is_global=True
                            )
                        ))
                    else:
                        # per-country tasks as before
                        for country in countries.keys():
                            planned.append(PlannedTask(
                                key=TaskKey(
                                    model=model_type,
                                    nn_version=nn_ver,
                                    quantile=q,
                                    horizon=h,
                                    country=country,
                                    window_start=str(wstart.date()),
                                    window_end=str(wend.date()),
                                    is_global=False
                                )
                            ))
    return planned, paths


def memory_probe_for_group(
    sample_key: TaskKey,
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
) -> int:
    """
    Builds a minimal model for the sample_key with probe=True and measures RSS delta.
    """
    time_col = cfg["data"].get("time_col", "TIME")
    target = cfg["data"]["target"]
    ws = pd.Timestamp(sample_key.window_start)
    we = pd.Timestamp(sample_key.window_end)
    lags = cfg["data"].get("lags")

    def _probe():
        if sample_key.is_global and sample_key.model == "nn":
            # For global models, load data from all countries
            all_train_data = []
            country_names = []
            
            for country, country_file in paths["countries"].items():
                df = ensure_time_sorted(read_df(country_file), time_col)
                train_df = df.loc[(df[time_col] >= ws) & (df[time_col] <= we)].copy()
                if len(train_df) >= 12 and target in df.columns:  # Minimal check
                    all_train_data.append(train_df)
                    country_names.append(country)
            
            if len(all_train_data) > 0:
                nn_params = cfg_for_nn_version(cfg, sample_key.nn_version)
                _ = _build_nn_global(
                    all_train_data, target, sample_key.quantile, sample_key.horizon, lags, 
                    nn_params, seed=int(cfg.get("seed", 42)), probe=True, 
                    country_names=country_names
                )
        else:
            # Non-global models (existing logic)
            country_file = paths["countries"][sample_key.country]
            df = ensure_time_sorted(read_df(country_file), time_col)
            train_df = df.loc[(df[time_col] >= ws) & (df[time_col] <= we)].copy()

            if sample_key.model == "lqr":
                _ = _build_lqr(train_df, target, sample_key.quantile, sample_key.horizon, lags, cfg_for_lqr(cfg), seed=int(cfg.get("seed", 42)), ar_only=False, probe=True)
            elif sample_key.model == "ar-qr":
                _ = _build_lqr(train_df, target, sample_key.quantile, sample_key.horizon, lags, cfg_for_arqr(cfg), seed=int(cfg.get("seed", 42)), ar_only=True, probe=True)
            elif sample_key.model == "lqr-per-country":
                _ = _build_lqr_per_country(train_df, target, sample_key.quantile, sample_key.horizon, lags, cfg_for_lqr_per_country(cfg), seed=int(cfg.get("seed", 42)), ar_only=False, probe=True)
            elif sample_key.model == "ar-qr-per-country":
                _ = _build_lqr_per_country(train_df, target, sample_key.quantile, sample_key.horizon, lags, cfg_for_arqr_per_country(cfg), seed=int(cfg.get("seed", 42)), ar_only=True, probe=True)
            elif sample_key.model == "nn":
                nn_params = cfg_for_nn_version(cfg, sample_key.nn_version)
                _ = _build_nn(train_df, target, sample_key.quantile, sample_key.horizon, lags, nn_params, seed=int(cfg.get("seed", 42)), probe=True)
            else:
                raise ValueError(f"Unknown model: {sample_key.model}")

    mb = estimate_memory_mb(_probe)
    mb += int(cfg["runtime"].get("mem_probe_fudge_mb", 200))
    return max(mb, 100)

def estimate_memory_for_tasks(planned: List[PlannedTask], cfg: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """
    To stay efficient, perform one probe per (model, nn_version, country, is_global) group and reuse its estimate.
    For global models, the memory usage will be higher but shared across all countries in the same window.
    """
    groups: Dict[Tuple[str, Optional[str], str, bool], List[int]] = {}
    for idx, pt in enumerate(planned):
        # For global models, we group by (model, nn_version, 'global', True)
        # For per-country models, we group by (model, nn_version, country, False)
        if pt.key.is_global:
            k = (pt.key.model, pt.key.nn_version, 'global', True)
        else:
            k = (pt.key.model, pt.key.nn_version, pt.key.country, False)
        groups.setdefault(k, []).append(idx)
    
    for (model, nnv, country_or_global, is_global), indices in tqdm(groups.items(), desc="Memory probe groups"):
        sample_idx = indices[0]
        sample_key = planned[sample_idx].key
        mb = memory_probe_for_group(sample_key, cfg, paths)
        for i in indices:
            planned[i].mem_est_mb = mb

# ----------------------------- Scheduler -----------------------------

class MemoryGate:
    def __init__(self, total_mb: int):
        self.total = total_mb
        self.avail = total_mb
        from threading import Condition
        self.cv = Condition()

    def acquire(self, need_mb: int):
        with self.cv:
            while need_mb > self.avail:
                self.cv.wait()
            self.avail -= need_mb

    def release(self, mb: int):
        with self.cv:
            self.avail += mb
            if self.avail > self.total:
                self.avail = self.total
            self.cv.notify_all()

def run_scheduler(planned: List[PlannedTask], cfg: Dict[str, Any], paths: Dict[str, Path]) -> None:
    # Load existing progress to skip done windows if not retraining
    progress_df = load_progress(paths["progress_parquet"])
    filtered: List[PlannedTask] = []
    for pt in planned:
        tk = pt.key
        # Skip if already done
        if not progress_df.empty:
            m = (
                (progress_df["MODEL"] == tk.model) &
                (progress_df["NN_VERSION"].fillna("") == (tk.nn_version or "")) &
                (progress_df["QUANTILE"] == tk.quantile) &
                (progress_df["HORIZON"] == tk.horizon) &
                (progress_df["COUNTRY"] == tk.country) &
                (progress_df["WINDOW_START"] == tk.window_start) &
                (progress_df["WINDOW_END"] == tk.window_end) &
                (progress_df.get("IS_GLOBAL", False) == tk.is_global) &
                (progress_df["STATUS"] == "done")
            )
            if m.any():
                continue
        # Skip if artifact exists and allow_reload without retrain
        if tk.is_global:
            model_dir = paths["models_root"] / f"{tk.model}-global" / (f"nn_version={tk.nn_version}" if tk.nn_version else "default") / f"q={tk.quantile}" / f"h={tk.horizon}" / f"window_{tk.window_start}_{tk.window_end}"
        else:
            model_dir = paths["models_root"] / tk.model / (f"nn_version={tk.nn_version}" if tk.nn_version else "default") / f"country={tk.country}" / f"q={tk.quantile}" / f"h={tk.horizon}" / f"window_{tk.window_start}_{tk.window_end}"
        model_path = model_dir / "model.pkl"
        if cfg["runtime"].get("allow_reload", True) and not cfg["runtime"].get("retrain_if_exists", False) and model_path.exists():
            # Mark as done to reflect cache
            row = {
                "MODEL": tk.model,
                "NN_VERSION": tk.nn_version,
                "QUANTILE": tk.quantile,
                "HORIZON": tk.horizon,
                "COUNTRY": tk.country,
                "WINDOW_START": tk.window_start,
                "WINDOW_END": tk.window_end,
                "STATUS": "done",
                "LAST_UPDATE": datetime.now().isoformat(),
                "ERROR_MSG": None,
                "MODEL_PATH": str(model_path),
                "FORECAST_ROWS": 0,
                "IS_GLOBAL": tk.is_global
            }
            upsert_progress_row(paths["progress_parquet"], FileLock(str(lock_path(paths["progress_parquet"]))), row)
            continue
        filtered.append(pt)

    if not filtered:
        logging.info("Nothing to do — all tasks are complete or cached.")
        return

    # Resource caps
    detected_cores = detect_cpu_cores()
    max_workers = detected_cores if str(cfg["runtime"].get("max_cores", "auto")) == "auto" else int(cfg["runtime"]["max_cores"])
    avail_bytes = detect_available_ram_bytes()
    if str(cfg["runtime"].get("max_ram_gb", "auto")) == "auto":
        cap_bytes = int(avail_bytes * float(cfg["runtime"].get("safety_ram_fraction", 0.8)))
    else:
        cap_bytes = int(float(cfg["runtime"]["max_ram_gb"]) * (1024 ** 3))
    mem_gate = MemoryGate(total_mb=int(cap_bytes / (1024 * 1024)))

    # Sort tasks by memory (descending) to pack better
    filtered.sort(key=lambda x: x.mem_est_mb, reverse=True)

    attempts: Dict[str, int] = {}
    next_i = 0
    running: Dict[Any, Tuple[PlannedTask, int]] = {}

    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_init, initargs=(bool(cfg["runtime"].get("thread_pinning", True)),)) as pool:
        pbar = tqdm(total=len(filtered), desc="Tasks", unit="task")
        while next_i < len(filtered) or running:
            # Launch while capacity
            launched = False
            while next_i < len(filtered) and len(running) < max_workers:
                pt = filtered[next_i]
                need = pt.mem_est_mb
                if need <= mem_gate.avail:
                  
                    mem_gate.acquire(need)
                    worker_paths = {
                        "progress_parquet": paths["progress_parquet"],
                        "errors_parquet": paths["errors_parquet"],
                        "models_root": paths["models_root"],
                        "forecasts_root": paths["forecasts_root"],
                        "countries": paths["countries"],
                    }
                    fut = pool.submit(execute_task, pt.key, cfg, worker_paths)
                    running[fut] = (pt, need)
                    next_i += 1
                    launched = True
                else:
                    break

            # Collect a completion (with periodic timeout to keep loop active)
            timeout = max(0.5, float(cfg["runtime"].get("progress_refresh_sec", 5)))
            collected = False
            for fut in as_completed(list(running.keys()), timeout=timeout):
                pt, need = running.pop(fut)
                mem_gate.release(need)
                try:
                    task, status, err, n_rows, rows = fut.result()
                except Exception as e:
                    task = pt.key
                    status = "failed"
                    err = f"{type(e).__name__}: {e}"
                    n_rows = 0
                    rows = None

                if status == "done" and rows is not None and n_rows > 0:
                    # Append to parquet for (q,h)
                    q = task.quantile
                    h = task.horizon
                    out_path = paths["forecasts_root"] / f"q={q}" / f"h={h}" / "rolling_window.parquet"
                    append_forecasts(out_path, rows)
                else:
                    # retry?
                    tid = task.id()
                    a = attempts.get(tid, 0)
                    if a < int(cfg["runtime"].get("retries", 1)):
                        attempts[tid] = a + 1
                        filtered.append(pt)
                pbar.update(1)
                collected = True
                break

            if not launched and not collected:
                # brief idle
                time.sleep(0.2)
        pbar.close()

    logging.info("Scheduling finished.")

# ----------------------------- Paths -----------------------------

def make_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
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
    # ensure dirs
    safe_mkdirs(paths["models_root"])
    safe_mkdirs(paths["forecasts_root"])
    safe_mkdirs(paths["progress_parquet"].parent)
    safe_mkdirs(paths["errors_parquet"].parent)
    safe_mkdirs(paths["logs_dir"])
    return paths

# ----------------------------- CLI -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser("Parallel Rolling-Window Quantile Forecast Runner (CPU-only)")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    ap.add_argument("--dry-run", action="store_true", help="Plan + memory probe only; do not run scheduler.")
    ap.add_argument("--only", type=str, default=None, help="Filter models, e.g. 'nn,lqr' or 'ar-qr'")
    args = ap.parse_args(argv)

    cfg = read_yaml(Path(args.config))

    # Defaults
    cfg.setdefault("runtime", {})
    cfg["runtime"].setdefault("thread_pinning", True)
    cfg["runtime"].setdefault("safety_ram_fraction", 0.8)
    cfg["runtime"].setdefault("mem_probe_fudge_mb", 200)
    cfg["runtime"].setdefault("max_cores", "auto")
    cfg["runtime"].setdefault("max_ram_gb", "auto")
    cfg["runtime"].setdefault("retries", 1)

    set_seeds(int(cfg.get("seed", 42)))
    paths = make_paths(cfg)
    setup_logging(paths["logs_dir"])

    # Optional model filtering
    if args.only:
        only = {x.strip() for x in args.only.split(",")}
        for m in cfg.get("models", []):
            m["enabled"] = m.get("type") in only

    # Plan
    planned, paths = plan_tasks(cfg, paths)
    if not planned:
        logging.info("No tasks planned. Check dates/horizons/data.")
        return 0

    # Memory probe
    logging.info("Estimating memory usage per (model, nn_version, country) group...")
    estimate_memory_for_tasks(planned, cfg, paths)

    if args.dry_run:
        # Show first few planned tasks with mem estimates
        print("---- DRY RUN (first 20 tasks) ----")
        for pt in planned[:20]:
            print({
                "MODEL": pt.key.model,
                "NN_VERSION": pt.key.nn_version,
                "Q": pt.key.quantile,
                "H": pt.key.horizon,
                "COUNTRY": pt.key.country,
                "WIN_START": pt.key.window_start,
                "WIN_END": pt.key.window_end,
                "IS_GLOBAL": pt.key.is_global,
                "EST_MB": pt.mem_est_mb,
            })
        return 0

    # Run
    run_scheduler(planned, cfg, paths)
    logging.info("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())