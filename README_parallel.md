# Parallel Quantile Forecasting Runner

This tool runs quantile forecasting models in parallel using rolling windows with memory estimation, resource management, and automatic resumption capabilities.

## Features

- **Memory Estimation**: Estimates memory usage per task to avoid oversubscription
- **Resource Management**: Respects CPU and RAM limits for parallel execution
- **Rolling Window Forecasting**: Implements rolling window out-of-sample forecasting
- **Model Caching**: Saves models and skips training when resuming
- **Progress Tracking**: Comprehensive progress monitoring and error logging
- **Automatic Resumption**: Continues from where it left off on restart
- **Error Handling**: Robust error handling with retry logic

## Supported Models

The runner uses your existing model implementations:

- **ar_qr**: Autoregressive Quantile Regression (using LQR with AR lags)
- **lqr**: Linear Quantile Regression (using your `src.lqr_api.LQR`)
- **ensemble_nn**: Factor Neural Networks (using your `src.ensemble_nn_api.EnsembleNNAPI`)

## Quick Start

### 1. Test the Setup

```bash
python3 test_parallel.py
```

This will verify that all dependencies and source files are available.

### 2. Create/Edit Configuration

Generate a default configuration:

```bash
python3 parallel_runner.py --write-config my_config.yaml
```

Or use the provided configuration:

```bash
configs/parallel_config.yaml
```

### 3. Run with Limited Resources (Recommended for Testing)

```bash
python3 parallel_runner.py --config configs/parallel_config.yaml --max-workers 2
```

### 4. Monitor Progress

In another terminal, monitor the progress:

```bash
python3 monitor_progress.py --watch
```

Or check progress once:

```bash
python3 monitor_progress.py
```

### 5. Run with Full Resources

```bash
python3 parallel_runner.py --config configs/parallel_config.yaml
```

## Configuration

### Key Configuration Sections

#### Runtime Settings
```yaml
runtime:
  allow_reload: true              # Load existing models instead of retraining
  retrain_if_exists: false        # Force retrain even if model exists
  max_cores: auto                 # CPU cores to use ('auto' or number)
  safety_ram_fraction: 0.8        # Use 80% of available RAM
  max_ram_gb: auto                # RAM limit ('auto' or number in GB)
  retries: 1                      # Retry failed tasks once
```

#### Data Settings
```yaml
data:
  target: prc_hicp_manr_CP00      # Target variable name
  lags: [1, 2, 3, 6, 12]         # Lag features to create
  horizons: [1, 3, 6, 12]        # Forecast horizons
  quantiles: [0.1, 0.5, 0.9]     # Quantiles to forecast
  missing: forward_fill_then_mean # Missing value handling
  scale: per_country              # Scaling policy
```

#### Rolling Window Settings
```yaml
rolling_window:
  size: 60                        # Window size in periods
  step: 1                         # Step size between windows
  start_date: "2018-01-01"        # First forecast date
  end_date: "2023-12-01"          # Last forecast date
```

#### Model Settings
```yaml
models:
  ar_qr:
    enabled: true
    p: 12                         # AR lags
    
  lqr:
    enabled: true
    alphas: [7.5, 10, 15, 20, 25, 30, 35]  # Regularization strengths
    solver: huberized
    rff_features: 1000            # Random Fourier Features
    
  ensemble_nn:
    enabled: true
    units_per_layer: [32, 32, 32, 32, 32]
    learning_rate: 5.0e-5
    epochs: 1500
    batch_size: 32
    patience: 50
    parallel_models: 10
```

## Output Structure

```
parallel_outputs/
├── forecasts/          # Forecast parquet files (one per model/quantile/horizon)
├── models/             # Trained model files (pkl format)
├── progress/           # Progress tracking files
│   ├── progress.parquet   # Main progress tracking
│   └── errors.parquet     # Error details
└── logs/               # Log files
```

## Monitoring and Resumption

### Progress Monitoring

The runner tracks progress for each task combination of:
- Model type (ar_qr, lqr, ensemble_nn)
- Country
- Quantile
- Horizon
- Rolling window period

Status values:
- `pending`: Not started yet
- `training`: Currently training
- `done`: Completed successfully
- `failed`: Failed with error

### Automatic Resumption

When restarting the runner:
- If `allow_reload: true` and `retrain_if_exists: false`: Skips completed tasks
- If `retrain_if_exists: true`: Forces retraining of all tasks
- Failed tasks are automatically retried up to `retries` times

### Error Handling

- Tasks that fail are marked as `failed` with error details
- Other tasks continue running
- Failed tasks can be retried by running again
- Memory estimation helps prevent out-of-memory errors

## Resource Management

### Memory Estimation

The runner estimates memory usage by:
1. Running a dry-run training iteration for each model type
2. Measuring peak memory usage
3. Adding a safety margin (`mem_probe_fudge_mb`)
4. Scheduling tasks to stay within memory limits

### CPU Management

- Detects available CPU cores automatically
- Respects `max_cores` setting
- Sets BLAS thread limits to prevent oversubscription
- Groups tasks by model type for efficient scheduling

## Forecast Output Format

Forecast files are saved as parquet with columns:
- `TIME`: Forecast date
- `COUNTRY`: Country code
- `TARGET`: Target variable name
- `MODEL`: Model type used
- `QUANTILE`: Quantile level
- `HORIZON`: Forecast horizon
- `WINDOW_START`: Training window start date
- `WINDOW_END`: Training window end date
- `FORECAST`: Forecast value

## Tips for Large-Scale Runs

1. **Start Small**: Test with `--max-workers 2` and limited horizons/quantiles
2. **Monitor Resources**: Use `monitor_progress.py --watch` to track progress
3. **Use Resumption**: The runner is designed to be stopped and restarted safely
4. **Check Logs**: Monitor log files for detailed information and errors
5. **Disk Space**: Ensure sufficient disk space for models and forecasts
6. **Memory**: Start with conservative memory settings and adjust as needed

## Troubleshooting

### Import Errors
Ensure all source files are in the `src/` directory and the working directory is correct.

### Memory Issues
- Reduce `max_cores` or increase `safety_ram_fraction`
- Increase `mem_probe_fudge_mb` for more conservative memory estimation

### Slow Performance
- Check if models are being cached properly (`allow_reload: true`)
- Reduce batch sizes or model complexity
- Monitor CPU and memory usage

### Failed Tasks
- Check error messages in progress tracking
- Review log files for detailed error information
- Consider reducing model complexity or data preprocessing issues

## Command Line Options

```bash
python3 parallel_runner.py --help
```

Available options:
- `--config CONFIG`: Path to YAML config file
- `--write-config FILE`: Write default config and exit
- `--max-workers N`: Override max workers setting
- `--dry-run`: Show what would be done without executing

## Dependencies

The runner uses your existing codebase and requires:
- All dependencies from your `src/` modules
- `pandas`, `numpy`, `torch`, `psutil`, `filelock`, `joblib`, `pyyaml`
- Your existing model APIs: `EnsembleNNAPI`, `LQR`
- Your existing utility functions for data loading and preprocessing
