# Per-Country LQR and AR-QR Models

This document explains the new per-country modeling functionality added to the quantile regression runner.

## Overview

The runner now supports estimating Linear Quantile Regression (LQR) and Autoregressive Quantile Regression (AR-QR) models separately for each country, rather than pooling all countries into a single model. This allows for country-specific parameter estimation and can potentially improve forecasting performance by capturing country-specific dynamics.

## New Model Types

### 1. `lqr-per-country`
- Estimates a separate LQR model for each country using only that country's data
- Uses all available features (not just autoregressive terms)
- Cross-validation is performed within each country's data

### 2. `ar-qr-per-country`
- Estimates a separate AR-QR model for each country using only that country's data
- Uses only autoregressive terms (target variable lags)
- Cross-validation is performed within each country's data

## Configuration

Add the new model types to your YAML configuration file:

```yaml
models:
  - type: lqr-per-country
    enabled: true
    params:
      solver: huberized
      alphas: [7.5, 10, 15, 20, 25, 30, 35]
      use_cv: true
      cv_splits: 5
  - type: ar-qr-per-country
    enabled: true
    params:
      solver: huberized
      alphas: [7.5, 10, 15, 20, 25, 30, 35]
      use_cv: true
      cv_splits: 5
```

## Key Differences from Pooled Models

### Pooled Models (`lqr`, `ar-qr`)
- Combine data from all countries into a single dataset
- Estimate one set of coefficients that applies to all countries
- Cross-validation uses all countries' data
- Assumes similar relationships across countries

### Per-Country Models (`lqr-per-country`, `ar-qr-per-country`)
- Use only individual country data for each model
- Estimate separate coefficients for each country
- Cross-validation uses only that country's data
- Allows for country-specific relationships

## Implementation Details

### Model Training
Each per-country model:
1. Loads only the specific country's data for the training window
2. Creates lagged features and forecast targets using that country's data only
3. Performs k-fold cross-validation within the country's data to select optimal alpha
4. Fits the final model using the selected alpha

### Prediction
- Uses the country-specific model to generate forecasts
- Same prediction interface as pooled models
- Results are stored with the same format for easy comparison

### Memory and Storage
- Each (country, quantile, horizon, window) combination gets its own model file
- Memory usage scales with number of countries
- Storage requirements increase proportionally with country count

## Example Usage

### Basic Run
```bash
python quant_runner.py --config configs/experiment_per_country.yaml
```

### Dry Run (Planning Only)
```bash
python quant_runner.py --config configs/experiment_per_country.yaml --dry-run
```

### Run Only Per-Country Models
```bash
python quant_runner.py --config configs/experiment_per_country.yaml --only "lqr-per-country,ar-qr-per-country"
```

## Performance Considerations

### Advantages
- Captures country-specific dynamics
- May improve forecast accuracy for heterogeneous countries
- Allows for country-specific feature importance
- Better suited when countries have different economic structures

### Disadvantages
- Requires more memory and storage
- Longer training time (scales with number of countries)
- May overfit with limited country-specific data
- Less data for cross-validation per model

## Comparison with Pooled Models

To compare performance between pooled and per-country approaches:

1. Enable both model types in configuration
2. Run the analysis
3. Compare forecast accuracy metrics in the output files
4. Results will be stored separately by model type for easy comparison

## Code Structure

The implementation adds:
- `_build_lqr_per_country()`: Builds per-country LQR/AR-QR models
- `cfg_for_lqr_per_country()`: Configuration parser for per-country LQR
- `cfg_for_arqr_per_country()`: Configuration parser for per-country AR-QR
- Updated task execution logic to handle new model types
- Memory probing support for per-country models

## Testing

Use the included test script to verify functionality:
```bash
python test_per_country.py
```

This will validate configuration parsing and import functionality.
