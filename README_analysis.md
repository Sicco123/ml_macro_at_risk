# Quantile Regression Results Analysis Tool

This comprehensive analysis tool processes the output from the quantile regression runner and produces detailed tables, visualizations, and statistical tests.

## Features

### 📊 **Pinball Loss Tables**
- **Format**: Both readable CSV and professional LaTeX tables
- **Structure**: Countries on Y-axis, Models on X-axis
- **Aggregation**: Includes averages across all countries
- **COVID Options**: Analysis with and without COVID period (2020-03 to 2021-12)
- **Output**: 
  - `tables/pinball_q{quantile}_h{horizon}_{incl/excl}_covid.csv`
  - `latex/pinball_q{quantile}_h{horizon}_{incl/excl}_covid.tex`

### 📈 **Forecast Visualizations**
- **Per Country**: Individual plots for each country
- **Flexible Quantiles**: Option for separate plots per quantile or combined
- **Horizon Separation**: Different plots for each forecast horizon
- **COVID Highlighting**: Visual indication of COVID-19 period
- **Quality Options**: 
  - **Speed**: PNG format, 100 DPI
  - **High Quality**: PDF format, 300 DPI
- **Output**: `figures/forecasts_{country}_h{horizon}_q{quantiles}.{png/pdf}`

### 🔍 **Statistical Tests**

#### **Diebold-Mariano Tests**
- **Purpose**: Compare forecast accuracy between models
- **Benchmark**: Configurable reference model
- **HAC Adjustment**: Newey-West adjustment for multi-step forecasts
- **Significance Levels**: 5% and 10% testing
- **COVID Options**: Analysis with and without COVID period
- **Output**: 
  - `tests/diebold_mariano_results.csv` (detailed results)
  - `tests/dm_test_q{quantile}_h{horizon}_{incl/excl}_covid.csv` (summary tables)

#### **Model Confidence Sets (MCS)**
- **Purpose**: Identify set of statistically equivalent "best" models
- **Method**: Hansen, Lunde, and Nason (2011) procedure
- **Bootstrap**: 1000 bootstrap replications
- **Significance**: Configurable α level (default 10%)
- **Output**:
  - `confidence_sets/mcs_results.csv` (detailed results)
  - `confidence_sets/mcs_q{quantile}_h{horizon}_{incl/excl}_covid.csv` (summary tables)

## Quick Start

### 1. **Basic Analysis**
```bash
python analyze_results.py --config configs/analysis_config.yaml
```

### 2. **High-Quality Output**
```bash
python quick_analysis.py --high-quality
```

### 3. **Specific Analysis Types**
```bash
# Only create tables
python quick_analysis.py --type tables

# Only create plots  
python quick_analysis.py --type plots

# Only run statistical tests
python quick_analysis.py --type tests

# Full analysis (default)
python quick_analysis.py --type full
```

## Configuration

### **Standard Configuration** (`configs/analysis_config.yaml`)
```yaml
input:
  forecasts_dir: "outputs/forecasts"

output:
  base_dir: "analysis_output"

analysis:
  benchmark_model: "ar-qr-per-country"  # Reference for DM tests
  mcs_alpha: 0.1                        # MCS significance level

plots:
  separate_quantiles: true   # Separate plot per quantile
  high_quality: false        # Speed over quality

covid:
  start_date: "2020-03-01"   # COVID period start
  end_date: "2021-12-31"     # COVID period end
```

### **High-Quality Configuration** (`configs/analysis_config_hq.yaml`)
```yaml
plots:
  separate_quantiles: false  # Combined quantile plots
  high_quality: true         # PDF format, 300 DPI
```

## Output Structure

```
analysis_output/
├── tables/                    # Pinball loss tables (CSV)
├── latex/                     # LaTeX table files
├── figures/                   # Forecast plots
├── tests/                     # Statistical test results
├── confidence_sets/           # Model confidence set results
└── analysis_report.md         # Summary report
```

## Advanced Usage

### **Custom Benchmark Model**
Edit the configuration file to change the benchmark:
```yaml
analysis:
  benchmark_model: "lqr"  # Use pooled LQR as benchmark
```

### **COVID Period Adjustment**
Modify COVID period dates:
```yaml
covid:
  start_date: "2020-02-01"  # Earlier start
  end_date: "2022-06-30"    # Later end
```

### **Multiple Configurations**
Create custom configuration files for different analyses:
```bash
# Analysis excluding specific periods
python analyze_results.py --config configs/my_custom_config.yaml
```

## Statistical Methodology

### **Pinball Loss**
For quantile τ, the pinball loss is:
```
L(y, q) = (y - q) * (τ - I(y < q))
```
where I(·) is the indicator function.

### **Diebold-Mariano Test**
Tests the null hypothesis that two forecasts have equal expected loss:
- **Statistic**: DM = d̄ / √(V̂ar(d̄))
- **HAC Adjustment**: Newey-West for multi-step forecasts
- **Distribution**: Asymptotically N(0,1)

### **Model Confidence Set**
Constructs the set of models that are not significantly worse than the best:
- **Procedure**: Sequential testing with bootstrap p-values
- **Controls**: Family-wise error rate at level α
- **Interpretation**: Models in MCS are statistically equivalent

## Requirements

### **Python Packages**
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels pyarrow
```

### **Data Structure**
The tool expects forecast data in the format produced by `quant_runner.py`:
- **Location**: `outputs/forecasts/q={quantile}/h={horizon}/rolling_window.parquet`
- **Columns**: TIME, COUNTRY, TRUE_DATA, FORECAST, HORIZON, QUANTILE, MODEL, etc.

## Troubleshooting

### **"No forecast data found"**
- Check that `outputs/forecasts/` contains parquet files
- Verify the correct path in configuration

### **"Only 1 model found, need at least 2"**
- Statistical tests require multiple models for comparison
- Run quantile regression with multiple model types enabled

### **Missing plots or tables**
- Check that output directories have write permissions
- Verify sufficient disk space for high-quality outputs

## Examples

### **Typical Workflow**
1. **Run quantile regression**: Generate forecasts with multiple models
2. **Quick analysis**: `python quick_analysis.py` for overview
3. **High-quality output**: `python quick_analysis.py --high-quality` for publication
4. **Custom analysis**: Modify configuration for specific requirements

### **Model Comparison Study**
```bash
# Enable multiple models in quantile runner
# Run: python quant_runner.py --config configs/multi_model_config.yaml

# Analyze results
python analyze_results.py --config configs/analysis_config.yaml

# Check Diebold-Mariano results
cat analysis_output/tests/diebold_mariano_results.csv

# Review Model Confidence Sets
cat analysis_output/confidence_sets/mcs_results.csv
```

### **LaTeX Integration**
Include generated tables in your LaTeX document:
```latex
\input{analysis_output/latex/pinball_q0.1_h1_incl_covid.tex}
```

## Performance

### **Speed Mode** (default)
- PNG plots at 100 DPI
- Fast processing for exploratory analysis
- Suitable for screen viewing

### **High-Quality Mode**
- PDF plots at 300 DPI  
- Publication-ready output
- Larger file sizes, longer processing time

## Citation

If you use this analysis tool in your research, please cite:
- **Diebold-Mariano**: Diebold, F.X. and Mariano, R.S. (1995)
- **Model Confidence Sets**: Hansen, P.R., Lunde, A. and Nason, J.M. (2011)
- **Quantile Regression**: Koenker, R. and Bassett Jr, G. (1978)
