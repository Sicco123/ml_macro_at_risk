# R Setup for hqreg Integration

This guide explains how to install the required R dependencies to use the R-based `hqreg` solver in the Linear Quantile Regression API.

## Prerequisites

1. **R Installation**: Make sure R is installed on your system
   - Download from: https://www.r-project.org/
   - Or install via package manager:
     ```bash
     # macOS with Homebrew
     brew install r
     
     # Ubuntu/Debian
     sudo apt-get install r-base r-base-dev
     
     # CentOS/RHEL
     sudo yum install R
     ```

## Step 1: Install R Package

Install the `hqreg` package in R:

```r
# Option 1: Install from CRAN (stable version)
install.packages("hqreg")

# Option 2: Install latest version from GitHub (requires devtools)
install.packages("devtools")
devtools::install_github("CY-dev/hqreg")
```

## Step 2: Install Python R Interface

Install `rpy2` to enable Python-R communication:

```bash
# Install rpy2
pip install rpy2

# If you encounter issues, try:
pip install rpy2==3.5.14  # Use specific version if needed
```

### Common Installation Issues

1. **macOS users**: You may need to install additional dependencies:
   ```bash
   # Install R development tools
   brew install pkg-config
   brew install --cask xquartz  # For X11 support
   ```

2. **Linux users**: Install R development headers:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install r-base-dev
   
   # CentOS/RHEL
   sudo yum install R-devel
   ```

3. **Windows users**: 
   - Install R from https://cran.r-project.org/bin/windows/base/
   - Install Rtools from https://cran.r-project.org/bin/windows/Rtools/
   - Make sure R is in your PATH

## Step 3: Verify Installation

Test the installation in Python:

```python
# Test R interface
try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    
    # Test hqreg package
    hqreg = importr('hqreg')
    print("✓ R interface and hqreg package successfully installed!")
    
except ImportError as e:
    print(f"✗ Installation issue: {e}")
```

## Step 4: Usage Example

Once installed, you can use the R-based solver:

```python
from src.lqr_api import LQR

# Initialize with R-based solver
lqr = LQR(
    data_list=your_data,
    target="your_target",
    quantiles=[0.1, 0.5, 0.9],
    forecast_horizons=[1, 4, 8],
    solver="hqreg",  # Use R-based solver
    alpha=0.5,       # Elastic-net mixing parameter
    preprocess="standardize",
    screen="ASR"
)

# Fit and predict as usual
lqr.fit()
predictions = lqr.predict(test_data)
```

## R-specific Parameters

When using `solver="hqreg"`, you can configure additional R-specific parameters:

- `gamma`: Huber loss tuning parameter (default: IQR(y)/10)
- `nlambda`: Number of lambda values for regularization path (default: 100)
- `lambda_min`: Smallest lambda value as fraction of lambda.max (default: 0.05)
- `preprocess`: Preprocessing technique ("standardize" or "rescale")
- `screen`: Screening rule ("ASR", "SR", or "none")
- `max_iter`: Maximum iterations (default: 10000)
- `eps`: Convergence threshold (default: 1e-7)

## Troubleshooting

### Error: "Failed to import R hqreg package"
```bash
# Start R and install hqreg manually
R
> install.packages("hqreg")
> quit()
```

### Error: "R interface (rpy2) is required"
```bash
# Reinstall rpy2 with verbose output
pip uninstall rpy2
pip install rpy2 --verbose
```

### Environment Variables (if needed)
```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export R_HOME=/usr/lib/R  # Adjust path as needed
export LD_LIBRARY_PATH=$R_HOME/lib:$LD_LIBRARY_PATH
```

For more details on hqreg, see: https://github.com/CY-dev/hqreg