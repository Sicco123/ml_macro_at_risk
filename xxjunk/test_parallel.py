#!/usr/bin/env python3
"""
Test script for the parallel runner
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Test the parallel runner with different configurations"""
    
    # Check if parallel_runner.py exists
    runner_path = Path("parallel_runner.py")
    if not runner_path.exists():
        print("Error: parallel_runner.py not found")
        return
    
    config_path = Path("configs/parallel_config.yaml")
    if not config_path.exists():
        print("Error: parallel_config.yaml not found")
        return
    
    print("="*60)
    print("TESTING PARALLEL QUANTILE FORECASTING RUNNER")
    print("="*60)
    
    # Test 1: Check that we can import everything
    print("\n1. Testing imports...")
    try:
        import parallel_runner
        print("✓ parallel_runner imported successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return
    
    # Test 2: Generate default config
    print("\n2. Testing config generation...")
    try:
        subprocess.run([
            sys.executable, "parallel_runner.py", 
            "--write-config", "test_config.yaml"
        ], check=True)
        print("✓ Default config generated successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Config generation failed: {e}")
        return
    
    # Test 3: Dry run with existing config
    print("\n3. Testing dry run...")
    try:
        result = subprocess.run([
            sys.executable, "parallel_runner.py",
            "--config", str(config_path),
            "--dry-run"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Dry run completed successfully")
        else:
            print(f"✗ Dry run failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
    except subprocess.TimeoutExpired:
        print("✗ Dry run timed out")
    except Exception as e:
        print(f"✗ Dry run failed: {e}")
    
    # Test 4: Check data directory
    print("\n4. Checking data availability...")
    data_path = Path("processed_per_country")
    if data_path.exists():
        parquet_files = list(data_path.glob("*.parquet"))
        print(f"✓ Found {len(parquet_files)} parquet files in {data_path}")
        
        if len(parquet_files) > 0:
            print("Sample files:", [f.name for f in parquet_files[:5]])
        else:
            print("⚠ No parquet files found - parallel runner may create synthetic data")
    else:
        print(f"⚠ Data directory {data_path} not found - parallel runner may create synthetic data")
    
    # Test 5: Check source code availability
    print("\n5. Checking source code...")
    src_path = Path("src")
    if src_path.exists():
        required_files = ["ensemble_nn_api.py", "lqr_api.py", "utils/__init__.py"]
        missing_files = []
        
        for file in required_files:
            if not (src_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"⚠ Missing source files: {missing_files}")
        else:
            print("✓ All required source files found")
    else:
        print("✗ Source directory not found")
        return
    
    print("\n" + "="*60)
    print("READY TO RUN PARALLEL FORECASTING")
    print("="*60)
    print()
    print("To run with limited resources (recommended for testing):")
    print(f"python parallel_runner.py --config {config_path} --max-workers 2")
    print()
    print("To run with full resources:")
    print(f"python parallel_runner.py --config {config_path}")
    print()
    print("To monitor progress, check the parallel_outputs/progress/ directory")
    print("Forecasts will be saved to parallel_outputs/forecasts/")
    print("Models will be saved to parallel_outputs/models/")

if __name__ == "__main__":
    main()
