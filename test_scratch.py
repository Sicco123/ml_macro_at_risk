#!/usr/bin/env python3
"""
Test script to verify scratch directory functionality
"""

import os
import sys
import yaml
from pathlib import Path

# Add the project directory to path so we can import from quant_runner_2
sys.path.insert(0, str(Path(__file__).parent))

from quant_runner_2 import make_paths, copy_scratch_to_home, cleanup_scratch

def test_scratch_functionality():
    """Test the scratch directory functionality"""
    
    # Create a test config with scratch directory
    test_config = {
        "io": {
            "output_root": "/home/skooiker/ml_macro_at_risk/test_output",
            "scratch_dir": "/scratch-local/skooiker/test_scratch",
            "models_dir": "models",
            "forecasts_dir": "forecasts",
            "progress_file": "progress/progress.parquet",
            "errors_file": "progress/errors.parquet",
            "logs_dir": "logs"
        },
        "data": {
            "path": "/home/skooiker/ml_macro_at_risk/processed_per_country"
        }
    }
    
    print("=" * 60)
    print("Testing scratch directory functionality")
    print("=" * 60)
    
    # Test path creation
    print("\n1. Testing path creation...")
    paths = make_paths(test_config)
    
    print(f"   Use scratch: {paths.get('use_scratch', False)}")
    print(f"   Working root: {paths['working_root']}")
    print(f"   Output root: {paths['output_root']}")
    print(f"   Models path: {paths['models_root']}")
    print(f"   Forecasts path: {paths['forecasts_root']}")
    print(f"   Progress path: {paths['progress_parquet']}")
    
    # Test creating some dummy files
    if paths.get('use_scratch', False):
        print("\n2. Creating test files in scratch...")
        
        # Create a dummy model file
        model_dir = paths['models_root'] / "test_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.pkl").write_text("dummy model data")
        
        # Create a dummy forecast file
        forecast_dir = paths['forecasts_root'] / "q=0.5" / "h=1"
        forecast_dir.mkdir(parents=True, exist_ok=True)
        (forecast_dir / "forecasts.parquet").write_text("dummy forecast data")
        
        # Create a dummy progress file
        (paths['progress_parquet']).parent.mkdir(parents=True, exist_ok=True)
        (paths['progress_parquet']).write_text("dummy progress data")
        
        print("   Created dummy files in scratch directory")
        
        print("\n3. Testing copy back to home...")
        copy_scratch_to_home(paths, "test_instance")
        
        print("\n4. Verifying files copied to home...")
        home_model = paths['output_root'] / "models" / "test_model" / "model.pkl"
        home_forecast = paths['output_root'] / "forecasts" / "q=0.5" / "h=1" / "forecasts.parquet"
        home_progress = paths['output_root'] / "progress" / "progress.parquet"
        
        print(f"   Model file exists in home: {home_model.exists()}")
        print(f"   Forecast file exists in home: {home_forecast.exists()}")
        print(f"   Progress file exists in home: {home_progress.exists()}")
        
        print("\n5. Testing cleanup...")
        cleanup_scratch(paths, "test_instance")
        print(f"   Scratch directory cleaned: {not paths['working_root'].exists()}")
        
    else:
        print("\n   Scratch directory not available - using home directory directly")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_scratch_functionality()
