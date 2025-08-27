#!/usr/bin/env python3
"""
Test script for per-country LQR and AR-QR functionality
"""

import sys
from pathlib import Path
import yaml

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def test_per_country_config():
    """Test that the per-country configuration is properly parsed"""
    
    # Load the per-country config
    config_path = project_root / "configs" / "experiment_per_country.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=== Testing Per-Country Configuration ===")
    print(f"Config loaded from: {config_path}")
    print(f"Number of model types: {len(config['models'])}")
    
    # Check that per-country models are present
    model_types = [m['type'] for m in config['models']]
    
    expected_types = ['ar-qr', 'lqr', 'lqr-per-country', 'ar-qr-per-country', 'nn']
    
    print(f"Found model types: {model_types}")
    print(f"Expected model types: {expected_types}")
    
    for expected in expected_types:
        if expected in model_types:
            print(f"✓ {expected} found")
        else:
            print(f"✗ {expected} missing")
    
    # Check enabled status
    enabled_models = [(m['type'], m['enabled']) for m in config['models']]
    print("\nEnabled status:")
    for model_type, enabled in enabled_models:
        status = "✓" if enabled else "✗"
        print(f"  {status} {model_type}: {enabled}")
    
    # Check per-country models are enabled
    lqr_pc_enabled = any(m['type'] == 'lqr-per-country' and m['enabled'] for m in config['models'])
    ar_qr_pc_enabled = any(m['type'] == 'ar-qr-per-country' and m['enabled'] for m in config['models'])
    
    print(f"\nPer-country models enabled:")
    print(f"  LQR per-country: {'✓' if lqr_pc_enabled else '✗'}")
    print(f"  AR-QR per-country: {'✓' if ar_qr_pc_enabled else '✗'}")
    
    return lqr_pc_enabled and ar_qr_pc_enabled

def test_imports():
    """Test that all required imports work"""
    print("\n=== Testing Imports ===")
    
    try:
        from quant_runner import (
            cfg_for_lqr_per_country, 
            cfg_for_arqr_per_country,
            _build_lqr_per_country
        )
        print("✓ Per-country configuration functions imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    try:
        from src.lqr_api import LQRModel
        print("✓ LQRModel imported successfully")
    except ImportError as e:
        print(f"✗ LQRModel import error: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("Testing Per-Country LQR and AR-QR Functionality")
    print("=" * 50)
    
    # Test configuration
    config_ok = test_per_country_config()
    
    # Test imports
    imports_ok = test_imports()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Configuration test: {'✓ PASS' if config_ok else '✗ FAIL'}")
    print(f"Imports test: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    
    overall_pass = config_ok and imports_ok
    print(f"Overall: {'✓ PASS' if overall_pass else '✗ FAIL'}")
    
    if overall_pass:
        print("\n✓ Per-country LQR and AR-QR functionality is ready!")
        print("  You can now run with:")
        print("  python quant_runner.py --config configs/experiment_per_country.yaml")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    
    return 0 if overall_pass else 1

if __name__ == "__main__":
    sys.exit(main())
