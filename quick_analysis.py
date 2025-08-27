#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Analysis Runner for Quantile Regression Results

This script provides easy commands to run different types of analysis.
"""

import argparse
import sys
from pathlib import Path
from analyze_results import QuantileAnalyzer

def run_analysis(config_path: str, analysis_type: str = "full"):
    """Run analysis with specified configuration and type."""
    
    analyzer = QuantileAnalyzer(config_path)
    
    print(f"Running {analysis_type} analysis...")
    print("=" * 50)
    
    # Always load data first
    analyzer.load_data()
    
    if analysis_type == "tables":
        analyzer.create_pinball_tables()
    elif analysis_type == "plots":
        analyzer.plot_forecasts_by_country()
    elif analysis_type == "tests":
        analyzer.run_diebold_mariano_tests()
        analyzer.run_model_confidence_sets()
    elif analysis_type == "full":
        analyzer.run_full_analysis()
    else:
        print(f"Unknown analysis type: {analysis_type}")
        return 1
    
    print("=" * 50)
    print("Analysis complete!")
    return 0

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Quick analysis runner")
    parser.add_argument("--type", choices=["tables", "plots", "tests", "full"], 
                       default="full", help="Type of analysis to run")
    parser.add_argument("--config", default="configs/analysis_config.yaml",
                       help="Path to analysis configuration file")
    parser.add_argument("--high-quality", action="store_true",
                       help="Use high quality configuration")
    
    args = parser.parse_args()
    
    # Use high quality config if requested
    if args.high_quality:
        config_path = "configs/analysis_config_hq.yaml"
        print("Using high-quality configuration")
    else:
        config_path = args.config
    
    return run_analysis(config_path, args.type)

if __name__ == "__main__":
    sys.exit(main())
