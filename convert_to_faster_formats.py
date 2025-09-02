#!/usr/bin/env python3
"""
Convert parquet files to faster formats for quant_runner_2.py

This script converts your parquet files to various faster formats:
1. Feather (fastest for pandas, good compression)
2. NumPy arrays (fastest for pure numeric data)
3. HDF5 (good for complex queries, compression)
4. Pickle (fastest Python-specific format)

Usage:
    python convert_to_faster_formats.py --format feather
    python convert_to_faster_formats.py --format numpy
    python convert_to_faster_formats.py --format hdf5
    python convert_to_faster_formats.py --format pickle
    python convert_to_faster_formats.py --format all
"""

import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_feather(input_dir: Path, output_dir: Path) -> None:
    """Convert parquet files to feather format (fastest pandas I/O)"""
    output_dir.mkdir(exist_ok=True)
    
    parquet_files = list(input_dir.glob("*.parquet"))
    logger.info(f"Converting {len(parquet_files)} parquet files to feather format...")
    
    total_start = time.time()
    for parquet_file in parquet_files:
        start_time = time.time()
        
        # Read parquet
        df = pd.read_parquet(parquet_file)
        
        # Write feather
        output_file = output_dir / parquet_file.with_suffix('.feather').name
        df.to_feather(output_file)
        
        end_time = time.time()
        logger.info(f"Converted {parquet_file.name} -> {output_file.name} in {end_time - start_time:.2f}s")
    
    total_end = time.time()
    logger.info(f"Feather conversion completed in {total_end - total_start:.2f}s total")

def convert_to_numpy(input_dir: Path, output_dir: Path) -> None:
    """Convert parquet files to memory-mapped numpy arrays (fastest for numeric data)"""
    output_dir.mkdir(exist_ok=True)
    
    parquet_files = list(input_dir.glob("*.parquet"))
    logger.info(f"Converting {len(parquet_files)} parquet files to numpy format...")
    
    total_start = time.time()
    for parquet_file in parquet_files:
        start_time = time.time()
        
        # Read parquet
        df = pd.read_parquet(parquet_file)
        
        # Save data as numpy array
        output_file = output_dir / parquet_file.with_suffix('.npy').name
        np.save(output_file, df.values)
        
        # Save column names separately
        columns_file = output_dir / parquet_file.with_suffix('.columns.txt').name
        with open(columns_file, 'w') as f:
            for col in df.columns:
                f.write(f"{col}\n")
        
        # Save index separately if it's not default
        if not isinstance(df.index, pd.RangeIndex) or df.index.name is not None:
            index_file = output_dir / parquet_file.with_suffix('.index.npy').name
            np.save(index_file, df.index.values)
        
        end_time = time.time()
        logger.info(f"Converted {parquet_file.name} -> {output_file.name} in {end_time - start_time:.2f}s")
    
    total_end = time.time()
    logger.info(f"NumPy conversion completed in {total_end - total_start:.2f}s total")

def convert_to_hdf5(input_dir: Path, output_dir: Path) -> None:
    """Convert parquet files to HDF5 format (good compression and complex queries)"""
    output_dir.mkdir(exist_ok=True)
    
    parquet_files = list(input_dir.glob("*.parquet"))
    logger.info(f"Converting {len(parquet_files)} parquet files to HDF5 format...")
    
    total_start = time.time()
    for parquet_file in parquet_files:
        start_time = time.time()
        
        # Read parquet
        df = pd.read_parquet(parquet_file)
        
        # Write HDF5 with compression
        output_file = output_dir / parquet_file.with_suffix('.h5').name
        df.to_hdf(output_file, key='data', mode='w', complib='blosc', complevel=9)
        
        end_time = time.time()
        logger.info(f"Converted {parquet_file.name} -> {output_file.name} in {end_time - start_time:.2f}s")
    
    total_end = time.time()
    logger.info(f"HDF5 conversion completed in {total_end - total_start:.2f}s total")

def convert_to_pickle(input_dir: Path, output_dir: Path) -> None:
    """Convert parquet files to pickle format (fastest Python-specific format)"""
    output_dir.mkdir(exist_ok=True)
    
    parquet_files = list(input_dir.glob("*.parquet"))
    logger.info(f"Converting {len(parquet_files)} parquet files to pickle format...")
    
    total_start = time.time()
    for parquet_file in parquet_files:
        start_time = time.time()
        
        # Read parquet
        df = pd.read_parquet(parquet_file)
        
        # Write pickle with highest protocol
        output_file = output_dir / parquet_file.with_suffix('.pkl').name
        df.to_pickle(output_file, protocol=5)  # Highest pickle protocol
        
        end_time = time.time()
        logger.info(f"Converted {parquet_file.name} -> {output_file.name} in {end_time - start_time:.2f}s")
    
    total_end = time.time()
    logger.info(f"Pickle conversion completed in {total_end - total_start:.2f}s total")

def benchmark_loading(input_dir: Path, format_dirs: dict, sample_file: str = "AT") -> None:
    """Benchmark loading times for different formats"""
    logger.info("Benchmarking loading times...")
    
    formats_to_test = []
    
    # Test parquet (original)
    parquet_file = input_dir / f"{sample_file}.parquet"
    if parquet_file.exists():
        formats_to_test.append(("parquet", parquet_file, lambda f: pd.read_parquet(f)))
    
    # Test feather
    if "feather" in format_dirs:
        feather_file = format_dirs["feather"] / f"{sample_file}.feather"
        if feather_file.exists():
            formats_to_test.append(("feather", feather_file, lambda f: pd.read_feather(f)))
    
    # Test numpy
    if "numpy" in format_dirs:
        numpy_file = format_dirs["numpy"] / f"{sample_file}.npy"
        columns_file = format_dirs["numpy"] / f"{sample_file}.columns.txt"
        if numpy_file.exists() and columns_file.exists():
            def load_numpy(f):
                data = np.load(f)
                with open(columns_file, 'r') as cf:
                    columns = [line.strip() for line in cf.readlines()]
                return pd.DataFrame(data, columns=columns)
            formats_to_test.append(("numpy", numpy_file, load_numpy))
    
    # Test HDF5
    if "hdf5" in format_dirs:
        hdf5_file = format_dirs["hdf5"] / f"{sample_file}.h5"
        if hdf5_file.exists():
            formats_to_test.append(("hdf5", hdf5_file, lambda f: pd.read_hdf(f, key='data')))
    
    # Test pickle
    if "pickle" in format_dirs:
        pickle_file = format_dirs["pickle"] / f"{sample_file}.pkl"
        if pickle_file.exists():
            formats_to_test.append(("pickle", pickle_file, lambda f: pd.read_pickle(f)))
    
    # Run benchmarks
    results = []
    for format_name, file_path, load_func in formats_to_test:
        times = []
        
        # Warm up
        try:
            _ = load_func(file_path)
        except Exception as e:
            logger.error(f"Failed to load {format_name}: {e}")
            continue
        
        # Benchmark multiple runs
        for _ in range(5):
            start_time = time.time()
            df = load_func(file_path)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        file_size_mb = file_path.stat().st_size / 1024 / 1024
        
        results.append((format_name, avg_time, std_time, file_size_mb))
        logger.info(f"{format_name:>8}: {avg_time:.3f}±{std_time:.3f}s, {file_size_mb:.1f}MB")
    
    # Show speedup compared to parquet
    if results:
        parquet_time = next((time for name, time, _, _ in results if name == "parquet"), None)
        if parquet_time:
            logger.info("\nSpeedup vs Parquet:")
            for name, time_taken, _, size in results:
                speedup = parquet_time / time_taken
                logger.info(f"{name:>8}: {speedup:.1f}x faster, {size:.1f}MB")

def main():
    parser = argparse.ArgumentParser(description="Convert parquet files to faster formats")
    parser.add_argument("--format", choices=["feather", "numpy", "hdf5", "pickle", "all"], 
                       default="feather", help="Format to convert to")
    parser.add_argument("--input-dir", type=Path, 
                       default="processed_per_country",
                       help="Input directory with parquet files")
    parser.add_argument("--output-base", type=Path,
                       default="processed_per_country",
                       help="Base output directory")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run loading benchmarks after conversion")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_base = Path(args.output_base)
    
    format_dirs = {}
    
    if args.format == "all":
        formats = ["feather", "numpy", "hdf5", "pickle"]
    else:
        formats = [args.format]
    
    for fmt in formats:
        output_dir = output_base / f"fast_{fmt}"
        format_dirs[fmt] = output_dir
        
        logger.info(f"Converting to {fmt} format...")
        start_time = time.time()
        
        if fmt == "feather":
            convert_to_feather(input_dir, output_dir)
        elif fmt == "numpy":
            convert_to_numpy(input_dir, output_dir)
        elif fmt == "hdf5":
            convert_to_hdf5(input_dir, output_dir)
        elif fmt == "pickle":
            convert_to_pickle(input_dir, output_dir)
        
        end_time = time.time()
        logger.info(f"{fmt.title()} conversion completed in {end_time - start_time:.2f}s")
    
    if args.benchmark:
        benchmark_loading(input_dir, format_dirs)

if __name__ == "__main__":
    main()
