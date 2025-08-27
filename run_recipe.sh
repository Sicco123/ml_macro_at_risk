# Test the setup
python3 test_parallel.py

# Run with limited resources (recommended first)
python3 parallel_runner.py --config configs/parallel_config.yaml --max-workers 2

# Monitor progress in another terminal
python3 monitor_progress.py --watch

# Run with full resources
python3 parallel_runner.py --config configs/parallel_config.yaml