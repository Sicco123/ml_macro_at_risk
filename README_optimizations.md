# Optimized Parallel Processing Guide

## Key Optimizations Made

### 1. Task Deduplication
- **Problem**: Workers were claiming tasks that were already completed, wasting time
- **Solution**: Added `filter_completed_tasks()` function that removes tasks marked as "done" in the progress file
- **Benefit**: Workers only work on remaining tasks, no duplicate effort

### 2. Worker-Based Task Distribution
- **Problem**: All workers competed for the same global task pool
- **Solution**: Added chunking system where tasks are distributed among workers based on worker index
- **Configuration**: Set `num_workers` in the YAML config file to hint the expected number of workers
- **Benefit**: Reduced contention and more even work distribution

### 3. Multiple Workers Per Chunk
- **Feature**: If you have more workers than chunks, multiple workers can work on the same chunk
- **Example**: With 14 chunks and 20 workers, workers 0 and 14 will work on chunk 0, workers 1 and 15 on chunk 1, etc.
- **Benefit**: Allows flexible scaling beyond the number of chunks

## Usage

### Single Worker
```bash
python quant_runner_2.py --config test_config.yaml --worker-index 0
```

### Multiple Workers (Manual)
```bash
# Start 14 workers
for i in {0..13}; do
    python quant_runner_2.py --config test_config.yaml --worker-index $i &
done
```

### Multiple Workers (Using Launcher)
```bash
# Launch all workers automatically
./launch_workers.sh test_config.yaml 14

# Or specify different number
./launch_workers.sh test_config.yaml 20
```

### Test Task Distribution (Dry Run)
```bash
python quant_runner_2.py --config test_config.yaml --worker-index 0 --dry-run
python quant_runner_2.py --config test_config.yaml --worker-index 5 --dry-run
```

### Monitor Progress
```bash
python monitor_progress.py
```

## Configuration Changes

Added to `test_config.yaml`:
```yaml
runtime:
  num_workers: 14  # Hint for task distribution
```

## How It Works

1. **Task Planning**: Generate all possible tasks (same as before)
2. **Filtering**: Remove tasks already completed (NEW)
3. **Chunking**: Divide remaining tasks into `num_workers` chunks (NEW)
4. **Assignment**: Each worker gets assigned to a chunk based on `worker_index % num_chunks` (NEW)
5. **Processing**: Workers claim and process tasks from their assigned chunk

## Benefits

- **No Duplicate Work**: Completed tasks are skipped entirely
- **Reduced Contention**: Workers focus on different task subsets
- **Better Scalability**: Can run more workers than configured chunks
- **Efficient Resource Usage**: Workers don't waste time on already-completed tasks

## Example Output

```
Total tasks planned: 2430
Tasks remaining after filtering completed: 1832
Tasks assigned to worker 0: 130
```

This shows that out of 2430 total tasks, 598 were already done, leaving 1832 remaining. Worker 0 gets 130 of those tasks to work on.

## Stop All Workers
```bash
pkill -f 'python quant_runner_2.py'
```
