#!/bin/bash

# Launch multiple workers for parallel processing
# Usage: ./launch_workers.sh <config_file> [num_workers]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file> [num_workers]"
    echo "Example: $0 test_config.yaml 14"
    exit 1
fi

CONFIG_FILE="$1"
NUM_WORKERS="${2:-14}"  # Default to 14 workers if not specified

echo "Launching $NUM_WORKERS workers with config: $CONFIG_FILE"

# create dir
#mkdir -p /scratch-local/skooiker/

# copy code
#cp -r /home/skooiker/ml_macro_at_risk /scratch-local/skooiker/
#cd /scratch-local/skooiker/ml_macro_at_risk || { echo "Failed to change directory"; exit 1; }

module load 2023 PyTorch/2.1.2-foss-2023a SciPy-bundle/2023.07-gfbf-2023a matplotlib/3.7.2-gfbf-2023a scikit-learn/1.3.1-gfbf-2023a PyYAML/6.0-GCCcore-12.3.0 statsmodels/0.14.1-gfbf-2023a tqdm/4.66.1-GCCcore-12.3.0 typing-extensions/4.9.0-GCCcore-12.3.0
source .venv/bin/activate

# Limit threading for scientific libraries
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
# PyTorch specific
export TORCH_NUM_THREADS=1

# create a command that kills all python quant_runner_2.py and copies results
KILL_CMD="pkill -f 'python quant_runner_2.py'"
COPY_CMD="cp -r outputs/ $HOME/ml_macro_at_risk/"

# Get available CPU cores for this process
AVAILABLE_CORES=$(taskset -cp $$ | cut -d: -f2 | tr -d ' ')
echo "Available CPU cores for this process: $AVAILABLE_CORES"

# Convert range to array (e.g., "112-127" becomes array of cores)
if [[ $AVAILABLE_CORES == *"-"* ]]; then
    # Handle range format like "112-127"
    START_CORE=$(echo $AVAILABLE_CORES | cut -d- -f1)
    END_CORE=$(echo $AVAILABLE_CORES | cut -d- -f2)
    CORES_ARRAY=()
    for ((core=$START_CORE; core<=$END_CORE; core++)); do
        CORES_ARRAY+=($core)
    done
else
    # Handle comma-separated format like "1,2,3"
    IFS=',' read -ra CORES_ARRAY <<< "$AVAILABLE_CORES"
fi

NUM_AVAILABLE_CORES=${#CORES_ARRAY[@]}
echo "Number of available cores: $NUM_AVAILABLE_CORES"

# Launch workers in background
for i in $(seq 0 $((NUM_WORKERS-1))); do
    # Use modulo to wrap around available cores
    CORE_INDEX=$((i % NUM_AVAILABLE_CORES))
    CORE_ID=${CORES_ARRAY[$CORE_INDEX]}
    echo "Starting worker $i on CPU core $CORE_ID..."
    # Pin each worker to its assigned CPU core
    taskset -c $CORE_ID python quant_runner_2.py --config "$CONFIG_FILE" --worker-index $i &
    
    # Small delay to avoid overwhelming the system at startup
    sleep 15
done

echo "All $NUM_WORKERS workers started in background"
echo "To monitor progress, check the log files in outputs/logs/"
echo "To stop all workers, run: pkill -f 'python quant_runner_2.py'"

# move results back to home



# Wait for all background jobs to complete
wait
#cp -r outputs/ $HOME/ml_macro_at_risk/
echo "All workers completed"
