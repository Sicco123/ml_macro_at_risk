#!/bin/bash

# Simple launcher for multiple single-core instances
# Usage: ./launch_single_core.sh <config_file> [num_instances]

CONFIG_FILE=$1
NUM_INSTANCES=${2:-1}

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file> [num_instances]"
    echo "Example: $0 configs/experiment_example.yaml 4"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Launching $NUM_INSTANCES single-core instances with config: $CONFIG_FILE"

for i in $(seq 1 $NUM_INSTANCES); do
    INSTANCE_ID="worker_${i}_$(date +%s)"
    echo "Starting instance $i with ID: $INSTANCE_ID"
    
    # Launch in background
    python quant_runner_2.py --config "$CONFIG_FILE" --instance-id "$INSTANCE_ID" &
    
    # Small delay between launches
    sleep 5
done

echo "All instances launched. Use 'jobs' to see running processes."
echo "Use 'kill %1 %2 %3...' to stop specific jobs, or 'killall python' to stop all."
