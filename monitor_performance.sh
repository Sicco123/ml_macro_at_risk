#!/bin/bash
# Performance monitoring script

echo "=== Performance Monitor ==="
echo "Time: $(date)"
echo ""

# Check number of active workers
WORKERS=$(pgrep -f "python quant_runner_2.py" | wc -l)
echo "Active workers: $WORKERS"

# Check task completion rate
if [ -f outputs/progress/completed_tasks.parquet ]; then
    COMPLETED=$(python -c "
import pandas as pd
try:
    df = pd.read_parquet('outputs/progress/completed_tasks.parquet')
    print(len(df))
except:
    print(0)
")
    echo "Completed tasks: $COMPLETED"
fi

# Check system load
echo "System load: $(uptime | awk -F'load average:' '{print $2}')"

# Check memory usage
echo "Memory usage: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"

# Check disk I/O
if command -v iostat >/dev/null 2>&1; then
    echo ""
    echo "=== Disk I/O (last 1 sec) ==="
    iostat -x 1 1 | tail -n +4
fi

echo ""
echo "=== Recent log activity ==="
if [ -f outputs/logs/*.log ]; then
    tail -n 10 outputs/logs/*.log 2>/dev/null | grep -E "(Completed|Failed|Batch progress|Timing)" | tail -8
fi
