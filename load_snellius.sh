
module load 2023 PyTorch/2.1.2-foss-2023a SciPy-bundle/2023.07-gfbf-2023a matplotlib/3.7.2-gfbf-2023a scikit-learn/1.3.1-gfbf-2023a PyYAML/6.0-GCCcore-12.3.0 statsmodels/0.14.1-gfbf-2023a tqdm/4.66.1-GCCcore-12.3.0 typing-extensions/4.9.0-GCCcore-12.3.0
source .venv/bin/activate

# Copy your code to local scratch (much faster local SSD)
export TMPDIR=/scratch-local/skooiker
mkdir -p $TMPDIR
cp -r $HOME/ml_macro_at_risk $TMPDIR/

# Change to local directory
cd $TMPDIR/ml_macro_at_risk

# Run your code from local storage
python your_training_script.py

# Copy results back to home (if needed)
cp -r results/ $HOME/ml_macro_at_risk/



# Test task distribution
python quant_runner_2.py --config test_config.yaml --worker-index 0 --dry-run

# Run single worker
python quant_runner_2.py --config test_config.yaml --worker-index 0

# Launch multiple workers
./launch_workers.sh test_config.yaml 14

# Monitor progress
python monitor_progress.py