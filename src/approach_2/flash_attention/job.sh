#!/bin/bash
#PBS -N stma
#PBS -q gpu
#PBS -l select=1:ncpus=100:host=compute3
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o output_log.txt

# --- Body of the Job ---

# 1. Load Modules
module load libs/libblas-3.10.0
module load libs/liblapack-3.10.1
module load compiler/anaconda3
module load compiler/cuda-11.2
module load compiler/gcc-11.2.0

# 2. Activate Environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flashenv

# 3. Change to Project Directory
cd ~/iml/iml-a3/

# 4. Run the Training Script
echo "Starting training..."
python3 -u src/approach_2/flash_attention/train.py 2>&1 | tee output_log.txt
echo "Training complete."
