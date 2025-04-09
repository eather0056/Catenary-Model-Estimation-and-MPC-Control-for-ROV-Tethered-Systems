#!/bin/bash
#SBATCH --job-name=SymbReg_PySR              # Job name
#SBATCH --partition=besteffort                      # Use 'gpu' partition to avoid interruption
#SBATCH --gres=gpu:a100-10:1                 # Request one a100-20 GPU
#SBATCH --mem=64G                            # RAM allocation
#SBATCH --cpus-per-task=10                   # Use up to 10 CPUs (default limit on 'all', not needed here)
#SBATCH --time=1-00:00:00                    # Run time (D-HH:MM:SS)

# === Logging ===
#SBATCH -o /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/symbreg_pysr_out.log   # STDOUT
#SBATCH -e /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/symbreg_pysr_err.log   # STDERR

# === Working Directory ===
cd /home/mundus/mdeowan698/Catenary_Dynamic/Cluster_run

# === Conda and Environment ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cab

# NEW: Use local, user-writable Julia depot
export JULIA_DEPOT_PATH=/home/mundus/mdeowan698/.julia_pysr_local

# === Run the Script ===
CUDA_LAUNCH_BLOCKING=1 python3 symbolic_regression_wandb.py
