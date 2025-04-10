#!/bin/bash
#SBATCH --job-name=SymbReg_PySR
#SBATCH --partition=gpu
#SBATCH --nodelist=mundus-mir-1       # Explicitly pick a working node
#SBATCH --gres=gpu:a100-10:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00:00
#SBATCH -o /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/symbreg_pysr_out.log
#SBATCH -e /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/symbreg_pysr_err.log

# === Setup ===
cd /home/mundus/mdeowan698/Catenary_Dynamic/cluster_run
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cad

# Set Julia paths
export JULIA_DEPOT_PATH=/home/mundus/mdeowan698/julia_depots/pysr_depot
export JULIA_PROJECT=@pysr-0.12.1
export JULIA_EXE=/home/mundus/mdeowan698/bin/julia9
export PATH=/home/mundus/mdeowan698/bin:$PATH

# === Optional: First time only ===
# python -c "import pysr; pysr.install()"

# === Run ===
CUDA_LAUNCH_BLOCKING=1 WANDB_MODE=disabled python symbolic_regression_wandb.py
