#!/bin/bash
#SBATCH --job-name=PySR_Train
#SBATCH --partition=all  # or gpu for guaranteed
#SBATCH --nodelist=mundus-mir-2
#SBATCH --cpus-per-task=64
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00
#SBATCH -o /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/symbreg_pysr_out.log
#SBATCH -e /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/symbreg_pysr_err.log

# === Setup ===
cd /home/mundus/mdeowan698/Catenary_Dynamic/cluster_run
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cad

# WANDB_MODE=disabled
# === Run ===
CUDA_LAUNCH_BLOCKING=1 python symbolic_regression_wandb.py
