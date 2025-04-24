#!/bin/bash
#SBATCH --job-name=Lg_C6_split_log
#SBATCH --partition=mundus     # or gpu for guaranteed, mundus
#SBATCH --cpus-per-task=64
#SBATCH --mem=180G
#SBATCH --time=4-00:00:00
#SBATCH -o /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/Lg_C6_split_log_out.log
#SBATCH -e /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/Lg_C6_split_log_err.log

# === Setup ===
cd /home/mundus/mdeowan698/Catenary_Dynamic
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cad

# WANDB_MODE=disabled
# === Run ===
CUDA_LAUNCH_BLOCKING=1 python LagrangianModelEstimator.py
