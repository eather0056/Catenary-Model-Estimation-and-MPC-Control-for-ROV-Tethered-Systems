#!/bin/bash
#SBATCH --job-name=dd_C6_all_50_s
#SBATCH --partition=mundus     # or gpu for guaranteed
#SBATCH --cpus-per-task=64
#SBATCH --mem=180G
#SBATCH --time=4-00:00:00
#SBATCH -o /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/dd_C6_all_50_s_out.log
#SBATCH -e /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/dd_C6_all_50_s_err.log

# === Setup ===
cd /home/mundus/mdeowan698/Catenary_Dynamic
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cad

# WANDB_MODE=disabled
# === Run ===
CUDA_LAUNCH_BLOCKING=1 python dd_cluster.py
