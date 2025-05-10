#!/bin/bash
#SBATCH --job-name=C6_all_dd_1k_nf
#SBATCH --partition=mundus     # or gpu for guaranteed
#SBATCH --cpus-per-task=64
#SBATCH --mem=180G
#SBATCH --time=4-00:00:00
#SBATCH -o /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/C6_all_dd_1k_nf_out.log
#SBATCH -e /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/C6_all_dd_1k_nf_err.log

# === Setup ===
cd /home/mundus/mdeowan698/Catenary_Dynamic
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cad

# WANDB_MODE=disabled
# === Run ===
CUDA_LAUNCH_BLOCKING=1 python dd_cluster.py
