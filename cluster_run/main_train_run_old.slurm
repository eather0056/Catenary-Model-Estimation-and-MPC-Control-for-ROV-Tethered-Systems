#!/bin/bash
#SBATCH --job-name=C6_2_1KIter_nf_es
#SBATCH --partition=mundus     # or gpu for guaranteed
#SBATCH --cpus-per-task=64
#SBATCH --mem=180G
#SBATCH --time=4-00:00:00
#SBATCH -o /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/C6_2_1KIter_nf_es_out.log
#SBATCH -e /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/C6_2_1KIter_nf_es_err.log

# === Setup ===
cd /home/mundus/mdeowan698/Catenary_Dynamic/cluster_run
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cad

# WANDB_MODE=disabled
# === Run ===
CUDA_LAUNCH_BLOCKING=1 python main_train.py
