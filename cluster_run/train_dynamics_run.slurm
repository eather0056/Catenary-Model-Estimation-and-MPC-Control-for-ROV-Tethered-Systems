#!/bin/bash
#SBATCH --job-name=SymbReg_PySR
#SBATCH --partition=mundus 
#SBATCH --gres=gpu:a100-10:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00:00
#SBATCH -o /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/50_C6_train_dynamic_out.log
#SBATCH -e /home/mundus/mdeowan698/Catenary_Dynamic/Results/logs/50_C6_train_dynamic_err.log

# === Setup ===
cd /home/mundus/mdeowan698/Catenary_Dynamic/cluster_run
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cad

# WANDB_MODE=disabled
# === Run ===
CUDA_LAUNCH_BLOCKING=1 python train_dynamics.py
