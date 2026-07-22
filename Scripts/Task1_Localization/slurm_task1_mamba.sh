#!/bin/bash
#SBATCH --job-name=task1_mamba
#SBATCH --output=Scripts/Task1_Localization/experiments/logs/mamba_%j.out
#SBATCH --error=Scripts/Task1_Localization/experiments/logs/mamba_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --nodelist=gnode10
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:22000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p Scripts/Task1_Localization/experiments/logs
mkdir -p Scripts/Task1_Localization/experiments/checkpoints

# Task 1 usa latest.sif (mambapy disponibile lì via pip install --user)
apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    /shared/sifs/latest.sif \
    bash /workspace/Scripts/Task1_Localization/run_task1_train_mamba.sh
