#!/bin/bash
#SBATCH --job-name=task3_retrieval
#SBATCH --output=experiments/logs/task3_retrieval_%j.out
#SBATCH --error=experiments/logs/task3_retrieval_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --nodelist=gnode10
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:22000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL

mkdir -p experiments/logs experiments/checkpoints

apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    /shared/sifs/latest.sif \
    bash /workspace/Scripts/Task3_Object_Retrieval/run_task3.sh
