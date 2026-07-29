#!/bin/bash
#SBATCH --job-name=task1_mamba
#SBATCH --output=Scripts/Task1_Localization/experiments/task1_bellomo/logs/mamba_%j.out
#SBATCH --error=Scripts/Task1_Localization/experiments/task1_bellomo/logs/mamba_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --nodelist=gnode10
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:22000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Addestramento Mamba per Task 1 (Bellomo)


FEATURES_DIR="data/Features/Bellomo/Training"
VAL_DIR="data/Features/Bellomo/Validation"
OUTPUT_DIR="Scripts/Task1_Localization/experiments/task1_bellomo"
RUN_NAME="task1_mamba_bellomo_v1"

mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo "======================================================"
echo "Task 1 — Mamba Training: Bellomo"
echo "  features_dir : $FEATURES_DIR"
echo "  val_dir      : $VAL_DIR"
echo "  output_dir   : $OUTPUT_DIR"
echo "  run_name     : $RUN_NAME"
echo "======================================================"

apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    /shared/sifs/latest.sif \
    python -m Scripts.Task1_Localization.task1_train \
        --features_dir "$FEATURES_DIR" \
        --val_features_dir "$VAL_DIR" \
        --model mamba \
        --d_model 256 \
        --num_layers 4 \
        --d_state 16 \
        --dropout 0.2 \
        --batch_size 4 \
        --epochs 100 \
        --lr 3e-4 \
        --weight_decay 1e-2 \
        --label_smoothing 0.1 \
        --patience 20 \
        --num_workers 4 \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME"

echo "======================================================"
echo "Training completato: Bellomo"
echo "======================================================"
