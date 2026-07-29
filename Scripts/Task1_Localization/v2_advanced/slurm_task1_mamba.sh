#!/bin/bash
#SBATCH --job-name=task1_mamba
#SBATCH --output=Scripts/Task1_Localization/experiments/task1_bellomo_v2/logs/mamba_%j.out
#SBATCH --error=Scripts/Task1_Localization/experiments/task1_bellomo_v2/logs/mamba_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --nodelist=gnode10
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:22000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Addestramento Mamba v2 (Bellomo)


export WANDB_MODE=offline

# Crea in anticipo la cartella di log per evitare errori
mkdir -p Scripts/Task1_Localization/experiments/task1_bellomo_v2/logs

FEATURES_DIR="data/Features/Bellomo/Training"
VAL_DIR="data/Features/Bellomo/Validation"
OUTPUT_DIR="Scripts/Task1_Localization/experiments/task1_bellomo_v2"
RUN_NAME="task1_mamba_bellomo_v2_weighted"

echo "======================================================"
echo "Avvio Training V2 Avanzato (Weighted Loss): Bellomo"
echo "  Features Dir    : $FEATURES_DIR"
echo "  Val Features Dir: $VAL_DIR"
echo "  Output Dir      : $OUTPUT_DIR"
echo "  Run Name        : $RUN_NAME"
echo "======================================================"

apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    /shared/sifs/latest.sif \
    python -m Scripts.Task1_Localization.v2_advanced.task1_train \
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
echo "Training completato: Bellomo V2"
echo "======================================================"
