#!/bin/bash
#SBATCH --job-name=task1_mamba
#SBATCH --output=experiments/task1_%x/logs/mamba_%j.out
#SBATCH --error=experiments/task1_%x/logs/mamba_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --nodelist=gnode10
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:22000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ── Uso ────────────────────────────────────────────────────────────────────────
# Bellomo:
#   sbatch --export=DATASET=bellomo Scripts/Task1_Localization/slurm_task1_mamba.sh
#
# Monastero:
#   sbatch --export=DATASET=monastero Scripts/Task1_Localization/slurm_task1_mamba.sh
# ──────────────────────────────────────────────────────────────────────────────

# Default a bellomo se non specificato
DATASET=${DATASET:-bellomo}

if [ "$DATASET" = "bellomo" ]; then
    FEATURES_DIR="data/Task1_Features_Original/Bellomo_Train"
    VAL_DIR="data/Task1_Features_Original/Bellomo_Val"
    OUTPUT_DIR="experiments/task1_bellomo"
    RUN_NAME="task1_mamba_bellomo_v1"
elif [ "$DATASET" = "monastero" ]; then
    FEATURES_DIR="data/Task1_Features_Original/Monastero_Train"
    VAL_DIR="data/Task1_Features_Original/Monastero_Val"
    OUTPUT_DIR="experiments/task1_monastero"
    RUN_NAME="task1_mamba_monastero_v1"
else
    echo "ERRORE: DATASET deve essere 'bellomo' o 'monastero'. Ricevuto: $DATASET"
    exit 1
fi

mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo "======================================"
echo "Task 1 — Mamba Training: $DATASET"
echo "  features_dir : $FEATURES_DIR"
echo "  val_dir      : $VAL_DIR"
echo "  output_dir   : $OUTPUT_DIR"
echo "======================================"

apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    /shared/sifs/latest.sif \
    python -m Scripts.Task1_Localization.task1_train \
        --features_dir "$FEATURES_DIR" \
        --val_features_dir "$VAL_DIR" \
        --model mamba \
        --d_model 512 \
        --num_layers 6 \
        --d_state 32 \
        --dropout 0.2 \
        --batch_size 8 \
        --epochs 100 \
        --lr 3e-4 \
        --weight_decay 1e-2 \
        --label_smoothing 0.1 \
        --patience 20 \
        --num_workers 4 \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME"

echo "======================================"
echo "Training completato: $DATASET"
echo "======================================"
