#!/bin/bash
#SBATCH --job-name=task1_mamba
#SBATCH --output=experiments/task1_bellomo/logs/mamba_%j.out
#SBATCH --error=experiments/task1_bellomo/logs/mamba_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --nodelist=gnode10
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:22000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ── Iperparametri Mamba (Task 1 — Bellomo only) ────────────────────────────────
#
# Analisi OOM (22 GB VRAM disponibili, bf16-mixed):
#   Sequenze Bellomo: min=779, max=3099, media=1724 frame
#   Worst case batch_size=8: tensor [8, 3099, 512] = ~25.5 MB per layer
#   6 layer × ~25.5 MB = ~153 MB attivazioni — ben dentro i 22 GB.
#   → batch_size=8 è SICURO. Nessun rischio OOM.
#
# Nota sul dataset:
#   get_dataloaders() filtra automaticamente i file video-clip (.mp4_frames_features.pt)
#   quando esistono file room-level (_features.pt), evitando duplicati nel training.
#   Bellomo Train ha 22 stanze → 22 sequenze room-level effettive.
# ──────────────────────────────────────────────────────────────────────────────

FEATURES_DIR="data/Features/Bellomo/Training"
VAL_DIR="data/Features/Bellomo/Validation"
OUTPUT_DIR="Scripts/Task1_Localizationexperiments/task1_bellomo"
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

echo "======================================================"
echo "Training completato: Bellomo"
echo "======================================================"
