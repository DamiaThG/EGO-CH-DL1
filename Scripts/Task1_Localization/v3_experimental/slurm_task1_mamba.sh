#!/bin/bash
#SBATCH --job-name=task1_mamba
#SBATCH --output=Scripts/Task1_Localization/experiments/task1_bellomo_v3/logs/mamba_%j.out
#SBATCH --error=Scripts/Task1_Localization/experiments/task1_bellomo_v3/logs/mamba_%j.err
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

export WANDB_MODE=offline

# Crea in anticipo la cartella di log per evitare errori
mkdir -p Scripts/Task1_Localization/experiments/task1_bellomo_v3/logs

FEATURES_DIR="data/Features/Bellomo/Training"
VAL_DIR="data/Features/Bellomo/Validation"
OUTPUT_DIR="Scripts/Task1_Localization/experiments/task1_bellomo_v3"
RUN_NAME="task1_mamba_bellomo_v3_mstcn"

echo "======================================================"
echo "Avvio Training V3 (MS-TCN + MLP): Bellomo"
echo "  Features Dir    : $FEATURES_DIR"
echo "  Val Features Dir: $VAL_DIR"
echo "  Output Dir      : $OUTPUT_DIR"
echo "  Run Name        : $RUN_NAME"
echo "======================================================"

apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    /shared/sifs/latest.sif \
    python -m Scripts.Task1_Localization.v3_experimental.task1_train \
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
echo "Training completato: Bellomo V3"
echo "======================================================"
