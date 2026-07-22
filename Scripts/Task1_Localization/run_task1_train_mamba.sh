#!/bin/bash
# Runner script Task 1 — Mamba Training
# Da eseguire DENTRO il container latest.sif
# Uso: ./Scripts/Task1_Localization/run_task1_train_mamba.sh [--argomenti extra]

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$(pwd):$PYTHONPATH

# Porta la shell alla root del progetto
cd "$(dirname "$0")/../.."

echo "======================================"
echo "Task 1 — Mamba Training (Bellomo)"
echo "======================================"

# ---- BELLOMO ----
python Scripts/Task1_Localization/task1_train.py \
    --features_dir data/Task1_Features/Bellomo_Features \
    --model mamba \
    --d_model 256 \
    --num_layers 4 \
    --d_state 16 \
    --dropout 0.1 \
    --batch_size 4 \
    --epochs 50 \
    --lr 1e-3 \
    --run_name "mamba_bellomo_d256_l4" \
    "$@"

echo "======================================"
echo "Task 1 — Mamba Training (Monastero)"
echo "======================================"

# ---- MONASTERO ----
python Scripts/Task1_Localization/task1_train.py \
    --features_dir data/Task1_Features/Monastero_Features \
    --model mamba \
    --d_model 256 \
    --num_layers 4 \
    --d_state 16 \
    --dropout 0.1 \
    --batch_size 4 \
    --epochs 50 \
    --lr 1e-3 \
    --run_name "mamba_monastero_d256_l4" \
    "$@"

echo "======================================"
echo "Training completati per entrambi i dataset!"
echo "======================================"
