#!/bin/bash
# Script per eseguire una Grid Search (Sweep) sequenziale degli iperparametri per Mamba.
# Da eseguire DENTRO il container latest.sif

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/../.."

# Configurazioni da esplorare
MODELS=(128 256)
LAYERS=(2 4)
LEARNING_RATES=(1e-3 5e-4)

echo "========================================================="
echo "Avvio SWEEP automatico degli iperparametri per Task 1"
echo "========================================================="

# Dataset Bellomo
for d_model in "${MODELS[@]}"; do
    for l in "${LAYERS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            RUN_NAME="mamba_bellomo_d${d_model}_l${l}_lr${lr}"
            echo "--- Training Bellomo: $RUN_NAME ---"
            
            python Scripts/Task1_Localization/task1_train.py \
                --features_dir data/Task1_Features/Bellomo \
                --model mamba \
                --d_model $d_model \
                --num_layers $l \
                --lr $lr \
                --epochs 50 \
                --run_name $RUN_NAME
        done
    done
done

# Dataset Monastero
for d_model in "${MODELS[@]}"; do
    for l in "${LAYERS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            RUN_NAME="mamba_monastero_d${d_model}_l${l}_lr${lr}"
            echo "--- Training Monastero: $RUN_NAME ---"
            
            python Scripts/Task1_Localization/task1_train.py \
                --features_dir data/Task1_Features/Monastero \
                --model mamba \
                --d_model $d_model \
                --num_layers $l \
                --lr $lr \
                --epochs 50 \
                --run_name $RUN_NAME
        done
    done
done

echo "========================================================="
echo "Sweep completato!"
echo "========================================================="
