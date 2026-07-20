#!/bin/bash
# Runner script for feature extraction on GCluster
# To be executed inside Apptainer latest.sif

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get the script directory and move to the project root (up 2 levels since we are in Scripts/Task1_Localization)
cd "$(dirname "$0")/../.."

echo "======================================"
echo "Starting Feature Extraction for Task 1"
echo "======================================"

# ----- BELLOMO -----
echo "[1/2] Processing Bellomo_Small..."
mkdir -p "Scripts/Task1_Localization/Bellomo_Features"
python Scripts/Task1_Localization/phase2_extract_features.py \
    --frames_dir "data/Bellomo_Small/Training" \
    --output_dir "Scripts/Task1_Localization/Bellomo_Features"

# ----- MONASTERO -----
echo "[2/2] Processing Monastero_Benedettini_Small..."
mkdir -p "Scripts/Task1_Localization/Monastero_Features"
python Scripts/Task1_Localization/phase2_extract_features.py \
    --frames_dir "data/Monastero_Benedettini_Small/1_Monastero_Benedettini_Training" \
    --output_dir "Scripts/Task1_Localization/Monastero_Features"

echo "======================================"
echo "Feature Extraction Completed!"
echo "Features saved in Scripts/Task1_Localization/Bellomo_Features and Scripts/Task1_Localization/Monastero_Features"
echo "======================================"
