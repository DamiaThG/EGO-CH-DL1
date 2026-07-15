#!/bin/bash
# Runner script for feature extraction on GCluster
# To be executed inside Apptainer latest.sif

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default paths (can be overridden by arguments)
#FRAMES_DIR=${1:-"data/Bellomo_Small/Training"}
#ANNOTATIONS_DIR=${2:-"data/Points_Of_Interest_Recognition/Palazzo Bellomo/Training/bbox_annotations"}
#LABELS_FILE=${3:-"data/Object_Retrieval/Palazzo Bellomo/labels.txt"}
#OUTPUT_DIR=${4:-"data/Bellomo_Features"}
FRAMES_DIR=${1:-"data/Points_Of_Interest_Recognition/Monastero dei Benedettini/Training"}
ANNOTATIONS_DIR=${2:-"data/Points_Of_Interest_Recognition/Monastero dei Benedettini/Training"}
LABELS_FILE=${3:-"data/Object_Retrieval/Monastero dei Benedettini/labels.txt"}
OUTPUT_DIR=${4:-"data/Monastero_Features"}

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting Feature Extraction..."
python Scripts/phase2_extract_features.py \
    --frames_dir "$FRAMES_DIR" \
    --annotations_dir "$ANNOTATIONS_DIR" \
    --labels_file "$LABELS_FILE" \
    --output_dir "$OUTPUT_DIR"
echo "Feature Extraction Completed!"
