#!/bin/bash
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SITES=("Monastero dei Benedettini" "Palazzo Bellomo")
FEAT_DIRS=("Scripts/Task3_Object_Retrieval/Monastero_Features" "Scripts/Task3_Object_Retrieval/Bellomo_Features")

for i in "${!SITES[@]}"; do
    SITE="${SITES[$i]}"
    FEAT_DIR="${FEAT_DIRS[$i]}"

    echo "=========================================================="
    echo " processing SITE: $SITE"
    echo "=========================================================="

    echo "--- Extracting Gallery Features ---"
    python3 Scripts/Task3_Object_Retrieval/task3_extract_patch_features.py \
        --patches_dir "data/Object_Retrieval/$SITE/Training" \
        --output_file "$FEAT_DIR/gallery_features.pt"

    echo "--- Extracting Query Features ---"
    python3 Scripts/Task3_Object_Retrieval/task3_extract_patch_features.py \
        --patches_dir "data/Object_Retrieval/$SITE/Test" \
        --output_file "$FEAT_DIR/query_features.pt"

    echo "--- Evaluating Retrieval ---"
    python3 Scripts/Task3_Object_Retrieval/task3_evaluate_retrieval.py \
        --gallery_features "$FEAT_DIR/gallery_features.pt" \
        --query_features "$FEAT_DIR/query_features.pt" \
        --labels_file "data/Object_Retrieval/$SITE/labels.txt" \
        --query_labels_dir "data/Object_Retrieval/$SITE/Test/labels"
        
    echo ""
done
