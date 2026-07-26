#!/bin/bash
#SBATCH --job-name=task1_eval
#SBATCH --output=Scripts/Task1_Localization/experiments/task1_bellomo_v3/logs/eval_%j.out
#SBATCH --error=Scripts/Task1_Localization/experiments/task1_bellomo_v3/logs/eval_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --nodelist=gnode10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Trova automaticamente il checkpoint
CHECKPOINT_PATH=$(find Scripts/Task1_Localization/experiments/task1_bellomo_v3/checkpoints -name "*.ckpt" | head -n 1)
TEST_DIR="data/Features/Bellomo/Test"

RESULTS_DIR="Scripts/Task1_Localization/experiments/task1_bellomo_v3/results"
mkdir -p "$RESULTS_DIR"
SAVE_PATH="${RESULTS_DIR}/eval_mamba_v3_${SLURM_JOB_ID}.json"

echo "======================================================"
echo "Valutazione Task 1: Bellomo (V3 Experimental)"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Test Dir  : $TEST_DIR"
echo "  Save Path : $SAVE_PATH"
echo "======================================================"

apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    /shared/sifs/latest.sif \
    python -m Scripts.Task1_Localization.v3_experimental.task1_evaluate \
        --checkpoint "$CHECKPOINT_PATH" \
        --test_dir "$TEST_DIR" \
        --dataset "bellomo" \
        --save_path "$SAVE_PATH"

echo "======================================================"
echo "Valutazione Completata"
echo "======================================================"
