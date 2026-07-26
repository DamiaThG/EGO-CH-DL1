#!/bin/bash
#SBATCH --job-name=task1_eval
#SBATCH --output=Scripts/Task1_Localization/experiments/task1_bellomo/logs/eval_%j.out
#SBATCH --error=Scripts/Task1_Localization/experiments/task1_bellomo/logs/eval_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --nodelist=gnode10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

CHECKPOINT_PATH="Scripts/Task1_Localization/experiments/task1_bellomo/checkpoints/task1_mamba_bellomo_v1_best_val/ASF1=0.9889.ckpt"
TEST_DIR="data/Features/Bellomo/Test"

RESULTS_DIR="Scripts/Task1_Localization/experiments/task1_bellomo/results"
SAVE_PATH="${RESULTS_DIR}/eval_mamba_v1_${SLURM_JOB_ID}.json"

echo "======================================================"
echo "Valutazione Task 1: Bellomo"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Test Dir  : $TEST_DIR"
echo "  Save Path : $SAVE_PATH"
echo "======================================================"

apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    /shared/sifs/latest.sif \
    python -m Scripts.Task1_Localization.task1_evaluate \
        --checkpoint "$CHECKPOINT_PATH" \
        --test_dir "$TEST_DIR" \
        --dataset "bellomo" \
        --save_path "$SAVE_PATH"

echo "======================================================"
echo "Valutazione Completata"
echo "======================================================"
