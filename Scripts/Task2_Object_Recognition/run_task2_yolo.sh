#!/bin/bash
#SBATCH --job-name=yolo_task2
#SBATCH --output=logs/yolo_task2_%j.out
#SBATCH --error=logs/yolo_task2_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --nodelist=gnode10
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:22000
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Crea la cartella dei log se non esiste
mkdir -p logs

echo "============================================="
echo "Inizio job YOLO Task 2"
echo "Nodo assegnato: $SLURMD_NODENAME"
echo "Data: $(date)"
echo "============================================="

# Esegui lo script python all'interno del container di sistema
# NOTA: Usiamo /shared/sifs/latest.sif come indicato negli alias del corso
apptainer exec --nv /shared/sifs/latest.sif python Scripts/Task2_Object_Recognition/task2_train_yolo.py \
    --data data/yolo_dataset/dataset.yaml \
    --weights yolov8n.pt \
    --epochs 100 \
    --batch 32

echo "============================================="
echo "Job terminato!"
echo "Data: $(date)"
echo "============================================="
