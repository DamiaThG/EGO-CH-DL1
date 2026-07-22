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

# Controllo se siamo già dentro Apptainer
if [ -n "$APPTAINER_NAME" ] || [ -n "$SINGULARITY_NAME" ]; then
    echo "Rilevato ambiente Apptainer. Avvio diretto di Python..."
    python Scripts/Task2_Object_Recognition/task2_train_yolo.py \
        --data data/yolo_dataset/dataset.yaml \
        --weights yolov8x.pt \
        --epochs 100 \
        --batch 16 \
        --name task2_poi_xlarge
else
    echo "Avvio tramite Apptainer exec..."
    apptainer exec --nv /shared/sifs/latest.sif python Scripts/Task2_Object_Recognition/task2_train_yolo.py \
        --data data/yolo_dataset/dataset.yaml \
        --weights yolov8x.pt \
        --epochs 100 \
        --batch 16 \
        --name task2_poi_xlarge
fi

echo "============================================="
echo "Job terminato!"
echo "Data: $(date)"
echo "============================================="
