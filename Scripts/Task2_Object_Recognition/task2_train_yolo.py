from ultralytics import YOLO
import argparse
from pathlib import Path

def train_yolo(data_yaml, weights='yolov8n.pt', epochs=100, batch=16, name='task2_poi', resume=False):
    print(f"Avvio addestramento YOLOv8...")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Pesi Iniziali: {weights}")
    
    # Caricamento del modello
    model = YOLO(weights)
    
    # Addestramento
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        name=name,
        device=0,
        plots=True,
        save=True,
        exist_ok=True,
        resume=resume
    )
    
    print("\nAddestramento completato con successo!")
    print(f"Tutti i grafici (Loss, mAP, Confusion Matrix) e i pesi finali sono stati salvati nella cartella: runs/detect/{name}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script di addestramento YOLOv8 per Task 2")
    parser.add_argument('--data', type=str, default='data/yolo_dataset/dataset.yaml', help='Percorso del file dataset.yaml')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Pesi pre-addestrati (es. yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=100, help='Numero di epoche')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--name', type=str, default='task2_poi', help='Nome della cartella dei risultati')
    parser.add_argument('--resume', action='store_true', help='Riprendi l\'addestramento interrotto dall\'ultimo checkpoint')
    args = parser.parse_args()
    
    train_yolo(args.data, args.weights, args.epochs, args.batch, name=args.name, resume=args.resume)
