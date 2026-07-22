import os
import shutil
import random
from pathlib import Path
import yaml

def format_yolo_dataset(src_dir_path, dest_dir_path, split_ratio=(0.8, 0.1, 0.1)):
    src_dir = Path(src_dir_path)
    dest_dir = Path(dest_dir_path)
    
    # Crea la struttura delle cartelle YOLO
    for split in ['train', 'val', 'test']:
        os.makedirs(dest_dir / 'images' / split, exist_ok=True)
        os.makedirs(dest_dir / 'labels' / split, exist_ok=True)
        
    print(f"Scansione della directory: {src_dir}")
    
    # Trova tutte le immagini
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(src_dir.rglob(f"*{ext}")))
        image_paths.extend(list(src_dir.rglob(f"*{ext.upper()}")))
        
    print(f"Trovate {len(image_paths)} immagini totali.")
    
    # Crea un dizionario veloce per trovare le label
    print("Scansione per trovare i file delle annotazioni (.txt)...")
    txt_paths = list(src_dir.rglob("*.txt"))
    # Mappa: nome_file (senza estensione) -> Path completo del file txt
    label_map = {p.stem: p for p in txt_paths}
    
    valid_pairs = []
    max_class_id = -1
    
    # Associa ogni immagine alla sua label
    for img_path in image_paths:
        img_stem = img_path.stem
        if img_stem in label_map:
            lbl_path = label_map[img_stem]
            
            # Leggiamo la label per trovare il class_id massimo
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        try:
                            cls_id = int(parts[0])
                            max_class_id = max(max_class_id, cls_id)
                        except ValueError:
                            pass
                            
            valid_pairs.append((img_path, lbl_path))
            
    print(f"Trovate {len(valid_pairs)} coppie valide (Immagine + Bounding Box).")
    
    if len(valid_pairs) == 0:
        print("ERRORE: Nessuna coppia trovata. Verifica che i file .txt abbiano lo stesso nome delle immagini.")
        return

    # Shuffle per randomizzare i set
    random.seed(42)
    random.shuffle(valid_pairs)
    
    # Calcola gli indici di split
    n = len(valid_pairs)
    train_end = int(n * split_ratio[0])
    val_end = train_end + int(n * split_ratio[1])
    
    splits = {
        'train': valid_pairs[:train_end],
        'val': valid_pairs[train_end:val_end],
        'test': valid_pairs[val_end:]
    }
    
    # Copia i file nelle rispettive cartelle
    print("Copia dei file nella struttura YOLO...")
    for split_name, pairs in splits.items():
        print(f"Generazione set {split_name} ({len(pairs)} file)...")
        for img_path, lbl_path in pairs:
            # Destinazioni
            img_dest = dest_dir / 'images' / split_name / img_path.name
            lbl_dest = dest_dir / 'labels' / split_name / lbl_path.name
            
            # Copia fisica
            shutil.copy2(img_path, img_dest)
            shutil.copy2(lbl_path, lbl_dest)
            
    # Crea il file dataset.yaml
    num_classes = max_class_id + 1 if max_class_id >= 0 else 1
    class_names = [f"class_{i}" for i in range(num_classes)]
    
    yaml_content = {
        'path': str(dest_dir), # Percorso relativo (più sicuro tra Windows e Linux)
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': num_classes,
        'names': class_names
    }
    
    yaml_path = dest_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"\nFormattazione completata con successo!")
    print(f"Dataset YOLO generato in: {dest_dir}")
    print(f"File YAML creato: {yaml_path}")
    print(f"Numero di classi rilevate: {num_classes}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Formatta il dataset EGO-CH per YOLOv8')
    parser.add_argument('--src', type=str, default='data/Points_Of_Interest_Recognition', help='Percorso del dataset originale')
    parser.add_argument('--dest', type=str, default='data/yolo_dataset', help='Percorso di destinazione per YOLO')
    args = parser.parse_args()
    
    format_yolo_dataset(args.src, args.dest)
