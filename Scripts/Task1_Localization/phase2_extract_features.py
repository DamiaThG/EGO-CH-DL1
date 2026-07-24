import os
import re
import json
import torch
import argparse
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def get_backbone(project_root):
    weights_dir = os.path.join(project_root, "weights")
    os.environ['TORCH_HOME'] = weights_dir
    torch.hub.set_dir(weights_dir)
    
    hub_repo_dir = os.path.join(weights_dir, "facebookresearch_dinov2_main")
    
    if os.path.exists(hub_repo_dir):
        print(f"Caricamento DINOv2 in modalità OFFLINE da {hub_repo_dir}")
        model = torch.hub.load(hub_repo_dir, 'dinov2_vits14', source='local')
    else:
        print(f"Caricamento DINOv2 in modalità ONLINE (scaricamento in {weights_dir})")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_global_mapping(filepath):
    mapping = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                path = parts[0]
                label = int(float(parts[-1]))
                # Usa nome_cartella/nome_file.jpg come chiave per evitare collisioni di frame0001.jpg
                key = f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
                mapping[key] = label
    return mapping

def find_local_label_file(seq_dir):
    # Cerca un file txt che assomiglia a gt.txt o Labels.txt nella cartella o nel parent
    candidates = []
    
    # 1. Dentro la cartella stessa
    for f in seq_dir.glob("*.txt"):
        if f.name.lower().startswith("gt") or "label" in f.name.lower():
            candidates.append(f)
            
    if len(candidates) == 1:
        return candidates[0]
        
    # 2. Nel parent
    for f in seq_dir.parent.glob("*.txt"):
        if f.name.lower().startswith("gt") or "label" in f.name.lower():
            candidates.append(f)
            
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # Euristica: se ci sono più gt.txt (es. Test1.txt, Test2.txt), cerca il numero nel nome della directory (es. Video1)
        dir_num = re.search(r'\d+', seq_dir.name)
        if dir_num:
            num_str = dir_num.group()
            for c in candidates:
                c_num = re.search(r'\d+', c.name)
                if c_num and c_num.group() == num_str:
                    return c
        return candidates[0] # Fallback
        
    return None

def get_frame_idx(f):
    m = re.search(r'(\d+)', f.name)
    return int(m.group(1)) if m else 0

def main():
    parser = argparse.ArgumentParser(description="Extract features for Task 1: Room-based Localization")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory contenente i frame (es. data/Bellomo/Training)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory di output per i file .pt")
    parser.add_argument("--labels_mapping", type=str, default=None, help="(Opzionale) Path a un file txt globale (es. training.txt) con path e label")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizzando il device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    model = get_backbone(project_root).to(device)
    print("DINOv2 (dinov2_vits14) inizializzata (modalità feature extraction).")
    
    frames_path = Path(args.frames_dir)
    
    global_mapping = {}
    if args.labels_mapping:
        if os.path.exists(args.labels_mapping):
            global_mapping = load_global_mapping(args.labels_mapping)
            print(f"Caricato mapping globale con {len(global_mapping)} entry.")
        else:
            print(f"ATTENZIONE: Il file mapping globale {args.labels_mapping} non esiste!")
    
    # Trova tutti i file JPG e raggruppali per cartella padre (sequenza)
    image_files = list(frames_path.rglob('*.jpg'))
    sequences = {}
    for img_file in image_files:
        parent_dir = img_file.parent
        if parent_dir not in sequences:
            sequences[parent_dir] = []
        sequences[parent_dir].append(img_file)
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    unique_rooms = set()
    
    for seq_dir, jpg_files in tqdm(sequences.items(), desc="Processing Sequences"):
        out_features = []
        out_room_labels = []
        out_frame_ids = []
        
        video_id = seq_dir.name
        
        jpg_files.sort(key=get_frame_idx)
        
        # Modalità Sequenziale Locale
        local_labels = []
        if not global_mapping:
            label_file = find_local_label_file(seq_dir)
            if label_file:
                print(f"[{video_id}] Trovato label file locale: {label_file.name}")
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # Prende solo l'ultima parte per sicurezza (supporta sia 'X' che 'percorso X')
                            parts = line.split()
                            local_labels.append(int(float(parts[-1])))
            else:
                print(f"ATTENZIONE: Nessun file label trovato per la sequenza {seq_dir.name}. Uso label fittizia 0.")
        
        for i, img_path in enumerate(jpg_files):
            frame_idx = get_frame_idx(img_path)
            
            # Determina la label
            room_id = 0
            if global_mapping:
                key = f"{img_path.parent.name}/{img_path.name}"
                if key in global_mapping:
                    room_id = global_mapping[key]
                else:
                    # Tenta un matching parziale
                    fallback_found = False
                    for k, v in global_mapping.items():
                        if img_path.name == k.split('/')[-1]:
                            room_id = v
                            fallback_found = True
                            break
                    if not fallback_found:
                        print(f"ATTENZIONE: Label non trovata per {key}. Uso 0.")
            else:
                if i < len(local_labels):
                    room_id = local_labels[i]
                else:
                    # Silenziamo l'errore per ogni frame, stampiamo solo una volta alla fine se serve
                    room_id = local_labels[-1] if local_labels else 0
                    
            unique_rooms.add(room_id)
            
            try:
                img = Image.open(img_path).convert("RGB")
                img_t = transform(img).unsqueeze(0).to(device)
            except Exception as e:
                print(f"Errore immagine {img_path}: {e}")
                continue
            
            with torch.no_grad():
                feat = model(img_t).squeeze(0).cpu()
                
            out_features.append(feat)
            out_room_labels.append(room_id)
            out_frame_ids.append(frame_idx)
                
        if len(out_features) > 0:
            data = {
                "video_id": video_id,
                "features": torch.stack(out_features),
                "room_labels": torch.tensor(out_room_labels, dtype=torch.long),
                "frame_ids": torch.tensor(out_frame_ids, dtype=torch.long)
            }
            out_file = Path(args.output_dir) / f"{video_id}_features.pt"
            torch.save(data, out_file)
            print(f"Salvati {len(out_features)} tensori in {out_file}")
        else:
            print(f"Nessuna immagine valida trovata per {video_id}")

    # Salva un finto room_mapping.json con le etichette reali trovate (per compatibilità col Dataloader)
    room_mapping = {str(r): r for r in unique_rooms}
    room_mapping_file = Path(args.output_dir) / "room_mapping.json"
    with open(room_mapping_file, 'w') as f:
        json.dump(room_mapping, f, indent=4)
    print(f"Mappatura Room salvata in {room_mapping_file} (Totale {len(room_mapping)} ambienti unici trovati)")

if __name__ == "__main__":
    main()
