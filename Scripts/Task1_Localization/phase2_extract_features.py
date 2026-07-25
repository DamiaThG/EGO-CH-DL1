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
    sequential_list = []
    is_sequential = True
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) >= 2:
                is_sequential = False
                # Il path potrebbe contenere spazi (es. "6_Aula Santo Mazzarino")
                # quindi ricongiungiamo tutte le parti tranne l'ultima (che è la label)
                path = " ".join(parts[:-1])
                label = int(float(parts[-1]))
                
                # Usa nome_cartella/nome_file.jpg come chiave per evitare collisioni
                key = f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
                mapping[key] = label
            elif len(parts) == 1:
                sequential_list.append(int(float(parts[0])))
                
    if is_sequential and len(sequential_list) > 0:
        return sequential_list
    return mapping

def find_local_label_file(seq_dir):
    candidates = []
    for f in seq_dir.glob("*.txt"):
        if f.name.lower().startswith("gt") or "label" in f.name.lower():
            candidates.append(f)
    if len(candidates) == 1: return candidates[0]
        
    for f in seq_dir.parent.glob("*.txt"):
        if f.name.lower().startswith("gt") or "label" in f.name.lower():
            candidates.append(f)
            
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        dir_num = re.search(r'\d+', seq_dir.name)
        if dir_num:
            num_str = dir_num.group()
            for c in candidates:
                c_num = re.search(r'\d+', c.name)
                if c_num and c_num.group() == num_str: return c
        return candidates[0]
    return None

def get_frame_idx(f):
    m = re.search(r'(\d+)', f.name)
    return int(m.group(1)) if m else 0

def main():
    parser = argparse.ArgumentParser(description="Extract features for Task 1: Room-based Localization")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory contenente i frame")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory di output per i file .pt di Training/Default")
    parser.add_argument("--labels_mapping", type=str, default=None, help="Path a training.txt")
    parser.add_argument("--val_labels_mapping", type=str, default=None, help="Path a validation.txt")
    parser.add_argument("--val_output_dir", type=str, default=None, help="Directory di output per i file .pt di Validation")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizzando il device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.val_output_dir:
        os.makedirs(args.val_output_dir, exist_ok=True)
        
    if args.val_labels_mapping and not args.val_output_dir:
        print("ATTENZIONE: Hai fornito --val_labels_mapping ma non --val_output_dir. I file val verranno persi!")
    
    project_root = Path(__file__).resolve().parent.parent.parent
    model = get_backbone(project_root).to(device)
    
    frames_path = Path(args.frames_dir)
    
    # Carica Mappings
    train_mapping = {}
    if args.labels_mapping and os.path.exists(args.labels_mapping):
        train_mapping = load_global_mapping(args.labels_mapping)
        print(f"Caricato mapping TRAIN con {len(train_mapping)} entry.")
        
    val_mapping = {}
    if args.val_labels_mapping and os.path.exists(args.val_labels_mapping):
        val_mapping = load_global_mapping(args.val_labels_mapping)
        print(f"Caricato mapping VAL con {len(val_mapping)} entry.")
    
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
        train_features, train_labels, train_ids = [], [], []
        val_features, val_labels, val_ids = [], [], []
        
        # Se la cartella si chiama solo "1" (es. Monastero Test), usa anche il nome del padre per evitare di sovrascrivere
        video_id = f"{seq_dir.parent.name}_{seq_dir.name}" if seq_dir.name.isdigit() else seq_dir.name
        
        # --- Resume Capability ---
        # Controlla se abbiamo già estratto questa sequenza
        out_file_train = Path(args.output_dir) / f"{video_id}_features.pt"
        if out_file_train.exists():
            # Per sicurezza, leggi anche le unique rooms per mantenere la coerenza del mapping finale
            try:
                data = torch.load(out_file_train, map_location="cpu", weights_only=False)
                for r in data["room_labels"].unique().tolist():
                    unique_rooms.add(r)
            except Exception:
                pass
            continue
            
        jpg_files.sort(key=get_frame_idx)
        
        # Modalità Sequenziale (se non c'è mapping globale)
        local_labels = []
        if not train_mapping and not val_mapping:
            label_file = find_local_label_file(seq_dir)
            if label_file:
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            local_labels.append(int(float(line.split()[-1])))
        
        for i, img_path in enumerate(jpg_files):
            frame_idx = get_frame_idx(img_path)
            room_id = -1
            target_split = 'train' # Default
            
            key = f"{img_path.parent.name}/{img_path.name}"
            
            if train_mapping or val_mapping:
                # Gestione Liste Sequenziali passate da riga di comando
                if isinstance(train_mapping, list) and len(train_mapping) > 0:
                    room_id = train_mapping[i] if i < len(train_mapping) else train_mapping[-1]
                    target_split = 'train'
                elif isinstance(val_mapping, list) and len(val_mapping) > 0:
                    room_id = val_mapping[i] if i < len(val_mapping) else val_mapping[-1]
                    target_split = 'val'
                else:
                    # Gestione Dizionari Globali (Bellomo Train/Val)
                    # Controlla prima se è in train
                    if isinstance(train_mapping, dict) and key in train_mapping:
                        room_id = train_mapping[key]
                        target_split = 'train'
                    # Controlla se è in validation
                    elif isinstance(val_mapping, dict) and key in val_mapping:
                        room_id = val_mapping[key]
                        target_split = 'val'
                    else:
                        # Fallback parziale
                        found = False
                        if isinstance(train_mapping, dict):
                            for k, v in train_mapping.items():
                                if img_path.name == k.split('/')[-1]:
                                    room_id = v; target_split = 'train'; found = True; break
                        if not found and isinstance(val_mapping, dict):
                            for k, v in val_mapping.items():
                                if img_path.name == k.split('/')[-1]:
                                    room_id = v; target_split = 'val'; found = True; break
                        
                        if not found:
                            print(f"Label non trovata per {key}, ignoro frame.")
                            continue
            else:
                # Usa file locale
                if i < len(local_labels):
                    room_id = local_labels[i]
                else:
                    room_id = local_labels[-1] if local_labels else 0
                    
            unique_rooms.add(room_id)
            
            try:
                img = Image.open(img_path).convert("RGB")
                img_t = transform(img).unsqueeze(0).to(device)
            except Exception as e:
                print(f"Errore {img_path}: {e}"); continue
                
            with torch.no_grad():
                feat = model(img_t).squeeze(0).cpu()
                
            if target_split == 'train':
                train_features.append(feat); train_labels.append(room_id); train_ids.append(frame_idx)
            else:
                val_features.append(feat); val_labels.append(room_id); val_ids.append(frame_idx)
                
        # Salva Train
        if len(train_features) > 0:
            data = {
                "video_id": video_id,
                "features": torch.stack(train_features),
                "room_labels": torch.tensor(train_labels, dtype=torch.long),
                "frame_ids": torch.tensor(train_ids, dtype=torch.long)
            }
            out_file = Path(args.output_dir) / f"{video_id}_features.pt"
            torch.save(data, out_file)
            
        # Salva Validation
        if len(val_features) > 0 and args.val_output_dir:
            data = {
                "video_id": video_id,
                "features": torch.stack(val_features),
                "room_labels": torch.tensor(val_labels, dtype=torch.long),
                "frame_ids": torch.tensor(val_ids, dtype=torch.long)
            }
            out_file_val = Path(args.val_output_dir) / f"{video_id}_features.pt"
            torch.save(data, out_file_val)

    # Crea room mapping
    room_mapping_dict = {str(r): r for r in unique_rooms}
    
    with open(Path(args.output_dir) / "room_mapping.json", 'w') as f:
        json.dump(room_mapping_dict, f, indent=4)
        
    if args.val_output_dir:
        with open(Path(args.val_output_dir) / "room_mapping.json", 'w') as f:
            json.dump(room_mapping_dict, f, indent=4)

if __name__ == "__main__":
    main()
