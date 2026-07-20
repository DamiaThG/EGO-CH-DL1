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

def main():
    parser = argparse.ArgumentParser(description="Extract features for Task 1: Room-based Localization")
    parser.add_argument("--frames_dir", type=str, required=True, help="Es: data/Bellomo_Small/Training")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory di output per i file .pt")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizzando il device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    model = get_backbone(project_root).to(device)
    print("DINOv2 (dinov2_vits14) inizializzata (modalità feature extraction).")
    
    frames_path = Path(args.frames_dir)
    
    # Trova tutti i file JPG e raggruppali per cartella padre (sequenza)
    image_files = list(frames_path.rglob('*.jpg'))
    sequences = {}
    for img_file in image_files:
        parent_dir = img_file.parent
        if parent_dir not in sequences:
            sequences[parent_dir] = []
        sequences[parent_dir].append(img_file)
        
    room_mapping = {}
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for seq_dir, jpg_files in tqdm(sequences.items(), desc="Processing Sequences"):
        room_name = seq_dir.name.split('_')[0]
        room_name = room_name.split('.')[0]
        
        if room_name not in room_mapping:
            room_mapping[room_name] = len(room_mapping)
        room_id = room_mapping[room_name]
        
        out_features = []
        out_room_labels = []
        out_frame_ids = []
        
        video_id = seq_dir.name
        
        # Ordina le immagini
        def get_frame_idx(f):
            m = re.search(r'(\d+)', f.name)
            return int(m.group(1)) if m else 0
            
        jpg_files.sort(key=get_frame_idx)
        
        for img_path in jpg_files:
            frame_idx = get_frame_idx(img_path)
            
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

    room_mapping_file = Path(args.output_dir) / "room_mapping.json"
    with open(room_mapping_file, 'w') as f:
        json.dump(room_mapping, f, indent=4)
    print(f"Mappatura Room salvata in {room_mapping_file} (Totale {len(room_mapping)} ambienti)")

if __name__ == "__main__":
    main()
