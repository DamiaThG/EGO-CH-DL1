import os
import re
import json
import torch
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
import argparse
from tqdm import tqdm

def load_labels_mapping(labels_path):
    """
    Legge il file labels.txt per mappare il POI (es. '1288.49') al suo ID (es. 50).
    """
    mapping = {}
    with open(labels_path, 'r') as f:
        next(f) # salta header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                name = parts[0]
                label_id = int(parts[1])
                mapping[name] = label_id
    return mapping

def parse_via_annotations(via_path):
    """
    Parsa il file txt in formato VIA.
    Restituisce: { "frame000001.jpg": [{"bbox": [x, y, w, h], "label": "1.1288.49"}, ...] }
    """
    frame_dict = {}
    with open(via_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split(maxsplit=5)
            if len(parts) < 6:
                continue
            filename = parts[0]
            rest = parts[5]
            
            shape_start = rest.find('{')
            shape_end = rest.find('}', shape_start) + 1
            attr_start = rest.find('{', shape_end)
            attr_end = rest.find('}', attr_start) + 1
            
            if shape_start != -1 and attr_start != -1:
                shape_str = rest[shape_start:shape_end]
                attr_str = rest[attr_start:attr_end]
                try:
                    shape_data = json.loads(shape_str)
                    attr_data = json.loads(attr_str)
                    
                    if "Label" in attr_data and shape_data.get("name") == "rect":
                        bbox = [shape_data["x"], shape_data["y"], shape_data["width"], shape_data["height"]]
                        label_str = attr_data["Label"]
                        
                        if filename not in frame_dict:
                            frame_dict[filename] = []
                        frame_dict[filename].append({"bbox": bbox, "label": label_str})
                except json.JSONDecodeError:
                    pass
    return frame_dict

def get_backbone():
    """
    Carica ConvNeXt Tiny e rimuove il classificatore lineare.
    """
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    # L'ultimo livello di model.classifier è un Linear(768, 1000). Lo sostituiamo con Identity.
    model.classifier[2] = torch.nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def process_sequence(seq_dir, via_path, poi_mapping, room_id, model, device, output_dir, target_size=224):
    frame_dict = parse_via_annotations(via_path)
    
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    video_id = seq_dir.name
    out_features = []
    out_room_labels = []
    out_poi_labels = []
    out_bboxes = []
    out_frame_ids = []
    
    jpg_files = sorted(list(seq_dir.glob("*.jpg")), key=lambda x: int(re.search(r'(\d+)', x.name).group(1)))
    
    for img_path in tqdm(jpg_files, desc=f"Processing {video_id}"):
        frame_idx = int(re.search(r'(\d+)', img_path.name).group(1))
        # Formato atteso nel VIA: "frame000001.jpg"
        expected_via_name = f"frame{frame_idx:06d}.jpg"
        
        if expected_via_name not in frame_dict:
            continue
            
        regions = frame_dict[expected_via_name]
        
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            old_w, old_h = img.size
            img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = model(img_t).squeeze(0).cpu() # [768]
            
        scale_x = target_size / old_w
        scale_y = target_size / old_h
        
        for region in regions:
            raw_label = region["label"]
            poi_key = raw_label
            # Le etichette VIA di Bellomo sono "1.1288.49", ma in labels.txt sono "1288.49"
            if poi_key.startswith("1."):
                poi_key = poi_key[2:]
            
            poi_id = poi_mapping.get(poi_key, -1)
            if poi_id == -1:
                poi_id = poi_mapping.get(raw_label, -1)
                
            x, y, w, h = region["bbox"]
            scaled_bbox = [x * scale_x, y * scale_y, w * scale_x, h * scale_y]
            
            out_features.append(feat)
            out_room_labels.append(room_id)
            out_poi_labels.append(poi_id)
            out_bboxes.append(scaled_bbox)
            out_frame_ids.append(frame_idx)
            
    if len(out_features) > 0:
        data = {
            "video_id": video_id,
            "features": torch.stack(out_features),
            "room_labels": torch.tensor(out_room_labels, dtype=torch.long),
            "poi_labels": torch.tensor(out_poi_labels, dtype=torch.long),
            "bboxes": torch.tensor(out_bboxes, dtype=torch.float32),
            "frame_ids": torch.tensor(out_frame_ids, dtype=torch.long)
        }
        out_file = Path(output_dir) / f"{video_id}_features.pt"
        torch.save(data, out_file)
        print(f"Salvati {len(out_features)} tensori in {out_file}")
    else:
        print(f"Nessuna annotazione valida trovata per {video_id}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", type=str, required=True, help="Es: data/Bellomo_Small/Training")
    parser.add_argument("--annotations_dir", type=str, required=True, help="Es: data/Points Of Interest.../bbox_annotations")
    parser.add_argument("--labels_file", type=str, required=True, help="Es: data/Object Retrieval/Palazzo Bellomo/labels.txt")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory di output per i file .pt")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizzando il device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Mappatura POI
    poi_mapping = load_labels_mapping(args.labels_file)
    print(f"Caricate {len(poi_mapping)} label POI ufficiali.")
    
    model = get_backbone().to(device)
    print("ConvNeXt-Tiny inizializzata (modalità feature extraction).")
    
    frames_path = Path(args.frames_dir)
    ann_path = Path(args.annotations_dir)
    
    # Trova tutte le sequenze video (cartelle) ordinate
    seq_dirs = sorted([d for d in frames_path.iterdir() if d.is_dir()], key=lambda x: x.name)
    
    # Costruiamo il dizionario delle Room (Ambienti)
    room2id = {seq.name: i for i, seq in enumerate(seq_dirs)}
    with open(Path(args.output_dir) / "room_mapping.json", 'w') as f:
        json.dump(room2id, f, indent=4)
    print(f"Mappatura Room salvata in {args.output_dir}/room_mapping.json (Totale {len(room2id)} ambienti)")
    
    for seq_dir in seq_dirs:
        via_file = ann_path / f"{seq_dir.name}.txt"
        if via_file.exists():
            room_id = room2id[seq_dir.name]
            process_sequence(seq_dir, via_file, poi_mapping, room_id, model, device, args.output_dir)
        else:
            print(f"ATTENZIONE: File annotazioni {via_file} mancante per {seq_dir.name}")

if __name__ == "__main__":
    main()
