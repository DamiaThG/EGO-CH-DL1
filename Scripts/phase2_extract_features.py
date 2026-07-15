import os
import re
import json
import torch
import argparse
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

def load_labels_mapping(labels_path):
    mapping = {}
    with open(labels_path, 'r') as f:
        first_line = f.readline()
        if not any(char.isdigit() for char in first_line):
            pass
        else:
            f.seek(0)
            
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                name = parts[0]
                label_id = int(parts[1])
                mapping[name] = label_id
    return mapping

def parse_via_annotations(via_path):
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
            
            m = re.search(r'(\d+)', filename)
            if not m:
                continue
            frame_idx = int(m.group(1))
            
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
                    
                    if shape_data.get("name") == "rect":
                        bbox = [shape_data["x"], shape_data["y"], shape_data["width"], shape_data["height"]]
                        label_str = attr_data.get("Label", None)
                        
                        if frame_idx not in frame_dict:
                            frame_dict[frame_idx] = []
                        frame_dict[frame_idx].append({"bbox": bbox, "label": label_str})
                except json.JSONDecodeError:
                    pass
    return frame_dict

def get_backbone(weights_path=None):
    model = models.convnext_tiny()
    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    model.classifier[2] = torch.nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def find_annotation_file(ann_path, seq_name):
    base_name = re.sub(r'(\.mp4|_frames|\.mp4_frames)$', '', seq_name)
    exact_path = ann_path / f"{base_name}.txt"
    if exact_path.exists():
        return exact_path
        
    ann_files = [f for f in ann_path.iterdir() if f.is_file() and f.name.endswith('.txt')]
    m = re.match(r'^(\d+)_', base_name)
    if m:
        prefix_mod = f"{m.group(1)}.0_"
        mod_base = base_name.replace(m.group(0), prefix_mod)
        mod_path = ann_path / f"{mod_base}.txt"
        if mod_path.exists():
            return mod_path
            
    m_prefix = re.match(r'^(\d+(?:\.\d+)?)', base_name)
    if not m_prefix:
        return None
        
    num_part = m_prefix.group(1)
    if '.' not in num_part:
        num_part = f"{num_part}.0"
        
    is_s = base_name.endswith('_S')
    for f in ann_files:
        f_name = f.stem
        f_is_s = f_name.endswith('_S')
        if is_s != f_is_s:
            continue
            
        f_m_prefix = re.match(r'^(\d+(?:\.\d+)?)', f_name)
        if not f_m_prefix:
            continue
        f_num_part = f_m_prefix.group(1)
        if '.' not in f_num_part:
            f_num_part += ".0"
            
        if num_part == f_num_part:
            return f
    return None

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
    
    poi_mapping = load_labels_mapping(args.labels_file)
    print(f"Caricate {len(poi_mapping)} label POI ufficiali.")
    
    project_root = Path(__file__).parent.parent
    local_weights = project_root / "weights" / "convnext_tiny-983f1562.pth"
    
    model = get_backbone(weights_path=str(local_weights)).to(device)
    print("ConvNeXt-Tiny inizializzata (modalità feature extraction).")
    
    frames_path = Path(args.frames_dir)
    ann_path = Path(args.annotations_dir)
    
    seq_dirs = sorted([d for d in frames_path.iterdir() if d.is_dir()])
    
    is_monastero = "Monastero" in frames_path.name or "Monastero" in str(frames_path)
    if is_monastero:
        print("Rilevata modalità YOLO (Monastero dei Benedettini).")
    else:
        print("Rilevata modalità VIA (Palazzo Bellomo).")
        
    room_mapping = {}
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for seq_dir in tqdm(seq_dirs, desc="Processing Sequences"):
        room_name = seq_dir.name.split('_')[0]
        room_name = room_name.split('.')[0]
        
        if room_name not in room_mapping:
            room_mapping[room_name] = len(room_mapping)
        room_id = room_mapping[room_name]
        
        out_features = []
        out_room_labels = []
        out_poi_labels = []
        out_bboxes = []
        out_frame_ids = []
        
        video_id = seq_dir.name
        
        if is_monastero:
            images_dir = seq_dir / "images"
            labels_dir = seq_dir / "labels"
            
            if not images_dir.exists() or not labels_dir.exists():
                print(f"ATTENZIONE: Cartelle images/ o labels/ mancanti in {seq_dir.name}")
                continue
                
            jpg_files = sorted([f for f in images_dir.iterdir() if f.name.endswith('.jpg')])
            for img_path in jpg_files:
                txt_path = labels_dir / f"{img_path.stem}.txt"
                if not txt_path.exists():
                    continue
                
                m = re.search(r'(\d+)', img_path.name)
                frame_idx = int(m.group(1)) if m else 0
                
                try:
                    img = Image.open(img_path).convert("RGB")
                    old_w, old_h = img.size
                    img_t = transform(img).unsqueeze(0).to(device)
                except Exception as e:
                    print(f"Errore immagine {img_path}: {e}")
                    continue
                    
                with torch.no_grad():
                    feat = model(img_t).squeeze(0).cpu()
                
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            w_norm = float(parts[3])
                            h_norm = float(parts[4])
                            
                            w_orig = w_norm * old_w
                            h_orig = h_norm * old_h
                            x_orig = (x_center * old_w) - (w_orig / 2)
                            y_orig = (y_center * old_h) - (h_orig / 2)
                            
                            scale_x = 224 / old_w
                            scale_y = 224 / old_h
                            
                            scaled_bbox = [x_orig * scale_x, y_orig * scale_y, w_orig * scale_x, h_orig * scale_y]
                            
                            out_features.append(feat)
                            out_room_labels.append(room_id)
                            out_poi_labels.append(class_id)
                            out_bboxes.append(scaled_bbox)
                            out_frame_ids.append(frame_idx)
            
        else:
            anno_file = find_annotation_file(ann_path, seq_dir.name)
            if not anno_file:
                print(f"ATTENZIONE: File annotazioni mancante per la sequenza {seq_dir.name}")
                continue
                
            frame_dict = parse_via_annotations(anno_file)
            jpg_files = sorted(list(seq_dir.glob("*.jpg")), key=lambda x: int(re.search(r'(\d+)', x.name).group(1)))
            
            for img_path in jpg_files:
                frame_idx = int(re.search(r'(\d+)', img_path.name).group(1))
                if frame_idx not in frame_dict:
                    continue
                    
                regions = frame_dict[frame_idx]
                
                try:
                    img = Image.open(img_path).convert("RGB")
                    old_w, old_h = img.size
                    img_t = transform(img).unsqueeze(0).to(device)
                except Exception as e:
                    print(f"Errore immagine {img_path}: {e}")
                    continue
                
                with torch.no_grad():
                    feat = model(img_t).squeeze(0).cpu()
                    
                scale_x = 224 / old_w
                scale_y = 224 / old_h
                
                for region in regions:
                    raw_label = region["label"]
                    if raw_label is None:
                        m_seq = re.match(r'^(\d+(?:\.\d+)?)', seq_dir.name)
                        if m_seq:
                            raw_label = m_seq.group(1)
                        else:
                            continue
                            
                    poi_id = poi_mapping.get(str(raw_label), -1)
                    if poi_id == -1:
                        m_prefix = re.match(r'^\d+\.(.+)', str(raw_label))
                        if m_prefix:
                            poi_id = poi_mapping.get(m_prefix.group(1), -1)
                        
                    if poi_id == -1:
                        continue
                        
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
            out_file = Path(args.output_dir) / f"{video_id}_features.pt"
            torch.save(data, out_file)
            print(f"Salvati {len(out_features)} tensori in {out_file}")
        else:
            print(f"Nessuna annotazione valida trovata per {video_id}")

    room_mapping_file = Path(args.output_dir) / "room_mapping.json"
    with open(room_mapping_file, 'w') as f:
        json.dump(room_mapping, f, indent=4)
    print(f"Mappatura Room salvata in {room_mapping_file} (Totale {len(room_mapping)} ambienti)")

if __name__ == "__main__":
    main()
