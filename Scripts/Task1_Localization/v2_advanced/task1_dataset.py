# Gestione del dataset e dataloader per caricare le feature DINOv2
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class Task1FeatureDataset(Dataset):

    def __init__(self, features_dir: str, file_list=None):
        self.features_dir = Path(features_dir)
        mapping_file = self.features_dir / 'room_mapping.json'
        if not mapping_file.exists():
            raise FileNotFoundError(f'room_mapping.json non trovato in {features_dir}.\nAssicurati che la Phase 2 (estrazione feature) sia stata completata.')
        with open(mapping_file, 'r') as f:
            self.room_mapping = json.load(f)
        self.num_classes = len(self.room_mapping)
        if file_list is not None:
            self.pt_files = sorted([self.features_dir / fname for fname in file_list])
        else:
            self.pt_files = sorted(self.features_dir.glob('*.pt'))
        if len(self.pt_files) == 0:
            raise FileNotFoundError(f'Nessun file .pt trovato in {features_dir}.\nControlla che la Phase 2 abbia prodotto output in questa cartella.')
        self.data = []
        for pt_path in self.pt_files:
            data = torch.load(pt_path, map_location='cpu', weights_only=True)
            features = data['features'].float()
            labels = data['room_labels'].long()
            self.data.append((features, labels))
        if len(self.data) > 0:
            all_labels = torch.cat([labels for _, labels in self.data])
            class_counts = torch.bincount(all_labels, minlength=self.num_classes)
            class_counts = torch.where(class_counts == 0, torch.tensor(1), class_counts)
            total_frames = class_counts.sum()
            self.class_weights = total_frames / (self.num_classes * class_counts.float())
        else:
            self.class_weights = torch.ones(self.num_classes)
        print(f'[Task1FeatureDataset] {features_dir}\n  Sequenze: {len(self.pt_files)} | Classi: {self.num_classes}\n  Mapping: {self.room_mapping}')

    def __len__(self) -> int:
        return len(self.pt_files)

    def __getitem__(self, idx: int):
        data = torch.load(self.pt_files[idx], map_location='cpu', weights_only=True)
        features = data['features'].float()
        labels = data['room_labels'].long()
        assert features.shape[0] == labels.shape[0], f'Mismatch features/labels in {self.pt_files[idx]}: {features.shape[0]} vs {labels.shape[0]}'
        return (features, labels)

def collate_fn(batch):
    features_list, labels_list = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in features_list], dtype=torch.long)
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    max_len = features_padded.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return (features_padded, labels_padded, mask, lengths)

def get_dataloaders(features_dir: str, val_features_dir: str=None, batch_size: int=4, val_split: float=0.2, num_workers: int=4, seed: int=42):
    features_dir = Path(features_dir)
    all_files_raw = sorted([f.name for f in features_dir.glob('*.pt')])
    if len(all_files_raw) == 0:
        raise FileNotFoundError(f'Nessun file .pt in {features_dir}')
    room_level_files = [f for f in all_files_raw if f.endswith('_features.pt') and '.mp4' not in f]
    clip_only_files = [f for f in all_files_raw if '.mp4' in f]
    if room_level_files:
        all_files = room_level_files
        if clip_only_files:
            print(f'[get_dataloaders] Deduplication: trovati {len(room_level_files)} file room-level e {len(clip_only_files)} video-clip. Uso SOLO i file room-level per evitare duplicati.')
    else:
        all_files = all_files_raw
        print(f'[get_dataloaders] Nessun file room-level trovato, uso tutti i {len(all_files)} file .pt.')
    if val_features_dir is not None:
        val_dir = Path(val_features_dir)
        val_files_raw = sorted([f.name for f in val_dir.glob('*.pt')])
        if len(val_files_raw) == 0:
            raise FileNotFoundError(f'Nessun file .pt in {val_features_dir}')
        val_room_level = [f for f in val_files_raw if f.endswith('_features.pt') and '.mp4' not in f]
        val_clip_only = [f for f in val_files_raw if '.mp4' in f]
        if val_room_level:
            val_files = val_room_level
            if val_clip_only:
                print(f'[get_dataloaders] Val deduplication: trovati {len(val_room_level)} file room-level e {len(val_clip_only)} video-clip. Uso SOLO i file room-level per evitare duplicati.')
        else:
            val_files = val_files_raw
            print(f'[get_dataloaders] Val: nessun file room-level, uso tutti i {len(val_files)} file .pt.')
        print(f'[get_dataloaders] Modalità cartelle separate: Train={features_dir}, Val={val_features_dir}')
        train_files = all_files
        train_ds = Task1FeatureDataset(features_dir, file_list=train_files)
        val_ds = Task1FeatureDataset(val_dir, file_list=val_files)
    else:
        val_candidate = next((f for f in all_files if 'validation' in f.lower()), None)
        if val_candidate is not None:
            print(f'[get_dataloaders] Trovato file di validazione esplicito: {val_candidate}')
            val_files = [val_candidate]
            train_files = [f for f in all_files if f != val_candidate]
        else:
            rng = random.Random(seed)
            shuffled = all_files.copy()
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_split))
            val_files = shuffled[:n_val]
            train_files = shuffled[n_val:]
        print(f'[get_dataloaders] Split da singola cartella → Train: {len(train_files)} | Val: {len(val_files)} (seed={seed})')
        train_ds = Task1FeatureDataset(features_dir, file_list=train_files)
        val_ds = Task1FeatureDataset(features_dir, file_list=val_files)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    return (train_loader, val_loader, train_ds.num_classes)
if __name__ == '__main__':
    import sys
    import os
    features_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/Task1_Features/Bellomo'
    if not os.path.exists(features_dir):
        print(f'ERRORE: {features_dir} non esiste. Passa il path corretto come argomento.')
        sys.exit(1)
    print(f'\n=== Test DataLoader: {features_dir} ===')
    train_loader, val_loader, num_classes = get_dataloaders(features_dir, batch_size=2, val_split=0.2)
    print(f'Classi totali: {num_classes}')
    for features, labels, mask, lengths in train_loader:
        print(f'\nBatch train:')
        print(f'  features: {features.shape}  dtype={features.dtype}')
        print(f'  labels:   {labels.shape}   dtype={labels.dtype}')
        print(f'  mask:     {mask.shape}')
        print(f'  lengths:  {lengths.tolist()}')
        print(f'  Label range: min={labels[mask].min().item()} max={labels[mask].max().item()}')
        break
    print('\n✓ DataLoader funzionante.')