"""
Task 1 — Room-based Localization: DataLoader
Legge i file .pt delle feature DINOv2 pre-estratte e li serve al modello.

Struttura attesa della cartella features_dir:
    features_dir/
    ├── room_mapping.json          ← mappatura nome_stanza → id_intero
    ├── <seq_name>_features.pt     ← un file per sequenza/cartella training
    └── ...

Ogni file .pt contiene:
    {
        "video_id":    str            — nome della sequenza
        "features":    Tensor[N,384] — N frame × embedding DINOv2
        "room_labels": Tensor[N]     — ID stanza per ogni frame (long)
        "frame_ids":   Tensor[N]     — ID frame originale (long)
    }

Uso tipico:
    train_loader, val_loader, num_classes = get_dataloaders(
        features_dir="data/Task1_Features/Bellomo",
        batch_size=4,
        val_split=0.2
    )
"""
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class Task1FeatureDataset(Dataset):
    """
    Dataset PyTorch per Task 1.
    Carica i file .pt delle feature pre-estratte da DINOv2 (Phase 2).
    Ogni elemento è una sequenza (video training) completa con le sue label.

    Args:
        features_dir: cartella contenente room_mapping.json e i file .pt
        file_list:    lista opzionale di nomi file .pt da caricare;
                      se None carica tutti i .pt nella cartella
    """

    def __init__(self, features_dir: str, file_list=None):
        self.features_dir = Path(features_dir)

        # Carica mapping stanza → ID intero
        mapping_file = self.features_dir / "room_mapping.json"
        if not mapping_file.exists():
            raise FileNotFoundError(
                f"room_mapping.json non trovato in {features_dir}.\n"
                f"Assicurati che la Phase 2 (estrazione feature) sia stata completata."
            )
        with open(mapping_file, "r") as f:
            self.room_mapping = json.load(f)
        self.num_classes = len(self.room_mapping)

        # Lista dei file .pt da caricare
        if file_list is not None:
            self.pt_files = sorted([self.features_dir / fname for fname in file_list])
        else:
            self.pt_files = sorted(self.features_dir.glob("*.pt"))

        if len(self.pt_files) == 0:
            raise FileNotFoundError(
                f"Nessun file .pt trovato in {features_dir}.\n"
                f"Controlla che la Phase 2 abbia prodotto output in questa cartella."
            )

        print(
            f"[Task1FeatureDataset] {features_dir}\n"
            f"  Sequenze: {len(self.pt_files)} | Classi: {self.num_classes}\n"
            f"  Mapping: {self.room_mapping}"
        )

    def __len__(self) -> int:
        return len(self.pt_files)

    def __getitem__(self, idx: int):
        """
        Restituisce (features, labels) per una singola sequenza.
        features: Tensor[T, 384]  — embedding DINOv2 per ogni frame
        labels:   Tensor[T]       — ID stanza per ogni frame (long)
        """
        data = torch.load(self.pt_files[idx], map_location="cpu", weights_only=True)
        features = data["features"].float()    # [T, 384]
        labels = data["room_labels"].long()    # [T]

        # Sanity check
        assert features.shape[0] == labels.shape[0], (
            f"Mismatch features/labels in {self.pt_files[idx]}: "
            f"{features.shape[0]} vs {labels.shape[0]}"
        )

        return features, labels


# ─────────────────────────────────────────────────────────────────────────────
# Collate function (padding per batch di lunghezze variabili)
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Collate function per sequenze di lunghezza variabile.
    Esegue il padding delle sequenze alla lunghezza massima del batch.

    Restituisce:
        features_padded: Tensor[B, T_max, 384]
        labels_padded:   Tensor[B, T_max]       — valore -100 per frame paddati
        mask:            BoolTensor[B, T_max]   — True = frame reale
        lengths:         LongTensor[B]           — lunghezze originali
    """
    features_list, labels_list = zip(*batch)

    lengths = torch.tensor([f.shape[0] for f in features_list], dtype=torch.long)

    # padding_value=0.0 per le feature (non influenza il training)
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    # padding_value=-100 per le label → CrossEntropyLoss le ignora con ignore_index=-100
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    # Maschera booleana: True = frame reale, False = padding
    max_len = features_padded.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return features_padded, labels_padded, mask, lengths


# ─────────────────────────────────────────────────────────────────────────────
# Factory con split train/val
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    features_dir: str,
    val_features_dir: str = None,
    batch_size: int = 4,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Costruisce DataLoader per train e val. Se val_features_dir è specificata, carica
    il validation set da una cartella separata. Altrimenti cerca un file "validation" 
    nella cartella di train o esegue uno split random.

    Args:
        features_dir: path alla cartella del training (es. data/Task1_Features/Bellomo_Train)
        val_features_dir: path opzionale alla cartella validation (es. data/Task1_Features/Bellomo_Val)
        batch_size:   dimensione del batch per il training (val usa sempre batch_size=1)
        val_split:    frazione da usare come validation (solo se val_features_dir è None)
        num_workers:  worker per il DataLoader
        seed:         seed per la riproducibilità dello split

    Returns:
        (train_loader, val_loader, num_classes)
    """
    features_dir = Path(features_dir)

    all_files_raw = sorted([f.name for f in features_dir.glob("*.pt")])
    if len(all_files_raw) == 0:
        raise FileNotFoundError(f"Nessun file .pt in {features_dir}")

    # ── Deduplication filter ──────────────────────────────────────────────
    # Nel dataset Bellomo esistono due tipi di file:
    #   1) Room-level:  "8_Sala8_features.pt"          (intera stanza, tutti i frame)
    #   2) Video-clip:  "8.1_Sala8_Artwork.mp4_frames_features.pt"  (sotto-sequenza)
    # I file room-level già contengono i frame dei video-clip della stessa stanza,
    # quindi usarli insieme creerebbe duplicati (overfitting).
    # Strategia: se esistono file room-level, usa SOLO quelli;
    #            altrimenti usa tutti i file (fallback per dataset senza aggregazione).
    room_level_files = [f for f in all_files_raw if f.endswith("_features.pt") and ".mp4" not in f]
    clip_only_files  = [f for f in all_files_raw if ".mp4" in f]

    if room_level_files:
        all_files = room_level_files
        if clip_only_files:
            print(
                f"[get_dataloaders] Deduplication: trovati {len(room_level_files)} file room-level e "
                f"{len(clip_only_files)} video-clip. Uso SOLO i file room-level per evitare duplicati."
            )
    else:
        all_files = all_files_raw
        print(f"[get_dataloaders] Nessun file room-level trovato, uso tutti i {len(all_files)} file .pt.")

    # Modalità a due cartelle separate
    if val_features_dir is not None:
        val_dir = Path(val_features_dir)
        val_files = sorted([f.name for f in val_dir.glob("*.pt")])
        if len(val_files) == 0:
            raise FileNotFoundError(f"Nessun file .pt in {val_features_dir}")
            
        print(f"[get_dataloaders] Modalità cartelle separate: Train={features_dir}, Val={val_features_dir}")
        train_files = all_files
        train_ds = Task1FeatureDataset(features_dir, file_list=train_files)
        val_ds   = Task1FeatureDataset(val_dir, file_list=val_files)
        
    # Modalità a singola cartella (fallback)
    else:
        # Cerca esplicitamente un file di validazione
        val_candidate = next((f for f in all_files if "validation" in f.lower()), None)

        if val_candidate is not None:
            print(f"[get_dataloaders] Trovato file di validazione esplicito: {val_candidate}")
            val_files = [val_candidate]
            train_files = [f for f in all_files if f != val_candidate]
        else:
            # Fallback allo split random
            rng = random.Random(seed)
            shuffled = all_files.copy()
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_split))
            val_files = shuffled[:n_val]
            train_files = shuffled[n_val:]

        print(
            f"[get_dataloaders] Split da singola cartella → Train: {len(train_files)} | Val: {len(val_files)} "
            f"(seed={seed})"
        )

        train_ds = Task1FeatureDataset(features_dir, file_list=train_files)
        val_ds   = Task1FeatureDataset(features_dir, file_list=val_files)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    # Validation: batch_size=1 per gestire sequenze molto lunghe senza OOM
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_ds.num_classes


# ─────────────────────────────────────────────────────────────────────────────
# Test rapido (lanciare direttamente per verifica)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    features_dir = sys.argv[1] if len(sys.argv) > 1 else "data/Task1_Features/Bellomo"

    if not os.path.exists(features_dir):
        print(f"ERRORE: {features_dir} non esiste. Passa il path corretto come argomento.")
        sys.exit(1)

    print(f"\n=== Test DataLoader: {features_dir} ===")
    train_loader, val_loader, num_classes = get_dataloaders(
        features_dir, batch_size=2, val_split=0.2
    )
    print(f"Classi totali: {num_classes}")

    # Stampa un batch di training
    for features, labels, mask, lengths in train_loader:
        print(f"\nBatch train:")
        print(f"  features: {features.shape}  dtype={features.dtype}")
        print(f"  labels:   {labels.shape}   dtype={labels.dtype}")
        print(f"  mask:     {mask.shape}")
        print(f"  lengths:  {lengths.tolist()}")
        print(f"  Label range: min={labels[mask].min().item()} max={labels[mask].max().item()}")
        break

    print("\n✓ DataLoader funzionante.")
