---
name: Task1_Room_Localization_Implementation
description: Guida implementativa completa, passo per passo, per sviluppare il Task 1 (Room-based Localization) del progetto EGO-CH usando Mamba SSM come modello principale, con confronto contro baseline MLP e LSTM. Da leggere per intero prima di scrivere una singola riga di codice.
---

# IMPLEMENTATION SKILL — Task 1: Room-based Localization

> **Quando attivare questa skill:** ogni volta che devi scrivere, modificare, debuggare o estendere qualsiasi parte del codice di Task 1 (training, valutazione, dataloader, modello).

---

## 0. Contesto Assoluto — Leggi Prima di Tutto

### Il Progetto
Progetto per il corso di Deep Learning (DMI, UniCT). Implementazione del Task 1 del benchmark **EGO-CH** (paper: *"EGO-CH: Dataset and Fundamental Tasks for Visitors Behavioral Understanding Using Egocentric Vision"*) con un modello alternativo alla baseline VGG-19+KNN degli autori.

### Il Task
**Room-based Localization:** dato un video egocentric di un visitatore di un sito culturale, classificare ogni frame con l'ID della stanza (ambiente) in cui si trova il visitatore. Il problema è una **classificazione su sequenze temporali** (non frame-per-frame isolati).

### Stato Attuale del Progetto
- ✅ **Phase 1 DONE:** subsampling frame (29.97/24fps → 6fps) — script: `Scripts/Task1_Localization/phase1_subsample_frames.py`
- ✅ **Phase 2 DONE:** estrazione feature DINOv2 — script: `Scripts/Task1_Localization/phase2_extract_features.py`
- ✅ **Feature estratte sul cluster:** `/home/mssdmn01t05c351v/ProgDL1/data/Task1_Features`
- ❌ **Phase 3 DA FARE:** DataLoader + Modello Mamba + Training + Valutazione

### Path Fondamentali sul Cluster
```
Root del progetto Task 1:  /home/mssdmn01t05c351v/ProgDL1/
Feature Bellomo:           /home/mssdmn01t05c351v/ProgDL1/data/Task1_Features/Bellomo/
Feature Monastero:         /home/mssdmn01t05c351v/ProgDL1/data/Task1_Features/Monastero/
Script Task 1:             /home/mssdmn01t05c351v/ProgDL1/Scripts/Task1_Localization/
Pesi DINOv2 (già salvati): /home/mssdmn01t05c351v/ProgDL1/weights/facebookresearch_dinov2_main/
Container (tutti i task):  /shared/sifs/latest.sif
```

---

## 1. Struttura dei Dati di Input (Phase 2 Output)

### Formato dei file `.pt`
Ogni sequenza (sottocartella del dataset di training) genera un file `.pt`. Caricalo così:

```python
import torch
data = torch.load("path/to/sequence_features.pt")

# Campi disponibili:
data["video_id"]      # str — nome della cartella/sequenza
data["features"]      # Tensor [N, 384] — N frame, embedding DINOv2 dim=384
data["room_labels"]   # Tensor [N] dtype=torch.long — ID stanza per ogni frame
data["frame_ids"]     # Tensor [N] dtype=torch.long — ID frame originale (per debug)
```

### File `room_mapping.json`
Ogni dataset (Bellomo, Monastero) ha il suo `room_mapping.json` nella cartella delle feature.

```json
{
  "1": 0,   // nome stanza estratto dalla directory → ID intero 0-indexed
  "2": 1,
  "10": 2,
  ...
}
```

**Attenzione:** il mapping viene generato dinamicamente durante la Phase 2, quindi l'ID intero assegnato può NON coincidere con il numero della stanza. Usa SEMPRE `room_mapping.json` come riferimento.

### Struttura attesa delle cartelle delle feature
```
/home/mssdmn01t05c351v/ProgDL1/data/Task1_Features/
├── Bellomo/
│   ├── room_mapping.json
│   ├── 1.0_Sala1_..._features.pt
│   ├── 1.1_Sala1_..._features.pt
│   ├── 2.0_Sala2_..._features.pt
│   └── ...
└── Monastero/
    ├── room_mapping.json
    ├── 1.0_Antirefettorio_..._features.pt
    └── ...
```

> ⚠️ **Verifica sempre la struttura reale con `ls` prima di scrivere il DataLoader.** La Phase 2 potrebbe aver generato una struttura leggermente diversa da quella attesa. Il campo `video_id` dentro ogni `.pt` è la fonte di verità.

---

## 2. Struttura del Codice da Creare

Usa la **stessa filosofia di struttura** del resto del progetto:

```
Scripts/Task1_Localization/
├── phase1_subsample_frames.py      ← già esiste (non toccare)
├── phase2_extract_features.py      ← già esiste (non toccare)
├── run_extract_features.sh         ← già esiste (non toccare)
│
│   ── DA CREARE ──
├── task1_dataset.py                ← DataLoader PyTorch
├── task1_model_mamba.py            ← Modello Mamba SSM
├── task1_model_baselines.py        ← Modelli baseline (MLP, LSTM)
├── task1_metrics.py                ← Calcolo FF1 e ASF1
├── task1_train.py                  ← Training loop (PyTorch Lightning)
├── task1_evaluate.py               ← Valutazione su test set
├── run_task1_train_mamba.sh        ← Runner script (NO direttive SBATCH)
├── run_task1_train_baselines.sh    ← Runner script baseline
├── slurm_task1_mamba.sh            ← Wrapper SLURM (usa mamba_env.sif)
└── slurm_task1_baselines.sh        ← Wrapper SLURM (usa latest.sif)
```

> ⚠️ **Regola di architettura del cluster (dalla Cluster Skill):** Script runner (`run_*.sh`) = logica pura, NO `#SBATCH`. Wrapper SLURM (`slurm_*.sh`) = solo direttive `#SBATCH` + chiamata `apptainer exec`. **Non mescolarli mai.**

---

## 3. Passo 1 — `task1_dataset.py`: Il DataLoader

### Obiettivo
Caricare i file `.pt` pre-computati e servirli al modello in batch di sequenze. Le sequenze hanno lunghezze variabili → il batching richiede padding.

### Implementazione

```python
# task1_dataset.py
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence


class Task1FeatureDataset(Dataset):
    """
    Carica i file .pt delle feature DINOv2 pre-estratte.
    Ogni elemento è una sequenza (video) completa con le sue label.
    """
    def __init__(self, features_dir: str):
        self.features_dir = Path(features_dir)
        
        # Carica il mapping stanza → ID intero
        mapping_file = self.features_dir / "room_mapping.json"
        with open(mapping_file, 'r') as f:
            self.room_mapping = json.load(f)
        self.num_classes = len(self.room_mapping)
        
        # Lista tutti i file .pt (esclude room_mapping.json)
        self.pt_files = sorted(self.features_dir.glob("*.pt"))
        
        if len(self.pt_files) == 0:
            raise FileNotFoundError(f"Nessun file .pt trovato in {features_dir}")
        
        print(f"Dataset caricato: {len(self.pt_files)} sequenze, {self.num_classes} classi")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        data = torch.load(self.pt_files[idx], map_location='cpu')
        features = data["features"].float()    # [T, 384]
        labels = data["room_labels"].long()    # [T]
        return features, labels


def collate_fn(batch):
    """
    Gestisce sequenze di lunghezza variabile con padding.
    Restituisce anche una maschera per ignorare i frame paddati nella loss.
    """
    features_list, labels_list = zip(*batch)
    
    # Lunghezze originali (prima del padding)
    lengths = torch.tensor([f.shape[0] for f in features_list], dtype=torch.long)
    
    # Padding: porta tutte le sequenze alla lunghezza massima del batch
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    
    # Maschera booleana: True = frame reale, False = padding
    max_len = features_padded.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    
    return features_padded, labels_padded, mask, lengths


def get_dataloader(features_dir, batch_size=8, shuffle=True, num_workers=4):
    dataset = Task1FeatureDataset(features_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    ), dataset.num_classes
```

**Note implementative critiche:**
- `padding_value=-100` per le label → PyTorch `CrossEntropyLoss` ignora automaticamente gli indici con valore `-100` tramite il parametro `ignore_index=-100`.
- `map_location='cpu'` nel `torch.load` → il DataLoader gestisce il trasferimento su GPU.
- Usa `batch_size=1` se le sequenze sono troppo lunghe per la VRAM (sequenze da 18000+ frame).

---

## 4. Passo 2 — `task1_model_mamba.py`: Il Modello Principale

### Libreria Mamba: `mambapy` (unica opzione disponibile in ProgDL1)

> ⚠️ **`mamba-ssm` NON è accessibile** in questo progetto (appartiene al container `mamba_env.sif` di un altro progetto). Usare **esclusivamente `mambapy`** installata via `pip install --user` in `latest.sif`.

Tutti i job Task 1 (Mamba, MLP, LSTM) usano lo stesso container: **`latest.sif`**.

### Architettura del Modello

```
Input [B, T, 384]
    │
    ▼
Linear Projection → [B, T, d_model]
    │
    ▼
┌── N × MambaBlock ──┐
│  RMSNorm(x)        │
│  MambaLayer(x)     │  ← SSM con meccanismo di selezione
│  x = x + residual  │
└────────────────────┘
    │
    ▼
Final RMSNorm → [B, T, d_model]
    │
    ▼
Linear Head → [B, T, num_classes]
    │
    ▼
Output logits [B, T, num_classes]
```

### Implementazione

```python
# task1_model_mamba.py
import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig


class MambaRoomLocalizer(nn.Module):
    """
    Modello Mamba SSM per Room-based Localization.
    Backend: mambapy (installata via pip install --user in latest.sif)
    Input:  [B, T, input_dim]   (sequenza di feature DINOv2)
    Output: [B, T, num_classes] (logit per ogni frame)
    """
    def __init__(
        self,
        input_dim: int = 384,      # dimensione embedding DINOv2
        d_model: int = 256,        # dimensione interna Mamba
        num_layers: int = 4,       # numero di blocchi Mamba
        num_classes: int = 22,     # numero di stanze (22 Bellomo, 4 Monastero)
        d_state: int = 16,         # dimensione stato SSM
        d_conv: int = 4,           # dimensione conv interna Mamba
        expand: int = 2,           # fattore di espansione
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        
        print(f"MambaRoomLocalizer pronto | d_model={d_model} | layers={num_layers} | classi={num_classes}")

    def forward(self, x, lengths=None):
        # x: [B, T, input_dim]
        x = self.input_projection(x)      # [B, T, d_model]
        x = self.dropout(x)
        
        for layer in self.mamba_layers:
            x = layer(x)                  # [B, T, d_model]
        
        x = self.final_norm(x)
        logits = self.classifier(x)       # [B, T, num_classes]
        return logits


class MambaBlock(nn.Module):
    """
    Blocco Mamba singolo con pre-norm e skip connection.
    Usa mambapy come backend (disponibile in latest.sif via pip install --user).
    """
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # MambaConfig: n_layers=1 perché usiamo un blocco alla volta
        config = MambaConfig(
            d_model=d_model,
            n_layers=1,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand,
        )
        full_model = Mamba(config)
        # Estraiamo il singolo layer SSM dal modello completo
        self.mamba = full_model.layers[0]
    
    def forward(self, x):
        # Pre-norm + SSM + skip connection
        return x + self.mamba(self.norm(x))
```

> ⚠️ **Nota su mambapy:** a differenza di `mamba_ssm`, `mambapy` non richiede kernel CUDA custom, quindi funziona su `latest.sif` senza problemi. Le prestazioni sono leggermente inferiori ma i risultati sono equivalenti.

---

## 5. Passo 3 — `task1_model_baselines.py`: I Modelli di Confronto

Implementa MLP e LSTM con la **stessa interfaccia** del modello Mamba (stesso forward signature) per permettere confronti puliti.

```python
# task1_model_baselines.py
import torch
import torch.nn as nn


class MLPRoomLocalizer(nn.Module):
    """Baseline MLP frame-per-frame. Ignora il contesto temporale."""
    def __init__(self, input_dim=384, hidden_dim=256, num_classes=22, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, lengths=None):
        # x: [B, T, input_dim] → logits: [B, T, num_classes]
        return self.net(x)


class LSTMRoomLocalizer(nn.Module):
    """Baseline LSTM bidirezionale."""
    def __init__(self, input_dim=384, hidden_dim=256, num_layers=2,
                 num_classes=22, dropout=0.1, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(out_dim, num_classes)
    
    def forward(self, x, lengths=None):
        # x: [B, T, input_dim]
        if lengths is not None:
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output, _ = self.lstm(packed)
            output, _ = pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.lstm(x)
        return self.classifier(output)  # [B, T, num_classes]
```

---

## 6. Passo 4 — `task1_metrics.py`: FF1 e ASF1

Queste sono le metriche ufficiali del paper. **Devono essere implementate esattamente così** per garantire il confronto con la baseline degli autori.

```python
# task1_metrics.py
import torch
import numpy as np
from sklearn.metrics import f1_score


def compute_ff1(predictions: list, ground_truth: list) -> float:
    """
    Frame F1 score (FF1): F1 applicato frame per frame.
    
    Args:
        predictions: lista di array 1D (predizioni per ogni video del test set)
        ground_truth: lista di array 1D (label reali per ogni video del test set)
    
    Returns:
        FF1 score medio pesato per numero di frame
    """
    all_preds = np.concatenate(predictions)
    all_labels = np.concatenate(ground_truth)
    return f1_score(all_labels, all_preds, average='weighted', zero_division=0)


def compute_asf1(predictions: list, ground_truth: list,
                 overlap_threshold: float = 0.5) -> float:
    """
    Action Segment F1 (ASF1): F1 applicato ai segmenti temporali.
    Un segmento predetto è corretto se si sovrappone ≥ overlap_threshold
    con un segmento ground truth della stessa classe.
    
    Args:
        predictions: lista di array 1D (predizioni per ogni video)
        ground_truth: lista di array 1D (label reali per ogni video)
        overlap_threshold: soglia minima di IoU per considerare un segmento corretto
    
    Returns:
        ASF1 score medio sui video
    """
    video_f1s = []
    
    for preds, labels in zip(predictions, ground_truth):
        pred_segments = _extract_segments(preds)
        gt_segments = _extract_segments(labels)
        
        tp = 0
        fp = 0
        fn = 0
        
        gt_matched = set()
        
        for pred_seg in pred_segments:
            matched = False
            for i, gt_seg in enumerate(gt_segments):
                if i in gt_matched:
                    continue
                if pred_seg[2] == gt_seg[2]:  # stessa classe
                    iou = _segment_iou(pred_seg[:2], gt_seg[:2])
                    if iou >= overlap_threshold:
                        tp += 1
                        gt_matched.add(i)
                        matched = True
                        break
            if not matched:
                fp += 1
        
        fn = len(gt_segments) - len(gt_matched)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        video_f1s.append(f1)
    
    return float(np.mean(video_f1s))


def _extract_segments(labels: np.ndarray) -> list:
    """Converte un array di label frame-by-frame in lista di segmenti (start, end, class)."""
    if len(labels) == 0:
        return []
    segments = []
    start = 0
    current_class = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current_class:
            segments.append((start, i - 1, current_class))
            start = i
            current_class = labels[i]
    segments.append((start, len(labels) - 1, current_class))
    return segments


def _segment_iou(seg_a: tuple, seg_b: tuple) -> float:
    """Calcola l'IoU temporale tra due segmenti (start, end)."""
    intersection_start = max(seg_a[0], seg_b[0])
    intersection_end = min(seg_a[1], seg_b[1])
    intersection = max(0, intersection_end - intersection_start + 1)
    union = (seg_a[1] - seg_a[0] + 1) + (seg_b[1] - seg_b[0] + 1) - intersection
    return intersection / (union + 1e-8)
```

---

## 7. Passo 5 — `task1_train.py`: Il Training Loop

Usa **PyTorch Lightning** (`import lightning as L`) per strutturare il training. Rispetta le regole della Cluster Skill:
- Nessun progress bar visivo (`enable_progress_bar=False` nei logger)
- Log ogni N step per non intasare il file `.out` di SLURM
- Sempre `torch.set_float32_matmul_precision("high")` per sfruttare bf16 di L40S

```python
# task1_train.py
import os
import json
import argparse
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import locali (path relativo dalla root del progetto)
from Scripts.Task1_Localization.task1_dataset import get_dataloader
from Scripts.Task1_Localization.task1_metrics import compute_ff1, compute_asf1

import numpy as np


class Task1LightningModule(L.LightningModule):
    def __init__(self, model, num_classes, lr=1e-3, weight_decay=1e-4, max_epochs=50):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        
        # ignore_index=-100 → ignora i frame paddati
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        # Accumulatori per le metriche
        self._val_preds = []
        self._val_labels = []
        
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x, lengths=None):
        return self.model(x, lengths)

    def training_step(self, batch, batch_idx):
        features, labels, mask, lengths = batch
        # features: [B, T, 384], labels: [B, T], mask: [B, T]
        
        logits = self(features, lengths)              # [B, T, num_classes]
        
        # Reshape per la loss: [B*T, C] e [B*T]
        B, T, C = logits.shape
        loss = self.loss_fn(logits.reshape(B*T, C), labels.reshape(B*T))
        
        self.log('train/loss', loss, on_step=True, on_epoch=True,
                 prog_bar=False, batch_size=B)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels, mask, lengths = batch
        logits = self(features, lengths)              # [B, T, num_classes]
        
        B, T, C = logits.shape
        loss = self.loss_fn(logits.reshape(B*T, C), labels.reshape(B*T))
        self.log('val/loss', loss, on_epoch=True, prog_bar=True, batch_size=B)
        
        # Accumula predizioni per FF1/ASF1 a fine epoca
        preds = logits.argmax(dim=-1)  # [B, T]
        for b in range(B):
            L_b = lengths[b].item()
            self._val_preds.append(preds[b, :L_b].cpu().numpy())
            self._val_labels.append(labels[b, :L_b].cpu().numpy())

    def on_validation_epoch_end(self):
        if len(self._val_preds) == 0:
            return
        
        ff1 = compute_ff1(self._val_preds, self._val_labels)
        asf1 = compute_asf1(self._val_preds, self._val_labels)
        
        self.log('val/FF1', ff1, prog_bar=True)
        self.log('val/ASF1', asf1, prog_bar=True)
        
        print(f"\n[Epoch {self.current_epoch}] val/FF1={ff1:.4f} | val/ASF1={asf1:.4f}")
        
        self._val_preds.clear()
        self._val_labels.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    parser = argparse.ArgumentParser(description="Task 1 Training — Mamba Room Localizer")
    
    # Dataset
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Path alla cartella delle feature di training")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Path alla cartella delle feature di validation")
    
    # Modello
    parser.add_argument("--model", type=str, default="mamba",
                        choices=["mamba", "mlp", "lstm"],
                        help="Tipo di modello da usare")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_mamba_ssm", action="store_true",
                        help="Usa mamba_ssm (richiede mamba_env.sif)")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Output
    parser.add_argument("--output_dir", type=str,
                        default="Scripts/Task1_Localization/experiments")
    parser.add_argument("--run_name", type=str, default="task1_mamba")
    
    args = parser.parse_args()
    
    # ── Setup ──────────────────────────────────────────────────────────────
    torch.set_float32_matmul_precision("high")  # sfrutta bf16 su L40S
    
    os.makedirs(f"{args.output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.output_dir}/logs", exist_ok=True)
    
    # ── DataLoaders ────────────────────────────────────────────────────────
    train_loader, num_classes = get_dataloader(
        args.train_dir, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    val_loader, _ = get_dataloader(
        args.val_dir, batch_size=1,   # batch_size=1 in val per sequenze lunghe
        shuffle=False, num_workers=args.num_workers
    )
    
    print(f"Classi: {num_classes} | Train: {len(train_loader.dataset)} seqs | Val: {len(val_loader.dataset)} seqs")
    
    # ── Modello ────────────────────────────────────────────────────────────
    if args.model == "mamba":
        from Scripts.Task1_Localization.task1_model_mamba import MambaRoomLocalizer
        model = MambaRoomLocalizer(
            input_dim=384, d_model=args.d_model, num_layers=args.num_layers,
            num_classes=num_classes, d_state=args.d_state, dropout=args.dropout,
            use_mamba_ssm=args.use_mamba_ssm
        )
    elif args.model == "mlp":
        from Scripts.Task1_Localization.task1_model_baselines import MLPRoomLocalizer
        model = MLPRoomLocalizer(input_dim=384, hidden_dim=args.d_model,
                                 num_classes=num_classes, dropout=args.dropout)
    elif args.model == "lstm":
        from Scripts.Task1_Localization.task1_model_baselines import LSTMRoomLocalizer
        model = LSTMRoomLocalizer(input_dim=384, hidden_dim=args.d_model,
                                  num_classes=num_classes, dropout=args.dropout)
    
    # ── Lightning Module ────────────────────────────────────────────────────
    lit_model = Task1LightningModule(
        model=model, num_classes=num_classes,
        lr=args.lr, weight_decay=args.weight_decay, max_epochs=args.epochs
    )
    
    # ── Logger & Callbacks ──────────────────────────────────────────────────
    wandb_logger = WandbLogger(
        project="EGO-CH-Task1",
        name=args.run_name,
        save_dir=args.output_dir,
        offline=True  # WANDB_MODE=offline è già settato dal runner script
    )
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"{args.output_dir}/checkpoints",
        filename=f"{args.run_name}_best_{{val/ASF1:.4f}}",
        monitor="val/ASF1",
        mode="max",
        save_top_k=2
    )
    
    early_stop_cb = EarlyStopping(
        monitor="val/ASF1", mode="max", patience=15, verbose=True
    )
    
    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, early_stop_cb],
        enable_progress_bar=False,    # no progress bar → output leggibile nei log SLURM
        log_every_n_steps=10,
        precision="bf16-mixed",       # bf16 su L40S (gnode10)
        gradient_clip_val=1.0,        # gradient clipping — importante per Mamba su seq. lunghe
    )
    
    print(f"\nAvvio training: {args.model.upper()} | {args.epochs} epoche | lr={args.lr}")
    trainer.fit(lit_model, train_loader, val_loader)
    print(f"\nTraining completato. Checkpoint migliore: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
```

---

## 8. Passo 6 — Runner e Wrapper SLURM

### `run_task1_train_mamba.sh` (Runner — da eseguire dentro il container)

```bash
#!/bin/bash
# Runner script Task 1 — Mamba Training
# Da eseguire DENTRO il container latest.sif
# Uso: ./Scripts/Task1_Localization/run_task1_train_mamba.sh [--argomenti extra]
# I due dataset (Bellomo e Monastero) hanno feature in sottocartelle separate.

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Porta la shell alla root del progetto
cd "$(dirname "$0")/../.."

echo "======================================"
echo "Task 1 — Mamba Training (Bellomo)"
echo "======================================"

# ---- BELLOMO ----
# Le feature di Bellomo si trovano DENTRO la sottocartella Bellomo/
python Scripts/Task1_Localization/task1_train.py \
    --train_dir data/Task1_Features/Bellomo \
    --val_dir data/Task1_Features/Bellomo \
    --model mamba \
    --d_model 256 \
    --num_layers 4 \
    --d_state 16 \
    --dropout 0.1 \
    --batch_size 4 \
    --epochs 50 \
    --lr 1e-3 \
    --run_name "mamba_bellomo_d256_l4" \
    "$@"

echo "======================================"
echo "Training completato!"
echo "======================================"
```

### `slurm_task1_mamba.sh` (Wrapper SLURM — da usare con `sbatch`)

```bash
#!/bin/bash
#SBATCH --job-name=task1_mamba
#SBATCH --output=Scripts/Task1_Localization/experiments/logs/mamba_%j.out
#SBATCH --error=Scripts/Task1_Localization/experiments/logs/mamba_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --nodelist=gnode10
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:22000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p Scripts/Task1_Localization/experiments/logs
mkdir -p Scripts/Task1_Localization/experiments/checkpoints

# Task 1 usa latest.sif (mambapy disponibile lì via pip install --user)
apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    /shared/sifs/latest.sif \
    bash /workspace/Scripts/Task1_Localization/run_task1_train_mamba.sh
```

> ℹ️ **Nota container:** Task 1 usa `latest.sif` per **tutti i modelli** (Mamba, MLP, LSTM). `mambapy` è già installata lì. Non usare `mamba_env.sif` per questo progetto.

---

## 9. Passo 7 — `task1_evaluate.py`: Valutazione Finale

```python
# task1_evaluate.py
"""
Script di valutazione standalone.
Carica un checkpoint Lightning e valuta su un test set.
Produce tabella per-room come la Tabella 3 del paper.
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from Scripts.Task1_Localization.task1_dataset import Task1FeatureDataset
from Scripts.Task1_Localization.task1_metrics import compute_ff1, compute_asf1
from sklearn.metrics import f1_score


def evaluate(checkpoint_path, test_dir, model_type="mamba", use_mamba_ssm=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carica dataset
    dataset = Task1FeatureDataset(test_dir)
    num_classes = dataset.num_classes
    
    # Carica modello dal checkpoint Lightning
    from Scripts.Task1_Localization.task1_train import Task1LightningModule
    lit_model = Task1LightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    lit_model.eval()
    model = lit_model.model.to(device)
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            features, labels = dataset[i]
            features = features.unsqueeze(0).to(device)    # [1, T, 384]
            logits = model(features)                        # [1, T, num_classes]
            preds = logits.argmax(dim=-1).squeeze(0)       # [T]
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Metriche globali
    ff1 = compute_ff1(all_preds, all_labels)
    asf1 = compute_asf1(all_preds, all_labels)
    
    # Metriche per-room (come Tabella 3 del paper)
    flat_preds = np.concatenate(all_preds)
    flat_labels = np.concatenate(all_labels)
    
    # Carica room_mapping per nome delle classi
    with open(Path(test_dir) / "room_mapping.json") as f:
        room_mapping = json.load(f)
    id_to_name = {v: k for k, v in room_mapping.items()}
    
    per_class_f1 = f1_score(flat_labels, flat_preds,
                            average=None, zero_division=0)
    
    print("\n" + "="*60)
    print(f"RISULTATI VALUTAZIONE — {checkpoint_path}")
    print("="*60)
    print(f"{'Room':<30} {'FF1':>8}")
    print("-"*40)
    for class_id, f1 in enumerate(per_class_f1):
        room_name = id_to_name.get(class_id, f"Class_{class_id}")
        print(f"{room_name:<30} {f1:>8.4f}")
    print("-"*40)
    print(f"{'AVG FF1':<30} {ff1:>8.4f}")
    print(f"{'AVG ASF1':<30} {asf1:>8.4f}")
    print("="*60)
    
    # Confronto con baseline del paper
    print("\n--- CONFRONTO CON BASELINE DEL PAPER (VGG19+KNN) ---")
    print(f"{'Metrica':<15} {'Nostra':<10} {'Paper':<10} {'Delta':>8}")
    
    # Modifica questi valori in base al dataset che stai valutando
    paper_ff1_bellomo = 0.81
    paper_asf1_bellomo = 0.59
    
    print(f"{'FF1':<15} {ff1:<10.4f} {paper_ff1_bellomo:<10.4f} {ff1-paper_ff1_bellomo:>+8.4f}")
    print(f"{'ASF1':<15} {asf1:<10.4f} {paper_asf1_bellomo:<10.4f} {asf1-paper_asf1_bellomo:>+8.4f}")
    
    return {"ff1": ff1, "asf1": asf1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="mamba")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.test_dir, args.model)
```

---

## 10. Ordine di Esecuzione Completo

```bash
# ── SUL CLUSTER — ORDINE OBBLIGATORIO ─────────────────────────────────────

# 1. Verifica la struttura reale delle feature estratte
#    I due dataset sono in SOTTOCARTELLE SEPARATE:
ls /home/mssdmn01t05c351v/ProgDL1/data/Task1_Features/
# atteso: Bellomo/  Monastero/

ls /home/mssdmn01t05c351v/ProgDL1/data/Task1_Features/Bellomo/
# atteso: room_mapping.json  + file .pt delle sequenze

ls /home/mssdmn01t05c351v/ProgDL1/data/Task1_Features/Monastero/
# atteso: room_mapping.json  + file .pt delle sequenze

# 2. Verifica il contenuto di un file .pt per capire la struttura esatta
python3 -c "
import torch, os
d = '/home/mssdmn01t05c351v/ProgDL1/data/Task1_Features/Bellomo/'
files = [f for f in os.listdir(d) if f.endswith('.pt')]
data = torch.load(os.path.join(d, files[0]))
for k,v in data.items():
    print(k, type(v), getattr(v,'shape',''))
"

# 3. Carica i room_mapping per capire il numero di classi per dataset
cat /home/mssdmn01t05c351v/ProgDL1/data/Task1_Features/Bellomo/room_mapping.json
cat /home/mssdmn01t05c351v/ProgDL1/data/Task1_Features/Monastero/room_mapping.json

# 4. Training Mamba su Bellomo (via SLURM batch — raccomandato)
cd /home/mssdmn01t05c351v/ProgDL1
sbatch Scripts/Task1_Localization/slurm_task1_mamba.sh

# 5. Oppure training interattivo (debug veloce) — usa latest.sif (NON mamba-docker)
apptainer-gpu-xl
cd /home/mssdmn01t05c351v/ProgDL1
./Scripts/Task1_Localization/run_task1_train_mamba.sh --epochs 3 --batch_size 2

# 6. Monitora il job
squeue --me
tail -f Scripts/Task1_Localization/experiments/logs/mamba_<JOB_ID>.out

# 7. Valutazione finale
apptainer-gpu-xl
cd /home/mssdmn01t05c351v/ProgDL1
python Scripts/Task1_Localization/task1_evaluate.py \
    --checkpoint Scripts/Task1_Localization/experiments/checkpoints/mamba_bellomo_best.ckpt \
    --test_dir data/Task1_Features/Bellomo
```

---

## 11. Problemi Comuni e Soluzioni

### OOM (Out of Memory)
- Ridurre `batch_size` a 1 o 2
- Ridurre `d_model` da 256 a 128
- Ridurre `num_layers` da 4 a 2
- Aggiungere `torch.cuda.empty_cache()` dopo la validation

### `RuntimeError: Expected contiguous tensor`
- Aggiungere `.contiguous()` prima del forward Mamba: `x = x.contiguous()`

### `ModuleNotFoundError: mambapy`
- `mambapy` si installa con `pip install --user mambapy` dentro il container.
- Verifica: `python3 -c "from mambapy.mamba import Mamba; print('OK')"`
- Se non installata: `pip install --user mambapy` (persiste in `~/.local/`)

### `PYTHONNOUSERSITE` blocks user packages
- NON impostare `PYTHONNOUSERSITE=1` nei runner script (vedi Cluster Skill §6)

### Sequenze test molto lunghe (>10000 frame)
- Usa `batch_size=1` in valutazione
- Valuta la possibilità di spezzare le sequenze in chunk durante l'inferenza (non durante il training)

---

## 12. Checklist Finale Prima di Eseguire

- [ ] Verificata struttura reale dei file `.pt` sul cluster con `torch.load`
- [ ] Verificato che i file `.pt` di Bellomo siano in `data/Task1_Features/Bellomo/` e quelli di Monastero in `data/Task1_Features/Monastero/`
- [ ] Verificato `room_mapping.json` e numero classi reale per ogni dataset
- [ ] **Tutti i job usano `latest.sif`** (mai `mamba_env.sif` per ProgDL1)
- [ ] Runner scripts (`run_*.sh`) senza direttive `#SBATCH`
- [ ] Wrapper SLURM (`slurm_*.sh`) con sole direttive `#SBATCH` + `apptainer exec /shared/sifs/latest.sif`
- [ ] `WANDB_MODE=offline` nel runner
- [ ] `torch.set_float32_matmul_precision("high")` nel training
- [ ] `ignore_index=-100` nella CrossEntropyLoss
- [ ] `enable_progress_bar=False` nel Trainer Lightning
- [ ] Monitor su `val/ASF1` (non FF1) per ModelCheckpoint — è la metrica principale

---

*Documento creato il 22/07/2026. Aggiornare ad ogni modifica architetturale significativa.*
