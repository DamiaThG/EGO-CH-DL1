# Progetto EGO-CH — Deep Learning

Repository contenente il codice e i materiali relativi al progetto del corso di Deep Learning per il dataset **EGO-CH** (Egocentric Cultural Heritage).

## Autori e Task
- **Damiano Messina**: Task 1 — Room Localization
- **Claudio Nuncibello**: Task 2 — POI Recognition
- **Emanuele Barbagallo**: Task 3 — Object Retrieval

---

## Struttura del Repository

```
EGO-CH-DL1/
├── Scripts/
│   ├── Task1_Localization/       # Codice per la localizzazione della stanza (Mamba)
│   │   ├── models/               # Architetture di rete (Mamba, varianti)
│   │   ├── v2_advanced/          # Esperimenti e configurazioni v2
│   │   └── v3_experimental/      # Esperimenti e configurazioni v3
│   ├── Task2_Object_Recognition/ # Codice per l'Object/POI Recognition (YOLOv8)
│   └── Task3_Object_Retrieval/   # Codice per l'Object Retrieval (Feature extraction & similarity)
└── relazione.pdf                 # PDF compilato della relazione
```

---

## Panoramica dei Task

### Task 1: Room Localization
Obiettivo: Classificare in sequenza temporale la stanza in cui si trova l'utente a partire dalle feature dei frame video (estratte con backbone visuali come DINOv2).
- Modello principale: **Mamba** (Selective State Space Model) per catturare le dipendenze temporali sulle sequenze video.
- Script principali:
  - `task1_train.py`: Addestramento del modello Mamba.
  - `task1_evaluate.py`: Valutazione con metriche frame-level (FF1) e segment-level (ASF1).
  - `slurm_task1_mamba.sh`: Script di sottomissione per cluster SLURM (container Apptainer).

### Task 2: POI Recognition
Obiettivo: Rilevare e classificare i punti di interesse (POI) e gli oggetti rilevanti presenti negli iframe del dataset.
- Modello principale: **YOLOv8** (You Only Look Once).
- Script principali:
  - `task2_format_yolo.py`: Conversione del dataset nel formato annotazioni YOLO (`.txt` per immagine).
  - `task2_train_yolo.py`: Addestramento di YOLOv8 (nano/small/medium/large/pose) sugli iframe annotati.
  - `run_task2_yolo.sh`: Execution pipeline per l'addestramento e la validazione.

### Task 3: Object Retrieval
Obiettivo: Dato un crop o query di un oggetto/POI, recuperare le immagini o i frame corrispondenti nel dataset ordinati per similarità.
- Approccio: Estrazione di feature locali/patch (DINOv2/CLIP) e calcolo della similarità tramite Cosine Distance / KNN.
- Script principali:
  - `task3_extract_patch_features.py`: Estrazione e salvataggio dei descrittori di patch per le immagini target.
  - `task3_evaluate_retrieval.py`: Calcolo delle metriche di retrieval (mAP, Precision@k, Recall@k).
  - `run_task3.sh`: Execution script per l'eseguibile di retrieval.

---

## Requisiti ed Esecuzione

I requisiti principali includono PyTorch, PyTorch Lightning, Ultralytics (per YOLO), mambapy (per Mamba) e scikit-learn.

### Esempio Esecuzione Task 1 (Mamba)
```bash
python -m Scripts.Task1_Localization.task1_train --batch_size 8 --epochs 50
```

### Esempio Esecuzione Task 2 (YOLOv8)
```bash
python Scripts/Task2_Object_Recognition/task2_train_yolo.py --model yolo11m.pt --epochs 50
```

### Esempio Esecuzione Task 3 (Retrieval)
```bash
bash Scripts/Task3_Object_Retrieval/run_task3.sh
```

---

## Relazione
La relazione completa che illustra l'architettura dei modelli, i dettagli sperimentali e i risultati ottenuti è fornita in `relazione.pdf`.
