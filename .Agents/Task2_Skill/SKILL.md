---
name: Task2_POI_Recognition
description: Linee guida e decisioni architetturali per il Task 2 (Point of Interest Recognition)
---

# Skill: Task 2 - Point of Interest (POI) Recognition

Questa documentazione definisce la pipeline operativa ufficiale per il Task 2 del progetto EGO-CH, riflettendo le decisioni implementative prese per l'architettura.

## 1. Origine dei Dati
- **Dataset Utilizzati:** Vengono utilizzate esclusivamente le immagini raw ad alta risoluzione dalla cartella `data/Points_Of_Interest_Recognition`.
- **Nessuna sovrapposizione:** Non vengono mai lette né le cartelle `_Small` del Task 1, né le cartelle `Object_Retrieval` del Task 3.

## 2. Fase di Preprocessing e Riformattazione
- **Nessuna Feature Extraction Esplicita:** A differenza degli altri task, per il Task 2 non vengono estratti embedding offline (niente DINOv2 in fase di preprocessamento). Le feature verranno estratte end-to-end (dalle immagini raw alle predizioni) dalla rete neurale convoluzionale del modello di object detection.
- **Formattazione Dataset (YOLO):** L'unica fase di preprocessamento consiste nella conversione dei dati (immagini e label .txt) in un formato YOLOv8 standard. Verranno create le strutture:
  - `images/train`, `images/val`, `images/test`
  - `labels/train`, `labels/val`, `labels/test`
- **Gestione Disomogeneità:** Lo script `task2_format_yolo.py` (in via di sviluppo) si occuperà di gestire le enormi differenze strutturali tra:
  - *Bellomo:* Struttura piatta, immagini e bounding box separate in cartelle genitrici.
  - *Monastero:* Struttura altamente gerarchica, con immagini e annotazioni annidate per POI specifici.

## 3. Training e Modello
- **Modello Scelto:** Verrà addestrato **YOLOv8-nano** (oppure YOLOv11/YOLOv10 a seconda della disponibilità) a partire dai pesi pre-addestrati.
- **Obiettivo:** Predire le bounding box dei Punti di Interesse (POI) direttamente dai frame, restituendo coordinate `(x_center, y_center, w, h)` e la classe del POI.
