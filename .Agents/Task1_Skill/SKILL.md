---
name: Task1_Room_Localization
description: Linee guida e decisioni architetturali per il Task 1 (Room-based Localization)
---

# Skill: Task 1 - Room-based Localization

Questa documentazione definisce la pipeline operativa ufficiale per il Task 1 del progetto EGO-CH, riflettendo le decisioni implementative prese per l'architettura.

## 1. Origine dei Dati
- **Dataset Utilizzati:** Vengono utilizzate esclusivamente le cartelle `_Small` (es. `Bellomo_Small` e `Monastero_Benedettini_Small`).
- **Nessuna Bounding Box:** A differenza della pipeline originale, per il Task 1 non vengono lette né elaborate le annotazioni delle bounding box (POI), in quanto il task richiede solo la localizzazione della stanza a livello globale.

## 2. Fase di Preprocessing (Estrazione Feature)
- **Modello Vision Foundation:** Il modello originale ConvNeXt è stato sostituito con **DINOv2** (`dinov2_vits14`), che produce embedding di dimensione 384.
- **Modalità Offline:** Poiché i nodi di calcolo del cluster non hanno accesso a Internet, DINOv2 viene caricato in modalità `source='local'` da una cartella `weights/` pre-scaricata sulla radice del progetto, impostando opportunamente la variabile d'ambiente `TORCH_HOME`.
- **Iterazione Ricorsiva:** L'estrazione scorre in modo ricorsivo (tramite `rglob`) tutti i file `.jpg` all'interno dell'albero di directory. Questo permette di gestire le strutture piatte (es. `Bellomo`) e quelle altamente annidate (es. `Monastero_Benedettini_Training/Training/5_Antirefettorio/5.0_Antirefettorio`).
- **Etichettatura Automatica (Room ID):** L'etichetta della stanza (Room Label) viene dedotta dinamicamente dal nome della cartella padre che contiene le immagini (es. da `10.1_Sala10_...` viene estratto `10`). Non è più necessario il file `training.txt`.

## 3. Output
Lo script genera dei file tensoriali `.pt` per ogni sequenza/cartella, contenenti:
- `video_id`: nome della cartella sequenza.
- `features`: tensore PyTorch [N, 384] con le feature estratte da DINOv2.
- `room_labels`: tensore 1D [N] con l'ID testuale/numerico della stanza.
- `frame_ids`: tensore 1D [N] con gli ID originali del frame.
Questi file andranno a sostituire le vecchie cartelle `_Features` e verranno salvati all'interno della cartella `Task1_Localization/`.
