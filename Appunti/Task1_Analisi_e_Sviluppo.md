# Task 1 — Room-based Localization: Analisi e Sviluppo

> **Data di creazione:** 22 Luglio 2026  
> **Scopo:** Documento di riferimento completo per ricostruire il lavoro svolto nei minimi dettagli.  
> **Dataset:** EGO-CH — Video egocentrici in siti culturali siciliani (Palazzo Bellomo, Monastero dei Benedettini)

---

## 1. Il Task nel Paper

### Definizione del Task
Il **Task 1 — Room-based Localization** consiste nel classificare, per ogni frame di un video egocentric acquisito da un visitatore di un sito culturale, in quale ambiente (stanza) si trova il visitatore.

In altre parole: dato un flusso di frame del video di una visita, l'algoritmo deve produrre una **segmentazione temporale del video**, dove ogni segmento è associato a una delle stanze del sito. Questo è quindi un problema di **classificazione per frame + smoothing temporale**.

### Dataset Utilizzato

#### Palazzo Bellomo (Siracusa)
- **22 ambienti** (stanze)
- 191 Points of Interest (POI)
- Training: 57 video brevi (1.4 min media), 140.286 frame totali
- Test: 10 video lunghi (31.27 min media, max 50.23 min)
- Risoluzione: 1280×720 @ 29.97 fps
- ⚠️ **Nessuna classe "negativa"** (il visitatore è sempre in un ambiente annotato)

#### Monastero dei Benedettini (Catania)
- **4 ambienti** (Antirefettorio, Aula S. Mazzarino, Cucina, Ventre)
- 35 POI (inclusi elementi architettonici come pavimenti, corridoi)
- Training: 48 video (2.2 min media)
- Validation: 20 video (3.5 min media)
- Test: 455 video brevi + 60 visite di visitatori reali (entro 3 mesi)
- Risoluzione: 1216×684 / 1408×792 @ 24/30 fps
- ✅ **Presenza di classe "negativa"** (visitatore in area non classificata)

---

## 2. La Baseline degli Autori

### Architettura (Figura 5 del paper)
La baseline è descritta nel paper con riferimento al lavoro precedente [3, 19] degli stessi autori e si articola in **3 stadi**:

```
Video Frames
     │
     ▼
[1] DISCRIMINATION
     VGG-19 (pre-trained ImageNet)
     → features per frame → K-NN classificazione frame-by-frame
     │
     ▼
[2] NEGATIVE REJECTION  (solo per Monastero, dove esistono ambienti "negativi")
     Sogliatura con parametro K
     │
     ▼
[3] SEQUENTIAL MODELING
     Algoritmo di smoothing temporale (HMM-like)
     con iperparametro ε (ottimizzato su validation set)
     │
     ▼
Segmentazione temporale: [ Sala1 | Sala3 | ... | Sala7 ]
```

### Dettagli Implementativi
| Componente | Dettaglio |
|---|---|
| Feature extractor | VGG-19 pre-trainato su ImageNet |
| Classifier | K-NN (nearest neighbor sui feature) |
| Smoothing | Algoritmo temporale con iperparametro ε |
| ε ottimale (Bellomo) | `10^-273` (grid search su val set) |
| ε ottimale (Monastero) | `10^-89` |
| K ottimale (Monastero) | `K = 100` |

### Metriche di Valutazione
Gli autori usano **due metriche complementari**, entrambe basate sull'F1 score:

- **FF1 (Frame F1):** F1 score applicato ai singoli frame. Misura l'accuratezza della classificazione a livello di frame, ma **non valuta la coerenza temporale**.
- **ASF1 (Action Segment F1):** F1 score applicato ai segmenti temporali. Misura la capacità del modello di produrre **segmentazioni coerenti** con il ground truth. È la metrica più sfidante e significativa.

Entrambe sono normalizzate in [0, 1].

### Risultati Baseline (Paper, Tabella 3)

#### Palazzo Bellomo
| Metrica | Valore |
|---|---|
| **AVG FF1** | **0.81** |
| **AVG ASF1** | **0.59** |

Risultati buoni su FF1, molto più bassi su ASF1 → il modello classifica bene i singoli frame ma fatica a produrre segmenti temporali puliti.

#### Monastero dei Benedettini
| Metrica | Valore |
|---|---|
| **AVG FF1** | **0.68** |
| **AVG ASF1** | **0.40** |

Risultati più bassi a causa di: presenza di negatives, variazione delle condizioni di luce (dataset raccolto su 3 mesi), blur.

---

## 3. La Nostra Pipeline (Phase 1 e Phase 2 già implementate)

La fase di preprocessing (fase 1 e 2) è già stata completata e i dati sono disponibili sul cluster.

### Phase 1 — Subsampling dei Frame
- Script: `Scripts/Task1_Localization/phase1_subsample_frames.py`
- Riduce i frame da 29.97fps/24fps → **6fps** (fattore ~1/4)
- Itera ricorsivamente su tutte le sottocartelle
- Produce le cartelle `_Small`

### Phase 2 — Estrazione Feature con DINOv2
- Script: `Scripts/Task1_Localization/phase2_extract_features.py`
- Modello: **DINOv2 `dinov2_vits14`** (Vision Transformer S/14)
- Embedding: dimensione **384**
- Caricamento: **offline** da `weights/facebookresearch_dinov2_main` via `TORCH_HOME`
- Iterazione ricorsiva con `rglob('*.jpg')`
- Etichettatura automatica della stanza dal nome della cartella padre (es. `10.1_Sala10_...` → `10`)

### Output (già generato sul cluster)
- **Path sul cluster:** `/home/mssdmn01t05c351v/ProgDL1/data/Task1_Features`
- Formato: file `.pt` per sequenza, con campi:
  - `video_id`: nome della cartella sequenza
  - `features`: tensore `[N, 384]` (N frame × 384 dim DINOv2)
  - `room_labels`: tensore `[N]` con l'ID intero della stanza
  - `frame_ids`: tensore `[N]` con gli ID dei frame originali
- File aggiuntivo: `room_mapping.json` (mappatura nome_stanza → id_intero)

> **Nota:** Sostituisce il VGG-19 degli autori con DINOv2, che produce rappresentazioni molto più ricche grazie al self-supervised learning su dati non etichettati.

---

## 4. Proposte di Modelli Alternativi

L'obiettivo è costruire un modello che, **dato il tensore di feature** estratte da DINOv2 (una sequenza di embedding `[T, 384]`), produca una **classificazione per frame** (e opzionalmente la segmentazione temporale).

Questo è un problema di **classificazione su sequenze temporali**.

Di seguito i candidati valutati:

---

### Modello A — MLP / Linear Probe (baseline minima)
Un semplice classificatore lineare o MLP applicato frame-per-frame, senza modellazione temporale.
- **Pro:** Semplicissimo, rapido, ottimo come lower bound
- **Contro:** Ignora completamente il contesto temporale → basso ASF1
- **Utilità:** Serve come punto di partenza per misurare il guadagno degli altri modelli

---

### Modello B — LSTM / GRU (RNN classiche)
Reti ricorrenti che modellano la sequenza in modo incrementale.
- **Pro:** Ben consolidate, facili da trainare, bassa memoria
- **Contro:** Soffrono di vanishing gradient su sequenze molto lunghe; i video di test durano fino a 50 minuti (molto lunghe anche a 6fps → ~18000 frame)
- **Utilità:** Buon riferimento "classico" pre-Transformer/Mamba

---

### Modello C — Transformer (self-attention)
Encoder Transformer con self-attention sulla dimensione temporale.
- **Pro:** Cattura dipendenze a lungo raggio; eccellente nel video understanding
- **Contro:** Complessità quadratica O(T²) in memoria e computazione — con video da 50 min a 6fps = ~18.000 frame, è **proibitivo senza windowing**
- **Utilità:** Punto di riferimento "moderno" ma computazionalmente costoso

---

### Modello D — **Mamba (SSM — State Space Model)** ⭐ SCELTA PRINCIPALE

Mamba è un'architettura basata su **Selective State Space Models (S6)**, proposta nel 2023 da Gu & Dao, che supera i Transformer su sequenze lunghe con complessità **lineare O(T)**.

#### Perché Mamba è la scelta giusta per questo task

1. **Sequenze lunghissime:** I video di test hanno fino a ~18.000 frame (50 min × 6fps). Transformer → O(18000²) = infeasible. Mamba → O(18000) = fattibile.
2. **Egocentric video:** La letteratura recente (2024-2025) mostra che Mamba supera i Transformer in task di video understanding egocentric:
   - *Mamba-OTR* (2025): specifico per egocentric untrimmed video, supera Transformer-based baseline in accuracy e efficienza
   - *VideoMamba* (ECCV 2024): superiore a 3D CNN e Transformer standard su video recognition
   - *Video Mamba Suite* (2024): framework generale per video tasks con Mamba
3. **Memoria temporale selettiva:** Il meccanismo di "selection" di Mamba permette di ricordare informazioni rilevanti e dimenticare il rumore — fondamentale in un video dove il visitatore è in transizione tra stanze o guarda oggetti irrilevanti.
4. **Stato ricorrente efficiente:** A differenza di LSTM, Mamba aggiorna lo stato in modo **parallelo** durante il training (efficiente come un Transformer) ma si comporta come un **RNN** a inferenza (bassa memoria).
5. **Adatto alla segmentazione temporale:** Il concetto di "stato interno" di Mamba mappa naturalmente sul concetto di "stanza corrente" nella localizzazione.

#### Riferimenti chiave
- *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (Gu & Dao, NeurIPS 2023)
- *VideoMamba: State Space Model for Efficient Video Understanding* (Li et al., ECCV 2024)
- *Mamba-OTR: Egocentric Online Take and Release Detection* (2025)
- *Video Mamba Suite* (2024)

---

### Modello E — Hybrid: proiezione lineare + Mamba
Piccola proiezione MLP sulle feature DINOv2 + Mamba per la modellazione temporale.
- **Utilità:** Più parametri, potenzialmente più espressivo

---

## 5. Confronto Riassuntivo

| Modello | Comp. Temporale | Sequenze Lunghe | Parallelismo Train | Adatto al Task |
|---|---|---|---|---|
| MLP | ❌ Nessuna | ✅ | ✅ | Come baseline |
| LSTM/GRU | ✅ Ricorrente | ⚠️ Limitata | ❌ Lento | Buon riferimento |
| Transformer | ✅ Global attention | ❌ O(T²) | ✅ | Troppo costoso |
| **Mamba** | ✅ SSM selettivo | ✅ O(T) | ✅ | **Ottimale** |

---

## 6. Architettura Proposta (Mamba per Task 1)

```
Input: sequenza di feature DINOv2 → shape [T, 384]
         │
         ▼
  Linear Projection: [T, 384] → [T, d_model]  (es. d_model = 256)
         │
         ▼
  ┌─────────────────────────────┐
  │   N × Mamba Block           │
  │   - SSM (S6)                │
  │   - Normalization (RMSNorm) │
  │   - Skip connections        │
  └─────────────────────────────┘
         │
         ▼
  Classification Head: [T, d_model] → [T, num_classes]
         │
         ▼
  Output: logits per frame → shape [T, num_rooms]
         │
         ▼
  (Opzionale) Post-processing temporale (smoothing / CRF / argmax con finestra)
```

### Iperparametri da Esplorare
| Parametro | Valori da testare |
|---|---|
| `d_model` (dim. proiezione) | 128, 256, 512 |
| `n_layers` (blocchi Mamba) | 2, 4, 6 |
| `d_state` (dim. stato SSM) | 16, 32 |
| `d_conv` (conv interna Mamba) | 4 (default) |
| `dropout` | 0.1, 0.2 |
| `learning_rate` | 1e-3, 5e-4 |

---

## 7. Metriche di Valutazione

Seguiamo le stesse metriche degli autori per un confronto diretto:

- **FF1 (Frame F1):** F1 score per singolo frame (classe-wise, poi media pesata)
- **ASF1 (Action Segment F1):** F1 score su segmenti temporali

### Target da superare
| Dataset | FF1 baseline | ASF1 baseline |
|---|---|---|
| Palazzo Bellomo | 0.81 | 0.59 |
| Monastero dei Benedettini | 0.68 | 0.40 |

---

## 8. Prossimi Passi

- [ ] Implementare il DataLoader che legge i file `.pt` da `/home/mssdmn01t05c351v/ProgDL1/data/Task1_Features`
- [ ] Implementare il modello Mamba (`mamba-ssm` o implementazione manuale se non disponibile)
- [ ] Implementare il training loop con loss `CrossEntropyLoss` per frame
- [ ] Implementare il calcolo di FF1 e ASF1
- [ ] Eseguire esperimenti con MLP e LSTM come baseline di confronto
- [ ] Comparare i risultati con la baseline del paper (VGG19 + KNN)
- [ ] Eventuale post-processing temporale (smoothing con finestra scorrevole)

---

## 9. Note Tecniche sul Cluster

- **Path features estratte:** `/home/mssdmn01t05c351v/ProgDL1/data/Task1_Features`
- **Ambiente:** Apptainer `latest.sif`
- **Variabili di ambiente necessarie:**
  ```bash
  export WANDB_MODE=offline
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```
- **DINOv2 weights (già scaricati):** `weights/facebookresearch_dinov2_main`

---

*Documento generato il 22/07/2026. Aggiornare man mano che si procede con l'implementazione.*
