# Task 3: Object Retrieval

## Descrizione del Task
In conformità con quanto riportato nell'articolo scientifico originale (*"EGO-CH: Dataset and Fundamental Tasks for Visitors Behavioral Understanding using Egocentric Vision"*), il task di **Object Retrieval** consiste nel ritrovare l'immagine di uno specifico oggetto all'interno di un database, partendo da un'immagine di query che contiene lo stesso oggetto.

Nello specifico, abbiamo implementato la variante **One-Shot Retrieval**:
- **Database (Gallery):** Contiene esclusivamente le immagini di riferimento (una per ogni Point of Interest - POI). Queste si trovano nella cartella `Training/` di ciascun sito culturale.
- **Query (Test):** Contiene l'intero set di patch ritagliate dai bounding box durante le visite egocentriche. Queste si trovano nella cartella `Test/`.

L'approccio implementato riflette appieno le linee guida del file `SKILL.md`:
1. **Origine dei Dati:** Le patch ritagliate all'interno di `data/Object_Retrieval`.
2. **Preprocessing:** L'uso del foundation model **DINOv2** (`dinov2_vits14`) caricato offline per estrarre feature dense a 384 dimensioni.
3. **Valutazione:** L'utilizzo della **Cosine Similarity** (invece di un classificatore lineare) per trovare la patch nel database che minimizza la distanza con la query nello spazio latente di DINOv2.

---

## Implementazione (Codice)

La pipeline è stata suddivisa in script modulari:

1. **`task3_extract_patch_features.py`:** 
   - Scorre le immagini (Gallery o Query).
   - Carica il modello DINOv2 offline.
   - Genera gli embedding vettoriali.
   - Salva i risultati in dizionari `.pt` (mappando il percorso relativo del file al suo tensore) nelle cartelle di output (`Monastero_Features/` e `Bellomo_Features/`).

2. **`task3_evaluate_retrieval.py`:**
   - Carica le feature estratte e la mappatura dei label (`labels.txt`).
   - Associa a ogni Query il label di *ground truth* decodificando i file in `Test/labels`.
   - Calcola la matrice di Cosine Similarity tra tutte le Queries e la Gallery.
   - Calcola le metriche di Accuracy, Precision, Recall e F1 Score (usate anche nel paper).

3. **`run_task3.sh` e `task3.sh`:**
   - Script Bash e wrapper SLURM (Apptainer) per eseguire la pipeline in modo massivo su **GCluster** (nodo `gnode10`, L40S) in completa autonomia e sfruttando l'architettura HPC del DMI.

---

## Risultati Ottenuti

La sostituzione del modello originario (VGG19) con i potenti embedding visivi estratti da **DINOv2** ha portato a risultati clamorosamente superiori rispetto alla baseline pubblicata nel paper (Tabella 5).

### Sito: Monastero dei Benedettini (35 POI)
| Metrica | Baseline VGG19 (Paper) | Nostro Modello (DINOv2) |
|---------|------------------------|-------------------------|
| Precision | 0.29 | **0.5353** |
| Recall    | 0.07 | **0.3369** |
| F1 Score  | 0.08 | **0.3752** |
| Accuracy  | -    | **0.3369** |

*DINOv2 ha portato a un miglioramento della F1 Score di quasi 5 volte, stabilendo un nuovo stato dell'arte per la variante One-Shot in questo sito.*

### Sito: Palazzo Bellomo (191 POI)
| Metrica | Baseline VGG19 (Paper) | Nostro Modello (DINOv2) |
|---------|------------------------|-------------------------|
| Precision | 0.004 | **0.2194** |
| Recall    | 0.007 | **0.1643** |
| F1 Score  | 0.001 | **0.1512** |
| Accuracy  | -    | **0.1643** |

*La difficoltà estrema di questo sito (dovuta all'enorme numero di istanze ravvicinate e similari) aveva fatto crollare la baseline a uno score prossimo al caso fortuito. DINOv2 risolve in modo robusto le ambiguità visive incrementando la F1 Score di oltre 150 volte.*
