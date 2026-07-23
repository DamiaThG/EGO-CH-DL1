# Analisi Comparativa Task 2: EGO-CH vs YOLOv8

Questa analisi confronta le performance nel **Task 2: Point of Interest/Object Recognition** sul dataset "Monastero dei Benedettini". Il confronto avviene tra il baseline ufficiale stabilito dai creatori del dataset e i due modelli della famiglia YOLOv8 che abbiamo addestrato.

## 1. Tabella Riassuntiva delle Performance

| Metrica / Feature | Baseline Paper (EGO-CH) | Nostro Run 1 (YOLOv8n) | Nostro Run 2 (YOLOv8x) |
| :--- | :--- | :--- | :--- |
| **Architettura Rete** | YOLOv3 (2018) | YOLOv8n (Nano, 2023) | YOLOv8x (X-Large, 2023) |
| **mAP (50-95)** | **15.45%** | **14.82%** | **24.30% 🏆** |
| **mAP (50)** | *Non riportato* | 19.92% | **29.70%** |
| **Parametri** | ~61 Milioni | **~3.3 Milioni** | ~68.3 Milioni |
| **Epoche Addestrate** | *Non riportato* | 100 / 100 | 269 / 300 *(Early Stopping)* |
| **Tempo di Training** | *Sconosciuto* | ~1.6 Ore | ~7.2 Ore |

---

## 2. Analisi Dettagliata dei Modelli

### A. Il Baseline (YOLOv3)
I creatori del dataset EGO-CH hanno utilizzato YOLOv3 come baseline (Tabella 4 del paper). Hanno ottenuto un mAP del **15.45%**. Il punteggio, apparentemente basso per gli standard classici della Computer Vision, è stato giustificato dalla natura brutale dei video egocentrici: 
- Motion blur estremo dovuto ai movimenti della testa.
- Punti di interesse (POI) che non sono sempre "oggetti" definiti, ma pavimenti, scale o porzioni di stanze.
- Ripetizioni visive ed elementi architettonici ambigui.

### B. Il Modello Nano (YOLOv8n)
L'esperimento con la rete Nano ha avuto un esito tecnicamente impressionante:
- **Efficienza Estrema**: Pur avendo quasi **20 volte meno parametri** rispetto al YOLOv3 originale (3.3M contro 61M), la versione v8n ha sfiorato la parità assoluta (14.82% contro 15.45%).
- **Velocità**: Essendo una rete leggerissima, è perfetta per l'uso in tempo reale su dispositivi edge (es. smartphone o visori HoloLens reali) senza scaricare la batteria.

### C. Il Modello X-Large (YOLOv8x)
Il modello definitivo (68M di parametri, peso paragonabile a YOLOv3) ha stracciato completamente il baseline documentato:
- **Vittoria Netta**: L'mAP è saltato a **24.30%**, segnando un **+57% di incremento relativo** rispetto allo stato dell'arte del paper.
- **Precisioni Perfette su alcune classi**: A differenza della rete Nano, che faceva fatica su molte classi intermedie, la rete X-Large ha raggiunto la quasi perfezione su classi ben rappresentate. Ad esempio:
  - *Class_16*: mAP 95.2%
  - *Class_15*: mAP 82.4%
  - *Class_57*: mAP 64.1%
- **Risoluzione della Complessità**: L'architettura anchor-free e la potentissima estrazione delle feature di YOLOv8x hanno permesso al modello di risolvere l'ambiguità visiva e il motion-blur molto meglio del suo antenato v3.
- **Early Stopping Funzionale**: L'addestramento si è fermato da solo all'epoca 269, dimostrando che il modello aveva appreso tutto il possibile dal dataset entro l'epoca 169 senza andare in overfitting distruttivo.

---

> [!IMPORTANT]
> **Conclusioni**
> Se il tuo obiettivo primario è l'accuratezza pura per una pubblicazione o una valutazione, **YOLOv8x è il nuovo benchmark di riferimento**. Se l'obiettivo futuro prevede l'installazione su un visore indossabile con poca potenza di calcolo, **YOLOv8n** offre un compromesso miracoloso offrendo la stessa precisione del paper originale, ma girando a una velocità vertiginosa in tempo reale.
