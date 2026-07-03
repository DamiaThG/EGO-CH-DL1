# Skill: Pipeline Ibrida di Preprocessing Video e Estrazione Feature (EGO-CH)

Questa documentazione definisce la pipeline operativa ufficiale in due fasi per la preparazione dei dati del progetto EGO-CH. Seguire rigorosamente questo documento per garantire la sincronizzazione tra annotazioni e dati visivi.

## Visione Generale dell'Infrastruttura

La pipeline risolve il problema dei dataset disomogenei (video originali a framerate diversi) e riduce drasticamente il carico di rete e computazionale, dividendo il lavoro in due fasi distinte:

1.  *Fase 1 (In Locale):* Sottocampionamento dinamico per allineare temporalmente i video a un Framerate Fisso (Target FPS) di $6 \text{ fps}$, preservando la mappa delle annotazioni originali, seguito da compressione.
2.  *Fase 2 (Sul Cluster HPC):* Estrazione pura delle feature spaziali dai frame campionati tramite backbone pre-addestrata (es. ConvNeXt), generando i tensori finali pre-computati.

---

## Fase 1: Elaborazione in Locale (Allineamento Temporale)

Lo scopo di questo script Python da lanciare sulle workstation locali è uniformare lo scorrere del tempo per tutti i video e ridurre il peso (in GB) prima del caricamento sul cluster.

### 1.1 Configurazione Target
*   *Target FPS:* $6 \text{ fps}$ (valore vincolante per garantire coerenza temporale al modello a valle).
*   *Formato Salvataggio:* JPEG compresso (es. qualità 85).
*   *Resize Spaziale (Soft-resize):* Mantenere l'aspetto originario (Aspect Ratio 16:9). Ridurre l'asse più lungo (solitamente la larghezza) a un massimo di $1280 \text{ pixel}$ per evitare file inutilmente pesanti, senza compromettere la precisione necessaria per successivi ritagli dei POI (Punti di Interesse). *Non forzare il resize a $224 \times 224$ in questa fase.*

### 1.2 Algoritmo di Sottocampionamento Dinamico
Non deve essere utilizzato un intervallo fisso (es. "prendi 1 frame ogni 5"). Lo script deve leggere dinamicamente il framerate nativo di ciascun video (⁠ cv2.CAP_PROP_FPS ⁠) e calcolare l'intervallo teorico per quel specifico video:

$$ \text{Intervallo} = \frac{\text{FPS Originali (es. 29.97 o 24)}}{\text{Target FPS (6.0)}} $$

Il codice manterrà un contatore a virgola mobile (⁠ next_target_frame ⁠) e salverà il frame solo quando l'indice corrente (intero) supera o eguaglia il target calcolato.

### 1.3 Regola Aurea del Nomenclatore (Critico per le Annotazioni)
I frame estratti *DEVONO* essere nominati conservando rigorosamente il loro *Indice (ID) di frame originale* all'interno del video.
*   Errato: ⁠ video_01_frame_1.jpg ⁠, ⁠ video_01_frame_2.jpg ⁠ (Il subsampling distrugge l'indice, rompendo il legame col file CSV).
*   Corretto: ⁠ frame_000000.jpg ⁠, ⁠ frame_000005.jpg ⁠, ⁠ frame_000010.jpg ⁠ (L'ID nel nome corrisponde esattamente all'ID originale nel file CSV annotato).

---

## Fase 2: Estrazione Feature sul Cluster HPC

Questa fase prenderà in input l'output della Fase 1 (i JPEG alleggeriti) caricato sul cluster e genererà i tensori finali ⁠ .pt ⁠ (PyTorch).

### 2.1 Configurazione della Backbone
*   *Modello (Esempio):* ⁠ convnext_tiny ⁠ (da ⁠ torchvision.models ⁠).
*   *Pesi:* Pre-addestrati su ImageNet-1K (⁠ ConvNeXt_Tiny_Weights.IMAGENET1K_V1 ⁠).
*   *Modifica Topologica:* Rimuovere il classificatore finale (⁠ nn.Linear ⁠ di output). In ConvNeXt, modificare l'attributo ⁠ classifier ⁠ per fermarsi prima del layer lineare, in modo che restituisca l'embedding puro di dimensione 768.
*   *Stato:* Modalità ⁠ eval() ⁠, gradienti disattivati (⁠ requires_grad=False ⁠).

### 2.2 Sincronizzazione e Annotazioni
Il codice dovrà iterare sui file JPEG del dataset filtrato (Fase 1). Per ogni file:
1.  Leggere l'immagine (es. ⁠ frame_000005.jpg ⁠).
2.  Estrarre l'ID del frame dal nome del file (es. l'intero ⁠ 5 ⁠).
3.  Consultare il file CSV delle annotazioni corrispondente al video in elaborazione.
4.  Cercare la riga dove ⁠ frame_id == 5 ⁠ o il timestamp equivalente, per recuperare le label (Ambiente, POI, Bounding Box originali, da ri-scalare in base al soft-resize di Fase 1).

### 2.3 Trasformazioni Spaziali e Inferenza Offline
1.  Applicare il crop/resize finale richiesto dalla backbone ($224 \times 224$ o simili).
2.  Passare il tensore dell'immagine (forma ⁠ [1, 3, 224, 224] ⁠) alla backbone.
3.  Ottenere l'embedding $Z$ di forma ⁠ [1, 768] ⁠.
4.  Rimuovere la dimensione batch: ⁠ Z = Z.squeeze(0) ⁠ ottenendo un vettore 1D.

### 2.4 Output (Pacchettizzazione)
Lo script genererà un singolo file di output (es. ⁠ video_01_features.pt ⁠) contenente un dizionario PyTorch con le liste di dati sincronizzate:

```python
{
    "video_id": "video_01",
    "features": Tensor_2D,        # Forma: [N_Frame_Estratti, 768]
    "room_labels": Tensor_1D,     # Forma: [N_Frame_Estratti] (ID Ambienti)
    "poi_labels": Tensor_1D,      # Forma: [N_Frame_Estratti] (ID POI)
    "bboxes": Tensor_2D,          # Forma: [N_Frame_Estratti, 4] (BBox ricalcolate)
    "frame_ids": Tensor_1D        # Forma: [N_Frame_Estratti] (Gli ID originali, per debug)
}