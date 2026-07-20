---
name: Task3_Object_Retrieval
description: Linee guida e decisioni architetturali per il Task 3 (Object Retrieval)
---

# Skill: Task 3 - Object Retrieval

Questa documentazione definisce la pipeline operativa ufficiale per il Task 3 del progetto EGO-CH, riflettendo le decisioni implementative prese per l'architettura.

## 1. Origine dei Dati
- **Dataset Utilizzati:** Vengono utilizzate esclusivamente le immagini (generalmente crop o patch ritagliate) all'interno della cartella `data/Object_Retrieval`.
- **Nessuna sovrapposizione:** Non vengono lette le cartelle degli altri task. L'obiettivo qui è fare matching visuale tra istanze dello stesso oggetto/quadro (es. da angolazioni o condizioni di luce diverse).

## 2. Fase di Preprocessing (Estrazione Patch Features)
- **Modello Vision Foundation:** Come per il Task 1, viene utilizzato **DINOv2** (`dinov2_vits14`) per generare embedding vettoriali estremamente densi e semanticamente ricchi di dimensione 384.
- **Estrazione per Patch:** Lo script `task3_extract_patch_features.py` passerà a DINOv2 non il frame intero, ma le immagini degli oggetti isolati (crop).
- **Modalità Offline:** Il caricamento di DINOv2 avverrà offline tramite la cartella `weights/` per bypassare i firewall del cluster.

## 3. Modello a Valle e Metriche
- **Nessun Classifier Lineare:** A differenza della Room-based localization, il Task di Retrieval si basa tipicamente sul calcolo della distanza (es. Cosine Similarity) tra gli embedding.
- **Obiettivo:** Dato un query patch, trovare nel database le patch (oggetti) che minimizzano la distanza vettoriale nello spazio latente di DINOv2.
