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


def evaluate(checkpoint_path, test_dir, model_type="mamba"):
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
    
    # Questi valori andrebbero parametrizzati per Bellomo o Monastero
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
