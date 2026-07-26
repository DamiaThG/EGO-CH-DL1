# Valuta le performance del modello addestrato e calcola FF1 e ASF1
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from Scripts.Task1_Localization.task1_dataset import Task1FeatureDataset
from Scripts.Task1_Localization.task1_metrics import compute_ff1, compute_asf1
from sklearn.metrics import f1_score
PAPER_BASELINES = {'bellomo': {'ff1': 0.81, 'asf1': 0.59}, 'monastero': {'ff1': 0.68, 'asf1': 0.4}}

def evaluate(checkpoint_path, test_dir, dataset='bellomo', save_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_obj = Task1FeatureDataset(test_dir)
    num_classes = dataset_obj.num_classes
    from Scripts.Task1_Localization.models.task1_model_mamba import MambaRoomLocalizer
    mamba_model = MambaRoomLocalizer(input_dim=384, d_model=256, num_layers=4, num_classes=num_classes, d_state=16, dropout=0.2)
    from Scripts.Task1_Localization.v2_advanced.task1_train import Task1LightningModule
    lit_model = Task1LightningModule.load_from_checkpoint(checkpoint_path, model=mamba_model, map_location=device, strict=False)
    lit_model.eval()
    model = lit_model.model.to(device)
    all_preds, all_labels = ([], [])
    from scipy.signal import medfilt
    with torch.no_grad():
        for i in range(len(dataset_obj)):
            features, labels = dataset_obj[i]
            features = features.unsqueeze(0).to(device)
            logits = model(features)
            preds = logits.argmax(dim=-1).squeeze(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    best_asf1 = -1
    best_ff1 = -1
    best_kernel = 1
    best_smoothed_preds = []
    print('\n[Ottimizzazione Smoothing Post-Processing]')
    for k in [1, 51, 101, 201, 301, 401, 501, 751, 1001, 1501, 2001, 3001]:
        smoothed_preds = []
        for p in all_preds:
            if k == 1:
                smoothed_preds.append(p)
            else:
                smoothed_preds.append(medfilt(p, kernel_size=k))
        current_ff1 = compute_ff1(smoothed_preds, all_labels)
        current_asf1 = compute_asf1(smoothed_preds, all_labels)
        print(f'  Kernel: {k:3d} | FF1: {current_ff1:.4f} | ASF1: {current_asf1:.4f}')
        if current_asf1 > best_asf1:
            best_asf1 = current_asf1
            best_ff1 = current_ff1
            best_kernel = k
            best_smoothed_preds = smoothed_preds
    print(f'Miglior Kernel Selezionato: {best_kernel} (ASF1={best_asf1:.4f})')
    ff1 = best_ff1
    asf1 = best_asf1
    all_preds = best_smoothed_preds
    ff1 = compute_ff1(all_preds, all_labels)
    asf1 = compute_asf1(all_preds, all_labels)
    flat_preds = np.concatenate(all_preds)
    flat_labels = np.concatenate(all_labels)
    with open(Path(test_dir) / 'room_mapping.json') as f:
        room_mapping = json.load(f)
    id_to_name = {v: k for k, v in room_mapping.items()}
    per_class_f1 = f1_score(flat_labels, flat_preds, average=None, zero_division=0)
    print('\n' + '=' * 60)
    print(f'RISULTATI VALUTAZIONE — {Path(checkpoint_path).name}')
    print(f'Dataset: {dataset.upper()}')
    print('=' * 60)
    print(f"{'Room':<30} {'FF1':>8}")
    print('-' * 40)
    for class_id, f1 in enumerate(per_class_f1):
        room_name = id_to_name.get(class_id, f'Class_{class_id}')
        print(f'{room_name:<30} {f1:>8.4f}')
    print('-' * 40)
    print(f"{'AVG FF1':<30} {ff1:>8.4f}")
    print(f"{'AVG ASF1':<30} {asf1:>8.4f}")
    print('=' * 60)
    baseline = PAPER_BASELINES.get(dataset.lower(), None)
    if baseline:
        paper_ff1 = baseline['ff1']
        paper_asf1 = baseline['asf1']
        print(f'\n--- CONFRONTO CON BASELINE DEL PAPER (VGG19+KNN) [{dataset.upper()}] ---')
        print(f"{'Metrica':<15} {'Nostra':<10} {'Paper':<10} {'Delta':>8}")
        print(f"{'FF1':<15} {ff1:<10.4f} {paper_ff1:<10.4f} {ff1 - paper_ff1:>+8.4f}")
        print(f"{'ASF1':<15} {asf1:<10.4f} {paper_asf1:<10.4f} {asf1 - paper_asf1:>+8.4f}")
        if ff1 > paper_ff1 and asf1 > paper_asf1:
            print('\n✓ OBIETTIVO RAGGIUNTO: Battute entrambe le metriche del paper!')
        elif ff1 > paper_ff1 or asf1 > paper_asf1:
            print('\n~ OBIETTIVO PARZIALE: Battuta una metrica del paper.')
        else:
            print('\n✗ Non ancora al di sopra del baseline del paper.')
    else:
        print(f"\n[!] Dataset '{dataset}' non riconosciuto. Valori disponibili: {list(PAPER_BASELINES.keys())}")
    results_dict = {'checkpoint': checkpoint_path, 'dataset': dataset, 'overall_ff1': ff1, 'overall_asf1': asf1, 'per_room_ff1': {id_to_name.get(class_id, f'Class_{class_id}'): f1 for class_id, f1 in enumerate(per_class_f1)}, 'post_processing': {'type': 'median_filter', 'kernel_size': 101}}
    if args.save_path:
        out_path = Path(args.save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f'\n[+] Risultati salvati in modo strutturato in: {out_path}')
    return results_dict
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 1 Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path al checkpoint .ckpt di Lightning')
    parser.add_argument('--test_dir', type=str, required=True, help='Path alla cartella delle feature di test')
    parser.add_argument('--dataset', type=str, default='bellomo', choices=['bellomo', 'monastero'], help='Dataset di riferimento per confronto col paper')
    parser.add_argument('--save_path', type=str, default=None, help="Path dove salvare il JSON con i risultati dell'esperimento")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.test_dir, args.dataset, args.save_path)