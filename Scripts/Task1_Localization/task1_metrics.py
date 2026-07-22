"""
Task 1 — Room-based Localization: Metriche di Valutazione

Implementa le due metriche ufficiali del paper EGO-CH:
  - FF1  (Frame F1):          F1 score per singolo frame
  - ASF1 (Action Segment F1): F1 score su segmenti temporali contigui

Queste metriche sono definite nel paper (sezione 4.1) e nel lavoro
precedente degli autori [3, 19]. I valori baseline da battere sono:
  - Palazzo Bellomo:          FF1=0.81, ASF1=0.59
  - Monastero dei Benedettini: FF1=0.68, ASF1=0.40
"""
import numpy as np
from sklearn.metrics import f1_score


# ─────────────────────────────────────────────────────────────────────────────
# FF1 — Frame F1
# ─────────────────────────────────────────────────────────────────────────────

def compute_ff1(predictions: list, ground_truth: list) -> float:
    """
    Calcola il Frame F1 score (FF1) su tutti i frame del test set.

    FF1 è l'F1 score applicato ai singoli frame. Non valuta la coerenza
    temporale dei segmenti (per questo esiste ASF1).

    Args:
        predictions:  lista di array 1D numpy — predizioni per ogni sequenza
        ground_truth: lista di array 1D numpy — label reali per ogni sequenza

    Returns:
        FF1 score (float in [0, 1]), media pesata per numero di frame
    """
    all_preds  = np.concatenate(predictions)
    all_labels = np.concatenate(ground_truth)
    return float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))


# ─────────────────────────────────────────────────────────────────────────────
# ASF1 — Action Segment F1
# ─────────────────────────────────────────────────────────────────────────────

def compute_asf1(
    predictions: list,
    ground_truth: list,
    overlap_threshold: float = 0.5,
) -> float:
    """
    Calcola l'Action Segment F1 score (ASF1) su tutti i video del test set.

    ASF1 misura la capacità del modello di produrre segmenti temporali
    coerenti con il ground truth. Un segmento predetto è considerato
    corretto se:
      1. ha la stessa classe del segmento GT
      2. la sua sovrapposizione temporale (IoU) con il segmento GT è ≥ threshold

    Args:
        predictions:       lista di array 1D numpy — predizioni per ogni video
        ground_truth:      lista di array 1D numpy — label reali per ogni video
        overlap_threshold: soglia IoU temporale (default 0.5 come nel paper)

    Returns:
        ASF1 score medio tra tutti i video (float in [0, 1])
    """
    if len(predictions) == 0:
        return 0.0

    video_f1s = []

    for preds, labels in zip(predictions, ground_truth):
        pred_segments = _extract_segments(preds)
        gt_segments   = _extract_segments(labels)

        tp = 0
        fp = 0
        gt_matched = set()

        for pred_seg in pred_segments:
            matched = False
            for i, gt_seg in enumerate(gt_segments):
                if i in gt_matched:
                    continue
                # Stessa classe e IoU sufficiente
                if pred_seg[2] == gt_seg[2]:
                    iou = _segment_iou(pred_seg[:2], gt_seg[:2])
                    if iou >= overlap_threshold:
                        tp += 1
                        gt_matched.add(i)
                        matched = True
                        break
            if not matched:
                fp += 1

        fn = len(gt_segments) - len(gt_matched)

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        video_f1s.append(f1)

    return float(np.mean(video_f1s))


# ─────────────────────────────────────────────────────────────────────────────
# Utilità interne
# ─────────────────────────────────────────────────────────────────────────────

def _extract_segments(labels: np.ndarray) -> list:
    """
    Converte un array di label frame-by-frame in una lista di segmenti.
    Ogni segmento è una tupla (start_frame, end_frame, class_id).

    Esempio:
        [0,0,0,1,1,2,2,2] → [(0,2,0), (3,4,1), (5,7,2)]
    """
    if len(labels) == 0:
        return []

    segments = []
    start = 0
    current_class = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != current_class:
            segments.append((start, i - 1, int(current_class)))
            start = i
            current_class = labels[i]

    segments.append((start, len(labels) - 1, int(current_class)))
    return segments


def _segment_iou(seg_a: tuple, seg_b: tuple) -> float:
    """
    Calcola l'Intersection over Union temporale tra due segmenti.
    seg_a e seg_b sono tuple (start, end) inclusive.
    """
    intersection_start = max(seg_a[0], seg_b[0])
    intersection_end   = min(seg_a[1], seg_b[1])
    intersection = max(0, intersection_end - intersection_start + 1)
    union = (seg_a[1] - seg_a[0] + 1) + (seg_b[1] - seg_b[0] + 1) - intersection
    return intersection / (union + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Funzione combinata per il reporting
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(predictions: list, ground_truth: list) -> dict:
    """
    Calcola FF1 e ASF1 in un'unica chiamata.

    Returns:
        dict con chiavi "ff1" e "asf1"
    """
    return {
        "ff1":  compute_ff1(predictions, ground_truth),
        "asf1": compute_asf1(predictions, ground_truth),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test rapido
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    print("=== Test Metriche Task 1 ===\n")

    # Caso perfetto
    preds  = [np.array([0, 0, 1, 1, 2, 2])]
    labels = [np.array([0, 0, 1, 1, 2, 2])]
    metrics = compute_all_metrics(preds, labels)
    print(f"Caso perfetto   → FF1={metrics['ff1']:.4f}  ASF1={metrics['asf1']:.4f}")
    assert metrics["ff1"] == 1.0 and metrics["asf1"] == 1.0

    # Caso sbagliato
    preds  = [np.array([1, 1, 1, 1, 1, 1])]
    labels = [np.array([0, 0, 1, 1, 2, 2])]
    metrics = compute_all_metrics(preds, labels)
    print(f"Caso errato     → FF1={metrics['ff1']:.4f}  ASF1={metrics['asf1']:.4f}")

    # Caso realistico con rumore
    np.random.seed(42)
    preds_noisy = [np.random.choice([0, 1, 2], size=100)]
    labels_gt   = [np.array([0]*30 + [1]*40 + [2]*30)]
    metrics = compute_all_metrics(preds_noisy, labels_gt)
    print(f"Caso rumoroso   → FF1={metrics['ff1']:.4f}  ASF1={metrics['asf1']:.4f}")

    print("\n✓ Metriche funzionanti.")
