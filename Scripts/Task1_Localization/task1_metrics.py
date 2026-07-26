# Implementazione delle metriche custom (Frame-level F1 e Action Segment F1)
import numpy as np
from sklearn.metrics import f1_score

def compute_ff1(predictions: list, ground_truth: list) -> float:
    all_preds = np.concatenate(predictions)
    all_labels = np.concatenate(ground_truth)
    return float(f1_score(all_labels, all_preds, average='weighted', zero_division=0))

def compute_asf1(predictions: list, ground_truth: list, overlap_threshold: float=0.5) -> float:
    if len(predictions) == 0:
        return 0.0
    video_f1s = []
    for preds, labels in zip(predictions, ground_truth):
        pred_segments = _extract_segments(preds)
        gt_segments = _extract_segments(labels)
        tp = 0
        fp = 0
        gt_matched = set()
        for pred_seg in pred_segments:
            matched = False
            for i, gt_seg in enumerate(gt_segments):
                if i in gt_matched:
                    continue
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
        precision = tp / (tp + fp + 1e-08)
        recall = tp / (tp + fn + 1e-08)
        f1 = 2 * precision * recall / (precision + recall + 1e-08)
        video_f1s.append(f1)
    return float(np.mean(video_f1s))

def _extract_segments(labels: np.ndarray) -> list:
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
    intersection_start = max(seg_a[0], seg_b[0])
    intersection_end = min(seg_a[1], seg_b[1])
    intersection = max(0, intersection_end - intersection_start + 1)
    union = seg_a[1] - seg_a[0] + 1 + (seg_b[1] - seg_b[0] + 1) - intersection
    return intersection / (union + 1e-08)

def compute_all_metrics(predictions: list, ground_truth: list) -> dict:
    return {'ff1': compute_ff1(predictions, ground_truth), 'asf1': compute_asf1(predictions, ground_truth)}
if __name__ == '__main__':
    import numpy as np
    print('=== Test Metriche Task 1 ===\n')
    preds = [np.array([0, 0, 1, 1, 2, 2])]
    labels = [np.array([0, 0, 1, 1, 2, 2])]
    metrics = compute_all_metrics(preds, labels)
    print(f"Caso perfetto   → FF1={metrics['ff1']:.4f}  ASF1={metrics['asf1']:.4f}")
    assert metrics['ff1'] == 1.0 and metrics['asf1'] == 1.0
    preds = [np.array([1, 1, 1, 1, 1, 1])]
    labels = [np.array([0, 0, 1, 1, 2, 2])]
    metrics = compute_all_metrics(preds, labels)
    print(f"Caso errato     → FF1={metrics['ff1']:.4f}  ASF1={metrics['asf1']:.4f}")
    np.random.seed(42)
    preds_noisy = [np.random.choice([0, 1, 2], size=100)]
    labels_gt = [np.array([0] * 30 + [1] * 40 + [2] * 30)]
    metrics = compute_all_metrics(preds_noisy, labels_gt)
    print(f"Caso rumoroso   → FF1={metrics['ff1']:.4f}  ASF1={metrics['asf1']:.4f}")
    print('\n✓ Metriche funzionanti.')