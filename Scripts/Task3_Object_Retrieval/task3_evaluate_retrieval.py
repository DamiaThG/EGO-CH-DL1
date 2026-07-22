import argparse
import torch
import json
import os
import re
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def load_labels_mapping(labels_file):
    mapping = {}
    with open(labels_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]: # Skip header "Name\tLabel"
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                label = int(parts[1])
                mapping[name] = label
    return mapping

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery_features", type=str, required=True, help="Path to gallery features .pt (from Training)")
    parser.add_argument("--query_features", type=str, required=True, help="Path to query features .pt (from Test)")
    parser.add_argument("--labels_file", type=str, required=True, help="Path to labels.txt (maps POI to class id)")
    parser.add_argument("--query_labels_dir", type=str, required=True, help="Path to Test/labels directory")
    args = parser.parse_args()

    # Load mappings
    poi_to_label = load_labels_mapping(args.labels_file)
    
    # Load features
    print("Loading gallery features...")
    gallery_data = torch.load(args.gallery_features)
    print("Loading query features...")
    query_data = torch.load(args.query_features)
    
    # Process Gallery
    gallery_feats = []
    gallery_labels = []
    for rel_path, feat in gallery_data.items():
        # filename like "5.1_Porta Aula Santo Mazzarino.jpg" or "1288.150.jpg"
        filename = os.path.basename(rel_path)
        if '_' in filename:
            poi_id = filename.split('_')[0]
        else:
            poi_id = os.path.splitext(filename)[0]
        if poi_id in poi_to_label:
            gallery_feats.append(feat.numpy())
            gallery_labels.append(poi_to_label[poi_id])
        else:
            print(f"Warning: POI {poi_id} not found in labels mapping for {filename}")
            
    gallery_feats = np.array(gallery_feats)
    gallery_labels = np.array(gallery_labels)
    
    # Process Queries
    # query_data has keys like "100/000009_0.jpg"
    visits = {}
    for rel_path, feat in query_data.items():
        parts = Path(rel_path).parts
        if len(parts) < 2: continue
        visit_id = parts[0]
        filename = parts[1]
        if visit_id not in visits:
            visits[visit_id] = []
        visits[visit_id].append((filename, feat))
        
    query_feats = []
    query_true_labels = []
    
    print("Processing queries and ground truth...")
    for visit_id, items in visits.items():
        # Sort items naturally to match the labels file
        items.sort(key=lambda x: natural_sort_key(x[0]))
        
        label_file = os.path.join(args.query_labels_dir, f"{visit_id}.txt")
        if not os.path.exists(label_file):
            print(f"Warning: Label file {label_file} not found. Skipping visit {visit_id}.")
            continue
            
        with open(label_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        if len(lines) != len(items):
            print(f"Warning: Visit {visit_id} has {len(items)} images but {len(lines)} labels. Some might be missing or failed to extract.")
            
        for i, (filename, feat) in enumerate(items):
            if i < len(lines):
                true_label = int(lines[i])
                # We skip negative class if there's any? Actually in Monastero negative frames are present but retrieval is only on POIs?
                # Actually, the paper says the Test folder contains extracted image patches from bounding boxes, so these are all POIs!
                # Wait, what if the true_label is negative? Let's just keep them all.
                query_feats.append(feat.numpy())
                query_true_labels.append(true_label)
                
    query_feats = np.array(query_feats)
    query_true_labels = np.array(query_true_labels)
    
    print(f"Gallery size: {len(gallery_feats)}, Query size: {len(query_feats)}")
    
    if len(query_feats) == 0 or len(gallery_feats) == 0:
        print("Not enough data to evaluate.")
        return
        
    print("Computing Cosine Similarities...")
    # similarity matrix: [num_queries, num_gallery]
    sim_matrix = cosine_similarity(query_feats, gallery_feats)
    
    # Find best match in gallery for each query
    best_match_indices = np.argmax(sim_matrix, axis=1)
    predicted_labels = gallery_labels[best_match_indices]
    
    # Evaluation
    acc = accuracy_score(query_true_labels, predicted_labels)
    # The paper evaluates Precision, Recall, F1. 
    # Use macro or weighted? The table 5 doesn't specify, but often macro is used for such tasks. Let's compute both.
    precision = precision_score(query_true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(query_true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(query_true_labels, predicted_labels, average='weighted', zero_division=0)
    
    print("\n--- Results (One-Shot Retrieval) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f} (Weighted)")
    print(f"Recall:    {recall:.4f} (Weighted)")
    print(f"F1 Score:  {f1:.4f} (Weighted)")

if __name__ == '__main__':
    main()
