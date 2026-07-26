"""
Script temporaneo — genera i file sidecar _rooms.json per le sequenze
già estratte che non ce l'hanno (create prima dell'introduzione del resume).

Uso: python fix_sidecars.py
Da lanciare DENTRO il container, dalla root del progetto.
"""
import json
import torch
from pathlib import Path

test_dir = Path("/home/mssdmn01t05c351v/ProgDL1/data/Features/Monastero/Test")

pt_files = sorted(test_dir.glob("*_features.pt"))
print(f"Trovati {len(pt_files)} file .pt in {test_dir}\n")

for pt_file in pt_files:
    video_id = pt_file.stem.replace("_features", "")
    sidecar = test_dir / f"{video_id}_rooms.json"

    if sidecar.exists():
        print(f"  [SKIP] {video_id} — sidecar già presente")
        continue

    print(f"  [GEN]  {video_id}...", end=" ", flush=True)
    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    rooms = sorted(data["room_labels"].unique().tolist())
    del data  # libera subito la RAM prima di passare al prossimo

    with open(sidecar, "w") as f:
        json.dump(rooms, f)
    print(f"rooms={rooms}")

print("\nFatto! Ora puoi rilanciare l'estrazione normalmente.")
