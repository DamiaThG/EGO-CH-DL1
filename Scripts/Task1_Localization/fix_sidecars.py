"""
Genera i file sidecar _rooms.json per le sequenze estratte sprovviste.
"""
import json
import torch
from pathlib import Path

test_dir = Path("data/Features/Monastero/Test")

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
    del data

    with open(sidecar, "w") as f:
        json.dump(rooms, f)
    print(f"rooms={rooms}")

print("\nFatto!")

