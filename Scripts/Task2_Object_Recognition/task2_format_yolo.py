import os
import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Format dataset for YOLOv8")
    parser.add_argument("--source_images", type=str, required=True, help="Path to Bellomo_Small or Monastero_Benedettini_Small")
    parser.add_argument("--source_annotations", type=str, required=True, help="Path to bbox_annotations directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output YOLO dataset directory")
    args = parser.parse_args()

    # Create YOLO directory structure
    os.makedirs(os.path.join(args.output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "labels", "val"), exist_ok=True)

    print(f"Dataset YOLO formatter ready at {args.output_dir}")
    print("TODO: Implement parsing of VIA annotations to YOLO format.")
    
if __name__ == '__main__':
    main()
