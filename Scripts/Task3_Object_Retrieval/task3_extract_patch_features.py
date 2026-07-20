import os
import argparse
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def get_dinov2():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches_dir", type=str, required=True, help="Directory containing the cropped patches")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for patch features")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    model = get_dinov2().to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Extraction for Task 3 patches from {args.patches_dir} started.")
    
    # TODO: Implement iteration over patches, extraction, and save to a dictionary/tensors
    
if __name__ == '__main__':
    main()
