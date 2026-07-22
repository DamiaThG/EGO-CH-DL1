import os
import argparse
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def get_backbone(project_root):
    weights_dir = os.path.join(project_root, "weights")
    os.environ['TORCH_HOME'] = weights_dir
    torch.hub.set_dir(weights_dir)
    
    hub_repo_dir = os.path.join(weights_dir, "facebookresearch_dinov2_main")
    
    if os.path.exists(hub_repo_dir):
        print(f"Caricamento DINOv2 in modalità OFFLINE da {hub_repo_dir}")
        model = torch.hub.load(hub_repo_dir, 'dinov2_vits14', source='local')
    else:
        print(f"Caricamento DINOv2 in modalità ONLINE (scaricamento in {weights_dir})")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches_dir", type=str, required=True, help="Directory containing the cropped patches")
    parser.add_argument("--output_file", type=str, required=True, help="Output .pt file for patch features (dictionary path->tensor)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Utilizzando il device: {device}")
    
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    model = get_backbone(project_root).to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Extraction for Task 3 patches from {args.patches_dir} started.")
    
    patches_dir = Path(args.patches_dir)
    image_files = list(patches_dir.rglob('*.jpg')) + list(patches_dir.rglob('*.png'))
    
    if not image_files:
        print(f"Nessuna immagine trovata in {args.patches_dir}")
        return
        
    features_dict = {}
    
    for img_path in tqdm(image_files, desc="Estrazione features"):
        try:
            img = Image.open(img_path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Errore immagine {img_path}: {e}")
            continue
            
        with torch.no_grad():
            feat = model(img_t).squeeze(0).cpu()
            
        rel_path = str(img_path.relative_to(patches_dir))
        features_dict[rel_path] = feat

    torch.save(features_dict, args.output_file)
    print(f"Salvate {len(features_dict)} features in {args.output_file}")
    
if __name__ == '__main__':
    main()
