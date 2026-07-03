import os
import re
from pathlib import Path
from PIL import Image

def process_dataset(input_dir, output_dir, original_fps=24.0, target_fps=6.0, quality=85):
    interval = original_fps / target_fps
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Trova tutte le immagini .jpg in tutte le sottocartelle
    # Assumiamo che i frame siano salvati come .jpg
    image_files = list(input_path.rglob('*.jpg'))
    
    # Raggruppa le immagini per cartella (ogni cartella è una sequenza video)
    sequences = {}
    for img_file in image_files:
        parent_dir = img_file.parent
        if parent_dir not in sequences:
            sequences[parent_dir] = []
        sequences[parent_dir].append(img_file)
        
    print(f"Trovate {len(sequences)} sequenze (cartelle) da elaborare.")
    
    for seq_dir, frames in sequences.items():
        # Ordina i frame usando il numero nel nome file (es. frame000005.jpg -> 5)
        def get_frame_id(f):
            match = re.search(r'frame(\d+)\.jpg', f.name)
            return int(match.group(1)) if match else -1
            
        # Filtra i file che non seguono il pattern
        valid_frames = [f for f in frames if get_frame_id(f) != -1]
        valid_frames.sort(key=get_frame_id)
        
        if not valid_frames:
            continue
            
        print(f"Elaborazione: {seq_dir} ({len(valid_frames)} frame trovati)")
        
        # Determiniamo il primo frame ID. Impostiamo il next_target_frame.
        first_frame_id = get_frame_id(valid_frames[0])
        next_target_frame = float(first_frame_id)
        
        # Creiamo la struttura della cartella di output speculare
        relative_dir = seq_dir.relative_to(input_path)
        out_seq_dir = output_path / relative_dir
        out_seq_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        for frame_file in valid_frames:
            current_id = get_frame_id(frame_file)
            
            if current_id >= next_target_frame:
                # Salva il frame
                out_file_path = out_seq_dir / frame_file.name
                
                try:
                    with Image.open(frame_file) as img:
                        # Assicuriamoci che sia RGB per salvare in JPEG
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(out_file_path, "JPEG", quality=quality)
                    saved_count += 1
                except Exception as e:
                    print(f"Errore nel salvataggio di {frame_file}: {e}")
                
                # Avanza il target
                next_target_frame += interval
                
        print(f"  -> Salvati {saved_count} frame.")

if __name__ == "__main__":
    import sys
    
    input_directory = "Dataset_Monastero"
    output_directory = "Dataset_Monastero_Small"
    
    if not os.path.exists(input_directory):
        print(f"Errore: La directory di input {input_directory} non esiste!")
        sys.exit(1)
        
    print(f"Inizio elaborazione. FPS originali: 24.0, FPS Target: 6.0")
    process_dataset(input_directory, output_directory, original_fps=24.0, target_fps=6.0, quality=85)
    print("Elaborazione completata!")
