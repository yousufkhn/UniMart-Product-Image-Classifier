from PIL import Image
import os
from tqdm import tqdm

base_dir = "dataset"
size = (224, 224)

for category in os.listdir(base_dir):
    folder = os.path.join(base_dir, category)
    if os.path.isdir(folder):
        for file in tqdm(os.listdir(folder), desc=f"Processing {category}"):
            file_path = os.path.join(folder, file)
            try:
                img = Image.open(file_path).convert("RGB")
                img = img.resize(size)
                img.save(file_path)
            except:
                os.remove(file_path)  # delete bad/corrupt files
