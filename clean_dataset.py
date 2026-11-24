# clean_dataset.py
import os
from PIL import Image, UnidentifiedImageError

DATA_DIR = "Data/train"
TEST_DIR = "Data/test"

def clean_folder(folder):
    removed = 0
    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            # Skip mask files
            if "mask" in f.lower():
                os.remove(path)
                removed += 1
                print(f"Deleted mask file: {path}")
                continue
            # Skip corrupted images
            try:
                Image.open(path).convert("RGB")
            except UnidentifiedImageError:
                os.remove(path)
                removed += 1
                print(f"Deleted corrupted file: {path}")
    return removed

print("Cleaning dataset...")
n1 = clean_folder(DATA_DIR)
n2 = clean_folder(TEST_DIR)
print(f"Done. Deleted {n1+n2} bad files (mask/corrupted).")
