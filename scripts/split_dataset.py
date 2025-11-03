import os
import shutil
import random

# Set your paths
DATASET_DIR = "dataset"
OUTPUT_DIR = "data_split"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_category(category_path, category_name):
    images = os.listdir(category_path)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split_name, file_list in splits.items():
        split_folder = os.path.join(OUTPUT_DIR, split_name, category_name)
        create_dir(split_folder)
        for img_file in file_list:
            src = os.path.join(category_path, img_file)
            dst = os.path.join(split_folder, img_file)
            shutil.copy2(src, dst)

    print(f"{category_name}: {n_total} → train: {n_train}, val: {n_val}, test: {len(splits['test'])}")

def main():
    random.seed(42)
    categories = os.listdir(DATASET_DIR)
    for category in categories:
        category_path = os.path.join(DATASET_DIR, category)
        if os.path.isdir(category_path):
            split_category(category_path, category)

    print("\n✅ Dataset successfully split and copied to 'data_split/' folder.")

if __name__ == "__main__":
    main()
