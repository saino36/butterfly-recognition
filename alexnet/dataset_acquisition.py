"""
Tahap 1: Dataset Acquisition
---------------
Dataset: Butterfly Classification Dataset (Kaggle)
"""

import os
import kagglehub
import shutil

# ==============================
# Configuration
# ==============================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")

# Create dataset directory if not exists
os.makedirs(DATASET_DIR, exist_ok=True)

# ==============================
# Download Dataset
# ==============================
print("Downloading butterfly dataset from Kaggle...")

downloaded_path = kagglehub.dataset_download(
    "pnkjgpt/butterfly-classification-dataset"
)

print("Downloaded dataset path:", downloaded_path)

# ==============================
# Move Dataset to Project Folder
# ==============================
if not os.listdir(DATASET_DIR):
    shutil.copytree(downloaded_path, DATASET_DIR, dirs_exist_ok=True)
    print("Dataset successfully copied to:", DATASET_DIR)
else:
    print("Dataset already exists in:", DATASET_DIR)

# ==============================
# Dataset Overview
# ==============================
TRAIN_DIR = os.path.join(DATASET_DIR, "Train")

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError("Train directory not found inside dataset/raw/")

class_folders = [
    folder for folder in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, folder))
]

print("\nDataset Summary")
print("----------------------")
print("Total Classes:", len(class_folders))
print("Sample Classes:", class_folders[:5])


