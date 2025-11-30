from pathlib import Path

# Project root is the folder where this config.py lives
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_CSV = PROJECT_ROOT / "data" / "dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "trick_classifier.joblib"

# Trick labels (must match subfolder names in data/raw)
TRICKS = ["pop_shuvit", "kickflip"]
