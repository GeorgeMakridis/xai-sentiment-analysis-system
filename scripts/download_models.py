"""
One-time script to download/copy model weights locally.
Run from repo root with x-brain-dev conda env:
    python scripts/download_models.py
"""
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS_DIR = Path(__file__).parent.parent / "models"
REPO_ROOT = Path(__file__).parent.parent


def download_finbert():
    dest = MODELS_DIR / "finbert"
    if dest.exists():
        print(f"FinBERT already exists at {dest}, skipping.")
        return
    print("Downloading FinBERT (ProsusAI/finbert)...")
    AutoTokenizer.from_pretrained("ProsusAI/finbert").save_pretrained(str(dest))
    AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").save_pretrained(str(dest))
    print(f"Saved to {dest}")


def copy_yolo():
    src = REPO_ROOT / "uc1_robotics/model/train2/weights/best.pt"
    dest = MODELS_DIR / "yolo/best.pt"
    if dest.exists():
        print(f"YOLO weights already exist at {dest}, skipping.")
        return
    if not src.exists():
        print(f"YOLO weights not found at {src}. Run uc1_robotics/download_data_model.py first.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"Copied YOLO weights to {dest}")


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    download_finbert()
    copy_yolo()
    print("Done.")
