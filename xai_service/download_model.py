#!/usr/bin/env python3
"""
Simple script to download FinBERT model files directly
"""

import os
import requests
import json
from pathlib import Path

def download_finbert_files():
    """Download FinBERT model files directly"""
    try:
        # Create models directory
        models_dir = Path('/app/shared_data/models')
        models_dir.mkdir(exist_ok=True)
        
        finbert_path = models_dir / 'finbert'
        finbert_path.mkdir(exist_ok=True)
        
        print("Downloading FinBERT model files...")
        
        # Base URL for FinBERT model
        base_url = "https://huggingface.co/ProsusAI/finbert/resolve/main"
        
        # Files to download
        files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt"
        ]
        
        for file in files:
            url = f"{base_url}/{file}"
            file_path = finbert_path / file
            
            print(f"Downloading {file}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded {file}")
        
        print("FinBERT model files downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Download failed: {e}")
        return False

if __name__ == "__main__":
    success = download_finbert_files()
    exit(0 if success else 1) 