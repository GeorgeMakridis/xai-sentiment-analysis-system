#!/usr/bin/env python3
"""
Script to download FinBERT model with proper error handling
"""

import os
import sys
import requests
from pathlib import Path

def download_finbert():
    """Download FinBERT model using direct download approach"""
    try:
        # Set environment variables
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['TRANSFORMERS_CACHE'] = '/app/shared_data/models'
        
        # Create models directory
        models_dir = Path('/app/shared_data/models')
        models_dir.mkdir(exist_ok=True)
        
        finbert_path = models_dir / 'finbert'
        finbert_path.mkdir(exist_ok=True)
        
        print("Downloading FinBERT model...")
        
        # Try using huggingface_hub directly
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="ProsusAI/finbert",
                local_dir=str(finbert_path),
                local_dir_use_symlinks=False
            )
            print("FinBERT downloaded successfully using huggingface_hub")
            return True
        except Exception as e:
            print(f"huggingface_hub download failed: {e}")
        
        # Fallback: try transformers with different settings
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            print("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                "ProsusAI/finbert",
                cache_dir=str(models_dir),
                local_files_only=False,
                trust_remote_code=True,
                use_fast=True
            )
            
            print("Downloading model...")
            model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert",
                cache_dir=str(models_dir),
                local_files_only=False,
                trust_remote_code=True
            )
            
            print("Saving model and tokenizer...")
            model.save_pretrained(str(finbert_path))
            tokenizer.save_pretrained(str(finbert_path))
            
            print("FinBERT downloaded successfully using transformers")
            return True
            
        except Exception as e:
            print(f"Transformers download failed: {e}")
            return False
            
    except Exception as e:
        print(f"Download failed: {e}")
        return False

if __name__ == "__main__":
    success = download_finbert()
    sys.exit(0 if success else 1) 