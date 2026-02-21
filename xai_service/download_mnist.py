#!/usr/bin/env python3
"""
Script to download MNIST dataset and convert it to CSV format for testing image data functionality
"""

import os
import sys
import base64
import io
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json

# Try to import torchvision for MNIST
try:
    import torchvision
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: torchvision not available. Install with: pip install torchvision")

def download_mnist(sample_size=1000, output_dir=None):
    """
    Download MNIST dataset and convert to CSV format
    
    Args:
        sample_size: Number of samples to include (default: 1000, max: 70000)
        output_dir: Directory to save the CSV file (default: shared_volume/uploads)
    """
    if not TORCHVISION_AVAILABLE:
        print("Error: torchvision is required to download MNIST dataset")
        print("Install it with: pip install torchvision")
        return False
    
    try:
        # Set output directory
        if output_dir is None:
            # Try to use shared volume path (for Docker)
            if os.path.exists('/app/shared_data'):
                output_dir = Path('/app/shared_data/uploads')
            else:
                # Use local path - ensure it's relative to project root or absolute
                local_path = Path('shared_volume/uploads')
                # If running from xai_service directory, go up one level
                if not local_path.exists() and Path('../shared_volume/uploads').exists():
                    output_dir = Path('../shared_volume/uploads').resolve()
                else:
                    output_dir = local_path.resolve() if local_path.exists() else local_path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify the directory was created successfully
        if not output_dir.exists():
            raise Exception(f"Failed to create output directory: {output_dir}")
        
        print(f"Downloading MNIST dataset (sample size: {sample_size})...")
        print("This may take a few minutes on first run...")
        
        # Download MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Download training set
        trainset = torchvision.datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        # Download test set
        testset = torchvision.datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        print("MNIST dataset downloaded successfully!")
        
        # Combine train and test sets
        all_images = []
        all_labels = []
        
        # Add training samples
        train_size = min(sample_size // 2, len(trainset))
        print(f"Processing {train_size} training samples...")
        for i in range(train_size):
            image, label = trainset[i]
            all_images.append(image)
            all_labels.append(label)
        
        # Add test samples
        test_size = min(sample_size - train_size, len(testset))
        print(f"Processing {test_size} test samples...")
        for i in range(test_size):
            image, label = testset[i]
            all_images.append(image)
            all_labels.append(label)
        
        print(f"Converting {len(all_images)} images to base64 format...")
        
        # Convert images to base64
        image_data_list = []
        labels_list = []
        predictions_list = []
        confidence_list = []
        
        for idx, (image_tensor, label) in enumerate(zip(all_images, all_labels)):
            # Convert tensor to PIL Image
            # MNIST images are 1x28x28, need to convert to 28x28 for PIL
            image_np = image_tensor.squeeze().numpy()
            # Convert to 0-255 range
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np, mode='L')
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            base64_str = base64.b64encode(image_bytes).decode('utf-8')
            data_uri = f"data:image/png;base64,{base64_str}"
            
            image_data_list.append(data_uri)
            labels_list.append(int(label))
            
            # Add some mock predictions for testing (simulate 90% accuracy)
            if np.random.random() > 0.1:  # 90% correct predictions
                predictions_list.append(int(label))
                confidence_list.append(round(np.random.uniform(0.85, 0.99), 2))
            else:  # 10% wrong predictions
                wrong_label = np.random.randint(0, 10)
                predictions_list.append(wrong_label)
                confidence_list.append(round(np.random.uniform(0.5, 0.75), 2))
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(all_images)} images...")
        
        # Create DataFrame
        df = pd.DataFrame({
            'image_data': image_data_list,
            'label': labels_list,
            'prediction': predictions_list,
            'confidence': confidence_list,
            'digit': labels_list,  # Alias for label
            'is_correct': [l == p for l, p in zip(labels_list, predictions_list)]
        })
        
        # Add some metadata columns
        df['image_id'] = [f"mnist_{i:05d}" for i in range(len(df))]
        df['split'] = ['train' if i < train_size else 'test' for i in range(len(df))]
        
        # Save to CSV
        output_file = output_dir / 'mnist_dataset.csv'
        print(f"\nSaving to {output_file}...")
        df.to_csv(output_file, index=False)
        
        # Also save a smaller sample for quick testing
        sample_file = output_dir / 'mnist_sample_100.csv'
        df_sample = df.head(100)
        df_sample.to_csv(sample_file, index=False)
        
        # Print summary
        print("\n" + "="*50)
        print("MNIST Dataset Conversion Complete!")
        print("="*50)
        print(f"Total samples: {len(df)}")
        print(f"Training samples: {train_size}")
        print(f"Test samples: {test_size}")
        print(f"Labels distribution:")
        print(df['label'].value_counts().sort_index())
        print(f"\nAccuracy (mock predictions): {df['is_correct'].mean()*100:.1f}%")
        print(f"\nOutput files:")
        print(f"  - Full dataset: {output_file}")
        print(f"  - Sample (100): {sample_file}")
        print("\nYou can now upload these CSV files to test image data functionality!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"Error downloading/processing MNIST: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_mnist_metadata(output_dir=None):
    """Create metadata JSON file for MNIST dataset"""
    if output_dir is None:
        if os.path.exists('/app/shared_data'):
            output_dir = Path('/app/shared_data/uploads')
        else:
            output_dir = Path('shared_volume/uploads')
    
    output_dir = Path(output_dir)
    metadata = {
        'dataset_name': 'MNIST',
        'description': 'Handwritten digit recognition dataset',
        'num_classes': 10,
        'image_size': '28x28',
        'color_mode': 'grayscale',
        'classes': [str(i) for i in range(10)],
        'class_names': {
            '0': 'Zero',
            '1': 'One',
            '2': 'Two',
            '3': 'Three',
            '4': 'Four',
            '5': 'Five',
            '6': 'Six',
            '7': 'Seven',
            '8': 'Eight',
            '9': 'Nine'
        },
        'format': 'CSV with base64 encoded images',
        'columns': {
            'image_data': 'Base64 encoded PNG image',
            'label': 'True label (0-9)',
            'prediction': 'Predicted label (0-9)',
            'confidence': 'Prediction confidence (0-1)',
            'digit': 'Alias for label',
            'is_correct': 'Whether prediction matches label',
            'image_id': 'Unique image identifier',
            'split': 'train or test'
        }
    }
    
    metadata_file = output_dir / 'mnist_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_file}")
    return metadata_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and convert MNIST dataset')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Number of samples to include (default: 1000, max: 70000)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: shared_volume/uploads)')
    parser.add_argument('--metadata-only', action='store_true',
                       help='Only create metadata file')
    
    args = parser.parse_args()
    
    if args.metadata_only:
        create_mnist_metadata(args.output_dir)
    else:
        success = download_mnist(
            sample_size=min(args.sample_size, 70000),
            output_dir=args.output_dir
        )
        
        if success:
            create_mnist_metadata(args.output_dir)
            sys.exit(0)
        else:
            sys.exit(1)


