# MNIST Dataset for Testing

## Overview

The MNIST (Modified National Institute of Standards and Technology) dataset is a classic image classification dataset containing 70,000 grayscale images of handwritten digits (0-9). It's perfect for testing the image data functionality of the dashboard.

## Quick Start

### Option 1: Via API (Recommended)

**From Dashboard:**
```javascript
// In browser console or via API call
fetch('/api/download-mnist', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        sample_size: 1000  // Number of samples (100-10000)
    })
})
```

**Direct API Call:**
```bash
curl -X POST http://localhost:8000/download-mnist \
  -H "Content-Type: application/json" \
  -d '{"user_id": "admin", "sample_size": 1000}'
```

### Option 2: Via Script

**Run the download script directly:**
```bash
# From project root
cd xai_service
python download_mnist.py --sample-size 1000

# Or with custom output directory
python download_mnist.py --sample-size 1000 --output-dir /path/to/output
```

**Command-line options:**
- `--sample-size`: Number of samples to include (default: 1000, max: 70000)
- `--output-dir`: Output directory (default: shared_volume/uploads)
- `--metadata-only`: Only create metadata file without downloading

## Dataset Format

The script converts MNIST to CSV format with the following columns:

| Column | Description |
|--------|-------------|
| `image_data` | Base64-encoded PNG image (data URI format) |
| `label` | True label (0-9) |
| `prediction` | Mock predicted label (90% accuracy) |
| `confidence` | Prediction confidence score (0-1) |
| `digit` | Alias for label |
| `is_correct` | Whether prediction matches label (boolean) |
| `image_id` | Unique image identifier |
| `split` | Dataset split (train/test) |

## Output Files

After downloading, you'll find:

1. **`mnist_dataset.csv`** - Full dataset with specified sample size
2. **`mnist_sample_100.csv`** - Quick test sample (100 images)
3. **`mnist_metadata.json`** - Dataset metadata and documentation

## Usage in Dashboard

1. **Download MNIST** (via API or script)
   - Files are saved to `shared_volume/uploads/` folder
   - Main file: `mnist_dataset.csv`
   - Sample file: `mnist_sample_100.csv`
   - Metadata: `mnist_metadata.json`

2. **Upload the CSV file** through the dashboard
   - Upload `mnist_dataset.csv` or `mnist_sample_100.csv`
   - System automatically detects it as image data

3. **Generate data exploration plots** using natural language:
   - "Show class distribution" - See distribution of digits 0-9
   - "Show train test distribution" - See split between train and test
   - "Show dataset overview" - Get comprehensive dataset summary
   - "Show class balance analysis" - Analyze class imbalance
   - "Show image statistics" - See image dimensions and quality metrics
   - "Show sample images by class" - Visual inspection of images

4. **All plots are stored with metadata** for chat reference
   - Chat can answer questions about your plots
   - Plot summaries are stored in vector database

## Data Exploration Plots

The system provides comprehensive **data-only exploration plots** that analyze the dataset itself, not model performance. These plots help you understand your data before training or evaluating models.

### Available Data Exploration Plots

#### 1. Class Distribution
**Query Examples:**
- "Show class distribution"
- "Show label distribution"
- "Show image classification distribution"

**What it shows:**
- Bar chart with count of images per class (0-9)
- Percentage distribution for each class
- Total number of images and classes

**Use case:** Understand how balanced your dataset is across different classes.

#### 2. Train/Test Split Distribution
**Query Examples:**
- "Show train test distribution"
- "Show train vs test split"
- "Plot image count by split"

**What it shows:**
- Bar chart comparing train vs test counts
- Optional: Class distribution within each split (stacked view)
- Percentages for each split

**Use case:** Verify your data split is balanced and understand the distribution across splits.

#### 3. Dataset Overview
**Query Examples:**
- "Show dataset overview"
- "Show dataset summary"
- "Display dataset information"

**What it shows:**
- Summary table with total images, number of classes, column information
- Train/test split counts
- List of classes present
- Column types and metadata

**Use case:** Quick overview of your entire dataset structure.

#### 4. Class Balance Analysis
**Query Examples:**
- "Show class balance"
- "Analyze class imbalance"
- "Show class balance analysis"

**What it shows:**
- Class distribution (counts and percentages)
- Imbalance ratio indicator (max/min class count)
- Balance status (well balanced, moderately imbalanced, highly imbalanced)

**Use case:** Identify if your dataset has class imbalance issues that might affect model training.

#### 5. Image Statistics
**Query Examples:**
- "Show image statistics"
- "Show image dimensions"
- "Display image quality metrics"

**What it shows:**
- Image dimensions distribution (width/height)
- Aspect ratio distribution
- File size distribution
- Color channel statistics (if RGB)
- Image quality metrics (sharpness, contrast)

**Use case:** Understand the characteristics and quality of your image data.

#### 6. Sample Images by Class
**Query Examples:**
- "Show sample images by class"
- "Display images organized by class"
- "Show images from each class"

**What it shows:**
- Grid of sample images (2-3 per class)
- Images organized by their class labels
- Visual inspection of data quality

**Use case:** Visually inspect your data to ensure quality and understand what each class looks like.

## Distinction: Data Plots vs XAI Plots

### Data Exploration Plots (Phase 1)
These plots analyze **the dataset itself**:
- ✅ Class distribution
- ✅ Train/test split
- ✅ Dataset overview
- ✅ Class balance
- ✅ Image statistics
- ✅ Sample images

**Purpose:** Understand your data before model training/evaluation.

### XAI/Model Performance Plots (Future)
These plots analyze **model predictions and performance**:
- Confusion matrix
- Accuracy metrics
- Misclassification analysis
- Confidence distributions
- Model predictions

**Purpose:** Understand how well your model performs and why it makes certain predictions.

## Example Queries

### Classification Distribution (Data Plot)
```
Query: "Show image classification distribution"
Result: Bar chart showing count of images for each digit (0-9) with percentages
```

### Train/Test Split (Data Plot)
```
Query: "Show train test distribution"
Result: Bar chart comparing train vs test counts, optionally showing class distribution within splits
```

### Dataset Overview (Data Plot)
```
Query: "Show dataset overview"
Result: Summary table with total images, classes, split information, and column details
```

### Class Balance (Data Plot)
```
Query: "Show class balance analysis"
Result: Class distribution with imbalance ratio indicator and balance status
```

### Confusion Matrix (XAI Plot - Future)
```
Query: "Create confusion matrix for image predictions"
Result: Interactive confusion matrix showing prediction accuracy
```

### Accuracy Analysis (XAI Plot - Future)
```
Query: "Show prediction accuracy by digit"
Result: Bar chart showing accuracy for each digit class
```

## Dataset Statistics

- **Total Images**: 70,000 (60,000 train + 10,000 test)
- **Image Size**: 28x28 pixels
- **Color Mode**: Grayscale
- **Classes**: 10 (digits 0-9)
- **Format**: CSV with base64-encoded images

## Testing Scenarios

### 1. Image Data Detection
- Upload `mnist_dataset.csv`
- Verify system detects it as `image` data type
- Check that image columns are identified

### 2. Plot Generation
- Test various plot queries
- Verify confusion matrix generation
- Test classification distribution plots

### 3. Chat Integration
- Ask questions about the dataset
- Query about generated plots
- Test RAG system with image data context

### 4. Performance Testing
- Test with different sample sizes (100, 1000, 5000)
- Verify plot generation speed
- Test with full dataset (70,000 samples)

## Requirements

- **torchvision**: For downloading MNIST
  ```bash
  pip install torchvision
  ```

- **Pillow**: For image processing (already in requirements)
- **pandas**: For CSV handling (already in requirements)
- **numpy**: For array operations (already in requirements)

## Troubleshooting

### Issue: "torchvision not available"
**Solution:**
```bash
pip install torchvision
```

### Issue: Download takes too long
**Solution:**
- Use smaller sample size (e.g., 100-500)
- Download happens only once, subsequent runs use cached data

### Issue: Memory errors with large datasets
**Solution:**
- Reduce sample size
- Use `mnist_sample_100.csv` for quick testing
- Process in batches if needed

### Issue: Images not displaying
**Solution:**
- Verify base64 encoding is correct
- Check that image_data column contains valid data URIs
- Ensure CSV file is properly formatted

## Integration with Image Plot Generator

The MNIST dataset works seamlessly with the `ImagePlotGenerator`:

1. **Automatic Detection**: System recognizes base64 image data
2. **Plot Generation**: Can generate all supported plot types
3. **Metadata Analysis**: Analyzes labels, predictions, confidence scores
4. **Classification Support**: Perfect for testing classification visualizations

## Example Workflow

```python
# 1. Download MNIST
POST /download-mnist
{"user_id": "admin", "sample_size": 1000}

# 2. Upload CSV
POST /api/upload-data
# Upload mnist_dataset.csv

# 3. Generate Plot
POST /generate-interactive-plot
{
    "user_id": "admin",
    "query": "Show classification distribution"
}

# 4. Chat about results
POST /api/chat
{
    "question": "What is the distribution of digits in the dataset?",
    "user_id": "admin"
}
```

## Next Steps

After testing with MNIST:

1. Try with your own image datasets
2. Test with different image formats
3. Experiment with various plot types
4. Test RAG chat with image data
5. Explore advanced visualizations

## References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Image Data Support Documentation](./IMAGE_DATA_SUPPORT.md)
- [Plot Generator Documentation](../xai_service/plot_generators/README.md)


