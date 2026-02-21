# Image Data Support

## Overview

The dashboard now supports image data analysis with automatic detection and specialized plot generation capabilities.

## Features

### 1. Automatic Image Data Detection

The system automatically detects image data by:
- **Column name patterns**: Columns with names containing `image`, `img`, `picture`, `photo`, `file_path`, `path`
- **Base64 image data**: Columns containing base64-encoded image data (starting with `data:image` or long base64 strings)
- **Image file paths**: Columns containing file paths with image extensions (`.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp`)

### 2. Image Plot Generator

The `ImagePlotGenerator` class provides specialized visualizations for image data:

#### Supported Plot Types:
- **Image Grid**: Display images organized by category
- **Classification Distribution**: Bar charts showing distribution of image classifications
- **Confusion Matrix**: For image classification models
- **Image Statistics**: Metadata analysis and statistics
- **Correlation Heatmaps**: For image metadata features
- **Classification Results**: Visualization of prediction results

#### Supported Queries:
- "Show image classification distribution"
- "Create confusion matrix for image predictions"
- "Display image grid by category"
- "Show image count by class"
- "Plot image metadata correlation"
- "Generate histogram of image labels"
- "Show classification results"
- "Create heatmap of image features"

## Usage

### 1. Upload Image Data

Upload a CSV or JSON file containing image data. The file should have:
- Image columns (file paths, base64 data, or image references)
- Metadata columns (labels, predictions, categories, scores, etc.)

**Example CSV structure:**
```csv
image_path,label,prediction,confidence
/path/to/image1.jpg,cat,dog,0.75
/path/to/image2.jpg,dog,dog,0.92
/path/to/image3.jpg,cat,cat,0.88
```

**Example with base64:**
```csv
image_data,label,category
data:image/jpeg;base64,/9j/4AAQSkZJRg...,cat,animal
data:image/png;base64,iVBORw0KGgoAAAANSUh...,dog,animal
```

### 2. Automatic Detection

When you upload data, the system will:
1. Automatically detect if the data contains images
2. Set the data type to `'image'`
3. Identify image columns and metadata columns
4. Store preprocessing information

### 3. Generate Plots

Use the chat interface or plot generation endpoint to create visualizations:

**Via Chat:**
```
User: "Show me the distribution of image classifications"
Bot: [Generates and displays classification distribution plot]
```

**Via API:**
```bash
POST /generate-interactive-plot
{
  "user_id": "admin",
  "query": "Create confusion matrix for image predictions"
}
```

## Implementation Details

### Files Added/Modified

1. **`xai_service/plot_generators/image_plot_generator.py`** (NEW)
   - `ImagePlotGenerator` class
   - Handles image data visualization
   - Supports LLM-based query interpretation
   - Fallback mechanisms for robustness

2. **`xai_service/plot_generators/registry.py`** (MODIFIED)
   - Added `'image': ImagePlotGenerator` to registry

3. **`xai_service/app.py`** (MODIFIED)
   - Enhanced `detect_data_type()` to recognize image data
   - Updated `ingest_data()` to handle image data type
   - Added image preprocessing info

4. **`xai_service/plot_generators/__init__.py`** (MODIFIED)
   - Exported `ImagePlotGenerator`

### Data Type Detection Priority

The detection algorithm checks data types in this order:
1. **Image** (highest priority) - Checks for image columns
2. **Time Series** - Checks for datetime columns/index
3. **Text** - Checks for long text strings
4. **Tabular** (default) - Falls back to tabular data

### Image Column Detection Logic

```python
# Column name patterns
if 'image' in col.lower() or 'img' in col.lower():
    → Image column

# Base64 detection
if value.startswith('data:image') or is_base64_string(value):
    → Image column

# File path detection
if '.jpg' in value.lower() or '.png' in value.lower():
    → Image column
```

## Example Use Cases

### 1. Image Classification Results

**Data:**
```csv
filename,true_label,predicted_label,confidence
img001.jpg,cat,cat,0.95
img002.jpg,dog,dog,0.87
img003.jpg,cat,dog,0.65
```

**Query:** "Create confusion matrix"
**Result:** Interactive confusion matrix showing classification accuracy

### 2. Image Metadata Analysis

**Data:**
```csv
image_id,width,height,file_size,format,category
img001,1920,1080,245678,JPEG,animal
img002,800,600,123456,PNG,person
```

**Query:** "Show correlation between image dimensions and file size"
**Result:** Correlation heatmap of image metadata

### 3. Image Category Distribution

**Data:**
```csv
image_path,category,subcategory
/path/img1.jpg,animal,cat
/path/img2.jpg,animal,dog
/path/img3.jpg,person,adult
```

**Query:** "Show image count by category"
**Result:** Bar chart showing distribution of images across categories

## Future Enhancements

1. **Actual Image Rendering**: Display images in grid layouts
2. **Image Feature Extraction**: Extract and visualize image features
3. **Image Similarity Analysis**: Compare and cluster similar images
4. **Image Quality Metrics**: Analyze image quality and characteristics
5. **Multi-modal Analysis**: Combine image and text metadata

## Troubleshooting

### Issue: Image data not detected

**Solution:**
- Ensure image columns have recognizable names (`image`, `img`, `path`, etc.)
- For base64 data, ensure it starts with `data:image` or is valid base64
- For file paths, ensure they contain image file extensions

### Issue: Plot generation fails

**Solution:**
- Check that metadata columns exist (labels, predictions, categories)
- Ensure data has at least some rows
- Try simpler queries first (e.g., "show distribution")

### Issue: Confusion matrix not generated

**Solution:**
- Ensure both `label`/`true_label` and `prediction`/`predicted_label` columns exist
- Check that labels and predictions are in the same format

## API Reference

### Generate Interactive Plot

```python
POST /generate-interactive-plot
{
    "user_id": "string",
    "query": "string"  # Natural language query
}

Response:
{
    "message": "Interactive plot generated successfully",
    "plot_html": "<html>...</html>",
    "plot_type": "classification_distribution",
    "metadata": {
        "data_mode": "image",
        "query": "...",
        "columns_used": ["label", "category"]
    }
}
```

## Integration with RAG System

The image plot generator integrates with the existing RAG (Retrieval-Augmented Generation) system:

- **Plot metadata is stored** in the vector database
- **Chat can answer questions** about generated plots
- **Plot history** can be retrieved and discussed
- **Context-aware responses** based on image data and plots

Example chat interaction:
```
User: "What plots have I generated for my image data?"
Bot: "You've generated a classification distribution plot showing 15 categories, 
      with 'cat' being the most common (45 images). You also created a confusion 
      matrix showing 92% accuracy."
```

## Testing

To test image data support:

1. Create a test CSV with image data:
```csv
image_path,label,prediction
test1.jpg,cat,cat
test2.jpg,dog,dog
test3.jpg,cat,cat
```

2. Upload via dashboard or API
3. Generate plots using natural language queries
4. Verify plots are generated correctly
5. Test chat integration for plot-related questions


