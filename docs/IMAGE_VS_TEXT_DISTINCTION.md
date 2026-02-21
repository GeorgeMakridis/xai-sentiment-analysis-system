# Image vs Text Data Distinction

## Overview

The system now clearly distinguishes between **IMAGE data** and **TEXT data** in visualizations, providing appropriate and useful plots for each data type.

## Key Improvements

### 1. **Visual Indicators**

#### Image Data Plots
- **📸 Emoji indicator** in plot titles
- **"IMAGE DATA"** annotations on plots
- **Blue color scheme** for image data indicators
- Clear labeling: "Image Class/Label", "Number of Images"

#### Text Data Plots  
- **📝 Emoji indicator** in plot titles
- **"TEXT DATA"** annotations on plots
- Clear labeling: "Sentiment", "Text Analysis"

### 2. **Image-Specific Visualizations**

#### Image Grid (Shows Actual Images)
- **Query**: "Show me the actual images", "Display image grid", "Show sample images"
- **Result**: HTML grid displaying actual images from the dataset
- **Features**:
  - Up to 20 sample images
  - Grouped by category/label
  - Each image shows its label
  - Responsive grid layout

#### Image Metadata Analysis
- **Query**: "Show image classification distribution", "Plot image count by class"
- **Result**: Bar chart showing distribution of image classifications
- **Features**:
  - Clearly marked as "IMAGE DATA VISUALIZATION"
  - Shows metadata about images (not the images themselves)
  - Color-coded by count

#### Confusion Matrix (Image Classification)
- **Query**: "Create confusion matrix for image predictions"
- **Result**: Heatmap showing classification accuracy
- **Features**:
  - Clearly marked as image classification
  - Shows predicted vs actual image classes

### 3. **Text-Specific Visualizations**

#### Sentiment Distribution
- **Query**: "Show sentiment distribution"
- **Result**: Histogram of sentiment scores
- **Features**:
  - Clearly marked as "TEXT DATA VISUALIZATION"
  - Shows sentiment analysis of text content

#### Sentiment Trends
- **Query**: "Show sentiment over time"
- **Result**: Line chart of sentiment trends
- **Features**:
  - Time-based analysis of text sentiment
  - Clearly marked as text data

## Query Interpretation

### Image Data Queries
The system recognizes image-specific requests:

- **"Show images"** → Image grid with actual images
- **"Display image grid"** → Image grid
- **"Show sample images"** → Image grid by category
- **"Image classification distribution"** → Metadata bar chart
- **"Confusion matrix"** → Image classification matrix

### Text Data Queries
The system recognizes text-specific requests:

- **"Show sentiment distribution"** → Sentiment histogram
- **"Sentiment over time"** → Sentiment trend line
- **"Word importance"** → Text feature analysis
- **"Sentiment by asset"** → Text-based comparison

## Visual Distinctions

### Image Data Plots Include:
```
📸 IMAGE DATA VISUALIZATION
Distribution of image classifications
This shows metadata about images, not the images themselves
```

### Text Data Plots Include:
```
📝 TEXT DATA VISUALIZATION
Sentiment analysis of text content
```

## Example Workflows

### Image Data Workflow
1. Upload MNIST CSV with base64 images
2. System detects: `data_type: "image"`
3. Query: "Show me the actual images"
4. Result: HTML grid with 20 sample images displayed
5. Query: "Show image classification distribution"
6. Result: Bar chart with 📸 indicator showing digit distribution

### Text Data Workflow
1. Upload sentiment CSV with text columns
2. System detects: `data_type: "text"`
3. Query: "Show sentiment distribution"
4. Result: Histogram with 📝 indicator showing sentiment scores

## Technical Implementation

### Data Type Detection
```python
# Image detection priority:
1. Column names: 'image', 'img', 'picture', 'photo'
2. Base64 data: 'data:image...' or long base64 strings
3. File paths: '.jpg', '.png', etc.

# Text detection:
1. Long text strings (avg length > 20 chars)
2. Sentiment columns
3. Text analysis columns
```

### Plot Generator Selection
```python
# Automatic routing:
- image data → ImagePlotGenerator
- text/sentiment data → SentimentPlotGenerator
- timeseries data → TimeSeriesPlotGenerator (future)
- tabular data → GenericPlotGenerator (future)
```

## Best Practices

### For Image Data:
- Use "Show images" or "Display image grid" to see actual images
- Use "Show distribution" for metadata analysis
- Use "Confusion matrix" for classification results

### For Text Data:
- Use "Show sentiment distribution" for sentiment analysis
- Use "Sentiment over time" for temporal trends
- Use "Word importance" for feature analysis

## Troubleshooting

### Issue: Images not showing in grid
**Solution**: 
- Verify base64 encoding is correct
- Check image_data column contains valid data URIs
- Try smaller sample size

### Issue: Wrong plot type generated
**Solution**:
- Be more specific in query ("show actual images" vs "show distribution")
- Check data type detection in ingestion response
- Verify correct plot generator is being used

### Issue: Can't distinguish image vs text plots
**Solution**:
- Look for 📸 (image) or 📝 (text) emoji in titles
- Check for "IMAGE DATA" or "TEXT DATA" annotations
- Review plot labels (Image Class vs Sentiment)


