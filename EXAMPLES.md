# Example Usage Guide

This guide provides step-by-step examples for common use cases.

## Table of Contents
1. [Basic Training Example](#basic-training-example)
2. [Classification Example](#classification-example)
3. [Image Thresholding Example](#image-thresholding-example)
4. [Advanced Usage](#advanced-usage)

---

## Basic Training Example

### Scenario
You have 5,000 labeled droplet images in three categories and want to train a classifier.

### Step 1: Organize Your Data

```
my_dataset/
├── Aggregate/
│   ├── droplet_001.png
│   ├── droplet_002.png
│   └── ... (1,500 images)
├── Condensate/
│   ├── droplet_501.png
│   └── ... (2,000 images)
└── Homogenous/
    ├── droplet_1001.png
    └── ... (1,500 images)
```

### Step 2: Configure train.ipynb

Open `train.ipynb` and modify the configuration cell:

```python
# Update this path
dataset_path = "/path/to/my_dataset"

# Adjust these parameters
max_per_class = 2000      # Maximum samples per class
batch_size = 500          # Batch size for feature extraction
dimred = True             # Set to False to skip dimensionality reduction
```

### Step 3: Run Training

Execute all cells in the notebook. Expected output:

```
Loading dataset...
Loading Aggregate: 100%|████████| 1500/1500
Loading Condensate: 100%|████████| 2000/2000
Loading Homogenous: 100%|████████| 1500/1500
Loaded 5000 images across 3 classes

Extracting features...
Extracting features: 100%|████████| 10/10 [02:15<00:00]
Features saved to cache

Training XGBoost classifier...
Model saved to: /path/to/my_dataset/xgb_trained_mobilenetv2_max2000.pkl

Evaluating model on test set...
Accuracy: 0.9680 (242 / 250 correct)
```

### Step 4: Review Results

The training produces:
- **Model file:** `xgb_trained_mobilenetv2_max2000.pkl`
- **Feature cache:** `feat_cache_klavs_2000_mobilenetv2.pkl` 
- **Performance metrics** printed in output
- **(Optional)** t-SNE and UMAP visualizations

**Expected Performance:**
- Accuracy: 95-98%
- F1-Score: 93-97% per class

---

## Classification Example

### Scenario
You have new microscopy TIF files and want to classify detected droplets.

### Step 1: Prepare Input Data

Ensure you have:
```
experiment_directory/
├── 1.tif                           # Multi-channel image
├── 1_647_droplet_boxes.pkl         # Pre-computed bounding boxes
├── ca_params.yml                   # Camera alignment parameters
├── metadata.csv                    # Experimental metadata
└── xgb_trained_mobilenetv2_max2000.pkl  # Trained model
```

### Step 2: Configure classify.ipynb

```python
# Dataset path
dataset_path = "/path/to/experiment_directory"

# Load trained classifier
trained_n = 2000
feature = f"mobilenetv2_max{trained_n}"
classifier = "xgb"

# Quadrant configuration (adjust for your setup)
classify_quadrant = {
    "experiment_name": (488, 647),  # (classify_channel, detect_channel)
}

# Processing flags
skip_droplet_classification = False
write_raw_droplets = False
skip_debug_ims = False
skip_combined_spreadsheet = False
```

### Step 3: Run Classification

Execute all cells. Expected output:

```
Loading classifier...
Classifier loaded successfully

STEP 1: Extracting quadrant images from multi-channel TIFs
Found 1 quadrant directories to process

STEP 2: Reading pre-computed droplet bounding boxes
Processing quadrants: 100%|████████| 1/1
Processing 647: 100%|████████| 150/150

Loading MobileNetV2 feature extractor (one-time initialization)...
Model loaded on device: cuda

STEP 4: Classifying droplets using XGBoost
Working on /path/to/experiment_directory/quadrant_3/
Loading droplets: 100%|████████| 150/150
Classifying: 100%|████████| 45/45 [00:30<00:00]

STEP 5: Creating annotated images with classification labels
Creating debug images: 100%|████████| 1/1

STEP 6: Creating combined analysis spreadsheet
Saved analysis to: /path/to/experiment_directory/analysis.csv

PIPELINE COMPLETE
```

### Step 4: Review Results

**Output files:**
- `quadrant_3/droplet_classification_pkl/*.pkl` - Raw predictions
- `quadrant_3/droplet_classification_im/*.tif` - Annotated images
- `analysis.csv` - Combined spreadsheet with classifications

**analysis.csv format:**
```csv
tif,frame,droplet_id,x,y,class
1,0,0,245,312,Homogenous
1,0,1,456,123,Aggregate
1,0,2,678,234,Condensate
...
```

---

## Image Thresholding Example

### Scenario
You need to segment condensate structures from grayscale microscopy images.

### Step 1: Organize Images

```
raw_images/
├── Homogenous/
│   ├── sample_001.tif
│   └── sample_002.tif
├── Aggregate/
│   └── ...
└── Condensate/
    └── ...
```

### Step 2: Configure image-thresholding.ipynb

```python
# Input path
root_path = '/path/to/raw_images'

# Output path
output_path = '/path/to/segmented_images'

# Morphological kernel size (3x3 is good default)
kernel_size = 3
```

### Step 3: Test Pipeline on Samples

Run the visualization cell first:

```python
# Display processing pipeline on 5 random samples
display_processing_pipeline(images, n_samples=5, kernel_size=3)
```

This shows you:
- Original image
- Histogram with detected threshold
- Binary segmentation result
- Morphologically processed result

**Verify the threshold looks good** before batch processing!

### Step 4: Batch Process

If the sample results look good, run batch processing:

```python
process_and_save_images(images, output_path, kernel_size=3)
```

Expected output:
```
Processed 10/250 images...
Processed 20/250 images...
...
Processing complete!
Total: 245 images
Homogenous: 80 images
Aggregate: 85 images
Condensate: 80 images
Saved to: /path/to/segmented_images
```

---

## Advanced Usage

### 1. Adjusting Training Parameters

For better performance with imbalanced datasets:

```python
# In train.ipynb
params = {
    "max_depth": 5,              # Increase for more complex models
    "eta": 0.1,                  # Lower learning rate (more conservative)
    "objective": "multi:softprob",
    "scale_pos_weight": 2.0      # Add this for imbalanced classes
}

clf = XGBWrapper(params, epochs=200)  # More epochs
```

### 2. Custom Feature Extraction

To use different layers or models:

```python
# In extract_features function
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
# Remove final classification layer
model.fc = torch.nn.Identity()
```

### 3. Processing Large Datasets

For datasets too large for memory:

```python
# In train.ipynb - process in chunks
for chunk_idx in range(0, total_images, chunk_size):
    chunk_images = images[chunk_idx:chunk_idx + chunk_size]
    chunk_features = get_features_batched(chunk_images)
    # Save chunk features incrementally
```

### 4. Optimizing Batch Size

Find optimal batch size for your GPU:

```python
import torch

# Test different batch sizes
for bs in [100, 200, 500, 1000]:
    try:
        test_batch = torch.randn(bs, 3, 224, 224).cuda()
        model(test_batch)
        print(f"Batch size {bs}: OK")
    except RuntimeError as e:
        print(f"Batch size {bs}: Out of memory")
        break
```

### 5. Multi-Class Confidence Thresholds

For more conservative predictions:

```python
# In classify_droplets function
def predict_with_threshold(classifier, X, threshold=0.7):
    dtest = xgb.DMatrix(X)
    probs = classifier.model.predict(dtest)  # Get probabilities
    max_probs = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    
    # Set low-confidence predictions as "uncertain"
    predictions[max_probs < threshold] = -1  
    
    return classifier.label_encoder.inverse_transform(predictions)
```

### 6. Cross-Validation for Robustness

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    X_train_fold = X[train_idx]
    y_train_fold = [labels[i] for i in train_idx]
    
    clf = XGBWrapper(params, epochs=100)
    clf.fit(X_train_fold, y_train_fold)
    
    # Evaluate on validation fold
    # ...
```

---

## Common Workflows

### Workflow 1: New Project Setup
1. Collect and label training data
2. Run `train.ipynb` → Get trained model
3. Use model in `classify.ipynb` for new data

### Workflow 2: Model Retraining
1. Classify new data with existing model
2. Manually review and correct misclassifications
3. Add corrected samples to training set
4. Retrain model with expanded dataset

### Workflow 3: Dataset Curation
1. Use `image-thresholding.ipynb` to segment images
2. Manually inspect segmented images
3. Organize good quality images for training
4. Train classifier on curated dataset

---

## Performance Benchmarks

Typical runtimes on different hardware:

### Training (5,000 images)
- **CPU only:** 15-20 minutes
- **CPU + GPU (CUDA):** 5-8 minutes
- **With cached features:** 2-3 minutes

### Classification (10,000 droplets)
- **CPU only:** 10-15 minutes
- **CPU + GPU (CUDA):** 2-4 minutes
- **With optimizations:** 1-2 minutes

### Thresholding (1,000 images)
- **CPU:** 3-5 minutes
- (No GPU acceleration for this pipeline)

---

## Tips and Best Practices

1. **Always cache features** when training - saves enormous time on reruns
2. **Test on small subset first** before processing large datasets
3. **Verify thresholds visually** before batch processing
4. **Use GPU** when available - 5-10x speedup for feature extraction
5. **Balance your training data** - aim for similar numbers per class
6. **Keep raw data** - never overwrite original images
7. **Document parameters** used for each trained model
8. **Validate on held-out data** - don't evaluate on training set

---

## Need More Help?

- See [README.md](README.md) for general information
- See [INSTALLATION.md](INSTALLATION.md) for setup issues
- Check [GitHub Issues](https://github.com/your-username/droplet-classification/issues)
