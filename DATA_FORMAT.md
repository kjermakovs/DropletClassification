# Data Format Specification

This document describes the expected data formats and directory structures for all pipelines.

## Table of Contents
1. [Training Data Format](#training-data-format)
2. [Classification Data Format](#classification-data-format)
3. [Thresholding Data Format](#thresholding-data-format)
4. [File Format Specifications](#file-format-specifications)

---

## Training Data Format

### Directory Structure

```
dataset_root/
├── Aggregate/
│   ├── image_0001.png
│   ├── image_0002.png
│   └── ...
├── Condensate/
│   ├── image_0001.png
│   └── ...
└── Homogenous/
    ├── image_0001.png
    └── ...
```

### Requirements
- **Classes:** Each subdirectory name represents a class label
- **Images:** Any standard image format (PNG, JPG, TIF, TIFF)
- **Naming:** No specific naming convention required
- **Size:** Images will be resized to 224×224 for MobileNetV2
- **Format:** Grayscale or RGB (converted to RGB internally)
- **Balance:** Aim for similar number of samples per class (recommended)

### Recommendations
- **Minimum samples:** 500-1000 per class for good performance
- **Maximum samples:** Limited by memory (use `max_per_class` parameter)
- **Image quality:** Remove blurry, corrupted, or ambiguous samples
- **Consistent imaging:** Same microscope settings for all images

### Example Dataset Statistics
```
Total images: 15,562
├── Aggregate: 5,125 images (33%)
├── Condensate: 5,218 images (34%)
└── Homogenous: 5,219 images (33%)

Image properties:
- Size range: 48×48 to 512×512 pixels
- Format: PNG, grayscale
- Bit depth: 8-bit
```

---

## Classification Data Format

### Directory Structure

```
experiment_root/
├── 1.tif                               # Multi-channel microscopy image
├── 2.tif
├── 1_647_droplet_boxes.pkl             # Bounding boxes for image 1, channel 647
├── 2_647_droplet_boxes.pkl
├── ca_params.yml                       # Camera alignment parameters
├── experiment_metadata.csv             # Optional: metadata
├── xgb_trained_mobilenetv2_*.pkl      # Trained classifier model
└── quadrant_3/                         # Auto-generated during processing
    ├── raw/
    │   ├── 1_0.tif                     # Frame 0 from image 1
    │   ├── 1_1.tif
    │   └── ...
    ├── droplet_detection_pkl/
    ├── droplet_classification_pkl/
    └── droplet_classification_im/
```

### Multi-Channel TIF Format

**Structure:**
- Multi-frame TIFF file
- Each frame is a time point or z-slice
- Contains 4 quadrants arranged in 2×2 grid

**Quadrant Layout:**
```
┌─────────┬─────────┐
│    0    │    1    │  Quadrant 0: Empty or reference
│         │ (647nm) │  Quadrant 1: Channel 647nm
├─────────┼─────────┤  Quadrant 2: Channel 546nm
│ (546nm) │ (488nm) │  Quadrant 3: Channel 488nm
│    2    │    3    │
└─────────┴─────────┘
```

### Bounding Box Format (.pkl)

**File naming:** `{tif_id}_{channel}_droplet_boxes.pkl`

**Content:** Python dictionary
```python
{
    'image_key_0_647': [
        (x, y, width, height),  # Bounding box 1
        (x, y, width, height),  # Bounding box 2
        # ...
    ],
    'image_key_1_647': [
        # Bounding boxes for frame 1
    ],
    # ...
}
```

**Key format:** `{prefix}_{frame_id}_{channel}`

**Coordinates:**
- `x, y`: Top-left corner of bounding box
- `width, height`: Box dimensions in pixels
- Coordinates are relative to the full image (before quadrant extraction)

### Camera Alignment Parameters (ca_params.yml)

```yaml
quadrant_align_info:
  ref: 1  # Reference quadrant (647nm channel)
  
  # Quadrant positions (normalized coordinates)
  qpos:
    0: [0.0, 0.0, 0.5, 0.5]      # [x_min, y_min, x_max, y_max]
    1: [0.5, 0.0, 1.0, 0.5]
    2: [0.0, 0.5, 0.5, 1.0]
    3: [0.5, 0.5, 1.0, 1.0]
  
  # Affine transformation matrices for alignment
  warp:
    "2": [[1.0, 0.0, -2.3], [0.0, 1.0, 1.5]]  # 2×3 transformation matrix
    "3": [[1.0, 0.0, 1.2], [0.0, 1.0, -0.8]]
```

### Metadata CSV Format (Optional)

```csv
tif,frame,droplet_id,x,y,area,intensity,additional_columns...
1,0,0,245.3,312.7,1234,45678,...
1,0,1,456.1,123.4,987,34567,...
1,1,0,234.5,456.7,1100,42000,...
2,0,0,789.0,234.5,1050,38900,...
```

**Required columns:**
- `tif`: TIF file ID (integer)
- `frame`: Frame number within TIF (integer)
- `droplet_id`: Unique droplet ID within frame (integer)

**Optional columns:**
- `x, y`: Droplet centroid coordinates
- Any additional experimental metadata

---

## Thresholding Data Format

### Directory Structure

```
input_root/
├── Homogenous/
│   ├── sample_001.tif
│   ├── sample_002.tif
│   └── ...
├── Aggregate/
│   ├── sample_101.tif
│   └── ...
└── Condensate/
    ├── sample_201.tif
    └── ...
```

### Requirements
- **Categories:** Subdirectory names for organizing images
- **Images:** TIF, TIFF, PNG, JPG formats supported
- **Format:** Grayscale images (single channel)
- **Bit depth:** 8-bit or 16-bit (will be normalized)
- **Size:** Any size (processed as-is, no resizing)

### Output Structure

After processing:
```
output_root/
├── Homogenous/
│   ├── sample_001.tif          # Binary segmented version
│   └── ...
├── Aggregate/
│   └── ...
└── Condensate/
    └── ...
```

**Output format:**
- Binary images (0 = background, 255 = foreground)
- Same filename as input
- Same size as input
- 8-bit grayscale TIF

---

## File Format Specifications

### Supported Image Formats

| Format | Extension | Read | Write | Notes |
|--------|-----------|------|-------|-------|
| PNG | .png | ✓ | ✓ | Lossless, recommended for training |
| JPEG | .jpg, .jpeg | ✓ | ✗ | Lossy, not recommended |
| TIFF | .tif, .tiff | ✓ | ✓ | Multi-frame support, high bit-depth |
| BMP | .bmp | ✓ | ✗ | Uncompressed |

### Pickle Files (.pkl)

**Python pickle format** for serializing:
- Bounding boxes
- Trained models
- Classification results
- Feature caches

**Loading example:**
```python
import pickle

with open('file.pkl', 'rb') as f:
    data = pickle.load(f)
```

**Compatibility:**
- Python 3.11+ (using pickle protocol 5)
- Not cross-platform with Python 2.x
- Binary format (not human-readable)

### YAML Files (.yml)

**Human-readable configuration files**

**Loading example:**
```python
import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)
```

### CSV Files (.csv)

**Comma-separated values** for tabular data

**Requirements:**
- UTF-8 encoding
- Comma delimiter
- Header row with column names
- No special characters in column names

---

## Data Validation

### Validation Checklist

**Before Training:**
- [ ] All class directories contain images
- [ ] No corrupted images (test with `PIL.Image.open()`)
- [ ] Reasonable class balance (within 2-3x of each other)
- [ ] Images open without errors
- [ ] Consistent image properties within dataset

**Before Classification:**
- [ ] TIF files are multi-frame
- [ ] Bounding box files exist for all TIF files
- [ ] ca_params.yml is present and valid
- [ ] Trained model file exists
- [ ] Quadrant configuration matches your setup

**Before Thresholding:**
- [ ] Images are grayscale
- [ ] Category subdirectories exist
- [ ] Output directory has write permissions
- [ ] Sufficient disk space for output

### Automated Validation Scripts

**Check training data:**
```python
from pathlib import Path
from PIL import Image

def validate_training_data(root_path):
    classes = ['Aggregate', 'Condensate', 'Homogenous']
    
    for cls in classes:
        cls_path = Path(root_path) / cls
        if not cls_path.exists():
            print(f"❌ Missing directory: {cls}")
            continue
        
        images = list(cls_path.glob('*.png')) + list(cls_path.glob('*.tif'))
        print(f"✓ {cls}: {len(images)} images")
        
        # Test first few images
        for img_path in images[:5]:
            try:
                Image.open(img_path)
            except Exception as e:
                print(f"  ❌ Corrupted: {img_path.name}")

validate_training_data('/path/to/dataset')
```

**Check classification data:**
```python
import yaml
from pathlib import Path

def validate_classification_data(root_path):
    root = Path(root_path)
    
    # Check for required files
    if not (root / 'ca_params.yml').exists():
        print("❌ Missing ca_params.yml")
    else:
        print("✓ Found ca_params.yml")
    
    # Check TIF and PKL pairs
    tif_files = list(root.glob('*.tif'))
    print(f"✓ Found {len(tif_files)} TIF files")
    
    for tif in tif_files:
        tif_id = tif.stem
        pkl_pattern = f"{tif_id}_*_droplet_boxes.pkl"
        pkl_files = list(root.glob(pkl_pattern))
        
        if not pkl_files:
            print(f"  ❌ Missing bounding boxes for {tif.name}")
        else:
            print(f"  ✓ {tif.name}: {len(pkl_files)} channel(s)")

validate_classification_data('/path/to/experiment')
```

---

## Common Issues

### Issue: "No such file or directory"
- Check that paths use correct separators for your OS
- Use absolute paths or ensure working directory is correct
- Verify case sensitivity on Linux/macOS

### Issue: "Cannot identify image file"
- Image file may be corrupted
- Format may not be supported
- Try opening with another program to verify

### Issue: "Index out of range" during classification
- Bounding boxes may be outside image boundaries
- Check coordinate system matches (top-left origin)
- Verify image hasn't been cropped since box computation

### Issue: "Key error" in classification
- TIF IDs in filenames don't match metadata
- Frame indices don't match
- Droplet IDs are inconsistent

---

## Best Practices

1. **Use lossless formats** (PNG, TIF) for training data
2. **Keep original data** separate from processed versions
3. **Document your naming conventions** in a separate file
4. **Validate data** before starting long processing runs
5. **Use version control** for configuration files (YAML, CSV)
6. **Back up processed results** (models, features, classifications)
7. **Include README** in data directories explaining structure

---

## Example Datasets

Small example datasets are available for testing:

```bash
# Download example training data (500 images, 50MB)
wget https://example.com/example_training_data.zip

# Download example classification data (10 TIF files, 200MB)
wget https://example.com/example_classification_data.zip
```

See [EXAMPLES.md](EXAMPLES.md) for usage with example data.
