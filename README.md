# Droplet Classification and Image Thresholding - Supplementary Code

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains supplementary code for the Nature article on biomolecular condensate classification using deep learning and machine learning techniques.

## Overview

This package provides automated pipelines for:
1. **Training** a droplet classifier using transfer learning (MobileNetV2 + XGBoost)
2. **Classifying** droplets in multi-channel microscopy images
3. **Thresholding** and segmenting condensate structures from microscopy data

## Repository Structure

```
code-refractored/
├── train.ipynb                    # Training pipeline for droplet classifier
├── classify.ipynb                 # Classification pipeline for new data
├── image-thresholding.ipynb       # Automated thresholding for condensate detection
├── environment.yml                # Conda environment specification
├── src/
│   └── data_processing.py         # Utility functions for image processing
└── README.md                      # This file
```

## Notebooks

### 1. `train.ipynb` - Model Training Pipeline

Trains an XGBoost classifier on deep features extracted from droplet images using MobileNetV2.

**Key Features:**
- Transfer learning with MobileNetV2 (pre-trained on ImageNet)
- Feature caching for efficient repeated runs
- XGBoost multi-class classification
- Comprehensive performance metrics (accuracy, precision, recall, F1-score)
- Optional 3D visualization with t-SNE and UMAP

**Input:**
- Directory structure with class subdirectories containing labeled droplet images:
  ```
  dataset/
  ├── Aggregate/
  ├── Condensate/
  └── Homogenous/
  ```

**Output:**
- Trained XGBoost model (`.pkl` file)
- Feature cache for reuse
- Performance metrics and classification report
- Optional dimensionality reduction visualizations

**Typical Runtime:** 5-20 minutes (depending on dataset size and GPU availability)

---

### 2. `classify.ipynb` - Droplet Classification Pipeline

Automated pipeline for detecting and classifying droplets in multi-channel microscopy images.

**Key Features:**
- Multi-channel quadrant extraction
- Pre-computed bounding box integration
- Batch feature extraction with cached MobileNetV2 (10-50x speedup)
- XGBoost classification
- Annotated visualization generation
- CSV output with classification results

**Input:**
- Multi-channel TIF images
- Pre-computed droplet bounding boxes (`.pkl` files)
- Trained classifier model
- Metadata CSV files

**Output:**
- Classified droplets saved as `.pkl` files
- Annotated debug images with classification labels
- Combined analysis spreadsheet (CSV)

**Performance Optimizations:**
- Model caching eliminates redundant loading (major speedup)
- Adaptive batch sizes (500 for GPU, 100 for CPU)
- Sequential processing optimized for Jupyter notebooks

**Typical Runtime:** 2-10 minutes (with optimizations)

---

### 3. `image-thresholding.ipynb` - Condensate Segmentation

Automated thresholding pipeline for segmenting biomolecular condensates from microscopy images.

**Key Features:**
- Kernel Density Estimation (KDE) for adaptive thresholding
- Morphological opening (erosion + dilation) for noise removal
- Batch processing with progress tracking
- Pipeline visualization for quality control

**Method:**
The KDE approach analyzes pixel intensity distributions to detect histogram tail inflection points, identifying the transition from signal to background.

**Input:**
- Images organized in category folders (Homogenous, Aggregate, Condensate)

**Output:**
- Binary segmented images
- Processing pipeline visualizations

**Typical Runtime:** 1-5 minutes (depending on dataset size)

---

## Installation

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- macOS, Linux, or Windows
- (Optional) NVIDIA GPU with CUDA support for faster processing

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/droplet-classification.git
   cd droplet-classification
   ```

2. **Create conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment:**
   ```bash
   conda activate droplet_classification_env
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

---

## Quick Start

### Training a Classifier

1. Organize your training images into class subdirectories
2. Open `train.ipynb` in Jupyter:
   ```bash
   jupyter notebook train.ipynb
   ```
3. Update the `dataset_path` variable to point to your data
4. Run all cells (Cell → Run All)
5. Trained model will be saved as `xgb_trained_mobilenetv2_*.pkl`

### Classifying New Data

1. Ensure you have:
   - Multi-channel TIF images
   - Pre-computed bounding boxes
   - Trained classifier model
2. Open `classify.ipynb`
3. Update configuration parameters:
   - `dataset_path`
   - `classify_quadrant` mapping
   - Model path
4. Run all cells
5. Results saved in:
   - `droplet_classification_pkl/` - Raw predictions
   - `droplet_classification_im/` - Annotated images
   - `analysis.csv` - Combined spreadsheet

### Image Thresholding

1. Open `image-thresholding.ipynb`
2. Set `root_path` to your image directory
3. Run pipeline visualization cells to verify threshold quality
4. Run batch processing to segment all images

---

### Output Files

**From Training:**
- `xgb_trained_mobilenetv2_max*.pkl` - Trained model
- `feat_cache_klavs_*_mobilenetv2.pkl` - Feature cache
- `model_performance_metrics.png` - Performance visualization (optional)

**From Classification:**
- `droplet_classification_pkl/*.pkl` - Classification results
- `droplet_classification_im/*.tif` - Annotated images
- `analysis.csv` - Complete analysis spreadsheet

**From Thresholding:**
- Segmented binary images in respective category folders

---

### Training Data
```
dataset/
├── Aggregate/
│   ├── image001.png
│   ├── image002.png
│   └── ...
├── Condensate/
│   └── ...
└── Homogenous/
    └── ...
```

### Classification Data
- Multi-channel TIF images
- Corresponding `.pkl` files with bounding boxes
- CSV metadata with columns: `tif`, `frame`, `droplet_id`

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_article_2025,
  title={Your Article Title},
  author={Author Names},
  journal={Nature},
  year={2025},
  volume={XXX},
  pages={XXX-XXX},
  doi={XX.XXXX/xxxxx}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [your.email@institution.edu]
