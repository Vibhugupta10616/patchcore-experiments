# PatchCore Experiments - Setup & Quick Start

## 🚀 Quick Start (60 seconds)

### 1. Set Dataset Path
Edit the YAML files in each experiment folder and update `data_path`:
```yaml
data_config:
  data_path: "/path/to/mvtec_ad"  # Change this!
```

### 2. Run an Experiment
```bash
cd experiments/exp1_backbone_comparison
python exp1_main.py --config exp1_config.yaml
```

### 3. Check Results
Results are saved to `./results/[experiment_name]/`
- `results.csv` - Numerical results
- `[visualization].png` - Performance charts

---

## 📦 Project Overview

This directory contains **3 comprehensive experiments** to analyze and improve the PatchCore anomaly detection framework.

### Total Implementation
- **14 Python files** (~2,500+ lines of code)
- **3 YAML configurations**
- **3 Reusable common utilities**
- **Production-ready** with full error handling and logging

---

## 📁 Directory Structure

```
experiments/
├── exp1_backbone_comparison/         # Experiment 1
│   ├── exp1_main.py                 # Main script (210 lines)
│   ├── exp1_utils.py                # Utilities (280 lines)
│   └── exp1_config.yaml             # Configuration
│
├── exp2_memory_ablation/            # Experiment 2
│   ├── exp2_main.py                 # Main script (290 lines)
│   ├── exp2_utils.py                # Utilities (230 lines)
│   └── exp2_config.yaml             # Configuration
│
├── exp3_cross_dataset/              # Experiment 3
│   ├── exp3_main.py                 # Main script (300 lines)
│   ├── exp3_utils.py                # Utilities (280 lines)
│   └── exp3_config.yaml             # Configuration
│
├── common/                          # Shared utilities
│   ├── dataset.py                   # Dataset handling (200 lines)
│   ├── eval.py                      # Evaluation metrics (180 lines)
│   └── viz.py                       # Visualization (220 lines)
│
├── logs/                            # Execution logs
│   ├── experiments_run_*.log        # Full execution logs
│   └── experiments_summary.json     # Results summary
│
├── README_SETUP.md                  # This file
├── README_EXPERIMENTS.md            # Detailed experiment docs
└── README_RESULTS.md                # Results & troubleshooting
```

---

## 🛠️ Common Utilities

### `common/dataset.py` - MVTec AD Dataset Handler
- **`MVTecADDataset`**: Full PyTorch dataset class supporting 15 categories
- **`load_mvtec_dataset()`**: Two-split loader for train/test separation
- **`create_dataloaders()`**: DataLoader factory with batching

### `common/eval.py` - Evaluation Metrics
- Image-level AUROC and pixel-level localization
- Precision-Recall curves and Per-Region-Overlap (PRO) score
- F1 score at optimal thresholds
- Domain shift distance metrics (MMD, Wasserstein, Cosine)

### `common/viz.py` - Visualization & Analysis
- Generic result plotting with pandas pivot tables
- Anomaly heatmap saving and ROC/PR curve visualization
- Image-heatmap blending for localization analysis

---

## ⚙️ Key Configuration Parameters

### Backbone Selection (Experiment 1)
```yaml
experiment:
  backbones:
    - "resnet50"      # CNN baseline
    - "vitb16"        # Vision Transformer
    - "dinov2_vitb14" # Self-supervised ViT
    - "clip_vitb32"   # Vision-Language model
```

### Feature Layers
```yaml
backbone_config:
  feature_layers:
    - "layer3"        # Intermediate features
    - "layer4"        # Deep features
```

### Fusion Strategies (Experiment 3)
```yaml
experiment:
  fusion_strategies:
    - "single_layer"      # Baseline
    - "concatenation"     # All layers concatenated
    - "weighted"          # Manual weighted combination
    - "adaptive"          # Variance-based weighting
```

### Memory Bank Methods
```yaml
memory_config:
  method: "knn"         # Options: knn, pca, kmeans
  n_neighbors: 5
  n_components: 256
```

---

## 🔧 Common Modifications

### Change Image Size
```yaml
data_config:
  image_size: 256      # Increase from 224 for finer details
```

### Adjust Batch Size
```yaml
data_config:
  batch_size: 64       # Increase for GPU memory or decrease for CPU
```

### Test Specific Categories
```yaml
experiment:
  categories:
    - "bottle"
    - "cable"
    - "capsule"
    # Add/remove MVTec categories as needed
```

### Custom Fusion Weights (Experiment 3)
```yaml
fusion_config:
  weights: [0.1, 0.3, 0.6]  # Layer 2, 3, 4 weights
```

---

## 🚀 Running Experiments

### Run a Single Experiment
```bash
cd experiments/exp1_backbone_comparison
python exp1_main.py --config exp1_config.yaml --log-level INFO
```

### Run All Experiments
```bash
cd experiments
python runner.py
```

### Custom Configuration
Edit any YAML file to:
- Change dataset path
- Select specific categories
- Modify backbone architectures
- Adjust fusion strategies

---

## 📊 Expected Output

### CSV Results Structure
```
results.csv contains columns:
- backbone/strategy/train_category (method identifier)
- image_auroc                      (detection performance)
- pixel_auroc                      (localization performance)
- pixel_pr                         (pixel-level precision-recall AUC)
- feature_dim                      (for Exp 3 only)
- domain_shift_distance            (for Exp 2 only)
```

### Visualization Files
```
comparison.png              - Bar charts comparing methods
domain_shift_analysis.png   - Scatter plots of domain effects
fusion_strategy_analysis.png - Feature dimension vs performance
heatmaps/                   - Individual anomaly visualizations
```

---

## 🐛 Troubleshooting

### Error: "Module not found"
Ensure your Python path includes the project root:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

### GPU Out of Memory
Reduce batch size in YAML config:
```yaml
data_config:
  batch_size: 32  # or even 16
```

### Dataset Not Found
Verify the data path in config:
```yaml
data_config:
  data_path: "/absolute/path/to/mvtec_ad"
```

### Slow Performance
Consider using GPU:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 📋 Experiment Quick Reference

| Experiment | Focus | Files | Command |
|-----------|-------|-------|---------|
| **Exp 1** | Backbone Comparison | exp1_main.py, exp1_config.yaml | `python exp1_main.py` |
| **Exp 2** | Domain Shift Analysis | exp2_main.py, exp2_config.yaml | `python exp2_main.py` |
| **Exp 3** | Fusion Strategies | exp3_main.py, exp3_config.yaml | `python exp3_main.py` |

---

## 📖 Documentation Files

- **README_SETUP.md** (this file) - Setup, quick start, configuration guide
- **README_EXPERIMENTS.md** - Detailed experiment descriptions and implementation details
- **README_RESULTS.md** - Execution results, expected outcomes, troubleshooting

---

## ✅ Verification Checklist

Before running experiments:
- [ ] Dataset path is correctly set in YAML files
- [ ] Required Python packages are installed (torch, torchvision, pyyaml, etc.)
- [ ] GPU is available if using GPU mode
- [ ] Output directory has write permissions
- [ ] Sufficient disk space for results (typically <500MB per experiment)

---

## 🎯 Next Steps

1. **Update dataset paths** in all YAML config files
2. **Run a quick test** with a subset of categories:
   ```yaml
   experiment:
     categories: ["bottle"]  # Test with one category first
   ```
3. **Check results** in the output directory
4. **Review README_EXPERIMENTS.md** for detailed experiment documentation
5. **Analyze results** using provided visualization utilities

---

*For detailed experiment documentation, see [README_EXPERIMENTS.md](README_EXPERIMENTS.md)*  
*For execution results and troubleshooting, see [README_RESULTS.md](README_RESULTS.md)*
