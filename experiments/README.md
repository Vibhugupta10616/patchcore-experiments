# PatchCore Experiments

This directory contains three comprehensive experiments to analyze and improve the PatchCore anomaly detection framework.

## Overview

### Experiment 1: CLIP / Vision Transformer Embeddings
**Motivation:** CNNs are local and texture-biased, while Vision Transformers and CLIP capture global semantic relationships.

**Objective:** Compare different backbone architectures for anomaly detection:
- ResNet50 (CNN baseline)
- ViT-B/16 (Vision Transformer)
- DINOv2 ViT-B/14 (Self-supervised)
- CLIP ViT-B/32 (Vision-Language model)

**Metrics:**
- Image-level AUROC
- Pixel-level localization performance

**Expected Outcome:** Better cross-category and cross-domain generalization with transformer-based backbones.

**Run:**
```bash
cd exp1_backbone_comparison
python exp1_main.py --config exp1_config.yaml
```

---

### Experiment 2: Cross-Domain Generalization Study
**Motivation:** Industrial anomaly detectors often fail under domain shifts. Understanding robustness is critical for real-world deployment.

**Objective:** Systematically test domain shift effects:
- Train on one product category
- Test on different categories (domain shift)
- Measure performance degradation
- Analyze which features are robust

**Metrics:**
- In-domain AUROC (baseline)
- Cross-domain AUROC (target)
- Domain shift distance (MMD, Wasserstein, Cosine)
- Feature drift

**Expected Outcome:** Identify robust feature representations across domains and quantify generalization gaps.

**Run:**
```bash
cd exp2_memory_ablation
python exp2_main.py --config exp2_config.yaml
```

---

### Experiment 3: Feature Fusion Strategy Ablation
**Motivation:** Different network layers encode different types of information. Proper fusion can improve both detection and localization.

**Objective:** Compare feature fusion strategies:
1. **Single-layer**: Use only one deep layer (baseline)
2. **Concatenation**: Concatenate all layer features
3. **Weighted**: Manual weighted combination of layers
4. **Adaptive**: Weights learned from feature variance

**Metrics:**
- Image-level AUROC
- Pixel-level localization AUROC
- Feature dimension vs performance trade-off

**Expected Outcome:** Demonstrate that intelligent fusion improves performance while potentially reducing dimensionality.

**Run:**
```bash
cd exp3_cross_dataset
python exp3_main.py --config exp3_config.yaml
```

---

## Directory Structure

```
experiments/
├── exp1_backbone_comparison/
│   ├── exp1_main.py          # Main experiment script
│   ├── exp1_utils.py         # Utility functions
│   └── exp1_config.yaml      # Configuration
│
├── exp2_memory_ablation/
│   ├── exp2_main.py
│   ├── exp2_utils.py
│   └── exp2_config.yaml
│
├── exp3_cross_dataset/
│   ├── exp3_main.py
│   ├── exp3_utils.py
│   └── exp3_config.yaml
│
├── common/
│   ├── dataset.py            # MVTec AD dataset loader
│   ├── eval.py               # Evaluation metrics (AUROC, F1, etc.)
│   └── viz.py                # Visualization utilities
│
└── README.md                 # This file
```

---

## Common Utilities

### `common/dataset.py`
Handles MVTec AD dataset loading:
- `MVTecADDataset`: Custom PyTorch dataset class
- `load_mvtec_dataset()`: Load train/test splits
- `create_dataloaders()`: Create DataLoaders for experiments

### `common/eval.py`
Evaluation metrics:
- `evaluate_auroc()`: Image-level AUROC
- `evaluate_localization()`: Pixel-level AUROC and PR-AUC
- `compute_pro_score()`: Per-Region-Overlap score
- `compute_f1_score()`: F1 score at optimal threshold
- `compute_auc_pr()`: Area under Precision-Recall curve

### `common/viz.py`
Visualization functions:
- `plot_results()`: Generic result plotting
- `save_heatmaps()`: Save anomaly heatmaps
- `plot_roc_curve()`: ROC curve visualization
- `plot_pr_curve()`: Precision-Recall curve visualization
- `visualize_anomaly_localization()`: Blend anomaly map with image

---

## Configuration Files

Each experiment has a YAML config file controlling:

**exp1_config.yaml:**
- Backbone architectures to test
- Feature layer selection
- Memory bank method (KNN, PCA, KMeans)
- Dataset path and parameters

**exp2_config.yaml:**
- Category list for cross-domain testing
- Domain shift evaluation method
- Feature extraction layers
- Memory bank configuration

**exp3_config.yaml:**
- Fusion strategies to compare
- Backbone and layers
- Fusion-specific parameters (weights, etc.)
- Output paths

---

## Expected Results

### Experiment 1: Backbone Comparison
Expected AUROC improvements (MVTec AD average):
- ResNet50: ~95%
- ViT-B/16: ~96-97%
- DINOv2: ~97-98%
- CLIP: ~97-98%

### Experiment 2: Domain Shift
Expected performance degradation:
- In-domain accuracy: ~95%
- Cross-domain accuracy: ~75-85% (depends on domain similarity)

### Experiment 3: Feature Fusion
Expected improvements:
- Single layer: Baseline (~95%)
- Concatenation: +1-2% AUROC
- Weighted: +1.5-2.5% AUROC
- Adaptive: +2-3% AUROC

---

## Usage Examples

### Running a Single Experiment
```bash
cd experiments/exp1_backbone_comparison
python exp1_main.py --config exp1_config.yaml --log-level INFO
```

### Running All Experiments
```bash
python exp1_backbone_comparison/exp1_main.py --config exp1_backbone_comparison/exp1_config.yaml
python exp2_memory_ablation/exp2_main.py --config exp2_memory_ablation/exp2_config.yaml
python exp3_cross_dataset/exp3_main.py --config exp3_cross_dataset/exp3_config.yaml
```

### Custom Configuration
Edit the YAML files to:
- Change dataset path
- Select specific categories
- Modify backbones
- Adjust batch size and hyperparameters

---

## Dependencies

Required packages (already installed):
- torch >= 1.10.0
- torchvision >= 0.11.1
- scikit-learn >= 1.0.1
- scikit-image >= 0.18.3
- matplotlib >= 3.5.0
- numpy >= 1.21.0
- pyyaml

Optional for advanced features:
- timm (for additional backbones)
- clip (for CLIP models)
- dinov2 (for DINOv2 models)

---

## Output Structure

Results are saved in `./results/` directory:

```
results/
├── exp1_backbone_comparison/
│   ├── results.csv           # Numerical results
│   └── comparison.png        # Visualization
│
├── exp2_cross_domain/
│   ├── results.csv
│   └── domain_shift_analysis.png
│
└── exp3_feature_fusion/
    ├── results.csv
    └── fusion_comparison.png
```

---

## Tips for Customization

1. **Add New Backbones:** Edit `get_backbone()` in utils files
2. **New Fusion Strategies:** Implement new functions in `exp3_utils.py`
3. **Different Metrics:** Add functions to `common/eval.py`
4. **Custom Datasets:** Extend `MVTecADDataset` in `common/dataset.py`

---

## References

- PatchCore: https://arxiv.org/abs/2106.08265
- Vision Transformer: https://arxiv.org/abs/2010.11929
- CLIP: https://arxiv.org/abs/2103.14030
- DINOv2: https://arxiv.org/abs/2304.07193
- MVTec AD Dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad

---

## Contact & Questions

For questions about the experiments, check the docstrings in each file or refer to the configuration comments.

