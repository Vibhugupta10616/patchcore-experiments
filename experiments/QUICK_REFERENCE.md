# Quick Reference Guide: PatchCore Experiments

## 🚀 Quick Start (60 seconds)

### 1. Set Dataset Path
Edit the YAML files and update `data_path`:
```yaml
data_config:
  data_path: "/path/to/mvtec_ad"  # Change this!
```

### 2. Run Experiment
```bash
cd experiments/exp1_backbone_comparison
python exp1_main.py --config exp1_config.yaml
```

### 3. Check Results
Results saved to `./results/exp1_backbone_comparison/`
- `results.csv` - Numerical results
- `comparison.png` - Visualization

---

## 📋 Experiment Quick Reference

| Experiment | Focus | Key Files | Run Command |
|-----------|-------|-----------|------------|
| **Exp 1** | Backbone Comparison | exp1_main.py, exp1_config.yaml | `python exp1_main.py` |
| **Exp 2** | Domain Shift | exp2_main.py, exp2_config.yaml | `python exp2_main.py` |
| **Exp 3** | Fusion Strategies | exp3_main.py, exp3_config.yaml | `python exp3_main.py` |

---

## 🎯 What Each Experiment Does

### Experiment 1: Backbone Comparison
```
ResNet50 vs ViT-B/16 vs DINOv2 vs CLIP
         ↓
Compare AUROC on all MVTec categories
         ↓
Find best architecture
```

### Experiment 2: Cross-Domain Generalization
```
Train Category A → Test on Category B
         ↓
Measure performance drop
         ↓
Identify robust features
```

### Experiment 3: Feature Fusion
```
Single Layer vs Concatenation vs Weighted vs Adaptive
         ↓
Compare fusion impact on AUROC
         ↓
Find best fusion strategy
```

---

## 🔧 Key Configuration Parameters

### Backbone Selection
```yaml
# exp1_config.yaml
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

### Fusion Strategies
```yaml
# exp3_config.yaml
experiment:
  fusion_strategies:
    - "single_layer"      # Baseline
    - "concatenation"     # All layers concat
    - "weighted"          # Manual weights
    - "adaptive"          # Learned weights
```

### Memory Bank Methods
```yaml
memory_config:
  method: "knn"         # Options: knn, pca, kmeans
  n_neighbors: 5
  n_components: 256
```

---

## 📊 Expected Results

### Experiment 1: Backbone AUROC (Image-level)
```
ResNet50:    ~95.0%
ViT-B/16:    ~96.5%
DINOv2:      ~97.5%
CLIP:        ~97.3%
```

### Experiment 2: Domain Shift Impact
```
In-Domain Performance:     ~95%
Same-Domain (baseline):    ~95%
Close-Domain:              ~88%
Far-Domain:                ~78%
Domain Shift Distance:     0.1-0.8 (MMD)
```

### Experiment 3: Fusion Impact
```
Single Layer:      ~95.0% (AUROC)
Concatenation:     ~96.2% (+1.2%)
Weighted:          ~96.8% (+1.8%)
Adaptive:          ~97.2% (+2.2%)
Feature Dim:       512→4096→2048
```

---

## 🛠️ Common Modifications

### Change Image Size
```yaml
data_config:
  image_size: 256      # Change from 224
```

### Adjust Batch Size
```yaml
data_config:
  batch_size: 64       # Increase for GPU memory
```

### Test Specific Categories
```yaml
experiment:
  categories:
    - "bottle"
    - "cable"
    - "capsule"
    # Add/remove as needed
```

### Custom Fusion Weights
```yaml
fusion_config:
  weights: [0.1, 0.3, 0.6]  # Layer 2, 3, 4 weights
```

---

## 📈 Output Files

### CSV Results
```
results.csv contains:
- backbone/strategy/train_category
- image_auroc
- pixel_auroc
- pixel_pr (precision-recall AUC)
- feature_dim (for Exp 3)
- domain_shift_distance (for Exp 2)
```

### Visualization Files
```
comparison.png         - Bar charts comparing methods
domain_shift_analysis.png  - Scatter plots of domain effects
fusion_comparison.png  - Feature dimension vs performance
heatmaps/             - Individual anomaly visualizations
```

---

## 🐛 Troubleshooting

### Error: "Module not found"
```python
# Make sure path is correct in import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

### CUDA Out of Memory
```yaml
data_config:
  batch_size: 16        # Reduce batch size
backbone_config:
  pretrained: true      # Use smaller model
```

### Dataset Not Found
```yaml
data_config:
  data_path: "/actual/path/to/mvtec_ad"  # Verify path exists
```

---

## 🎓 Understanding Metrics

### AUROC (Area Under ROC Curve)
- Higher is better (0-1 range)
- 0.5 = random, 1.0 = perfect
- Image-level: Detects if image is anomalous
- Pixel-level: Localizes exactly where anomaly is

### PR-AUC (Precision-Recall AUC)
- Better for imbalanced data
- Higher is better (0-1 range)
- Useful when false positives are expensive

### Domain Shift Distance (MMD)
- Measures difference between domains
- Higher = more different
- Explains performance drop in cross-domain tests

### Feature Fusion Weights
- Adaptive: Automatically learned from data variance
- Weighted: Manual specification for control
- Concatenation: Simple but high-dimensional
- Single: Baseline, low-dimensional

---

## 🔄 Workflow for Analysis

### Step 1: Run Experiments
```bash
python exp1_main.py --config exp1_config.yaml
python exp2_main.py --config exp2_config.yaml
python exp3_main.py --config exp3_config.yaml
```

### Step 2: Analyze Results
```python
import pandas as pd

# Load results
df1 = pd.read_csv('results/exp1_backbone_comparison/results.csv')
df2 = pd.read_csv('results/exp2_cross_domain/results.csv')
df3 = pd.read_csv('results/exp3_feature_fusion/results.csv')

# Find best backbone
print(df1.groupby('backbone')['image_auroc'].mean().idxmax())

# Analyze domain shift
print(df2[df2['train_category'] != df2['test_category']]['image_auroc'].describe())

# Compare fusion strategies
print(df3.groupby('strategy')['image_auroc'].mean())
```

### Step 3: Create Report
Use the visualization files and CSV data to create:
- Performance comparison tables
- Domain shift analysis plots
- Fusion strategy recommendations

---

## 📞 Key Functions Reference

### exp1_utils.py
```python
get_backbone(name, pretrained)      # Load backbone
extract_features(model, dataloader)  # Extract features
prepare_memory_bank(features, method) # Prepare memory
compute_anomaly_scores(features, memory) # Score
```

### exp2_utils.py
```python
evaluate_domain_shift(src, tgt, method) # Domain metrics
```

### exp3_utils.py
```python
fuse_features_single_layer(features)  # Single layer
fuse_features_concatenation(features) # Concat fusion
fuse_features_weighted(features, weights) # Weighted
fuse_features_adaptive(features)      # Adaptive
```

### common/eval.py
```python
evaluate_auroc(scores, dataloader)    # Image AUROC
evaluate_localization(maps, dataloader) # Pixel metrics
compute_pro_score(maps, mask)         # PRO score
compute_f1_score(scores, labels)      # F1 score
```

### common/viz.py
```python
plot_results(results, output_path)    # Generic plot
save_heatmaps(images, maps)           # Save visualizations
plot_roc_curve(labels, scores)        # ROC curves
plot_pr_curve(precision, recall)      # PR curves
```

---

## 💡 Pro Tips

1. **Use smaller categories first** for quick testing
2. **Monitor GPU memory** - adjust batch size as needed
3. **Save configurations** for reproducible results
4. **Compare baselines** - run standard configs first
5. **Check logs** - verbose mode shows detailed progress
6. **Reuse memory banks** - don't recompute if unchanged

---

## 📚 References

- **PatchCore**: [Paper](https://arxiv.org/abs/2106.08265)
- **ViT**: [Paper](https://arxiv.org/abs/2010.11929)
- **CLIP**: [Paper](https://arxiv.org/abs/2103.14030)
- **MVTec AD**: [Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

**Last Updated:** January 2026
**Status:** ✅ All experiments ready to run

