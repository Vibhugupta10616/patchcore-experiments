# Experiment 1: Backbone Comparison (ResNet50 vs DINOv2)

## Overview

Evaluates different backbone architectures for anomaly detection using **PatchCore** on real MVTec AD and VisA datasets. Demonstrates that modern self-supervised models (DINOv2) outperform traditional CNNs (ResNet50) in both in-domain and cross-domain anomaly detection.

**Key Questions:**
- How do self-supervised Vision Transformers (DINOv2) compare to CNNs (ResNet50)?
- What is the generalization gap between in-domain and cross-domain performance?
- Does DINOv2's better representation learning lead to more robust anomaly detection?

---

## Experiment Configuration

### Backbones Tested
1. **ResNet50** - CNN baseline (2048-D features, 49 patches per image)
2. **DINOv2 ViT-B/14** - Self-supervised Vision Transformer (768-D features, 256 patches per image)

### Datasets
- **In-Domain (MVTec AD)**: bottle, cable, hazelnut, leather, screw
- **Cross-Domain (VisA)**: candle, cashew, chewinggum, frito (zero-shot)

### Evaluation Metrics
- **Image-level AUROC**: Anomaly detection performance per image (0.0-1.0)
- **In-Domain Performance**: Train on normal MVTec images, test on mixed normal/anomaly
- **Cross-Domain Performance**: Zero-shot evaluation on unseen VisA categories
- **Generalization Gap**: Difference between in-domain and cross-domain (lower is better)

---

## Results

### Performance Comparison (2026-01-11)

| Backbone | In-Domain (MVTec) | Cross-Domain (VisA) | Gap | Winner |
|----------|------------------|---------------------|-----|--------|
| **DINOv2** | **0.9691** | **0.8990** | **0.0701** | ✅ |
| ResNet50 | 0.9392 | 0.8390 | 0.1002 | Baseline |

### Key Findings

🏆 **DINOv2 Superior Performance:**
- **In-Domain**: +2.99% advantage (better feature learning)
- **Cross-Domain**: +6.0% advantage (self-supervised pre-training transfers better)
- **Generalization Gap**: 30% smaller (0.0701 vs 0.1002) - more robust to domain shifts

**Why DINOv2 Wins:**
1. Self-supervised pre-training learns semantic features without labels
2. Vision Transformer architecture captures global context better than CNNs
3. Better generalization to unseen object categories (critical for real-world)

---

## File Descriptions

| File | Purpose |
|------|---------|
| `scripts/exp1_main.py` | Main orchestrator: loads config, runs backbones, generates visualizations |
| `scripts/exp1_utils.py` | Core functions: backbone loading, memory bank construction, evaluation |
| `scripts/exp1_config.yaml` | Configuration: backbones, datasets, paths, device settings |
| `logs/exp1_realmodels_cpu.log` | Experiment execution log with timing and per-category scores |
| `results/results_all_backbones.csv` | Combined CSV results (backbone, dataset, category, AUROC) |
| `visualizations/exp1_backbone_comparison.png` | 4-panel comparison: MVTec, VisA, per-category, gap analysis |

---

## Technical Summary

**PatchCore Method:**
- Extract patch-level features from backbone
- Build memory bank using k-center coreset (5% of patches)
- For test images: extract features → find k-NN (k=5) → compute anomaly scores → image-level AUROC

**Feature Properties:**
- ResNet50: 2048-D vectors, 7×7 spatial resolution, 49 patches/image
- DINOv2: 768-D vectors, 16×16 spatial resolution, 256 patches/image

---

## Performance Interpretation

**AUROC Scale:**
- 0.90+: Excellent | 0.80-0.90: Good | 0.70-0.80: Fair | <0.70: Poor

**Generalization Gap Analysis:**
- DINOv2 gap (0.0701): Only 7% performance drop on unseen domains
- ResNet50 gap (0.1002): 10% performance drop (42% worse generalization)
- **Insight**: DINOv2 learns more domain-agnostic features

---

**Status**: ✅ Complete with real models and real data  
**Last Updated**: January 11, 2026
