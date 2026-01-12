# Experiment 1: Dataset Documentation

## Overview
This experiment uses two publicly available anomaly detection datasets to evaluate backbone architectures.

## Datasets Used

### 1. MVTec AD (In-Domain Testing)
- **Type**: Real industrial anomaly detection dataset
- **Categories**: 5 object classes
  - bottle
  - cable
  - hazelnut
  - leather
  - screw
- **Purpose**: In-domain evaluation with normal training and anomaly testing
- **Location**: `../data/mvtec_ad/`

### 2. VisA (Cross-Domain Testing)
- **Type**: Real-world anomaly detection dataset (diverse categories)
- **Categories**: Multiple object classes (candle, cashew, capsules, macaroni, pcb, etc.)
- **Purpose**: Zero-shot cross-domain evaluation to assess generalization
- **Location**: `../data/visa/`

## Data Structure
```
data/
├── mvtec_ad/
│   ├── bottle/
│   ├── cable/
│   ├── hazelnut/
│   ├── leather/
│   └── screw/
├── visa/
│   ├── candle/
│   ├── capsules/
│   ├── cashew/
│   ├── macaroni1/
│   └── pcb1/
└── visa_pytorch/
    └── 1cls/
```

## Splits
- **Training**: Normal images only
- **Testing**: Mixed normal and anomaly images
- **In-Domain**: MVTec AD (seen categories)
- **Cross-Domain**: VisA (unseen categories - zero-shot)

## Data Preparation
Run the data preparation script:
```bash
python ../data/prepare_data.py
```

This script handles downloading and organizing both datasets for experiments.
