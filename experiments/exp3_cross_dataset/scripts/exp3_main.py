"""
Experiment 3: FAISS-Based Memory Compression & ANN Search
Replaces exact k-NN with FAISS IVF-PQ (Product Quantization) indexing.
- Uses real MVTec AD data with ResNet50 features
- Tests various PQ bits (4, 8, 16) and nprobe settings (1, 4, 8, 16)
- Measures: accuracy loss, speedup factor, RAM reduction
- Goal: Enable faster, memory-efficient inference
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List
import sys
import time
import json

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'common'))

from exp3_utils import (
    build_exact_knn_index,
    build_faiss_ivfpq_index,
    evaluate_index_performance,
    benchmark_search_speed
)

from common.dataset import MVTecADDataset

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """Extract features from ResNet50 intermediate layers."""
    
    def __init__(self, backbone='resnet50', layers=['layer2', 'layer3']):
        super().__init__()
        self.layers = layers
        
        if backbone == 'resnet50':
            # Use non-pretrained weights by default to avoid download issues; swap to pretrained=True if weights are available.
            model = models.resnet50(pretrained=False)
        elif backbone == 'wide_resnet50_2':
            model = models.wide_resnet50_2(pretrained=False)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Extract layer modules
        self.model_layers = nn.ModuleDict({
            'layer1': model.layer1,
            'layer2': model.layer2,
            'layer3': model.layer3,
            'layer4': model.layer4,
        })
        
        # Initial conv layers
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        
        self.eval()
        
    def forward(self, x):
        """Extract features from specified layers."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        features = {}
        for name, layer in self.model_layers.items():
            x = layer(x)
            if name in self.layers:
                features[name] = x
        
        return features


def extract_features(
    dataloader: DataLoader,
    extractor: FeatureExtractor,
    device: torch.device
) -> np.ndarray:
    """Extract and aggregate features from dataloader."""
    all_features = []
    all_labels = []
    
    extractor.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            
            # Extract multi-layer features
            layer_features = extractor(images)
            
            # Aggregate features from all layers
            aggregated = []
            for layer_name in sorted(layer_features.keys()):
                feat = layer_features[layer_name]
                # Average pool spatial dimensions
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                feat = feat.view(feat.size(0), -1)
                aggregated.append(feat)
            
            # Concatenate all layer features
            combined = torch.cat(aggregated, dim=1)
            all_features.append(combined.cpu().numpy())
            all_labels.append(labels)
    
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    
    # L2 normalize
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)
    
    return features, labels


def evaluate_with_real_labels(
    index: Dict,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    index_type: str,
    train_features: np.ndarray
) -> tuple:
    """Evaluate index performance with real anomaly labels."""
    test_features = test_features.astype(np.float32)
    
    if index_type == 'exact_knn':
        nbrs = index['nbrs']
        distances, _ = nbrs.kneighbors(test_features)
        scores = distances.mean(axis=1)
        memory_mb = (train_features.nbytes + 1000) / (1024 * 1024)
    
    elif index_type == 'faiss_ivfpq':
        faiss_index = index['index']
        distances, _ = faiss_index.search(test_features, k=5)
        scores = distances.mean(axis=1)
        compression = index['nbits'] / 32.0
        memory_mb = (train_features.nbytes * compression) / (1024 * 1024)
    
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Compute AUROC with real labels
    auroc = roc_auc_score(test_labels, scores)
    
    return auroc, memory_mb, scores


def main():
    parser = argparse.ArgumentParser(description='Exp3: FAISS Memory Compression & ANN Search')
    parser.add_argument('--config', default='exp3_config.yaml', help='Path to config file')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    args = parser.parse_args()
    
    # Setup logging
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f'exp3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("=" * 70)
    logger.info("Experiment 3: FAISS-Based Memory Compression & ANN Search")
    logger.info("=" * 70)
    
    # Load config
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    search_methods = config['experiment']['search_methods']
    pq_bits = config['experiment']['faiss_config']['pq_bits']
    nprobe_values = config['experiment']['faiss_config']['nprobe']
    n_clusters = config['experiment']['faiss_config']['n_clusters']
    m = config['experiment']['faiss_config']['m']
    categories = config['experiment']['categories']
    
    logger.info(f"✓ Categories: {categories}")
    logger.info(f"✓ Search methods: {search_methods}")
    logger.info(f"✓ PQ bits: {pq_bits}")
    logger.info(f"✓ Nprobe values: {nprobe_values}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"✓ Device: {device}")
    
    # Initialize feature extractor
    logger.info("\n[1/4] Loading ResNet50 feature extractor...")
    extractor = FeatureExtractor(backbone='resnet50', layers=['layer2', 'layer3'])
    extractor = extractor.to(device)
    
    # Data path
    data_path = Path(__file__).parent.parent / 'data' / 'mvtec_ad'
    logger.info(f"✓ Data path: {data_path}")
    
    all_results = []
    category_summaries = []
    
    # Process each category
    for cat_idx, category in enumerate(categories):
        logger.info("\n" + "=" * 70)
        logger.info(f"Category {cat_idx+1}/{len(categories)}: {category.upper()}")
        logger.info("=" * 70)
        
        try:
            # Load datasets
            logger.info(f"\n[2/4] Loading {category} dataset...")
            train_dataset = MVTecADDataset(
                root_path=str(data_path),
                category=category,
                split='train',
                image_size=224
            )
            
            test_dataset = MVTecADDataset(
                root_path=str(data_path),
                category=category,
                split='test',
                image_size=224
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
            
            logger.info(f"✓ Train samples: {len(train_dataset)}")
            logger.info(f"✓ Test samples: {len(test_dataset)}")
            
            # Extract features
            logger.info(f"\n[3/4] Extracting features for {category}...")
            train_features, _ = extract_features(train_loader, extractor, device)
            test_features, test_labels = extract_features(test_loader, extractor, device)
            
            logger.info(f"✓ Train features: {train_features.shape}")
            logger.info(f"✓ Test features: {test_features.shape}")
            logger.info(f"✓ Anomaly ratio: {test_labels.mean():.2%}")
            
            # Build indices and evaluate
            logger.info(f"\n[4/4] Building indices and evaluating...")
            
            # Baseline: Exact k-NN
            logger.info("\n>>> Exact k-NN (baseline)...")
            exact_index = build_exact_knn_index(train_features, k=5)
            baseline_auroc, baseline_memory, baseline_scores = evaluate_with_real_labels(
                exact_index, test_features, test_labels, 'exact_knn', train_features
            )
            baseline_speed = benchmark_search_speed(exact_index, test_features, n_runs=10)
            
            all_results.append({
                'category': category,
                'search_method': 'exact_knn',
                'pq_bits': None,
                'nprobe': None,
                'image_auroc': round(baseline_auroc, 4),
                'memory_mb': round(baseline_memory, 2),
                'search_time_ms': round(baseline_speed * 1000, 4),
                'speedup': 1.0,
                'memory_reduction': 1.0,
                'accuracy_loss_percent': 0.0
            })
            
            logger.info(f"✓ Baseline: AUROC={baseline_auroc:.4f}, Memory={baseline_memory:.1f}MB, Time={baseline_speed*1000:.2f}ms")
            
            # FAISS IVF-PQ variants
            logger.info("\n>>> FAISS IVF-PQ variants...")
            
            for bits in pq_bits:
                for nprobe in nprobe_values:
                    try:
                        logger.info(f"\n  Testing {bits}-bit, nprobe={nprobe}...")
                        
                        # Build index
                        faiss_index = build_faiss_ivfpq_index(
                            train_features,
                            m=m,
                            nbits=bits,
                            nprobe=nprobe,
                            n_clusters=n_clusters
                        )
                        
                        # Evaluate
                        faiss_auroc, faiss_memory, faiss_scores = evaluate_with_real_labels(
                            faiss_index, test_features, test_labels, 'faiss_ivfpq', train_features
                        )
                        faiss_speed = benchmark_search_speed(faiss_index, test_features, n_runs=10)
                        
                        # Compute metrics
                        speedup = baseline_speed / max(faiss_speed, 1e-6)
                        memory_reduction = baseline_memory / max(faiss_memory, 1e-3)
                        accuracy_loss = max(0, (baseline_auroc - faiss_auroc) * 100)
                        
                        all_results.append({
                            'category': category,
                            'search_method': 'faiss_ivfpq',
                            'pq_bits': bits,
                            'nprobe': nprobe,
                            'image_auroc': round(faiss_auroc, 4),
                            'memory_mb': round(faiss_memory, 2),
                            'search_time_ms': round(faiss_speed * 1000, 4),
                            'speedup': round(speedup, 2),
                            'memory_reduction': round(memory_reduction, 2),
                            'accuracy_loss_percent': round(accuracy_loss, 2)
                        })
                        
                        logger.info(f"    AUROC: {faiss_auroc:.4f} (loss: {accuracy_loss:.2f}%)")
                        logger.info(f"    Memory: {faiss_memory:.1f}MB ({memory_reduction:.1f}x reduction)")
                        logger.info(f"    Speed: {faiss_speed*1000:.2f}ms ({speedup:.1f}x speedup)")
                    
                    except Exception as e:
                        logger.error(f"    Error with {bits}-bit, nprobe={nprobe}: {e}")
                        continue
            
            # Category summary
            cat_results = [r for r in all_results if r['category'] == category]
            baseline = [r for r in cat_results if r['search_method'] == 'exact_knn'][0]
            faiss_results = [r for r in cat_results if r['search_method'] == 'faiss_ivfpq']
            
            if faiss_results:
                best_config = max(faiss_results, key=lambda x: x['image_auroc'])
                category_summaries.append({
                    'category': category,
                    'baseline_auroc': baseline['image_auroc'],
                    'best_faiss_auroc': best_config['image_auroc'],
                    'best_config': f"{best_config['pq_bits']}-bit, nprobe={best_config['nprobe']}",
                    'best_speedup': best_config['speedup'],
                    'best_memory_reduction': best_config['memory_reduction'],
                    'accuracy_loss': best_config['accuracy_loss_percent']
                })
        
        except Exception as e:
            logger.error(f"Error processing {category}: {e}", exc_info=True)
            continue
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'exp3_faiss_ann'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    df = pd.DataFrame(all_results)
    csv_path = output_dir / f'exp3_results_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Detailed results saved to: {csv_path}")
    
    # Save category summaries
    summary_df = pd.DataFrame(category_summaries)
    summary_path = output_dir / f'exp3_category_summary_{timestamp}.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"✓ Category summary saved to: {summary_path}")
    
    # Save configuration
    config_save_path = output_dir / f'exp3_config_{timestamp}.json'
    with open(config_save_path, 'w') as f:
        json.dump({
            'config': config,
            'timestamp': timestamp,
            'device': str(device),
            'categories_processed': categories
        }, f, indent=2)
    logger.info(f"✓ Configuration saved to: {config_save_path}")
    
    # Create visualizations
    logger.info("\n" + "=" * 70)
    logger.info("Creating visualizations...")
    logger.info("=" * 70)
    create_visualizations(df, summary_df, output_dir, timestamp)
    
    # Print final summary
    print_summary(df, summary_df)
    
    logger.info("\n" + "=" * 70)
    logger.info("Experiment 3 completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 70)
    
    return all_results

def create_visualizations(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path, timestamp: str):
    """Create comprehensive visualization plots."""
    
    # Get aggregated metrics across all categories
    exact_df = df[df['search_method'] == 'exact_knn']
    faiss_df = df[df['search_method'] == 'faiss_ivfpq']
    
    if len(faiss_df) == 0:
        logger.warning("No FAISS results to visualize")
        return
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Accuracy vs Speedup (main trade-off)
    ax1 = fig.add_subplot(gs[0, :2])
    for category in df['category'].unique():
        cat_faiss = faiss_df[faiss_df['category'] == category]
        ax1.scatter(cat_faiss['speedup'], cat_faiss['image_auroc'], 
                   label=category, s=80, alpha=0.6)
    
    baseline_auroc = exact_df['image_auroc'].mean()
    ax1.axhline(y=baseline_auroc, color='r', linestyle='--', linewidth=2, label='Exact k-NN baseline')
    ax1.set_xlabel('Speedup Factor', fontsize=11)
    ax1.set_ylabel('Image AUROC', fontsize=11)
    ax1.set_title('Accuracy vs Speedup Trade-off (All Categories)', fontsize=12, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Category-wise AUROC comparison
    ax2 = fig.add_subplot(gs[0, 2])
    categories = summary_df['category'].tolist()
    x_pos = np.arange(len(categories))
    baseline_aurocs = summary_df['baseline_auroc'].values
    best_aurocs = summary_df['best_faiss_auroc'].values
    
    width = 0.35
    ax2.bar(x_pos - width/2, baseline_aurocs, width, label='Exact k-NN', color='steelblue')
    ax2.bar(x_pos + width/2, best_aurocs, width, label='Best FAISS', color='coral')
    ax2.set_xlabel('Category', fontsize=10)
    ax2.set_ylabel('AUROC', fontsize=10)
    ax2.set_title('AUROC by Category', fontsize=11, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Memory reduction by PQ bits
    ax3 = fig.add_subplot(gs[1, 0])
    pq_bits = sorted(faiss_df['pq_bits'].unique())
    for bits in pq_bits:
        bits_data = faiss_df[faiss_df['pq_bits'] == bits].groupby('nprobe')['memory_reduction'].mean()
        ax3.plot(bits_data.index, bits_data.values, marker='o', label=f'{bits}-bit', linewidth=2)
    ax3.set_xlabel('Nprobe', fontsize=10)
    ax3.set_ylabel('Memory Reduction Factor', fontsize=10)
    ax3.set_title('Memory Reduction vs Nprobe', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Speedup by PQ bits
    ax4 = fig.add_subplot(gs[1, 1])
    for bits in pq_bits:
        bits_data = faiss_df[faiss_df['pq_bits'] == bits].groupby('nprobe')['speedup'].mean()
        ax4.plot(bits_data.index, bits_data.values, marker='s', label=f'{bits}-bit', linewidth=2)
    ax4.set_xlabel('Nprobe', fontsize=10)
    ax4.set_ylabel('Speedup Factor', fontsize=10)
    ax4.set_title('Speedup vs Nprobe', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Accuracy loss by PQ bits
    ax5 = fig.add_subplot(gs[1, 2])
    for bits in pq_bits:
        bits_data = faiss_df[faiss_df['pq_bits'] == bits].groupby('nprobe')['accuracy_loss_percent'].mean()
        ax5.plot(bits_data.index, bits_data.values, marker='^', label=f'{bits}-bit', linewidth=2)
    ax5.set_xlabel('Nprobe', fontsize=10)
    ax5.set_ylabel('Accuracy Loss (%)', fontsize=10)
    ax5.set_title('Accuracy Loss vs Nprobe', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Search time heatmap
    ax6 = fig.add_subplot(gs[2, 0])
    pivot_time = faiss_df.groupby(['nprobe', 'pq_bits'])['search_time_ms'].mean().unstack()
    im1 = ax6.imshow(pivot_time, cmap='YlOrRd', aspect='auto')
    ax6.set_xlabel('PQ Bits', fontsize=10)
    ax6.set_ylabel('Nprobe', fontsize=10)
    ax6.set_xticks(range(len(pivot_time.columns)))
    ax6.set_xticklabels(pivot_time.columns, fontsize=9)
    ax6.set_yticks(range(len(pivot_time.index)))
    ax6.set_yticklabels(pivot_time.index, fontsize=9)
    ax6.set_title('Search Time Heatmap (ms)', fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=ax6)
    
    # 7. Memory usage heatmap
    ax7 = fig.add_subplot(gs[2, 1])
    pivot_memory = faiss_df.groupby(['nprobe', 'pq_bits'])['memory_mb'].mean().unstack()
    im2 = ax7.imshow(pivot_memory, cmap='Blues', aspect='auto')
    ax7.set_xlabel('PQ Bits', fontsize=10)
    ax7.set_ylabel('Nprobe', fontsize=10)
    ax7.set_xticks(range(len(pivot_memory.columns)))
    ax7.set_xticklabels(pivot_memory.columns, fontsize=9)
    ax7.set_yticks(range(len(pivot_memory.index)))
    ax7.set_yticklabels(pivot_memory.index, fontsize=9)
    ax7.set_title('Memory Usage Heatmap (MB)', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax7)
    
    # 8. Speedup heatmap
    ax8 = fig.add_subplot(gs[2, 2])
    pivot_speedup = faiss_df.groupby(['nprobe', 'pq_bits'])['speedup'].mean().unstack()
    im3 = ax8.imshow(pivot_speedup, cmap='YlGn', aspect='auto')
    ax8.set_xlabel('PQ Bits', fontsize=10)
    ax8.set_ylabel('Nprobe', fontsize=10)
    ax8.set_xticks(range(len(pivot_speedup.columns)))
    ax8.set_xticklabels(pivot_speedup.columns, fontsize=9)
    ax8.set_yticks(range(len(pivot_speedup.index)))
    ax8.set_yticklabels(pivot_speedup.index, fontsize=9)
    ax8.set_title('Speedup Heatmap', fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax8)
    
    plt.suptitle('Experiment 3: FAISS IVF-PQ Analysis (Real MVTec Data)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plot_path = output_dir / f'exp3_faiss_analysis_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Visualization saved to: {plot_path}")
    plt.close()


def print_summary(df: pd.DataFrame, summary_df: pd.DataFrame):
    """Print experiment summary."""
    exact_df = df[df['search_method'] == 'exact_knn']
    faiss_df = df[df['search_method'] == 'faiss_ivfpq']
    
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3 SUMMARY")
    logger.info("=" * 70)
    
    logger.info(f"\nCategories tested: {', '.join(df['category'].unique())}")
    logger.info(f"Total configurations: {len(faiss_df)} FAISS variants per category")
    
    logger.info(f"\n{'Category':<15} {'Baseline AUROC':<15} {'Best FAISS':<15} {'Speedup':<10} {'Mem Reduction':<15}")
    logger.info("-" * 70)
    
    for _, row in summary_df.iterrows():
        logger.info(f"{row['category']:<15} {row['baseline_auroc']:<15.4f} "
                   f"{row['best_faiss_auroc']:<15.4f} {row['best_speedup']:<10.1f}x "
                   f"{row['best_memory_reduction']:<15.1f}x")
    
    logger.info("\nOverall Statistics:")
    logger.info(f"  Average baseline AUROC: {exact_df['image_auroc'].mean():.4f}")
    logger.info(f"  Average best FAISS AUROC: {summary_df['best_faiss_auroc'].mean():.4f}")
    logger.info(f"  Average accuracy loss: {summary_df['accuracy_loss'].mean():.2f}%")
    logger.info(f"  Average speedup: {summary_df['best_speedup'].mean():.1f}x")
    logger.info(f"  Average memory reduction: {summary_df['best_memory_reduction'].mean():.1f}x")
    
    # Best overall configuration
    best_row = faiss_df.loc[faiss_df['image_auroc'].idxmax()]
    logger.info(f"\nBest Overall Configuration:")
    logger.info(f"  Category: {best_row['category']}")
    logger.info(f"  Config: {best_row['pq_bits']:.0f}-bit, nprobe={best_row['nprobe']:.0f}")
    logger.info(f"  AUROC: {best_row['image_auroc']:.4f}")
    logger.info(f"  Speedup: {best_row['speedup']:.1f}x")
    logger.info(f"  Memory reduction: {best_row['memory_reduction']:.1f}x")
    
    # Fastest configuration
    fastest_row = faiss_df.loc[faiss_df['speedup'].idxmax()]
    logger.info(f"\nFastest Configuration:")
    logger.info(f"  Category: {fastest_row['category']}")
    logger.info(f"  Config: {fastest_row['pq_bits']:.0f}-bit, nprobe={fastest_row['nprobe']:.0f}")
    logger.info(f"  Speedup: {fastest_row['speedup']:.1f}x")
    logger.info(f"  AUROC: {fastest_row['image_auroc']:.4f}")
    logger.info(f"  Accuracy loss: {fastest_row['accuracy_loss_percent']:.2f}%")
    
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
