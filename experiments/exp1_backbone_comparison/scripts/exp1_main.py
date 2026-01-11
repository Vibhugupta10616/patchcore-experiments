"""
Experiment 1: Replace Backbone with CLIP-ViT or DINOv2
Tests whether modern self-supervised/CLIP representations improve anomaly detection.
- Replaces ResNet50 with CLIP-ViT-B/16 or DINOv2 ViT-B/14
- Evaluates in-domain (MVTec AD) and cross-domain zero-shot (VisA)
- Keeps PatchCore unchanged otherwise
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List
import sys

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from exp1_utils import (
    load_backbone_with_patchcore,
    evaluate_on_mvtec_with_memory_bank,
    evaluate_on_visa_with_memory_bank
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Exp1: Backbone Replacement (CLIP/DINOv2)')
    parser.add_argument('--config', default='exp1_config.yaml', help='Path to config file')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 70)
    logger.info("Experiment 1: Backbone Replacement (CLIP-ViT / DINOv2)")
    logger.info("=" * 70)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    data_path = Path(config['data_config']['data_path'])
    if not data_path.is_absolute():
        data_path = (script_dir / data_path).resolve()
    config['data_config']['data_path'] = str(data_path)
    
    visa_path = Path(config['data_config'].get('visa_path', '../data/visa_pytorch/1cls'))
    if not visa_path.is_absolute():
        visa_path = (script_dir / visa_path).resolve()
    config['data_config']['visa_path'] = str(visa_path)
    
    backbone_name = config['experiment'].get('backbone', 'dinov2_vitb14')
    logger.info(f"✓ Backbone: {backbone_name}")
    logger.info(f"✓ Data path: {config['data_config']['data_path']}")
    logger.info(f"✓ VisA path: {config['data_config']['visa_path']}")
    
    backbone_names = ['resnet50', 'dinov2_vitb14']
    categories_to_test = config['experiment']['categories'][:8]  # Use 8 categories
    
    logger.info(f"Backbones to evaluate: {backbone_names}")
    logger.info(f"Categories to evaluate: {categories_to_test}")
    logger.info(f"Configuration: {config['experiment']['name']}")
    
    all_results = []
    
    # Evaluate each backbone
    for backbone_name in backbone_names:
        logger.info("\n" + "=" * 70)
        logger.info(f"EVALUATING BACKBONE: {backbone_name}")
        logger.info("=" * 70)
        
        try:
            # Load backbone with PatchCore
            logger.info(f"\n[1/2] Loading backbone: {backbone_name}...")
            model, backbone_config = load_backbone_with_patchcore(
                backbone_name=backbone_name,
                device=config.get('device', 'cuda')
            )
            logger.info(f"Successfully loaded {backbone_name}")
            logger.info(f"Feature dimension: {backbone_config['feature_dim']}")
            
            device = torch.device(backbone_config['device'])
            
            # Evaluate on MVTec AD (in-domain)
            logger.info(f"\n[2/2] Evaluating on MVTec AD (in-domain) with {backbone_name}...")
            mvtec_results = evaluate_on_mvtec_with_memory_bank(
                model=model,
                config=backbone_config,
                data_path=config['data_config']['data_path'],
                categories=categories_to_test,
                backbone_name=backbone_name,
                device=device
            )
            logger.info(f"MVTec evaluation complete: {len(mvtec_results)} results")
            all_results.extend(mvtec_results)
            
            # Evaluate on VisA (cross-domain zero-shot using MVTec memory bank)
            logger.info(f"\nEvaluating on VisA (cross-domain zero-shot) with {backbone_name}...")
            visa_results = evaluate_on_visa_with_memory_bank(
                model=model,
                config=backbone_config,
                visa_path=config['data_config'].get('visa_path', '../../data/visa'),
                backbone_name=backbone_name,
                device=device
            )
            logger.info(f"VisA evaluation complete: {len(visa_results)} results")
            all_results.extend(visa_results)
            
        except Exception as e:
            logger.error(f"Error evaluating {backbone_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    results = all_results
    
    # Save results to main directory
    df = pd.DataFrame(results)
    
    # Construct output path from script location
    script_dir = Path(__file__).parent
    base_output = Path(config['output_path'])
    if not base_output.is_absolute():
        base_output = (script_dir / base_output).resolve()
    
    output_dir = base_output / 'exp1_backbone_comparison' / 'all_backbones'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / 'results_all_backbones.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to: {csv_path}")
    
    # Create comprehensive comparison visualizations
    logger.info("Creating comparison visualizations...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. MVTec performance by backbone
    ax1 = fig.add_subplot(gs[0, 0])
    mvtec_df = df[df['dataset'] == 'MVTec AD']
    if len(mvtec_df) > 0:
        mvtec_by_backbone = mvtec_df.groupby('backbone')['image_auroc'].mean().sort_values(ascending=False)
        mvtec_by_backbone.plot(ax=ax1, kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('In-Domain Performance: MVTec AD by Backbone', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Image AUROC')
        ax1.set_xlabel('Backbone')
        ax1.set_ylim([0.8, 1.0])
        ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.3, label='Baseline (0.95)')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # 2. VisA performance by backbone
    ax2 = fig.add_subplot(gs[0, 1])
    visa_df = df[df['dataset'] == 'VisA']
    if len(visa_df) > 0:
        visa_by_backbone = visa_df.groupby('backbone')['image_auroc'].mean().sort_values(ascending=False)
        visa_by_backbone.plot(ax=ax2, kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title('Cross-Domain Performance: VisA Zero-Shot by Backbone', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Image AUROC')
        ax2.set_xlabel('Backbone')
        ax2.set_ylim([0.7, 1.0])
        ax2.grid(axis='y', alpha=0.3)
    
    # 3. Category-wise comparison (MVTec)
    ax3 = fig.add_subplot(gs[1, 0])
    mvtec_by_cat = mvtec_df.pivot_table(values='image_auroc', index='category', columns='backbone', aggfunc='mean')
    if not mvtec_by_cat.empty:
        mvtec_by_cat.plot(ax=ax3, kind='bar', width=0.8)
        ax3.set_title('MVTec AD: Per-Category Performance', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Image AUROC')
        ax3.set_xlabel('Category')
        ax3.legend(title='Backbone', loc='lower right')
        ax3.set_ylim([0.8, 1.0])
        ax3.grid(axis='y', alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Generalization gap comparison
    ax4 = fig.add_subplot(gs[1, 1])
    if len(mvtec_df) > 0 and len(visa_df) > 0:
        mvtec_avg_by_bb = mvtec_df.groupby('backbone')['image_auroc'].mean()
        visa_avg_by_bb = visa_df.groupby('backbone')['image_auroc'].mean()
        
        backbones = list(mvtec_avg_by_bb.index)
        x = np.arange(len(backbones))
        width = 0.35
        
        ax4.bar(x - width/2, mvtec_avg_by_bb, width, label='MVTec (In-Domain)', color='#1f77b4')
        ax4.bar(x + width/2, visa_avg_by_bb, width, label='VisA (Cross-Domain)', color='#ff7f0e')
        
        ax4.set_title('Generalization Gap: In-Domain vs Cross-Domain', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Average Image AUROC')
        ax4.set_xlabel('Backbone')
        ax4.set_xticks(x)
        ax4.set_xticklabels(backbones)
        ax4.legend()
        ax4.set_ylim([0.7, 1.0])
        ax4.grid(axis='y', alpha=0.3)
        
        # Add gap annotations
        for i, bb in enumerate(backbones):
            gap = mvtec_avg_by_bb[bb] - visa_avg_by_bb[bb]
            ax4.text(i, max(mvtec_avg_by_bb[bb], visa_avg_by_bb[bb]) + 0.01, 
                    f'Gap: {gap:.3f}', ha='center', fontsize=9)
    
    plt.suptitle('Experiment 1: Backbone Comparison (ResNet50 vs DINOv2 vs CLIP-ViT)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plot_path = output_dir / 'exp1_backbone_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to: {plot_path}")
    plt.close()
    
    # Save individual backbone results
    for backbone in df['backbone'].unique():
        bb_df = df[df['backbone'] == backbone]
        bb_dir = Path(config['output_path']) / backbone
        bb_dir.mkdir(parents=True, exist_ok=True)
        bb_csv = bb_dir / 'results.csv'
        bb_df.to_csv(bb_csv, index=False)
        logger.info(f"Saved {backbone} results to: {bb_csv}")
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1 SUMMARY - BACKBONE COMPARISON")
    logger.info("=" * 70)
    
    df = pd.DataFrame(results)
    
    for backbone in sorted(df['backbone'].unique()):
        bb_df = df[df['backbone'] == backbone]
        
        logger.info(f"\n--- {backbone.upper()} ---")
        
        mvtec_bb = bb_df[bb_df['dataset'] == 'MVTec AD']
        visa_bb = bb_df[bb_df['dataset'] == 'VisA']
        
        if len(mvtec_bb) > 0:
            mvtec_avg = mvtec_bb['image_auroc'].mean()
            logger.info(f"  In-Domain (MVTec AD):        {mvtec_avg:.4f}")
        
        if len(visa_bb) > 0:
            visa_avg = visa_bb['image_auroc'].mean()
            logger.info(f"  Cross-Domain (VisA):         {visa_avg:.4f}")
        
        if len(mvtec_bb) > 0 and len(visa_bb) > 0:
            gap = mvtec_avg - visa_avg
            logger.info(f"  Generalization Gap:          {gap:.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("KEY FINDINGS:")
    logger.info("=" * 70)
    
    # Determine best backbone
    mvtec_all = df[df['dataset'] == 'MVTec AD']
    visa_all = df[df['dataset'] == 'VisA']
    
    if len(mvtec_all) > 0:
        best_mvtec = mvtec_all.groupby('backbone')['image_auroc'].mean().idxmax()
        best_mvtec_score = mvtec_all.groupby('backbone')['image_auroc'].mean().max()
        logger.info(f"Best In-Domain: {best_mvtec} ({best_mvtec_score:.4f})")
    
    if len(visa_all) > 0:
        best_visa = visa_all.groupby('backbone')['image_auroc'].mean().idxmax()
        best_visa_score = visa_all.groupby('backbone')['image_auroc'].mean().max()
        logger.info(f"Best Cross-Domain: {best_visa} ({best_visa_score:.4f})")
    
    if len(mvtec_all) > 0 and len(visa_all) > 0:
        gaps = []
        for backbone in df['backbone'].unique():
            mvtec_avg = mvtec_all[mvtec_all['backbone'] == backbone]['image_auroc'].mean()
            visa_avg = visa_all[visa_all['backbone'] == backbone]['image_auroc'].mean()
            gaps.append((backbone, mvtec_avg - visa_avg))
        
        best_generalization = min(gaps, key=lambda x: x[1])
        logger.info(f"Best Generalization (smallest gap): {best_generalization[0]} ({best_generalization[1]:.4f})")
    
    logger.info("=" * 70)
    
    return results


def generate_synthetic_results(backbone_name: str) -> List[Dict]:
    """Generate synthetic results for demonstration."""
    categories_mvtec = ["bottle", "cable", "capsule", "carpet", "grid"]
    categories_visa = ["candle", "cashew", "chewinggum", "frito"]
    
    # Performance depends on backbone
    perf_map = {
        'clip_vitb16': {'mvtec': 0.965, 'visa': 0.872},
        'dinov2_vitb14': {'mvtec': 0.975, 'visa': 0.885},
        'resnet50': {'mvtec': 0.930, 'visa': 0.780}
    }
    
    perf = perf_map.get(backbone_name, perf_map['clip_vitb16'])
    results = []
    np.random.seed(42)
    
    # MVTec results
    for cat in categories_mvtec:
        auroc = np.clip(perf['mvtec'] + np.random.normal(0, 0.03), 0.85, 0.99)
        results.append({
            'dataset': 'MVTec AD',
            'category': cat,
            'image_auroc': round(auroc, 4),
            'pixel_auroc': round(auroc - 0.04, 4),
            'backbone': backbone_name,
            'timestamp': datetime.now().isoformat()
        })
    
    # VisA results
    for cat in categories_visa:
        auroc = np.clip(perf['visa'] + np.random.normal(0, 0.04), 0.75, 0.95)
        results.append({
            'dataset': 'VisA',
            'category': cat,
            'image_auroc': round(auroc, 4),
            'pixel_auroc': round(auroc - 0.05, 4),
            'backbone': backbone_name,
            'timestamp': datetime.now().isoformat()
        })
    
    return results


if __name__ == '__main__':
    main()
