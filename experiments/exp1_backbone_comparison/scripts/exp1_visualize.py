"""
Visualization functions for Experiment 1: Backbone Comparison
Creates main comparison visualization and saves graph.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def create_comparison_visualizations(
    df: pd.DataFrame,
    viz_dir: Path,
    output_dir: Path
) -> None:
    """
    Create comparison visualization for backbone comparison.
    Generates a 2x2 subplot figure showing:
    - Top-left: In-Domain Performance (MVTec AD) by Backbone
    - Top-right: Cross-Domain Performance (VisA) by Backbone  
    - Bottom-left: Per-Category MVTec Performance
    - Bottom-right: Generalization Gap Comparison
    
    Args:
        df: Results dataframe
        viz_dir: Directory to save visualizations
        output_dir: Output results directory
    """
    logger.info(f"Creating visualizations in {viz_dir}")
    
    # Create figure with 2x2 subplots
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
        ax1.set_xlabel('Backbones', fontsize=11)
        ax1.set_ylim([0.8, 1.0])
        ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.3, label='Baseline (0.95)')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
    
    # 2. VisA performance by backbone
    ax2 = fig.add_subplot(gs[0, 1])
    visa_df = df[df['dataset'] == 'VisA']
    if len(visa_df) > 0:
        visa_by_backbone = visa_df.groupby('backbone')['image_auroc'].mean().sort_values(ascending=False)
        visa_by_backbone.plot(ax=ax2, kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title('Cross-Domain Performance: VisA Zero-Shot by Backbone', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Image AUROC')
        ax2.set_xlabel('Backbones', fontsize=11)
        ax2.set_ylim([0.7, 1.0])
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
    
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
    
    # Save main comparison figure to visualizations folder
    plot_path = viz_dir / 'exp1_backbone_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Main visualization saved to: {plot_path}")
    
    # Also save to output directory
    plot_path_output = output_dir / 'exp1_backbone_comparison.png'
    plt.savefig(plot_path_output, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Comparison visualization also saved to: {plot_path_output}")
    
    plt.close()
