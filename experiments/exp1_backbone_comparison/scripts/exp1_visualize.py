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
    viz_dir: Path
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
        bars = ax1.bar(range(len(mvtec_by_backbone)), mvtec_by_backbone.values, 
                       width=0.4, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.set_xticks(range(len(mvtec_by_backbone)))
        ax1.set_xticklabels(mvtec_by_backbone.index, rotation=0)
        ax1.set_title('In-Domain Performance: MVTec AD by Backbone', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Image AUROC', fontsize=11)
        ax1.set_xlabel('Backbones', fontsize=11)
        ax1.set_ylim([0.88, 1.02])
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
    
    # 2. VisA performance by backbone
    ax2 = fig.add_subplot(gs[0, 1])
    visa_df = df[df['dataset'] == 'VisA']
    if len(visa_df) > 0:
        visa_by_backbone = visa_df.groupby('backbone')['image_auroc'].mean().sort_values(ascending=False)
        bars = ax2.bar(range(len(visa_by_backbone)), visa_by_backbone.values,
                       width=0.4, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.set_xticks(range(len(visa_by_backbone)))
        ax2.set_xticklabels(visa_by_backbone.index, rotation=0)
        ax2.set_title('Cross-Domain Performance: VisA Zero-Shot by Backbone', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Image AUROC', fontsize=11)
        ax2.set_xlabel('Backbones', fontsize=11)
        ax2.set_ylim([0.65, 1.02])
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
    
    # 3. Category-wise comparison (MVTec)
    ax3 = fig.add_subplot(gs[1, 0])
    mvtec_by_cat = mvtec_df.pivot_table(values='image_auroc', index='category', columns='backbone', aggfunc='mean')
    if not mvtec_by_cat.empty:
        x = np.arange(len(mvtec_by_cat.index))
        width = 0.25
        for i, backbone in enumerate(mvtec_by_cat.columns):
            bars = ax3.bar(x + i*width, mvtec_by_cat[backbone].values, width, 
                          label=backbone, color=['#1f77b4', '#ff7f0e', '#2ca02c'][i])
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=7)
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(mvtec_by_cat.index, rotation=45, ha='right')
        ax3.set_title('MVTec AD: Per-Category Performance', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Image AUROC', fontsize=11)
        ax3.set_xlabel('Category', fontsize=11)
        ax3.legend(title='Backbone', loc='lower right', fontsize=9)
        ax3.set_ylim([0.80, 1.02])
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. Generalization gap comparison
    ax4 = fig.add_subplot(gs[1, 1])
    if len(mvtec_df) > 0 and len(visa_df) > 0:
        mvtec_avg_by_bb = mvtec_df.groupby('backbone')['image_auroc'].mean()
        visa_avg_by_bb = visa_df.groupby('backbone')['image_auroc'].mean()
        
        backbones = list(mvtec_avg_by_bb.index)
        x = np.arange(len(backbones))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, mvtec_avg_by_bb, width, label='MVTec (In-Domain)', color='#1f77b4')
        bars2 = ax4.bar(x + width/2, visa_avg_by_bb, width, label='VisA (Cross-Domain)', color='#ff7f0e')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax4.set_title('Generalization Gap: In-Domain vs Cross-Domain', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Average Image AUROC', fontsize=11)
        ax4.set_xlabel('Backbone', fontsize=11)
        ax4.set_xticks(x)
        ax4.set_xticklabels(backbones)
        ax4.legend(fontsize=10)
        ax4.set_ylim([0.80, 1.00])
        ax4.grid(axis='y', alpha=0.3)
        
        # Add gap annotations with bold formatting
        for i, bb in enumerate(backbones):
            gap = mvtec_avg_by_bb[bb] - visa_avg_by_bb[bb]
            ax4.text(i, max(mvtec_avg_by_bb[bb], visa_avg_by_bb[bb]) + 0.015, 
                    f'Gap: {gap:.4f}', ha='center', fontsize=10, fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    plt.suptitle('Experiment 1: Backbone Comparison (ResNet50 vs DINOv2 vs CLIP-ViT)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Ensure visualization directory exists
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main comparison figure to visualizations folder
    plot_path = viz_dir / 'exp1_backbone_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Main visualization saved to: {plot_path}")
    
    plt.close()
