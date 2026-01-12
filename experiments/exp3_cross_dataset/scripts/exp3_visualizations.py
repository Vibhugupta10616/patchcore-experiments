"""
Experiment 3: Visualization utilities
Creates comprehensive plots for FAISS analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_exp3_visualizations(results_df: pd.DataFrame, output_dir: Path, timestamp: str):
    """
    Create comprehensive visualization for Experiment 3 results.
    Generates a 2x2 grid showing:
    - Top-left: AUROC Comparison by Category
    - Top-right: Search Speedup vs Nprobe
    - Bottom-left: Memory Reduction vs Nprobe
    - Bottom-right: Accuracy Loss vs Nprobe
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save plots
        timestamp: Timestamp string for file naming
    """
    try:
        exact_df = results_df[results_df['search_method'] == 'exact_knn']
        faiss_df = results_df[results_df['search_method'] == 'faiss_ivfpq']
        
        if len(faiss_df) == 0:
            logger.warning("No FAISS results found for visualization")
            return
        
        # Create comprehensive figure with better scaling
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
        
        # 1. AUROC Comparison by Category
        ax1 = fig.add_subplot(gs[0, 0])
        categories = sorted(results_df['category'].unique())
        baseline_aurocs = [exact_df[exact_df['category'] == c]['image_auroc'].iloc[0] for c in categories]
        best_faiss_aurocs = [faiss_df[faiss_df['category'] == c]['image_auroc'].max() for c in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        bars1 = ax1.bar(x - width/2, baseline_aurocs, width, label='Exact k-NN', color='#1f77b4', alpha=0.8)
        bars2 = ax1.bar(x + width/2, best_faiss_aurocs, width, label='Best FAISS', color='#ff7f0e', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Category', fontsize=11)
        ax1.set_ylabel('AUROC', fontsize=11)
        ax1.set_title('AUROC Comparison by Category', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        ax1.set_ylim([0.0, 1.05])
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Speedup vs Nprobe
        ax2 = fig.add_subplot(gs[0, 1])
        pq_bits = sorted(faiss_df['pq_bits'].dropna().unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(pq_bits)))
        
        for bits, color in zip(pq_bits, colors):
            data = faiss_df[faiss_df['pq_bits'] == bits].groupby('nprobe')['speedup'].mean()
            ax2.plot(data.index, data.values, marker='o', label=f'{int(bits)}-bit', 
                    color=color, linewidth=2.5, markersize=8)
        
        ax2.set_xlabel('Nprobe', fontsize=11)
        ax2.set_ylabel('Speedup Factor (log scale)', fontsize=11)
        ax2.set_title('Search Speedup vs Nprobe', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. Memory Reduction vs Nprobe
        ax3 = fig.add_subplot(gs[1, 0])
        for bits, color in zip(pq_bits, colors):
            data = faiss_df[faiss_df['pq_bits'] == bits].groupby('nprobe')['memory_reduction'].mean()
            ax3.plot(data.index, data.values, marker='s', label=f'{int(bits)}-bit', 
                    color=color, linewidth=2.5, markersize=8)
        
        ax3.set_xlabel('Nprobe', fontsize=11)
        ax3.set_ylabel('Memory Reduction Factor', fontsize=11)
        ax3.set_title('Memory Reduction vs Nprobe', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. Accuracy Loss vs Nprobe
        ax4 = fig.add_subplot(gs[1, 1])
        for bits, color in zip(pq_bits, colors):
            data = faiss_df[faiss_df['pq_bits'] == bits].groupby('nprobe')['accuracy_loss_percent'].mean()
            ax4.plot(data.index, data.values, marker='^', label=f'{int(bits)}-bit', 
                    color=color, linewidth=2.5, markersize=8)
        
        ax4.set_xlabel('Nprobe', fontsize=11)
        ax4.set_ylabel('Accuracy Loss (%)', fontsize=11)
        ax4.set_title('Accuracy Loss vs Nprobe', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Experiment 3: FAISS IVF-PQ Performance Analysis', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Save plot
        plot_path = output_dir / f'exp3_comprehensive_analysis_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Comprehensive visualization saved: {plot_path}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}", exc_info=True)


if __name__ == "__main__":
    # Example usage
    print("This module provides visualization utilities for Experiment 3.")
    print("Import and use create_exp3_visualizations() function.")