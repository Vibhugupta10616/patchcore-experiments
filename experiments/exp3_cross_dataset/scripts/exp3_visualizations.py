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
    Create comprehensive visualizations for Experiment 3 results.
    
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
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. AUROC Comparison by Category
        ax1 = fig.add_subplot(gs[0, 0])
        categories = sorted(results_df['category'].unique())
        baseline_aurocs = [exact_df[exact_df['category'] == c]['image_auroc'].iloc[0] for c in categories]
        best_faiss_aurocs = [faiss_df[faiss_df['category'] == c]['image_auroc'].max() for c in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        ax1.bar(x - width/2, baseline_aurocs, width, label='Exact k-NN', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, best_faiss_aurocs, width, label='Best FAISS', color='coral', alpha=0.8)
        ax1.set_xlabel('Category')
        ax1.set_ylabel('AUROC')
        ax1.set_title('AUROC Comparison by Category')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Speedup vs Nprobe
        ax2 = fig.add_subplot(gs[0, 1])
        pq_bits = sorted(faiss_df['pq_bits'].dropna().unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(pq_bits)))
        
        for bits, color in zip(pq_bits, colors):
            data = faiss_df[faiss_df['pq_bits'] == bits].groupby('nprobe')['speedup'].mean()
            ax2.plot(data.index, data.values, marker='o', label=f'{int(bits)}-bit', 
                    color=color, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Nprobe')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Search Speedup vs Nprobe')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. Memory Reduction vs Nprobe
        ax3 = fig.add_subplot(gs[0, 2])
        for bits, color in zip(pq_bits, colors):
            data = faiss_df[faiss_df['pq_bits'] == bits].groupby('nprobe')['memory_reduction'].mean()
            ax3.plot(data.index, data.values, marker='s', label=f'{int(bits)}-bit', 
                    color=color, linewidth=2, markersize=6)
        
        ax3.set_xlabel('Nprobe')
        ax3.set_ylabel('Memory Reduction Factor')
        ax3.set_title('Memory Reduction vs Nprobe')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Accuracy Loss vs Nprobe
        ax4 = fig.add_subplot(gs[1, 0])
        for bits, color in zip(pq_bits, colors):
            data = faiss_df[faiss_df['pq_bits'] == bits].groupby('nprobe')['accuracy_loss_percent'].mean()
            ax4.plot(data.index, data.values, marker='^', label=f'{int(bits)}-bit', 
                    color=color, linewidth=2, markersize=6)
        
        ax4.set_xlabel('Nprobe')
        ax4.set_ylabel('Accuracy Loss (%)')
        ax4.set_title('Accuracy Loss vs Nprobe')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Search Time Heatmap
        ax5 = fig.add_subplot(gs[1, 1])
        if len(pq_bits) > 1:
            pivot_time = faiss_df.groupby(['nprobe', 'pq_bits'])['search_time_ms'].mean().unstack()
            im1 = ax5.imshow(pivot_time, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            ax5.set_xlabel('PQ Bits')
            ax5.set_ylabel('Nprobe')
            ax5.set_xticks(range(len(pivot_time.columns)))
            ax5.set_xticklabels([int(x) for x in pivot_time.columns])
            ax5.set_yticks(range(len(pivot_time.index)))
            ax5.set_yticklabels(pivot_time.index)
            ax5.set_title('Search Time Heatmap (ms)')
            plt.colorbar(im1, ax=ax5, shrink=0.8)
        else:
            ax5.text(0.5, 0.5, 'Insufficient data\nfor heatmap', transform=ax5.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax5.set_title('Search Time Heatmap (ms)')
        
        # 6. Speedup Heatmap
        ax6 = fig.add_subplot(gs[1, 2])
        if len(pq_bits) > 1:
            pivot_speedup = faiss_df.groupby(['nprobe', 'pq_bits'])['speedup'].mean().unstack()
            im2 = ax6.imshow(pivot_speedup, cmap='YlGn', aspect='auto', interpolation='nearest')
            ax6.set_xlabel('PQ Bits')
            ax6.set_ylabel('Nprobe')
            ax6.set_xticks(range(len(pivot_speedup.columns)))
            ax6.set_xticklabels([int(x) for x in pivot_speedup.columns])
            ax6.set_yticks(range(len(pivot_speedup.index)))
            ax6.set_yticklabels(pivot_speedup.index)
            ax6.set_title('Speedup Heatmap')
            plt.colorbar(im2, ax=ax6, shrink=0.8)
        else:
            ax6.text(0.5, 0.5, 'Insufficient data\nfor heatmap', transform=ax6.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax6.set_title('Speedup Heatmap')
        
        # 7. Performance Trade-off Scatter
        ax7 = fig.add_subplot(gs[2, :2])
        for category in categories:
            cat_data = faiss_df[faiss_df['category'] == category]
            ax7.scatter(cat_data['speedup'], cat_data['image_auroc'], 
                       label=category, s=80, alpha=0.7)
        
        # Add baseline line
        baseline_auroc = exact_df['image_auroc'].mean()
        ax7.axhline(y=baseline_auroc, color='red', linestyle='--', linewidth=2, 
                   label='Exact k-NN Baseline')
        
        ax7.set_xlabel('Speedup Factor')
        ax7.set_ylabel('AUROC')
        ax7.set_title('Performance Trade-off: Speed vs Accuracy')
        ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax7.grid(True, alpha=0.3)
        ax7.set_xscale('log')
        
        # 8. Summary Statistics
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        # Calculate summary stats
        avg_speedup = faiss_df['speedup'].mean()
        max_speedup = faiss_df['speedup'].max()
        avg_memory_reduction = faiss_df['memory_reduction'].mean()
        avg_accuracy_loss = faiss_df['accuracy_loss_percent'].mean()
        
        best_config_idx = faiss_df['image_auroc'].idxmax()
        best_config = faiss_df.loc[best_config_idx]
        
        summary_text = f"""Summary Statistics:
        
Average Speedup: {avg_speedup:.1f}x
Maximum Speedup: {max_speedup:.1f}x
Avg Memory Reduction: {avg_memory_reduction:.1f}x
Avg Accuracy Loss: {avg_accuracy_loss:.2f}%

Best Configuration:
Category: {best_config['category']}
Config: {best_config['pq_bits']:.0f}-bit, nprobe={best_config['nprobe']:.0f}
AUROC: {best_config['image_auroc']:.4f}
Speedup: {best_config['speedup']:.1f}x
Memory Reduction: {best_config['memory_reduction']:.1f}x"""
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.suptitle('Experiment 3: FAISS IVF-PQ Performance Analysis', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Save plot
        plot_path = output_dir / f'exp3_comprehensive_analysis_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Comprehensive visualization saved: {plot_path}")
        plt.close()
        
        # Create individual category plots
        create_category_plots(results_df, output_dir, timestamp)
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}", exc_info=True)


def create_category_plots(results_df: pd.DataFrame, output_dir: Path, timestamp: str):
    """Create individual plots for each category."""
    try:
        categories = sorted(results_df['category'].unique())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, category in enumerate(categories):
            if i >= len(axes):
                break
                
            ax = axes[i]
            cat_exact = results_df[(results_df['category'] == category) & 
                                  (results_df['search_method'] == 'exact_knn')]
            cat_faiss = results_df[(results_df['category'] == category) & 
                                  (results_df['search_method'] == 'faiss_ivfpq')]
            
            if len(cat_faiss) > 0:
                # Plot speedup vs accuracy
                ax.scatter(cat_faiss['speedup'], cat_faiss['image_auroc'], 
                          c=cat_faiss['pq_bits'], cmap='viridis', s=60, alpha=0.7)
                
                # Add baseline
                if len(cat_exact) > 0:
                    baseline_auroc = cat_exact['image_auroc'].iloc[0]
                    ax.axhline(y=baseline_auroc, color='red', linestyle='--', 
                              linewidth=2, alpha=0.7, label='Exact k-NN')
                
                ax.set_xlabel('Speedup Factor')
                ax.set_ylabel('AUROC')
                ax.set_title(f'{category.capitalize()} Category')
                ax.grid(True, alpha=0.3)
                ax.legend()
            else:
                ax.text(0.5, 0.5, f'No FAISS data\nfor {category}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{category.capitalize()} Category')
        
        # Hide unused subplots
        for i in range(len(categories), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Per-Category Analysis', fontsize=14, fontweight='bold', y=0.98)
        
        category_plot_path = output_dir / f'exp3_category_analysis_{timestamp}.png'
        plt.savefig(category_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Category analysis saved: {category_plot_path}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating category plots: {e}", exc_info=True)


if __name__ == "__main__":
    # Example usage
    print("This module provides visualization utilities for Experiment 3.")
    print("Import and use create_exp3_visualizations() function.")