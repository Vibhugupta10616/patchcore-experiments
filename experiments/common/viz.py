"""
Common visualization utilities for all experiments
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torchvision.transforms.functional import to_pil_image

logger = logging.getLogger(__name__)


def plot_results(
    results: List[Dict],
    output_path: str,
    metric_name: str = 'image_auroc'
):
    """
    Plot experiment results.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save plot
        metric_name: Metric to plot
    """
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(figsize=(12, 6))
    
    if 'category' in df.columns:
        pivot = df.pivot_table(
            values=metric_name,
            index='category',
            columns=[col for col in df.columns if col not in ['category', metric_name]][0]
        )
        pivot.plot(ax=axes, kind='bar')
    else:
        df.plot(ax=axes, kind='bar')
    
    axes.set_title(f'{metric_name.replace("_", " ").title()} Results')
    axes.set_ylabel(metric_name.replace("_", " ").title())
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {output_path}")
    plt.close()


def save_heatmaps(
    images: torch.Tensor,
    anomaly_maps: np.ndarray,
    masks: Optional[np.ndarray] = None,
    output_dir: str = './heatmaps',
    n_samples: int = 10,
    category: str = 'unknown'
):
    """
    Save anomaly heatmaps.
    
    Args:
        images: Input images [B, 3, H, W]
        anomaly_maps: Anomaly maps [B, H, W]
        masks: Ground truth masks [B, H, W]
        output_dir: Directory to save visualizations
        n_samples: Number of samples to visualize
        category: Category name for organization
    """
    output_dir = Path(output_dir) / category
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_vis = min(n_samples, len(images))
    
    for idx in range(n_vis):
        fig, axes = plt.subplots(1, 3 if masks is not None else 2, figsize=(12, 4))
        
        # Original image
        img = images[idx].cpu()
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Anomaly map with heatmap
        anom_map = anomaly_maps[idx]
        im = axes[1].imshow(anom_map, cmap='jet')
        axes[1].set_title('Anomaly Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Ground truth mask
        if masks is not None:
            mask = masks[idx]
            axes[2].imshow(mask, cmap='gray')
            axes[2].set_title('Ground Truth Mask')
            axes[2].axis('off')
        
        plt.tight_layout()
        save_path = output_dir / f'sample_{idx:04d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Heatmaps saved to {output_dir}")


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    output_path: str,
    title: str = 'ROC Curve'
):
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curve saved to {output_path}")


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    output_path: str,
    auc_score: Optional[float] = None,
    title: str = 'Precision-Recall Curve'
):
    """Plot precision-recall curve."""
    from sklearn.metrics import auc as auc_fn
    
    if auc_score is None:
        auc_score = auc_fn(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'PR (AUC = {auc_score:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="upper right")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"PR curve saved to {output_path}")


def visualize_anomaly_localization(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Create visualization of anomaly localization.
    
    Args:
        image: Input image [H, W, 3]
        anomaly_map: Anomaly map [H, W]
        mask: Ground truth mask [H, W]
        threshold: Threshold for visualization
    
    Returns:
        Composite visualization image
    """
    # Normalize anomaly map
    anom_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)
    
    # Create heatmap
    heatmap = cm.jet(anom_norm)[:, :, :3]
    
    # Blend with original image
    blended = 0.7 * image + 0.3 * (heatmap * 255)
    
    return blended.astype(np.uint8)

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(image.permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("GT Mask")
    plt.imshow(gt_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Predicted (score={score:.3f})")
    plt.imshow(pred_mask, cmap="jet")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
