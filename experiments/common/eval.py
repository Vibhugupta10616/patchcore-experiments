"""
Common evaluation utilities for all experiments
"""

import logging
from typing import Tuple, Dict, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate_auroc(
    anomaly_scores: np.ndarray,
    dataloader: DataLoader
) -> float:
    """
    Compute image-level AUROC.
    
    Args:
        anomaly_scores: Array of anomaly scores
        dataloader: DataLoader containing labels
    
    Returns:
        AUROC score
    """
    logger.info("Computing image-level AUROC...")
    
    labels = []
    for batch in dataloader:
        labels.extend(batch['label'].numpy())
    labels = np.array(labels[:len(anomaly_scores)])
    
    auroc = roc_auc_score(labels, anomaly_scores)
    logger.info(f"Image AUROC: {auroc:.4f}")
    
    return auroc


def evaluate_localization(
    anomaly_maps: np.ndarray,
    dataloader: DataLoader
) -> Tuple[float, Dict]:
    """
    Compute pixel-level AUROC and PR curves.
    
    Args:
        anomaly_maps: Pixel-level anomaly maps
        dataloader: DataLoader containing masks
    
    Returns:
        Tuple of (pixel_auroc, pr_curves_dict)
    """
    logger.info("Computing pixel-level AUROC...")
    
    pixel_labels = []
    for batch in dataloader:
        masks = batch.get('mask')
        if masks is not None:
            pixel_labels.extend(masks.numpy().flatten())
    
    if len(pixel_labels) == 0:
        logger.warning("No pixel-level labels available")
        return 0.0, {}
    
    pixel_labels = np.array(pixel_labels[:len(anomaly_maps.flatten())])
    anomaly_flat = anomaly_maps.flatten()[:len(pixel_labels)]
    
    # Compute pixel AUROC
    pixel_auroc = roc_auc_score(pixel_labels, anomaly_flat)
    logger.info(f"Pixel AUROC: {pixel_auroc:.4f}")
    
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(pixel_labels, anomaly_flat)
    pr_auc = auc(recall, precision)
    
    pr_curves = {
        'precision': precision,
        'recall': recall,
        'auc': pr_auc
    }
    
    logger.info(f"Pixel PR-AUC: {pr_auc:.4f}")
    
    return pixel_auroc, pr_curves


def compute_pro_score(
    anomaly_maps: np.ndarray,
    mask: np.ndarray,
    normalize: bool = True
) -> float:
    """
    Compute Per-Region-Overlap (PRO) score.
    
    Args:
        anomaly_maps: Predicted anomaly maps [H, W]
        mask: Ground truth mask [H, W]
        normalize: Whether to normalize scores
    
    Returns:
        PRO score
    """
    from scipy import ndimage
    
    # Threshold at multiple levels
    thresholds = np.arange(0, 1.01, 0.01)
    pro_scores = []
    
    for threshold in thresholds:
        prediction = (anomaly_maps > threshold).astype(int)
        
        # Label connected components
        labeled_pred, _ = ndimage.label(prediction)
        labeled_mask, _ = ndimage.label(mask)
        
        # Compute overlap for each prediction region
        overlaps = []
        for region_id in np.unique(labeled_pred):
            if region_id == 0:
                continue
            
            pred_region = (labeled_pred == region_id)
            overlap = np.sum(pred_region & mask) / (np.sum(pred_region) + 1e-6)
            overlaps.append(overlap)
        
        if overlaps:
            pro = np.mean(overlaps)
        else:
            pro = 0.0
        
        pro_scores.append(pro)
    
    return np.mean(pro_scores) if pro_scores else 0.0


def compute_f1_score(
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None
) -> float:
    """Compute F1 score at optimal or given threshold."""
    from sklearn.metrics import f1_score
    
    if threshold is None:
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)
        f1_scores = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr) + 1e-6)
        optimal_idx = np.argmax(f1_scores)
        threshold = thresholds[optimal_idx]
    
    predictions = (anomaly_scores >= threshold).astype(int)
    f1 = f1_score(labels, predictions)
    
    logger.info(f"F1 Score (threshold={threshold:.4f}): {f1:.4f}")
    return f1


def compute_auc_pr(
    labels: np.ndarray,
    scores: np.ndarray
) -> float:
    """Compute Area Under Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    return pr_auc

def evaluate_image_level(scores, labels):
    metrics = compute_imagewise_retrieval_metrics(scores, labels)
    return metrics["auroc"]

def evaluate_pixel_level(pred_masks, gt_masks):
    metrics = compute_pixelwise_retrieval_metrics(
        pred_masks, gt_masks
    )
    return metrics["auroc"]
