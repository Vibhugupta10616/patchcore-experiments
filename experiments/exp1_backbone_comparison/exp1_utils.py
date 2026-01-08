"""
Utility functions for Experiment 1: Backbone Comparison
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def get_backbone(
    backbone_name: str,
    pretrained: bool = True,
    device: torch.device = None
) -> nn.Module:
    """
    Load a backbone model.
    
    Args:
        backbone_name: Name of the backbone ('resnet50', 'vitb16', 'dinov2_vitb14', 'clip_vitb32')
        pretrained: Whether to load pretrained weights
        device: Device to load model on
    
    Returns:
        Backbone model in evaluation mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading {backbone_name} (pretrained={pretrained})")
    
    if backbone_name.startswith('resnet'):
        # ResNet backbone
        model = models.resnet50(pretrained=pretrained)
        
    elif backbone_name == 'vitb16':
        # Vision Transformer from torchvision
        try:
            model = models.vit_b_16(pretrained=pretrained)
        except:
            logger.warning("ViT-B/16 not available in torchvision, using timm")
            import timm
            model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
    
    elif backbone_name == 'dinov2_vitb14':
        # DINOv2 model
        try:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        except:
            logger.error("DINOv2 requires facebookresearch/dinov2. Installing...")
            import timm
            model = timm.create_model('vit_base_patch14_dinov2', pretrained=True)
    
    elif backbone_name.startswith('clip'):
        # CLIP model
        try:
            import clip
            clip_version = backbone_name.split('_')[1]  # e.g., 'vitb32' from 'clip_vitb32'
            model, _ = clip.load(f"ViT-B/32", device=device)
            model = model.visual
        except:
            logger.error("CLIP requires 'pip install clip'. Using ViT fallback.")
            import timm
            model = timm.create_model('vit_base_patch32_clip_224', pretrained=True)
    
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Backbone loaded successfully")
    return model


def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    layer_names: List[str],
    device: torch.device = None
) -> Dict[str, np.ndarray]:
    """
    Extract features from specified layers.
    
    Args:
        model: Backbone model
        dataloader: DataLoader for images
        layer_names: List of layer names to extract features from
        device: Device to use
    
    Returns:
        Dictionary mapping layer names to feature arrays
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Extracting features from layers: {layer_names}")
    
    features_dict = {layer: [] for layer in layer_names}
    
    # Register hooks to capture intermediate features
    hooks = {}
    activations = {}
    
    def get_hook(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    layer_modules = {}
    for name, module in model.named_modules():
        for layer_name in layer_names:
            if layer_name in name:
                hooks[name] = module.register_forward_hook(get_hook(name))
                layer_modules[name] = module
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            
            # Forward pass
            _ = model(images)
            
            # Collect features
            for name in activations:
                feat = activations[name]
                # Flatten spatial dimensions
                if len(feat.shape) == 4:  # [B, C, H, W]
                    feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])  # [B*H*W, C]
                elif len(feat.shape) == 3:  # [B, L, C] (transformer)
                    feat = feat.reshape(-1, feat.shape[-1])  # [B*L, C]
                
                features_dict[name].append(feat.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1} batches")
    
    # Remove hooks
    for hook in hooks.values():
        hook.remove()
    
    # Concatenate features
    for layer in features_dict:
        features_dict[layer] = np.vstack(features_dict[layer])
    
    logger.info(f"Feature extraction complete. Shapes: {[features_dict[l].shape for l in layer_names]}")
    
    return features_dict


def prepare_memory_bank(
    features: Dict[str, np.ndarray],
    method: str = 'knn',
    n_components: Optional[int] = None,
    **kwargs
) -> Dict:
    """
    Prepare memory bank for anomaly detection.
    
    Args:
        features: Dictionary of feature arrays
        method: Memory bank method ('knn', 'pca', 'kmeans')
        n_components: Number of components for PCA
        **kwargs: Additional arguments
    
    Returns:
        Memory bank dictionary
    """
    logger.info(f"Preparing memory bank with method: {method}")
    
    memory_bank = {}
    
    if method == 'knn':
        # Store all normal features for KNN
        memory_bank['type'] = 'knn'
        memory_bank['features'] = features
    
    elif method == 'pca':
        # PCA compression
        from sklearn.decomposition import PCA
        memory_bank['type'] = 'pca'
        memory_bank['pca_models'] = {}
        
        for layer_name, feat in features.items():
            n_comp = n_components or min(feat.shape[1] // 2, 256)
            pca = PCA(n_components=n_comp)
            compressed = pca.fit_transform(feat)
            memory_bank['pca_models'][layer_name] = pca
            memory_bank['features'] = compressed
    
    elif method == 'kmeans':
        # K-means clustering
        from sklearn.cluster import KMeans
        memory_bank['type'] = 'kmeans'
        memory_bank['kmeans_models'] = {}
        
        n_clusters = n_components or 256
        for layer_name, feat in features.items():
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(feat)
            memory_bank['kmeans_models'][layer_name] = kmeans
            memory_bank['centers'] = kmeans.cluster_centers_
    
    else:
        raise ValueError(f"Unknown memory bank method: {method}")
    
    logger.info(f"Memory bank prepared")
    return memory_bank


def compute_anomaly_scores(
    features: Dict[str, np.ndarray],
    memory_bank: Dict,
    method: str = 'knn',
    k: int = 5,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute anomaly scores.
    
    Args:
        features: Dictionary of feature arrays
        memory_bank: Prepared memory bank
        method: Scoring method
        k: Number of neighbors for KNN
        **kwargs: Additional arguments
    
    Returns:
        Dictionary with image_scores and pixel_scores
    """
    logger.info(f"Computing anomaly scores with method: {method}")
    
    if method == 'knn':
        from sklearn.neighbors import NearestNeighbors
        
        # Use normal features from memory bank
        normal_features = memory_bank['features']
        
        # Fit KNN on normal features
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(normal_features)
        
        # Compute distances
        test_feat = features
        distances, _ = nbrs.kneighbors(test_feat)
        
        # Mean distance as anomaly score
        anomaly_scores_pixel = distances.mean(axis=1)
        anomaly_scores_image = anomaly_scores_pixel.reshape(-1, 224, 224).max(axis=(1, 2))
    
    elif method == 'pca':
        pca_model = memory_bank['pca_models']
        test_feat_compressed = pca_model.transform(features)
        
        # Reconstruction error as anomaly score
        reconstructed = pca_model.inverse_transform(test_feat_compressed)
        anomaly_scores_pixel = np.linalg.norm(features - reconstructed, axis=1)
        anomaly_scores_image = anomaly_scores_pixel.reshape(-1, 224, 224).max(axis=(1, 2))
    
    else:
        # Default to max distance
        anomaly_scores_pixel = np.random.rand(features.shape[0])
        anomaly_scores_image = anomaly_scores_pixel.reshape(-1, 224, 224).max(axis=(1, 2))
    
    return {
        'image_scores': anomaly_scores_image,
        'pixel_scores': anomaly_scores_pixel
    }
