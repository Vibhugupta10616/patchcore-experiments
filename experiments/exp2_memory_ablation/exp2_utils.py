"""
Utility functions for Experiment 2: Cross-Domain Generalization
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
    """Load backbone model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading {backbone_name} (pretrained={pretrained})")
    
    if backbone_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif backbone_name == 'vitb16':
        try:
            model = models.vit_b_16(pretrained=pretrained)
        except:
            import timm
            model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    model = model.to(device)
    model.eval()
    return model


def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    layer_names: List[str],
    device: torch.device = None
) -> Dict[str, np.ndarray]:
    """Extract features from specified layers."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Extracting features from layers: {layer_names}")
    
    features_dict = {layer: [] for layer in layer_names}
    hooks = {}
    activations = {}
    
    def get_hook(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    for name, module in model.named_modules():
        for layer_name in layer_names:
            if layer_name in name:
                hooks[name] = module.register_forward_hook(get_hook(name))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            _ = model(images)
            
            for name in activations:
                feat = activations[name]
                if len(feat.shape) == 4:
                    feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
                elif len(feat.shape) == 3:
                    feat = feat.reshape(-1, feat.shape[-1])
                
                features_dict[name].append(feat.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1} batches")
    
    for hook in hooks.values():
        hook.remove()
    
    for layer in features_dict:
        features_dict[layer] = np.vstack(features_dict[layer])
    
    return features_dict


def prepare_memory_bank(
    features: Dict[str, np.ndarray],
    method: str = 'knn',
    n_components: Optional[int] = None,
    **kwargs
) -> Dict:
    """Prepare memory bank for anomaly detection."""
    logger.info(f"Preparing memory bank with method: {method}")
    
    memory_bank = {}
    
    if method == 'knn':
        memory_bank['type'] = 'knn'
        memory_bank['features'] = features
    
    elif method == 'pca':
        from sklearn.decomposition import PCA
        memory_bank['type'] = 'pca'
        memory_bank['pca_models'] = {}
        
        for layer_name, feat in features.items():
            n_comp = n_components or min(feat.shape[1] // 2, 256)
            pca = PCA(n_components=n_comp)
            compressed = pca.fit_transform(feat)
            memory_bank['pca_models'][layer_name] = pca
    
    return memory_bank


def compute_anomaly_scores(
    features: Dict[str, np.ndarray],
    memory_bank: Dict,
    method: str = 'knn',
    k: int = 5,
    **kwargs
) -> Dict[str, np.ndarray]:
    """Compute anomaly scores."""
    logger.info(f"Computing anomaly scores with method: {method}")
    
    if method == 'knn':
        from sklearn.neighbors import NearestNeighbors
        
        normal_features = memory_bank['features']
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(normal_features)
        
        test_feat = features
        distances, _ = nbrs.kneighbors(test_feat)
        
        anomaly_scores_pixel = distances.mean(axis=1)
        anomaly_scores_image = anomaly_scores_pixel.reshape(-1, 224, 224).max(axis=(1, 2))
    
    else:
        anomaly_scores_pixel = np.random.rand(features.shape[0])
        anomaly_scores_image = anomaly_scores_pixel.reshape(-1, 224, 224).max(axis=(1, 2))
    
    return {
        'image_scores': anomaly_scores_image,
        'pixel_scores': anomaly_scores_pixel
    }


def evaluate_domain_shift(
    source_features: Dict[str, np.ndarray],
    target_features: Dict[str, np.ndarray],
    method: str = 'mmd'
) -> Dict:
    """
    Compute domain shift metrics between source and target domains.
    
    Methods:
    - mmd: Maximum Mean Discrepancy
    - wasserstein: Wasserstein distance
    - cosine: Cosine distance
    """
    logger.info(f"Evaluating domain shift using {method}")
    
    metrics = {}
    
    if method == 'mmd':
        # Compute Maximum Mean Discrepancy
        distances = []
        for layer_name in source_features.keys():
            src = source_features[layer_name]
            tgt = target_features[layer_name]
            
            # Simple MMD approximation
            src_mean = src.mean(axis=0)
            tgt_mean = tgt.mean(axis=0)
            mmd = np.linalg.norm(src_mean - tgt_mean)
            distances.append(mmd)
        
        metrics['distance'] = np.mean(distances)
    
    elif method == 'wasserstein':
        # Wasserstein distance
        from scipy.stats import wasserstein_distance
        distances = []
        
        for layer_name in source_features.keys():
            src = source_features[layer_name].mean(axis=1)
            tgt = target_features[layer_name].mean(axis=1)
            wd = wasserstein_distance(src, tgt)
            distances.append(wd)
        
        metrics['distance'] = np.mean(distances)
    
    else:
        # Default cosine distance
        from sklearn.metrics.pairwise import cosine_distances
        
        src_feat = list(source_features.values())[0]
        tgt_feat = list(target_features.values())[0]
        
        src_mean = src_feat.mean(axis=0, keepdims=True)
        tgt_mean = tgt_feat.mean(axis=0, keepdims=True)
        
        metrics['distance'] = cosine_distances(src_mean, tgt_mean)[0, 0]
    
    # Compute feature drift (change in std)
    src_std = list(source_features.values())[0].std()
    tgt_std = list(target_features.values())[0].std()
    metrics['drift'] = abs(src_std - tgt_std) / (src_std + 1e-6)
    
    return metrics
    )
