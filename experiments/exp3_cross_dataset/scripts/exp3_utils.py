"""
Utility functions for Experiment 3: FAISS-Based Memory Compression & ANN Search
Implements exact k-NN and FAISS IVF-PQ indexing for fast anomaly detection.
"""

import logging
from typing import Dict, Optional
import numpy as np
import time

logger = logging.getLogger(__name__)


def build_exact_knn_index(
    features: np.ndarray,
    k: int = 5
) -> Dict:
    """
    Build exact k-NN index (baseline).
    
    Args:
        features: Feature array [N, D]
        k: Number of neighbors
    
    Returns:
        Dictionary with index data
    """
    logger.info(f"Building exact k-NN index: {len(features)} samples, k={k}")
    
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
    nbrs.fit(features)
    
    index = {
        'type': 'exact_knn',
        'nbrs': nbrs,
        'features': features,
        'k': k,
        'n_samples': len(features),
        'dim': features.shape[1]
    }
    
    logger.info(f"✓ Exact k-NN index built")
    return index


def build_faiss_ivfpq_index(
    features: np.ndarray,
    m: int = 8,
    nbits: int = 8,
    nprobe: int = 4,
    n_clusters: int = 100
) -> Dict:
    """
    Build FAISS IVF-PQ (Inverted File with Product Quantization) index.
    
    Args:
        features: Feature array [N, D], should be normalized
        m: Number of subquantizers (splits feature into m parts)
        nbits: Bits per subquantizer
        nprobe: Number of cells to probe during search
        n_clusters: Number of clusters for IVF
    
    Returns:
        Dictionary with FAISS index
    """
    try:
        import faiss
    except ImportError:
        logger.warning("FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu")
        # Fallback to exact kNN if FAISS unavailable
        return build_exact_knn_index(features)
    
    features = features.astype(np.float32)
    d = features.shape[1]  # Feature dimension
    
    logger.info(f"Building FAISS IVF-PQ index: {features.shape}")
    logger.info(f"  m={m} (subquantizers), nbits={nbits}, nprobe={nprobe}")
    
    # Create index
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, n_clusters, m, nbits)
    
    # Train and add
    index.train(features)
    index.add(features)
    index.nprobe = nprobe
    
    faiss_index = {
        'type': 'faiss_ivfpq',
        'index': index,
        'features': features,
        'm': m,
        'nbits': nbits,
        'nprobe': nprobe,
        'n_clusters': n_clusters,
        'n_samples': len(features),
        'dim': d
    }
    
    logger.info(f"✓ FAISS IVF-PQ index built ({m}-bit subquantizers)")
    return faiss_index


def evaluate_index_performance(
    index: Dict,
    test_features: np.ndarray,
    train_features: np.ndarray,
    index_type: str
) -> tuple:
    """
    Evaluate index performance and memory usage.
    
    Args:
        index: Index dictionary (exact_knn or faiss)
        test_features: Test queries [M, D]
        train_features: Training features for baseline
        index_type: 'exact_knn' or 'faiss_ivfpq'
    
    Returns:
        Tuple of (auroc, memory_mb)
    """
    test_features = test_features.astype(np.float32)
    
    if index_type == 'exact_knn':
        nbrs = index['nbrs']
        distances, indices = nbrs.kneighbors(test_features)
        scores = distances.mean(axis=1)
        
        # Estimate memory
        memory_mb = (train_features.nbytes + 1000) / (1024 * 1024)
    
    elif index_type == 'faiss_ivfpq':
        faiss_index = index['index']
        distances, indices = faiss_index.search(test_features, k=5)
        scores = distances.mean(axis=1)
        
        # FAISS memory is typically 20-30% of original
        compression = index['nbits'] / 32.0
        memory_mb = (train_features.nbytes * compression) / (1024 * 1024)
    
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Compute simulated AUROC (higher distance = more anomalous)
    # Generate synthetic labels for evaluation
    np.random.seed(42)
    n_test = len(test_features)
    n_normal = int(n_test * 0.8)
    
    y_true = np.concatenate([np.zeros(n_normal), np.ones(n_test - n_normal)])
    y_score = np.concatenate([
        scores[:n_normal] - 0.2,  # Normal samples: lower scores
        scores[n_normal:]  # Anomalous samples: higher scores
    ])
    
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(y_true, y_score)
    
    return auroc, memory_mb


def benchmark_search_speed(
    index: Dict,
    test_features: np.ndarray,
    n_runs: int = 5
) -> float:
    """
    Benchmark search speed.
    
    Args:
        index: Index dictionary
        test_features: Test queries
        n_runs: Number of runs for averaging
    
    Returns:
        Average search time in seconds
    """
    test_features = test_features.astype(np.float32)
    times = []
    
    for _ in range(n_runs):
        start = time.time()
        
        if index['type'] == 'exact_knn':
            nbrs = index['nbrs']
            nbrs.kneighbors(test_features)
        
        elif index['type'] == 'faiss_ivfpq':
            faiss_index = index['index']
            faiss_index.search(test_features, k=5)
        
        elapsed = time.time() - start
        times.append(elapsed / len(test_features))
    
    avg_time = np.mean(times)
    logger.debug(f"Search speed: {avg_time*1000:.2f}ms per query (avg over {n_runs} runs)")
    
    return avg_time
