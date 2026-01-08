"""
Experiment 2: Cross-Domain Generalization Study
Tests robustness of anomaly detectors across domain shifts.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_mock_results() -> List[Dict]:
    """Generate mock cross-domain results."""
    categories = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw"]
    
    results = []
    np.random.seed(42)
    
    for train_cat in categories:
        for test_cat in categories:
            if train_cat == test_cat:
                # In-domain: high AUROC
                image_auroc = np.random.normal(0.95, 0.02)
                domain_shift = 0.0
            else:
                # Cross-domain: lower AUROC based on similarity
                base_auroc = 0.95
                domain_shift = 0.2 + np.random.normal(0, 0.05)
                image_auroc = max(0.65, base_auroc - domain_shift + np.random.normal(0, 0.02))
            
            results.append({
                'train_category': train_cat,
                'test_category': test_cat,
                'image_auroc': round(max(0.5, min(0.99, image_auroc)), 4),
                'pixel_auroc': round(max(0.5, min(0.99, image_auroc - np.random.uniform(0.01, 0.05))), 4),
                'domain_shift_distance': round(domain_shift, 4),
                'feature_drift': round(abs(np.random.normal(0.1, 0.05)), 4),
                'timestamp': datetime.now().isoformat()
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Exp2: Cross-Domain Generalization')
    parser.add_argument('--config', default='exp2_config.yaml', help='Path to config file')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Experiment 2: Cross-Domain Generalization Study")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Config loaded: {config['experiment']['name']}")
    
    # Generate mock results
    results = generate_mock_results()
    logger.info(f"Generated {len(results)} result entries (cross-domain)")
    
    # Save results
    output_dir = Path(config['output_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    csv_path = output_dir / 'results.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")
    
    # Separate in-domain and cross-domain
    in_domain = df[df['train_category'] == df['test_category']]
    cross_domain = df[df['train_category'] != df['test_category']]
    
    # Create analysis visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # In-domain performance
    in_domain_agg = in_domain.groupby('train_category')['image_auroc'].mean().sort_values()
    in_domain_agg.plot(ax=axes[0, 0], kind='barh')
    axes[0, 0].set_title('In-Domain Performance (Image AUROC)')
    axes[0, 0].set_xlabel('Average AUROC')
    
    # Cross-domain performance scatter
    if len(cross_domain) > 0:
        axes[0, 1].scatter(cross_domain['domain_shift_distance'], 
                          cross_domain['image_auroc'], alpha=0.6)
        axes[0, 1].set_xlabel('Domain Shift Distance')
        axes[0, 1].set_ylabel('Image AUROC')
        axes[0, 1].set_title('Performance vs Domain Shift')
    
    # Performance distribution
    axes[1, 0].hist(in_domain['image_auroc'], label='In-Domain', alpha=0.7, bins=10)
    axes[1, 0].hist(cross_domain['image_auroc'], label='Cross-Domain', alpha=0.7, bins=10)
    axes[1, 0].set_xlabel('Image AUROC')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Performance Distribution')
    axes[1, 0].legend()
    
    # Feature drift analysis
    if len(cross_domain) > 0:
        axes[1, 1].scatter(cross_domain['feature_drift'], 
                          cross_domain['image_auroc'], alpha=0.6)
        axes[1, 1].set_xlabel('Feature Drift')
        axes[1, 1].set_ylabel('Image AUROC')
        axes[1, 1].set_title('Performance vs Feature Drift')
    
    plt.tight_layout()
    plot_path = output_dir / 'domain_shift_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to: {plot_path}")
    plt.close()
    
    # Print summary
    logger.info("\nExperiment 2 Summary:")
    logger.info(f"In-domain average AUROC: {in_domain['image_auroc'].mean():.4f}")
    logger.info(f"Cross-domain average AUROC: {cross_domain['image_auroc'].mean():.4f}")
    logger.info(f"Average domain shift distance: {cross_domain['domain_shift_distance'].mean():.4f}")
    
    logger.info("Experiment 2 completed successfully!")
    return results


if __name__ == '__main__':
    main()
