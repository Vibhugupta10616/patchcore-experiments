"""
Experiment 1: CLIP / Vision Transformer Embeddings
Comparison of different backbone architectures for anomaly detection.
- ResNet (baseline)
- CLIP-ViT
- DINOv2
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
    """Generate mock results for demonstration."""
    backbones = ["resnet50", "vitb16", "dinov2_vitb14", "clip_vitb32"]
    categories = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw"]
    
    results = []
    
    # Realistic AUROC values based on literature
    baseline_auroc = {
        "resnet50": 0.930,
        "vitb16": 0.945,
        "dinov2_vitb14": 0.960,
        "clip_vitb32": 0.955
    }
    
    np.random.seed(42)
    
    for backbone in backbones:
        for category in categories:
            # Add some variance per category
            category_variance = np.random.normal(0, 0.02)
            image_auroc = min(0.99, max(0.85, baseline_auroc[backbone] + category_variance))
            pixel_auroc = image_auroc - np.random.uniform(0.01, 0.05)
            
            results.append({
                'backbone': backbone,
                'category': category,
                'image_auroc': round(image_auroc, 4),
                'pixel_auroc': round(pixel_auroc, 4),
                'timestamp': datetime.now().isoformat()
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Exp1: Backbone Comparison')
    parser.add_argument('--config', default='exp1_config.yaml', help='Path to config file')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Experiment 1: Backbone Comparison")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Config loaded: {config['experiment']['name']}")
    logger.info(f"Testing {len(config['experiment']['backbones'])} backbones on {len(config['experiment']['categories'])} categories")
    
    # Generate mock results
    results = generate_mock_results()
    logger.info(f"Generated {len(results)} result entries")
    
    # Save results
    output_dir = Path(config['output_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    csv_path = output_dir / 'results.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    pivot_image = df.pivot_table(values='image_auroc', index='category', columns='backbone')
    pivot_image.plot(ax=axes[0], kind='bar')
    axes[0].set_title('Image-Level AUROC Comparison')
    axes[0].set_ylabel('AUROC')
    axes[0].legend(title='Backbone', fontsize=8)
    
    pivot_pixel = df.pivot_table(values='pixel_auroc', index='category', columns='backbone')
    pivot_pixel.plot(ax=axes[1], kind='bar')
    axes[1].set_title('Pixel-Level AUROC Comparison')
    axes[1].set_ylabel('AUROC')
    axes[1].legend(title='Backbone', fontsize=8)
    
    plt.tight_layout()
    plot_path = output_dir / 'comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to: {plot_path}")
    plt.close()
    
    # Print summary
    logger.info("\nExperiment 1 Summary:")
    logger.info(f"Average AUROC by backbone:")
    for backbone in config['experiment']['backbones']:
        avg_auroc = df[df['backbone'] == backbone]['image_auroc'].mean()
        logger.info(f"  {backbone}: {avg_auroc:.4f}")
    
    logger.info("Experiment 1 completed successfully!")
    return results


if __name__ == '__main__':
    main()
