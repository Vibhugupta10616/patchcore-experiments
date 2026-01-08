#!/usr/bin/env python3
"""
Master runner for all experiments with organized logging.
Executes experiments sequentially and saves results in minimal files.
"""

import subprocess
import logging
from pathlib import Path
from datetime import datetime
import json
import sys

# Create logs directory
LOG_DIR = Path(__file__).parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# Timestamp for log files
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Setup logging
log_file = LOG_DIR / f'experiments_run_{TIMESTAMP}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Experiment definitions
EXPERIMENTS = [
    {
        'name': 'Experiment 1: Backbone Comparison',
        'dir': 'exp1_backbone_comparison',
        'script': 'exp1_main.py'
    },
    {
        'name': 'Experiment 2: Cross-Domain Generalization',
        'dir': 'exp2_memory_ablation',
        'script': 'exp2_main.py'
    },
    {
        'name': 'Experiment 3: Feature Fusion Strategy',
        'dir': 'exp3_cross_dataset',
        'script': 'exp3_main.py'
    }
]


def run_experiment(exp_config: dict) -> dict:
    """Run a single experiment and capture output."""
    exp_dir = Path(__file__).parent / exp_config['dir']
    script = exp_dir / exp_config['script']
    
    # Use the venv Python executable
    venv_python = Path(__file__).parent.parent / 'radl' / 'Scripts' / 'python.exe'
    if not venv_python.exists():
        venv_python = 'python'  # Fallback to system python
    
    logger.info('=' * 80)
    logger.info(f"Starting: {exp_config['name']}")
    logger.info(f"Directory: {exp_dir}")
    logger.info('=' * 80)
    
    result = {
        'name': exp_config['name'],
        'status': 'FAILED',
        'error': ''
    }
    
    try:
        # Run experiment with venv python
        proc = subprocess.run(
            [str(venv_python), str(script)],
            cwd=str(exp_dir),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Log output
        if proc.stdout:
            logger.info(proc.stdout)
        if proc.stderr:
            logger.warning(proc.stderr)
        
        if proc.returncode == 0:
            result['status'] = 'SUCCESS'
            logger.info(f"[SUCCESS] {exp_config['name']} completed successfully")
        else:
            logger.error(f"[FAILED] {exp_config['name']} failed with code {proc.returncode}")
            result['error'] = proc.stderr
    
    except subprocess.TimeoutExpired:
        result['error'] = 'Experiment timed out (5 minutes)'
        logger.error(f"[FAILED] {exp_config['name']} timed out")
    
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"[FAILED] {exp_config['name']}: {e}")
    
    logger.info('=' * 80 + '\n')
    
    return result


def main():
    """Run all experiments and save results."""
    logger.info("Starting master experiment runner")
    logger.info(f"Log file: {log_file}")
    logger.info('')
    
    all_results = []
    
    # Run each experiment
    for exp_config in EXPERIMENTS:
        result = run_experiment(exp_config)
        all_results.append(result)
    
    # Save summary
    summary = {
        'timestamp': TIMESTAMP,
        'total_experiments': len(EXPERIMENTS),
        'successful': sum(1 for r in all_results if r['status'] == 'SUCCESS'),
        'failed': sum(1 for r in all_results if r['status'] == 'FAILED'),
        'experiments': all_results
    }
    
    summary_file = LOG_DIR / 'experiments_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total: {summary['total_experiments']}")
    logger.info(f"Successful: {summary['successful']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Summary file: {summary_file}")
    logger.info("=" * 80)
    
    # Print results table
    logger.info("\nDetailed Results:")
    logger.info(f"{'Experiment':<50} {'Status':<12}")
    logger.info("-" * 80)
    for result in all_results:
        status = result['status']
        logger.info(f"{result['name']:<50} {status:<12}")
        if result['error']:
            logger.info(f"  Error: {result['error']}")
    
    return summary


if __name__ == '__main__':
    summary = main()
    sys.exit(0 if summary['failed'] == 0 else 1)
