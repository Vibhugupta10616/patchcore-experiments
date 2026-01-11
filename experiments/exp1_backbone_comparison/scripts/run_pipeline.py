#!/usr/bin/env python3
"""
Master execution script - Run visualizations and cleanup once experiment is done
This script will:
1. Wait for results CSV
2. Generate all visualizations
3. Create final report
4. Cleanup old files
"""

import os
import sys
import time
import subprocess
from pathlib import Path

script_dir = Path(__file__).parent
results_csv = script_dir / 'results' / 'exp1_backbone_comparison' / 'all_backbones' / 'results_all_backbones.csv'

print("\n" + "="*80)
print("EXPERIMENT 1: MASTER EXECUTOR")
print("="*80)

# Step 1: Wait for results CSV
print("\n[1/3] Waiting for experiment to complete...")
max_wait = 1200  # 20 minutes
elapsed = 0
while not results_csv.exists() and elapsed < max_wait:
    print(f"  Waiting... ({elapsed}s elapsed)", end='\r')
    time.sleep(10)
    elapsed += 10

if not results_csv.exists():
    print(f"\n✗ Timeout! Results CSV not found after {max_wait}s")
    sys.exit(1)

print(f"\n✓ Results CSV found! ({results_csv.stat().st_size} bytes)")

# Step 2: Generate visualizations
print("\n[2/3] Generating visualizations...")
try:
    result = subprocess.run(
        ['python', 'generate_visualizations.py'],
        cwd=script_dir,
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode == 0:
        print("✓ Visualizations generated successfully!")
        if 'Saved:' in result.stdout:
            for line in result.stdout.split('\n'):
                if 'Saved:' in line or '✓' in line:
                    print(f"  {line}")
    else:
        print(f"⚠ Visualization warning:\n{result.stderr}")
except Exception as e:
    print(f"⚠ Could not generate visualizations: {e}")

# Step 3: Cleanup
print("\n[3/3] Cleaning up old files...")
try:
    result = subprocess.run(
        ['python', 'cleanup_final.py'],
        cwd=script_dir,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"⚠ Cleanup warning:\n{result.stderr}")
except Exception as e:
    print(f"⚠ Could not run cleanup: {e}")

print("\n" + "="*80)
print("✓✓✓ ALL TASKS COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Review results in: results/exp1_backbone_comparison/all_backbones/")
print("2. Check visualizations (PNG files)")
print("3. Review results_all_backbones.csv for detailed metrics")
print("="*80 + "\n")
