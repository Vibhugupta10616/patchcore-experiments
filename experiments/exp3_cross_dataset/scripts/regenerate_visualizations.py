"""
Regenerate visualizations from existing results
"""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from exp3_visualizations import create_exp3_visualizations

# Load results
results_dir = Path(__file__).parent.parent / 'results'
results_file = list(results_dir.glob('exp3_results_*.csv'))[0]

print(f"Loading results from: {results_file}")
results_df = pd.read_csv(results_file)

# Create output directory
output_dir = Path(__file__).parent.parent / 'visualizations'
output_dir.mkdir(parents=True, exist_ok=True)

# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create visualizations
print("Regenerating visualizations...")
create_exp3_visualizations(results_df, output_dir, timestamp)
print(f"Visualizations saved to: {output_dir}")
