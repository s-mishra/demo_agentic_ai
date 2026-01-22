import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
root = Path(__file__).parent.parent
df = pd.read_csv(root / 'data/dengue_cases.csv', parse_dates=['date'])

# Create figure
plt.figure(figsize=(12, 5))
plt.plot(df['date'], df['cases'], marker='o', markersize=4, linewidth=1.5, color='#e63946')
plt.fill_between(df['date'], df['cases'], alpha=0.3, color='#e63946')

plt.title('Dengue Cases - Central Region (2024)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=11)
plt.ylabel('Weekly Cases', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
output_path = root / 'output/dengue_timeseries.png'
plt.savefig(output_path, dpi=150)
print(f"Plot saved to {output_path}")
