import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


with open('data.json', 'r') as f:
    data = json.load(f)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data.values())
df['Speedup'] = df['Speedup'].str.rstrip('x').astype(float)
speedup_pivot = df.pivot(index='NUM_TREES', columns='MAX_SIZE', values='Speedup')
recall_pivot = df.pivot(index='NUM_TREES', columns='MAX_SIZE', values='Recall')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Create heatmap for Speedup
sns.heatmap(speedup_pivot, ax=ax1, cmap='YlOrRd', annot=True, fmt='.2f', cbar_kws={'label': 'Speedup'})
ax1.set_title('Speedup Heatmap')
ax1.set_xlabel('MAX_SIZE')
ax1.set_ylabel('NUM_TREES')

# Create heatmap for Recall
sns.heatmap(recall_pivot, ax=ax2, cmap='YlGnBu', annot=True, fmt='.2f', cbar_kws={'label': 'Recall'})
ax2.set_title('Recall Heatmap')
ax2.set_xlabel('MAX_SIZE')
ax2.set_ylabel('NUM_TREES')

plt.tight_layout()
plt.show()
