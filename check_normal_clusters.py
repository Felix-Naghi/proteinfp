import pandas as pd
from pathlib import Path

all_clusters = {}
for i in range(1, 5):
    files = list(Path('data/grn/input').glob(f'*human{i}*/*human{i}*.csv'))
    if not files:
        files = list(Path('data/grn/input').glob(f'*human{i}*.csv'))
    if not files:
        continue
    df = pd.read_csv(files[0], index_col=0)
    counts = df['assigned_cluster'].value_counts()
    print(f'human{i}: {len(df)} cells')
    for cluster, count in counts.items():
        all_clusters[cluster] = all_clusters.get(cluster, 0) + count
        print(f'  {cluster:<20} {count}')
    print()

print('All cell types across 4 donors:')
for ct, count in sorted(all_clusters.items(), key=lambda x: -x[1]):
    print(f'  {ct:<20} {count}')
