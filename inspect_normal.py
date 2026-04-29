import pandas as pd
from pathlib import Path

for i in range(1, 5):
    pattern = f'*human{i}*/*human{i}*.csv'
    files = list(Path('data/grn/input').glob(pattern))
    if not files:
        pattern2 = f'*human{i}*.csv'
        files = list(Path('data/grn/input').glob(pattern2))
    if not files:
        print(f'human{i}: NOT FOUND')
        continue
    f = files[0]
    df = pd.read_csv(f, index_col=0, nrows=3)
    print(f'human{i}: {f.name}')
    print(f'  Shape (3 rows): {df.shape}')
    print(f'  Index name: {df.index.name}')
    print(f'  First index values: {df.index.tolist()}')
    print(f'  First cols: {df.columns.tolist()[:5]}')
    print()
