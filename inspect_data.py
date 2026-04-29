import pandas as pd
from pathlib import Path

DATA_PATH = Path('data/grn/input/GSE154778_dgeMtx.csv')

print(f'File size: {DATA_PATH.stat().st_size / 1e6:.1f} MB')
print('Reading first 5 rows...')

df = pd.read_csv(DATA_PATH, index_col=0, nrows=5)
print(f'Shape (5 rows): {df.shape}')
print(f'Index name: {df.index.name}')
print(f'First index values: {df.index.tolist()}')
print(f'First column names: {df.columns.tolist()[:5]}')

print('Getting full shape (may take 30s)...')
df_full = pd.read_csv(DATA_PATH, index_col=0)
print(f'Full shape: {df_full.shape}')
print(f'  Rows: {df_full.shape[0]}')
print(f'  Cols: {df_full.shape[1]}')
print(f'Value range: min={df_full.values.min()} max={df_full.values.max()}')
print(f'Sparsity: {(df_full.values == 0).mean():.1%} zeros')
print(f'First 5 row names: {df_full.index.tolist()[:5]}')
print(f'First 5 col names: {df_full.columns.tolist()[:5]}')
