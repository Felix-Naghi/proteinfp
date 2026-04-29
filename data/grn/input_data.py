"""
inspect_data.py
───────────────
Run this first to understand the PDAC data format.
Place in proteinFP root and run:
    python inspect_data.py
"""

import gzip
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data\grn\input\GSE154778_dgeMtx.csv")

if not DATA_PATH.exists():
    print(f"ERROR: File not found at {DATA_PATH}")
    print("Make sure you placed the file in data/grn/input/")
    exit(1)

print(f"File size: {DATA_PATH.stat().st_size / 1e6:.1f} MB")
print("Reading first few lines...")

# Peek at raw content
with gzip.open(DATA_PATH, 'rt') as f:
    lines = [f.readline() for _ in range(5)]

print("\nFirst 5 lines (raw):")
for i, line in enumerate(lines):
    print(f"  [{i}] {line[:120]}")

print("\nLoading with pandas (first 5 rows, first 10 cols)...")
try:
    df = pd.read_csv(DATA_PATH, index_col=0, nrows=5)
    print(f"\nShape (5 rows): {df.shape}")
    print(f"Index name: {df.index.name}")
    print(f"First index values: {df.index.tolist()}")
    print(f"First column names: {df.columns.tolist()[:10]}")
    print(f"\nDtypes: {df.dtypes.value_counts().to_dict()}")
except Exception as e:
    print(f"Error reading: {e}")

# Full shape
print("\nGetting full dimensions (may take a moment)...")
try:
    df_full = pd.read_csv(DATA_PATH, index_col=0)
    print(f"Full shape: {df_full.shape}")
    print(f"  Rows (likely genes): {df_full.shape[0]}")
    print(f"  Cols (likely cells): {df_full.shape[1]}")
    print(f"\nValue range: min={df_full.values.min():.2f} max={df_full.values.max():.2f}")
    print(f"Sparsity: {(df_full.values == 0).mean():.1%} zeros")
    print(f"\nFirst 5 gene names: {df_full.index.tolist()[:5]}")
    print(f"First 5 cell names: {df_full.columns.tolist()[:5]}")
    print(f"Last 5 gene names:  {df_full.index.tolist()[-5:]}")
except Exception as e:
    print(f"Error: {e}")