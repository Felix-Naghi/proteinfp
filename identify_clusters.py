import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

adata = sc.read_h5ad('data/grn/intermediate/preprocessed.h5ad')

# Marker genes for each cell type in PDAC
MARKERS = {
    'tumor_ductal':  ['KRT19', 'KRT8', 'KRT18', 'EPCAM', 'MUC1', 'KRAS'],
    'macrophage':    ['SPI1', 'AIF1', 'CD68', 'CSF1R', 'CLEC7A', 'MS4A7'],
    't_cell':        ['CD3D', 'CD3E', 'CD8A', 'CD4', 'FOXP3'],
    'b_cell':        ['CD79A', 'CD79B', 'MS4A1', 'CD19'],
    'fibroblast':    ['COL1A1', 'COL6A3', 'ACTA2', 'FAP', 'CDH11'],
    'acinar':        ['REG1A', 'PRSS1', 'CELA3A', 'AMY2A'],
    'endothelial':   ['PECAM1', 'VWF', 'CLDN5', 'CDH5'],
    'stellate':      ['RGS5', 'PDGFRB', 'MCAM'],
}

# Check which markers are in the dataset
genes_in_data = set(adata.var_names)
print('Markers found in dataset:')
for ct, markers in MARKERS.items():
    found = [m for m in markers if m in genes_in_data]
    print(f'  {ct:<15} {found}')

print()
print('Mean expression per cluster for key markers:')

if hasattr(adata.X, 'toarray'):
    X = adata.X.toarray()
else:
    X = np.array(adata.X)

expr = pd.DataFrame(X, columns=adata.var_names, index=adata.obs_names)
expr['cluster'] = adata.obs['leiden'].values

# For each cell type show top expressing clusters
for ct, markers in MARKERS.items():
    found = [m for m in markers if m in genes_in_data]
    if not found:
        continue
    mean_expr = expr.groupby('cluster')[found].mean().mean(axis=1)
    top2 = mean_expr.sort_values(ascending=False).head(3)
    print(f'  {ct:<15} top clusters: {top2.index.tolist()} scores: {[round(v,3) for v in top2.values]}')
