import scanpy as sc
import json
from pathlib import Path

adata = sc.read_h5ad('data/grn/intermediate/preprocessed.h5ad')

# Extract tumor ductal clusters
tumor_clusters = ['1', '6', '15']
tumor_mask = adata.obs['leiden'].isin(tumor_clusters)
adata_tumor = adata[tumor_mask].copy()

print(f'Tumor cells: {adata_tumor.n_obs}')
print(f'Patients represented:')
print(adata_tumor.obs['patient'].value_counts().to_dict())

# Save tumor subset
out = Path('data/grn/intermediate/tumor_cells.h5ad')
adata_tumor.write_h5ad(out)
print(f'Saved to {out}')
