import scanpy as sc
import numpy as np
import pandas as pd

adata = sc.read_h5ad('data/grn/intermediate/tumor_cells.h5ad')

if hasattr(adata.X, 'toarray'):
    X = adata.X.toarray()
else:
    X = np.array(adata.X)

expr = pd.DataFrame(X, columns=adata.var_names)
expr['cluster'] = adata.obs['leiden'].values
expr['patient'] = adata.obs['patient'].values

# Key discriminating markers
tumor_markers   = ['KRT19', 'KRT8', 'KRT18', 'EPCAM', 'KRAS', 'MYC', 'TP53']
acinar_markers  = ['PRSS1', 'REG1A', 'SPINK1', 'AMY2A', 'CELA3A']
immune_markers  = ['LYZ', 'SPI1', 'AIF1', 'CD68']

print('Per cluster mean expression:')
print()
for cluster in ['1', '6', '15']:
    sub = expr[expr['cluster'] == cluster]
    print(f'Cluster {cluster} ({len(sub)} cells):')
    for label, markers in [('Tumor', tumor_markers),
                            ('Acinar', acinar_markers),
                            ('Immune', immune_markers)]:
        found = [m for m in markers if m in expr.columns]
        scores = sub[found].mean().mean() if found else 0
        print(f'  {label:<8} score: {scores:.3f}  markers: {found}')
    print()
