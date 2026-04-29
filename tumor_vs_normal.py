import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Load normal ductal cells
print('Loading normal ductal cells...')
normal_dfs = []
for i in range(1, 5):
    files = list(Path('data/grn/input').glob(f'*human{i}*/*human{i}*.csv'))
    if not files:
        files = list(Path('data/grn/input').glob(f'*human{i}*.csv'))
    if not files:
        continue
    df = pd.read_csv(files[0], index_col=0)
    ductal = df[df['assigned_cluster'] == 'ductal'].drop(
        columns=['barcode', 'assigned_cluster'], errors='ignore'
    )
    normal_dfs.append(ductal)
    print(f'  human{i}: {len(ductal)} ductal cells')

normal_df = pd.concat(normal_dfs)
print(f'  Total normal ductal: {len(normal_df)} cells')

# Load tumor ductal cells (cluster 6)
print('Loading tumor ductal cells...')
import scanpy as sc
adata = sc.read_h5ad('data/grn/intermediate/preprocessed.h5ad')
tumor_mask = adata.obs['leiden'] == '6'
adata_tumor = adata[tumor_mask]

if hasattr(adata_tumor.X, 'toarray'):
    X = adata_tumor.X.toarray()
else:
    X = np.array(adata_tumor.X)

tumor_df = pd.DataFrame(X, columns=adata_tumor.var_names)
print(f'  Total tumor ductal: {len(tumor_df)} cells')

# Find common genes
common_genes = list(set(normal_df.columns) & set(tumor_df.columns))
print(f'  Common genes: {len(common_genes)}')

normal_common = normal_df[common_genes]
tumor_common  = tumor_df[common_genes]

# Our drug targets
targets = {
    'ATAD2':   'Q6PL18',
    'TOP2A':   'P11388',
    'STMN1':   'P16949',
    'SLC2A1':  'P11166',
    'CLSPN':   'Q9HAW4',
    'HELLS':   'Q9NRZ9',
    'LCN2':    'P80188',
    'CEACAM6': 'P40199',
}

print()
print('=' * 75)
print('  TUMOR vs NORMAL DUCTAL EXPRESSION COMPARISON')
print('=' * 75)
print(f'  {"Gene":<10} {"Normal mean":<14} {"Tumor mean":<13} '
      f'{"Fold change":<13} {"p-value":<12} {"Priority"}')
print(f'  {"-"*10} {"-"*13} {"-"*12} {"-"*12} {"-"*11} {"-"*10}')

results = []
for gene in targets:
    if gene not in common_genes:
        print(f'  {gene:<10} NOT IN COMMON GENES')
        continue

    normal_expr = normal_common[gene].values
    tumor_expr  = tumor_common[gene].values

    normal_mean = float(np.mean(normal_expr))
    tumor_mean  = float(np.mean(tumor_expr))

    # Log2 fold change (add pseudocount)
    fc = (tumor_mean + 0.01) / (normal_mean + 0.01)
    log2fc = np.log2(fc)

    # Mann-Whitney test
    stat, pval = stats.mannwhitneyu(
        tumor_expr, normal_expr, alternative='greater'
    )

    # Priority: high fold change + significant + druggable
    if log2fc > 2 and pval < 0.05:
        priority = 'HIGH'
    elif log2fc > 1 and pval < 0.05:
        priority = 'MEDIUM'
    elif log2fc > 0:
        priority = 'LOW'
    else:
        priority = 'NOT ELEVATED'

    results.append((gene, normal_mean, tumor_mean, log2fc, pval, priority))

results.sort(key=lambda x: -x[3])

for gene, nm, tm, fc, pval, pri in results:
    pval_str = f'{pval:.2e}'
    print(f'  {gene:<10} {nm:<14.3f} {tm:<13.3f} '
          f'{fc:<+13.2f} {pval_str:<12} {pri}')

print()
print('=' * 75)
print('  FINAL DRUG TARGET PRIORITY (tumor-specific + druggable):')
print('=' * 75)
high = [(g, fc) for g, nm, tm, fc, pval, pri in results if pri == 'HIGH']
med  = [(g, fc) for g, nm, tm, fc, pval, pri in results if pri == 'MEDIUM']

for rank, (gene, fc) in enumerate(sorted(high + med, key=lambda x: -x[1]), 1):
    uid = targets[gene]
    print(f'  {rank}. {gene:<10} log2FC={fc:+.2f}  UniProt={uid}')

import json
out = {g: {'log2fc': fc, 'normal_mean': nm, 'tumor_mean': tm,
           'pval': pval, 'priority': pri}
       for g, nm, tm, fc, pval, pri in results}
Path('data/grn/intermediate/tumor_vs_normal.json').write_text(
    json.dumps(out, indent=2)
)
print()
print('Saved to data/grn/intermediate/tumor_vs_normal.json')
