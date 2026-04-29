import scanpy as sc
import json, time
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor
from tqdm import tqdm
import joblib
from contextlib import contextmanager

# Extract pure tumor cluster 6 only
adata_full = sc.read_h5ad('data/grn/intermediate/preprocessed.h5ad')
tumor = adata_full[adata_full.obs['leiden'] == '6'].copy()
print(f'Pure tumor cells: {tumor.n_obs}')
print(f'Patients: {tumor.obs["patient"].value_counts().to_dict()}')

GENE_DATA    = json.loads(Path('data/grn/intermediate/hvg_genes.json').read_text())
PDAC_DRIVERS = GENE_DATA['pdac_drivers']

if hasattr(tumor.X, 'toarray'):
    X_mat = tumor.X.toarray()
else:
    X_mat = np.array(tumor.X)

genes = list(tumor.var_names)
expr  = pd.DataFrame(X_mat, columns=genes)

N_ESTIMATORS   = 150
MAX_DEPTH      = 8
RANDOM_STATE   = 42
MIN_IMPORTANCE = 1e-5
MIN_NORM_IMP   = 0.003

@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCB(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCB
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()

def _fit_target(target, predictor_genes, X_train, expr):
    y = expr[target].values
    if np.std(y) < 1e-9:
        return target, np.zeros(len(predictor_genes), dtype=np.float32)
    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        max_features='sqrt', random_state=RANDOM_STATE, n_jobs=1,
    )
    model.fit(X_train, y)
    return target, model.feature_importances_.astype(np.float32)

X_train = expr[genes].values
n_workers = max(1, __import__('os').cpu_count() - 1)

print(f'Running GENIE3 on {tumor.n_obs} pure tumor cells...')
t0 = time.time()

with tqdm_joblib(tqdm(total=len(genes), desc='GENIE3 pure tumor')):
    results = Parallel(n_jobs=n_workers, verbose=0)(
        delayed(_fit_target)(t, genes, X_train, expr)
        for t in genes
    )

print(f'Done in {(time.time()-t0)/60:.1f} minutes')

rows = []
for target, imps in results:
    total = imps.sum()
    if total < 1e-12:
        continue
    norm_imps = imps / total
    for pi, (raw_imp, norm_imp) in enumerate(zip(imps, norm_imps)):
        if raw_imp < MIN_IMPORTANCE or norm_imp < MIN_NORM_IMP:
            continue
        pred = genes[pi]
        if pred == target:
            continue
        rows.append({
            'Regulator':       pred,
            'Target':          target,
            'raw_importance':  float(raw_imp),
            'norm_importance': float(norm_imp),
            'is_pdac_driver':  pred in set(PDAC_DRIVERS),
        })

edges_df = pd.DataFrame(rows).sort_values('norm_importance', ascending=False)
out = Path('data/grn/intermediate/genie3_pure_tumor_edges.csv')
edges_df.to_csv(out, index=False)
print(f'Saved {len(edges_df)} edges')

print('Top 20 regulators:')
top = edges_df.groupby('Regulator')['norm_importance'].sum() \
              .sort_values(ascending=False).head(20)
for gene, score in top.items():
    flag = ' <-- PDAC DRIVER' if gene in set(PDAC_DRIVERS) else ''
    print(f'  {gene:<15} {score:.4f}{flag}')

# Save tumor h5ad
tumor.write_h5ad('data/grn/intermediate/pure_tumor_cells.h5ad')
