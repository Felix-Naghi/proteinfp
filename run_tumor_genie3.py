import json, time
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor
from tqdm import tqdm
import joblib
from contextlib import contextmanager

ROOT      = Path('.')
OUT_EDGES = ROOT / 'data/grn/intermediate/genie3_tumor_edges.csv'
GENE_DATA = json.loads(Path('data/grn/intermediate/hvg_genes.json').read_text())
PDAC_DRIVERS = GENE_DATA['pdac_drivers']

N_ESTIMATORS   = 100
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

def _fit_target(target, predictor_genes, X_train, expr_sub):
    y = expr_sub[target].values
    if np.std(y) < 1e-9:
        return target, np.zeros(len(predictor_genes), dtype=np.float32)
    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        max_features='sqrt', random_state=RANDOM_STATE, n_jobs=1,
    )
    model.fit(X_train, y)
    return target, model.feature_importances_.astype(np.float32)

print('Loading tumor cells...')
adata = sc.read_h5ad('data/grn/intermediate/tumor_cells.h5ad')
print(f'  {adata.n_obs} cells x {adata.n_vars} genes')

if hasattr(adata.X, 'toarray'):
    X_mat = adata.X.toarray()
else:
    X_mat = np.array(adata.X)

genes = list(adata.var_names)
expr  = pd.DataFrame(X_mat, columns=genes)

predictor_genes = genes
target_genes    = genes

X_train = expr[predictor_genes].values

print(f'Running GENIE3: {len(predictor_genes)} predictors x {len(target_genes)} targets...')
t0 = time.time()
n_workers = max(1, __import__('os').cpu_count() - 1)

with tqdm_joblib(tqdm(total=len(target_genes), desc='GENIE3 tumor')):
    results = Parallel(n_jobs=n_workers, verbose=0)(
        delayed(_fit_target)(t, predictor_genes, X_train, expr)
        for t in target_genes
    )

elapsed = time.time() - t0
print(f'Done in {elapsed/60:.1f} minutes')

rows = []
for target, imps in results:
    total = imps.sum()
    if total < 1e-12:
        continue
    norm_imps = imps / total
    for pi, (raw_imp, norm_imp) in enumerate(zip(imps, norm_imps)):
        if raw_imp < MIN_IMPORTANCE or norm_imp < MIN_NORM_IMP:
            continue
        pred = predictor_genes[pi]
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
edges_df.to_csv(OUT_EDGES, index=False)
print(f'Saved {len(edges_df)} edges to {OUT_EDGES}')

print('Top 15 regulators:')
top = edges_df.groupby('Regulator')['norm_importance'].sum().sort_values(ascending=False).head(15)
for gene, score in top.items():
    flag = ' <- PDAC DRIVER' if gene in set(PDAC_DRIVERS) else ''
    print(f'  {gene:<12} {score:.4f}{flag}')
