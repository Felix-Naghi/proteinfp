"""
grn/02_genie3.py
─────────────────
Module GRN-02 — GENIE3 edge inference for PDAC GRN.

Takes preprocessed scRNA-seq data and infers regulatory edges
using ExtraTrees regression (GENIE3 algorithm).

For each target gene, fits an ExtraTrees model using all predictor
genes as features. Feature importances become edge weights.

Input:  data/grn/intermediate/preprocessed.h5ad
Output: data/grn/intermediate/genie3_edges.csv

Runtime estimate on 14924 cells × 3036 genes:
  ~20-40 minutes on CPU (all cores)
  Use --max-cells 3000 for a quick test run (~3 minutes)

Usage:
    python grn/02_genie3.py
    python grn/02_genie3.py --max-cells 3000   # quick test
    python grn/02_genie3.py --max-targets 1000  # fewer targets
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor
from tqdm import tqdm
import joblib
from contextlib import contextmanager

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parent.parent
IN_H5AD      = ROOT / "data" / "grn" / "intermediate" / "preprocessed.h5ad"
IN_GENES     = ROOT / "data" / "grn" / "intermediate" / "hvg_genes.json"
OUT_EDGES    = ROOT / "data" / "grn" / "intermediate" / "genie3_edges.csv"
OUT_STATS    = ROOT / "data" / "grn" / "intermediate" / "genie3_stats.json"

# ── GENIE3 parameters ─────────────────────────────────────────────────────────

N_ESTIMATORS      = 100
MAX_DEPTH         = 8
RANDOM_STATE      = 42
MIN_IMPORTANCE    = 1e-5
MIN_NORM_IMP      = 0.005

# ── PDAC drivers (used as priority predictors) ────────────────────────────────

PDAC_DRIVERS = [
    "KRAS", "TP53", "SMAD4", "CDKN2A",
    "EGFR", "ERBB2", "MET", "BRAF", "RAF1",
    "MAP2K1", "MAPK1", "MAPK3", "MAPK8",
    "PIK3CA", "AKT1", "MTOR",
    "CCND1", "CDK4", "CDK6", "RB1", "E2F1",
    "BCL2", "BCL2L1", "BAX", "CASP3", "MDM2",
    "TGFB1", "SMAD2", "SMAD3", "SMAD4",
    "MYC", "JUN", "FOS",
    "CDH1", "VIM", "MMP2", "MMP9",
    "CD274", "ACTA2", "COL1A1",
]


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


def _fit_target(target: str, predictor_genes: list, X_train: np.ndarray,
                expr_sub: pd.DataFrame) -> tuple:
    y = expr_sub[target].values
    if np.std(y) < 1e-9:
        return target, np.zeros(len(predictor_genes), dtype=np.float32)
    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(X_train, y)
    return target, model.feature_importances_.astype(np.float32)


def run_genie3(
    max_cells:    int = 0,
    max_targets:  int = 0,
    n_workers:    int = -1,
) -> pd.DataFrame:

    print("=" * 65)
    print("  GRN-02: GENIE3 Edge Inference")
    print("=" * 65)

    # ── Load preprocessed data ────────────────────────────────────────────────
    print(f"\n[1/5] Loading preprocessed data...")
    adata = sc.read_h5ad(IN_H5AD)
    gene_data = json.loads(IN_GENES.read_text())
    final_genes = gene_data["final_genes"]
    pdac_drivers = gene_data["pdac_drivers"]

    print(f"  Cells: {adata.n_obs}  Genes: {adata.n_vars}")

    # ── Build expression DataFrame ────────────────────────────────────────────
    print(f"\n[2/5] Building expression matrix...")
    if hasattr(adata.X, "toarray"):
        X_mat = adata.X.toarray()
    else:
        X_mat = np.array(adata.X)

    expr_df = pd.DataFrame(X_mat, columns=adata.var_names, index=adata.obs_names)

    # Subsample cells if requested
    if max_cells and expr_df.shape[0] > max_cells:
        rng      = np.random.default_rng(RANDOM_STATE)
        idx      = rng.choice(expr_df.shape[0], size=max_cells, replace=False)
        expr_sub = expr_df.iloc[idx].copy()
        print(f"  Subsampled: {expr_df.shape[0]} → {max_cells} cells")
    else:
        expr_sub = expr_df.copy()
        print(f"  Using all {expr_sub.shape[0]} cells")

    # ── Define predictor and target gene sets ─────────────────────────────────
    print(f"\n[3/5] Defining predictor/target pools...")
    genes_available = set(expr_sub.columns)

    # Predictors: all genes (GENIE3 standard)
    predictor_genes = [g for g in final_genes if g in genes_available]

    # Targets: all genes
    target_genes = [g for g in final_genes if g in genes_available]
    if max_targets and len(target_genes) > max_targets:
        # Prioritize PDAC drivers in targets
        driver_targets  = [g for g in pdac_drivers if g in set(target_genes)]
        other_targets   = [g for g in target_genes if g not in set(driver_targets)]
        target_genes    = driver_targets + other_targets[:max_targets - len(driver_targets)]
        print(f"  Targets capped at {max_targets} "
              f"({len(driver_targets)} PDAC drivers prioritized)")

    print(f"  Predictors: {len(predictor_genes)}")
    print(f"  Targets:    {len(target_genes)}")

    X_train = expr_sub[predictor_genes].values

    # ── Run ExtraTrees for each target ────────────────────────────────────────
    print(f"\n[4/5] Running ExtraTrees ({N_ESTIMATORS} trees, "
          f"max_depth={MAX_DEPTH})...")
    print(f"  This may take 20-40 minutes for full dataset.")
    print(f"  Use --max-cells 3000 for a ~3 minute test run.\n")

    n_workers_actual = max(1, __import__('os').cpu_count() - 1) \
                       if n_workers == -1 else n_workers

    t0 = time.time()
    with tqdm_joblib(tqdm(total=len(target_genes), desc="GENIE3")):
        results = Parallel(n_jobs=n_workers_actual, verbose=0)(
            delayed(_fit_target)(t, predictor_genes, X_train, expr_sub)
            for t in target_genes
        )
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f} minutes")

    # ── Build edge dataframe ──────────────────────────────────────────────────
    print(f"\n[5/5] Filtering edges...")
    rows = []
    for target, imps in results:
        total = imps.sum()
        if total < 1e-12:
            continue
        norm_imps = imps / total
        for pi, (raw_imp, norm_imp) in enumerate(zip(imps, norm_imps)):
            if raw_imp  < MIN_IMPORTANCE:
                continue
            if norm_imp < MIN_NORM_IMP:
                continue
            predictor = predictor_genes[pi]
            if predictor == target:
                continue
            rows.append({
                "Regulator":      predictor,
                "Target":         target,
                "raw_importance": float(raw_imp),
                "norm_importance":float(norm_imp),
                "is_pdac_driver": predictor in set(pdac_drivers),
            })

    edges_df = pd.DataFrame(rows)
    print(f"  Raw edges:    {len(rows)}")

    if edges_df.empty:
        print("  WARNING: No edges survived filtering.")
        print("  Try lowering MIN_NORM_IMP in the script.")
        return edges_df

    # Sort by importance
    edges_df = edges_df.sort_values("norm_importance", ascending=False)

    # Save
    edges_df.to_csv(OUT_EDGES, index=False)
    print(f"  Saved {len(edges_df)} edges → {OUT_EDGES}")

    stats = {
        "n_edges":       len(edges_df),
        "n_regulators":  int(edges_df["Regulator"].nunique()),
        "n_targets":     int(edges_df["Target"].nunique()),
        "n_cells_used":  int(expr_sub.shape[0]),
        "n_genes_used":  len(predictor_genes),
        "runtime_min":   round(elapsed / 60, 2),
        "top_regulators": edges_df.groupby("Regulator")["norm_importance"]
                          .sum().sort_values(ascending=False).head(20)
                          .to_dict(),
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))

    print("\n" + "=" * 65)
    print(f"  GENIE3 complete")
    print(f"  Edges:      {len(edges_df)}")
    print(f"  Regulators: {edges_df['Regulator'].nunique()}")
    print(f"  Targets:    {edges_df['Target'].nunique()}")
    print(f"\n  Top 10 regulators by total importance:")
    top_regs = edges_df.groupby("Regulator")["norm_importance"].sum() \
                       .sort_values(ascending=False).head(10)
    for gene, score in top_regs.items():
        driver_flag = " ← PDAC driver" if gene in set(pdac_drivers) else ""
        print(f"    {gene:<12} {score:.4f}{driver_flag}")
    print("=" * 65)

    return edges_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRN-02: GENIE3 edge inference")
    parser.add_argument("--max-cells",   type=int, default=0,
                        help="Subsample to N cells (0=all). Use 3000 for quick test.")
    parser.add_argument("--max-targets", type=int, default=0,
                        help="Limit target genes (0=all)")
    parser.add_argument("--n-workers",   type=int, default=-1,
                        help="Parallel workers (-1=all cores)")
    args = parser.parse_args()

    run_genie3(
        max_cells   = args.max_cells,
        max_targets = args.max_targets,
        n_workers   = args.n_workers,
    )