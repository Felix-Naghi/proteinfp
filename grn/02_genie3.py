"""
grn/02_genie3.py
─────────────────
Module GRN-02 — GENIE3 edge inference for GRN reconstruction.

LAYER 1+2 CHANGES (disease-agnostic refactor):
  - Driver gene list now read from config/disease_config.yaml
  - Disease name used in output labels
  - No PDAC-specific strings remain in this file

Takes preprocessed scRNA-seq data and infers regulatory edges using
ExtraTrees regression (GENIE3 algorithm).

Input:  data/grn/intermediate/preprocessed.h5ad
Output: data/grn/intermediate/genie3_edges.csv

Usage:
    python grn/02_genie3.py
    python grn/02_genie3.py --max-cells 3000   # quick test (~3 min)
    python grn/02_genie3.py --disease-config path/to/other_disease.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parent.parent
IN_H5AD      = ROOT / "data" / "grn" / "intermediate" / "preprocessed.h5ad"
IN_GENES     = ROOT / "data" / "grn" / "intermediate" / "hvg_genes.json"
OUT_EDGES    = ROOT / "data" / "grn" / "intermediate" / "genie3_edges.csv"
OUT_STATS    = ROOT / "data" / "grn" / "intermediate" / "genie3_stats.json"

# ── GENIE3 parameters ──────────────────────────────────────────────────────────

N_ESTIMATORS   = 100
MAX_DEPTH      = 8
RANDOM_STATE   = 42
MIN_IMPORTANCE = 1e-5
MIN_NORM_IMP   = 0.005


# ── Disease config loader ──────────────────────────────────────────────────────

def _load_driver_genes(config_path: Path | None = None) -> tuple[list[str], str]:
    """
    Load driver genes and disease name from disease_config.yaml.
    Returns (driver_genes, disease_name).
    """
    if config_path is None:
        config_path = ROOT / "config" / "disease_config.yaml"

    if not config_path.exists():
        print(f"  WARNING: disease_config.yaml not found at {config_path}")
        print(f"  Proceeding without disease-specific driver gene prioritisation.")
        return [], "unknown"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    drivers      = cfg.get("driver_genes", [])
    disease_name = cfg["disease"]["name"]
    return drivers, disease_name


# ── Main GENIE3 function ───────────────────────────────────────────────────────

def run_genie3(
    max_cells:          int         = 0,
    max_targets:        int         = 0,
    n_workers:          int         = -1,
    disease_config_path: Path | None = None,
) -> pd.DataFrame:

    driver_genes, disease_name = _load_driver_genes(disease_config_path)

    print("=" * 65)
    print(f"  GRN-02: GENIE3 edge inference — {disease_name}")
    print("=" * 65)

    # ── Load preprocessed data ─────────────────────────────────────────────────
    if not IN_H5AD.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {IN_H5AD}\n"
            f"  Run grn/01_preprocess.py first."
        )

    print(f"\n[1/5] Loading preprocessed data...")
    adata = sc.read_h5ad(IN_H5AD)
    print(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # Load gene list from preprocessing output
    genes_data   = json.loads(IN_GENES.read_text()) if IN_GENES.exists() else {}
    final_genes  = genes_data.get("final_genes", list(adata.var_names))

    # Use driver genes from config (fallback to genes_data if config missing)
    if not driver_genes:
        driver_genes = genes_data.get("driver_genes", [])
        print(f"  Driver genes loaded from hvg_genes.json: {len(driver_genes)}")
    else:
        print(f"  Driver genes from disease_config: {len(driver_genes)}")

    # ── Extract expression matrix ──────────────────────────────────────────────
    print(f"\n[2/5] Extracting expression matrix...")
    if hasattr(adata.X, "toarray"):
        X_mat = adata.X.toarray()
    else:
        X_mat = np.array(adata.X)

    expr_df = pd.DataFrame(
        X_mat,
        columns=adata.var_names,
        index=adata.obs_names,
    )

    if max_cells and expr_df.shape[0] > max_cells:
        rng      = np.random.default_rng(RANDOM_STATE)
        idx      = rng.choice(expr_df.shape[0], size=max_cells, replace=False)
        expr_sub = expr_df.iloc[idx].copy()
        print(f"  Subsampled: {expr_df.shape[0]} → {max_cells} cells")
    else:
        expr_sub = expr_df.copy()
        print(f"  Using all {expr_sub.shape[0]} cells")

    # ── Define predictor/target gene sets ──────────────────────────────────────
    print(f"\n[3/5] Defining predictor/target pools...")
    genes_available = set(expr_sub.columns)
    predictor_genes = [g for g in final_genes if g in genes_available]
    target_genes    = [g for g in final_genes if g in genes_available]

    if max_targets and len(target_genes) > max_targets:
        # Prioritise known driver genes as targets
        driver_targets = [g for g in driver_genes if g in set(target_genes)]
        other_targets  = [g for g in target_genes
                          if g not in set(driver_targets)]
        target_genes   = (driver_targets
                          + other_targets[:max_targets - len(driver_targets)])
        print(f"  Targets capped at {max_targets} "
              f"({len(driver_targets)} {disease_name} drivers prioritised)")

    print(f"  Predictors: {len(predictor_genes)}")
    print(f"  Targets:    {len(target_genes)}")

    X_train = expr_sub[predictor_genes].values

    # ── Run ExtraTrees ─────────────────────────────────────────────────────────
    print(f"\n[4/5] Running ExtraTrees "
          f"({N_ESTIMATORS} trees, max_depth={MAX_DEPTH})...")
    print(f"  This may take 20-40 minutes for the full dataset.")

    t0 = time.time()

    def _fit_one(target_gene: str) -> list[tuple[str, str, float]]:
        y = expr_sub[target_gene].values
        if y.std() < 1e-8:
            return []
        predictors = [g for g in predictor_genes if g != target_gene]
        X = expr_sub[predictors].values
        model = ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        model.fit(X, y)
        edges = []
        for reg, imp in zip(predictors, model.feature_importances_):
            if imp >= MIN_IMPORTANCE:
                edges.append((reg, target_gene, imp))
        return edges

    all_edges: list[tuple[str, str, float]] = []
    results = Parallel(n_jobs=n_workers)(
        delayed(_fit_one)(g)
        for g in tqdm(target_genes, desc="  GENIE3", unit="gene")
    )
    for r in results:
        all_edges.extend(r)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min  |  {len(all_edges)} raw edges")

    # ── Build and filter edge dataframe ───────────────────────────────────────
    print(f"\n[5/5] Building edge table...")
    edges_df = pd.DataFrame(all_edges, columns=["Regulator", "Target", "importance"])

    if edges_df.empty:
        print(f"  WARNING: No edges found. "
              f"Try lowering MIN_IMPORTANCE in the script.")
        return edges_df

    # Normalise importance scores 0-1 within each target
    edges_df["norm_importance"] = (
        edges_df.groupby("Target")["importance"]
        .transform(lambda x: x / x.max() if x.max() > 0 else x)
    )
    edges_df = edges_df[edges_df["norm_importance"] >= MIN_NORM_IMP]
    edges_df = edges_df.sort_values("norm_importance", ascending=False)

    edges_df.to_csv(OUT_EDGES, index=False)
    print(f"  Saved {len(edges_df)} edges → {OUT_EDGES}")

    stats = {
        "disease":       disease_name,
        "n_edges":       len(edges_df),
        "n_regulators":  int(edges_df["Regulator"].nunique()),
        "n_targets":     int(edges_df["Target"].nunique()),
        "n_cells_used":  int(expr_sub.shape[0]),
        "n_genes_used":  len(predictor_genes),
        "runtime_min":   round(elapsed / 60, 2),
        "top_regulators": (
            edges_df.groupby("Regulator")["norm_importance"]
            .sum()
            .sort_values(ascending=False)
            .head(20)
            .to_dict()
        ),
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))

    print("\n" + "=" * 65)
    print(f"  GENIE3 complete — {disease_name}")
    print(f"  Edges: {len(edges_df)}  |  "
          f"Regulators: {edges_df['Regulator'].nunique()}  |  "
          f"Targets: {edges_df['Target'].nunique()}")
    print(f"\n  Top 10 regulators by total importance:")
    top_regs = (
        edges_df.groupby("Regulator")["norm_importance"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    driver_set = set(driver_genes)
    for gene, score in top_regs.items():
        flag = " ← driver gene" if gene in driver_set else ""
        print(f"    {gene:<12} {score:.4f}{flag}")
    print("=" * 65)

    return edges_df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRN-02: GENIE3 edge inference"
    )
    parser.add_argument(
        "--disease-config",
        default=None,
        help="Path to disease_config.yaml (default: config/disease_config.yaml)",
    )
    parser.add_argument("--max-cells",   type=int, default=0,
                        help="Subsample to N cells (0=all)")
    parser.add_argument("--max-targets", type=int, default=0,
                        help="Limit target genes (0=all)")
    parser.add_argument("--n-workers",   type=int, default=-1,
                        help="Parallel workers (-1=all cores)")
    args = parser.parse_args()

    config_path = Path(args.disease_config) if args.disease_config else None

    run_genie3(
        max_cells           = args.max_cells,
        max_targets         = args.max_targets,
        n_workers           = args.n_workers,
        disease_config_path = config_path,
    )