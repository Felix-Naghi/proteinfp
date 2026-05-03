"""
sim/step05_network_perturbation_patch.py
─────────────────────────────────────────
LAYER 1+2 PATCH for sim/step05_network_perturbation.py

This module provides two drop-in replacement functions that replace the
hardcoded PDAC-specific references in step05:

  1. load_baseline_expression_generic()
     Replaces load_baseline_expression() — reads tumor cluster ID
     from disease_config.yaml instead of hardcoding "6"

  2. build_normal_cell_state_generic()
     Replaces build_normal_cell_state() — reads normal cell type
     from disease_config.yaml instead of hardcoding "ductal" / GSE84133

HOW TO USE THIS PATCH
─────────────────────
In sim/step05_network_perturbation.py, add at the top:

    from sim.step05_network_perturbation_patch import (
        load_baseline_expression_generic  as load_baseline_expression,
        build_normal_cell_state_generic   as build_normal_cell_state,
    )

This replaces the two hardcoded functions without touching any other
logic in step05.  All ODE maths, GRN loading, and perturbation code
remain unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parent.parent
GRN_INT  = ROOT / "data" / "grn" / "intermediate"


# ── Disease config loader ──────────────────────────────────────────────────────

def _load_disease_cfg(config_path: Optional[Path] = None) -> dict:
    if config_path is None:
        config_path = ROOT / "config" / "disease_config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Generic baseline expression loader ────────────────────────────────────────

def load_baseline_expression_generic(
    genes:              list,
    gene_idx:           dict,
    disease_config_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Load tumor baseline expression from preprocessed scRNA-seq.

    Disease-agnostic replacement for the hardcoded PDAC version.
    Reads tumor_cluster_id from disease_config.yaml.

    Fallback chain:
      1. Leiden cluster matching tumor_cluster_id from config
      2. preprocessed.h5ad all-cell mean (if cluster not found)
      3. Uniform 0.3 baseline
    """
    cfg          = _load_disease_cfg(disease_config_path)
    tumor_cluster = str(cfg.get("data", {}).get("tumor_cluster_id", "6"))
    disease_name  = cfg.get("disease", {}).get("name", "unknown")

    print(f"  Loading baseline expression "
          f"(disease={disease_name}, cluster={tumor_cluster})...")

    try:
        import scanpy as sc

        h5ad_path = GRN_INT / "preprocessed.h5ad"
        if not h5ad_path.exists():
            raise FileNotFoundError(f"preprocessed.h5ad not found: {h5ad_path}")

        adata = sc.read_h5ad(h5ad_path)

        # Try Leiden cluster column first
        cluster_col = None
        for col in ("leiden", "cluster", "cell_type", "louvain"):
            if col in adata.obs.columns:
                cluster_col = col
                break

        if cluster_col and tumor_cluster in adata.obs[cluster_col].values:
            tumor_cells = adata[adata.obs[cluster_col] == tumor_cluster]
            print(f"  Using cluster '{tumor_cluster}' "
                  f"({tumor_cells.n_obs} cells)")
        else:
            print(f"  Cluster '{tumor_cluster}' not found — "
                  f"using all {adata.n_obs} cells as baseline")
            tumor_cells = adata

        if hasattr(tumor_cells.X, "toarray"):
            X = tumor_cells.X.toarray()
        else:
            X = np.array(tumor_cells.X)

        mean_expr = X.mean(axis=0)
        max_expr  = mean_expr.max()
        norm_expr = mean_expr / (max_expr + 1e-10)

        x0 = np.full(len(genes), 0.3, dtype=np.float32)
        for i, gene in enumerate(genes):
            if gene in tumor_cells.var_names:
                idx    = list(tumor_cells.var_names).index(gene)
                x0[i] = float(norm_expr[idx])

        print(f"  Loaded expression for {len(genes)} genes from scRNA-seq")
        return x0

    except Exception as e:
        print(f"  Baseline load failed ({e}) — using uniform 0.3")
        return np.full(len(genes), 0.3, dtype=np.float32)


# ── Generic normal cell state builder ─────────────────────────────────────────

def build_normal_cell_state_generic(
    genes:              list,
    gene_idx:           dict,
    disease_config_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Build normal (non-disease) cell baseline expression state.

    Disease-agnostic replacement for the hardcoded PDAC/ductal version.

    Source priority:
      1. normal_reference dataset from disease_config.yaml (if provided)
      2. Non-tumor clusters in preprocessed.h5ad
         (any cluster that is NOT the tumor_cluster_id)
      3. Uniform 0.2 fallback (represents low baseline activity)
    """
    cfg               = _load_disease_cfg(disease_config_path)
    data_cfg          = cfg.get("data", {})
    normal_cell_type  = data_cfg.get("normal_cell_type", "")
    normal_reference  = data_cfg.get("normal_reference", "")
    tumor_cluster     = str(data_cfg.get("tumor_cluster_id", "6"))
    disease_name      = cfg.get("disease", {}).get("name", "unknown")

    print(f"  Building normal cell state "
          f"(disease={disease_name}, normal_type='{normal_cell_type}')...")

    # ── Option 1: separate normal reference dataset ────────────────────────────
    if normal_reference:
        ref_path = ROOT / normal_reference
        if ref_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(ref_path, index_col=0)
                # Try to filter by normal_cell_type if there's an assignment column
                if "assigned_cluster" in df.columns and normal_cell_type:
                    df = df[df["assigned_cluster"] == normal_cell_type].drop(
                        columns=["assigned_cluster"], errors="ignore"
                    )
                    print(f"  Using '{normal_cell_type}' cells from {ref_path.name}"
                          f" ({len(df)} cells)")
                else:
                    df = df.select_dtypes(include=[float, int])
                    print(f"  Using all cells from {ref_path.name} "
                          f"({len(df)} cells)")

                mean_normal = df.mean()
                max_expr    = mean_normal.max()
                norm_normal = mean_normal / (max_expr + 1e-10)

                x_normal = np.full(len(genes), 0.2, dtype=np.float32)
                for i, gene in enumerate(genes):
                    if gene in norm_normal.index:
                        x_normal[i] = float(norm_normal[gene])
                print(f"  Normal state loaded from reference dataset")
                return x_normal
            except Exception as e:
                print(f"  Normal reference load failed ({e}) — trying h5ad")

    # ── Option 2: non-tumor clusters from preprocessed.h5ad ──────────────────
    try:
        import scanpy as sc

        h5ad_path = GRN_INT / "preprocessed.h5ad"
        if not h5ad_path.exists():
            raise FileNotFoundError()

        adata = sc.read_h5ad(h5ad_path)

        cluster_col = None
        for col in ("leiden", "cluster", "cell_type", "louvain"):
            if col in adata.obs.columns:
                cluster_col = col
                break

        if cluster_col:
            # Use everything that is NOT the tumor cluster
            normal_mask = adata.obs[cluster_col] != tumor_cluster

            # Further filter by normal_cell_type label if it exists
            if (normal_cell_type
                    and normal_cell_type in adata.obs[cluster_col].values):
                normal_mask = adata.obs[cluster_col] == normal_cell_type

            normal_cells = adata[normal_mask]
            if normal_cells.n_obs > 0:
                print(f"  Using {normal_cells.n_obs} non-tumor cells from h5ad")
                if hasattr(normal_cells.X, "toarray"):
                    X = normal_cells.X.toarray()
                else:
                    X = np.array(normal_cells.X)

                mean_expr = X.mean(axis=0)
                max_expr  = mean_expr.max()
                norm_expr = mean_expr / (max_expr + 1e-10)

                x_normal = np.full(len(genes), 0.2, dtype=np.float32)
                for i, gene in enumerate(genes):
                    if gene in normal_cells.var_names:
                        idx         = list(normal_cells.var_names).index(gene)
                        x_normal[i] = float(norm_expr[idx])
                return x_normal

    except Exception as e:
        print(f"  h5ad normal state load failed ({e})")

    # ── Option 3: uniform fallback ─────────────────────────────────────────────
    print(f"  Using uniform 0.2 baseline for normal cell state")
    return np.full(len(genes), 0.2, dtype=np.float32)