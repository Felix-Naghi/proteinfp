"""
grn/01_preprocess.py
─────────────────────
Module GRN-01 — scRNA-seq preprocessing for PDAC GRN reconstruction.

Input:  data/grn/input/GSE154778_dgeMtx.csv
        Rows = genes (22,217), Cols = cells (14,926)
        Raw UMI counts, 90% sparse

Output: data/grn/intermediate/preprocessed.h5ad
        Normalized, log-transformed, HVGs selected
        Cell metadata includes patient ID

Pipeline:
  1. Load CSV, transpose to cells × genes
  2. Extract patient IDs from cell barcodes (P03:1 → P03)
  3. Filter low-quality cells and genes
  4. Normalize per cell (target sum 1e4)
  5. Log1p transform
  6. Select highly variable genes (HVGs)
  7. Force-include known PDAC driver genes
  8. Save as h5ad + gene list

Usage:
    python grn/01_preprocess.py
    python grn/01_preprocess.py --n-hvg 3000
    python grn/01_preprocess.py --n-hvg 5000 --max-cells 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent
INPUT_CSV   = ROOT / "data" / "grn" / "input" / "GSE154778_dgeMtx.csv"
OUT_DIR     = ROOT / "data" / "grn" / "intermediate"
OUT_H5AD    = OUT_DIR / "preprocessed.h5ad"
OUT_GENES   = OUT_DIR / "hvg_genes.json"
OUT_STATS   = OUT_DIR / "preprocess_stats.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Known PDAC driver genes to force-include ──────────────────────────────────
# These are the key regulators in pancreatic cancer.
# Your ProteinFP pipeline has validated predictions for many of these.

PDAC_DRIVERS = [
    # Core PDAC mutations
    "KRAS", "TP53", "SMAD4", "CDKN2A",
    # RTK / RAS signaling
    "EGFR", "ERBB2", "ERBB3", "MET",
    "BRAF", "RAF1", "MAP2K1", "MAPK1", "MAPK3",
    # PI3K / AKT / mTOR
    "PIK3CA", "PIK3CB", "AKT1", "AKT2",
    "MTOR", "TSC1", "TSC2",
    # Cell cycle
    "CCND1", "CDK4", "CDK6", "RB1", "E2F1",
    # Apoptosis
    "BCL2", "BCL2L1", "BAX", "CASP3", "MDM2",
    # TGF-β / SMAD pathway
    "TGFB1", "TGFB2", "SMAD2", "SMAD3",
    # Transcription factors
    "MYC", "JUN", "FOS", "NF2",
    # Invasion / metastasis
    "MMP2", "MMP9", "CDH1", "VIM",
    # Immune checkpoints
    "CD274", "PDCD1", "CTLA4",
    # Stroma / TME
    "ACTA2", "FAP", "COL1A1",
]


def preprocess(
    input_csv:  Path = INPUT_CSV,
    n_hvg:      int  = 3000,
    max_cells:  int  = 0,       # 0 = use all
    min_genes:  int  = 200,
    min_cells:  int  = 10,
    random_seed: int = 42,
) -> sc.AnnData:

    print("=" * 65)
    print("  GRN-01: Preprocessing PDAC scRNA-seq")
    print("=" * 65)

    # ── Step 1: Load CSV ──────────────────────────────────────────────────────
    print(f"\n[1/7] Loading {input_csv.name}...")
    df = pd.read_csv(input_csv, index_col=0)
    print(f"  Raw matrix: {df.shape[0]} genes × {df.shape[1]} cells")

    # Transpose: scanpy expects cells × genes
    df = df.T
    print(f"  Transposed: {df.shape[0]} cells × {df.shape[1]} genes")

    # ── Step 2: Build AnnData + extract patient metadata ─────────────────────
    print("\n[2/7] Building AnnData + extracting patient IDs...")
    adata = sc.AnnData(df.astype(np.float32))

    # Cell barcodes look like "P03:1" → patient = "P03"
    patients = [bc.split(":")[0] if ":" in bc else bc for bc in adata.obs_names]
    adata.obs["patient"] = patients

    patient_counts = pd.Series(patients).value_counts()
    print(f"  Patients: {len(patient_counts)}")
    for pat, count in patient_counts.items():
        print(f"    {pat}: {count} cells")

    # ── Step 3: Subsample if requested ───────────────────────────────────────
    if max_cells and adata.n_obs > max_cells:
        print(f"\n[3/7] Subsampling {adata.n_obs} → {max_cells} cells...")
        rng   = np.random.default_rng(random_seed)
        idx   = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[idx].copy()
        print(f"  Kept {adata.n_obs} cells")
    else:
        print(f"\n[3/7] Using all {adata.n_obs} cells (no subsampling)")

    # ── Step 4: QC filtering ──────────────────────────────────────────────────
    print(f"\n[4/7] QC filtering (min_genes={min_genes}, min_cells={min_cells})...")
    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    print(f"  Cells:  {n_cells_before} → {adata.n_obs} "
          f"(removed {n_cells_before - adata.n_obs})")
    print(f"  Genes:  {n_genes_before} → {adata.n_vars} "
          f"(removed {n_genes_before - adata.n_vars})")

    # ── Step 5: Normalize + log transform ────────────────────────────────────
    print("\n[5/7] Normalizing (target sum=1e4) + log1p...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"  Value range after norm: "
          f"min={adata.X.min():.2f} max={adata.X.max():.2f}")

    # ── Step 6: Select HVGs ───────────────────────────────────────────────────
    print(f"\n[6/7] Selecting {n_hvg} highly variable genes...")
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")
    except Exception:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")

    hvg_genes = adata.var_names[adata.var["highly_variable"]].tolist()
    print(f"  HVGs selected: {len(hvg_genes)}")

    # ── Step 7: Force-include PDAC drivers ───────────────────────────────────
    print(f"\n[7/7] Force-including PDAC driver genes...")
    genes_in_data   = set(adata.var_names)
    present_drivers = [g for g in PDAC_DRIVERS if g in genes_in_data]
    missing_drivers = [g for g in PDAC_DRIVERS if g not in genes_in_data]
    already_hvg     = [g for g in present_drivers if g in set(hvg_genes)]
    rescued         = [g for g in present_drivers if g not in set(hvg_genes)]

    print(f"  PDAC drivers in dataset:  {len(present_drivers)}/{len(PDAC_DRIVERS)}")
    print(f"  Already HVG:              {len(already_hvg)}")
    print(f"  Rescued (force-included): {len(rescued)}")
    if rescued:
        print(f"    {rescued}")
    if missing_drivers:
        print(f"  Missing from data:        {missing_drivers}")

    final_genes = list(dict.fromkeys(hvg_genes + present_drivers))
    print(f"  Final gene pool:          {len(final_genes)}")

    # Subset to final gene pool
    adata_sub = adata[:, final_genes].copy()

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving preprocessed data...")
    adata_sub.write_h5ad(OUT_H5AD)
    print(f"  h5ad → {OUT_H5AD}")

    OUT_GENES.write_text(json.dumps({
        "hvg_genes":        hvg_genes,
        "pdac_drivers":     present_drivers,
        "rescued_drivers":  rescued,
        "final_genes":      final_genes,
        "n_cells":          adata_sub.n_obs,
        "n_genes":          adata_sub.n_vars,
    }, indent=2))
    print(f"  Gene list → {OUT_GENES}")

    stats = {
        "n_cells_raw":      int(df.shape[0]),
        "n_genes_raw":      int(df.shape[1]),
        "n_cells_final":    int(adata_sub.n_obs),
        "n_genes_final":    int(adata_sub.n_vars),
        "n_hvg":            len(hvg_genes),
        "n_drivers_found":  len(present_drivers),
        "n_rescued":        len(rescued),
        "patients":         patient_counts.to_dict(),
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))

    print("\n" + "=" * 65)
    print(f"  Preprocessing complete")
    print(f"  Final matrix: {adata_sub.n_obs} cells × {adata_sub.n_vars} genes")
    print(f"  Patients:     {len(patient_counts)}")
    print(f"  PDAC drivers: {len(present_drivers)} found, {len(rescued)} rescued")
    print("=" * 65)

    return adata_sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRN-01: Preprocess PDAC scRNA-seq")
    parser.add_argument("--n-hvg",     type=int, default=3000,
                        help="Number of highly variable genes (default 3000)")
    parser.add_argument("--max-cells", type=int, default=0,
                        help="Subsample to N cells (0=all, default 0)")
    parser.add_argument("--min-genes", type=int, default=200,
                        help="Min genes per cell (default 200)")
    parser.add_argument("--min-cells", type=int, default=10,
                        help="Min cells per gene (default 10)")
    args = parser.parse_args()

    adata = preprocess(
        n_hvg      = args.n_hvg,
        max_cells  = args.max_cells,
        min_genes  = args.min_genes,
        min_cells  = args.min_cells,
    )