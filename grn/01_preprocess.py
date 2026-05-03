"""
grn/01_preprocess.py
─────────────────────
Module GRN-01 — scRNA-seq preprocessing for GRN reconstruction.

LAYER 1+2 CHANGES (disease-agnostic refactor):
  - Input CSV path now comes from config/disease_config.yaml
  - Driver gene list now comes from config/disease_config.yaml
  - HVG/QC parameters now come from config/disease_config.yaml
  - Disease name used in all output labels and filenames
  - No PDAC-specific strings remain in this file

To switch disease: edit config/disease_config.yaml only.

Input:
  Specified in disease_config.yaml → data.scrnaseq_input
  Rows = genes, Cols = cells (raw UMI counts)
  OR a pre-processed .h5ad (skips steps 1-6)

Output:
  data/grn/intermediate/preprocessed.h5ad
  data/grn/intermediate/hvg_genes.json
  data/grn/intermediate/preprocess_stats.json

Usage:
    python grn/01_preprocess.py
    python grn/01_preprocess.py --n-hvg 5000 --max-cells 5000
    python grn/01_preprocess.py --disease-config path/to/other_disease.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import yaml

# ── Project root ───────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

# ── Paths ──────────────────────────────────────────────────────────────────────

OUT_DIR   = ROOT / "data" / "grn" / "intermediate"
OUT_H5AD  = OUT_DIR / "preprocessed.h5ad"
OUT_GENES = OUT_DIR / "hvg_genes.json"
OUT_STATS = OUT_DIR / "preprocess_stats.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Disease config loader ──────────────────────────────────────────────────────

def load_disease_config(config_path: Path | None = None) -> dict:
    """
    Load disease_config.yaml.  Falls back to config/disease_config.yaml
    relative to the project root if no path is given.
    """
    if config_path is None:
        config_path = ROOT / "config" / "disease_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Disease config not found: {config_path}\n"
            f"  Copy config/disease_config.yaml and edit it for your disease."
        )

    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Main preprocessing function ────────────────────────────────────────────────

def preprocess(
    disease_config_path: Path | None = None,
    # CLI overrides (take priority over config file values)
    n_hvg:      int | None = None,
    max_cells:  int        = 0,
    min_genes:  int | None = None,
    min_cells:  int | None = None,
    random_seed: int       = 42,
) -> sc.AnnData:
    """
    Preprocess a scRNA-seq count matrix for GRN reconstruction.

    All disease-specific settings (input file, driver genes, cell type labels)
    are read from disease_config.yaml.  Pass disease_config_path to use a
    non-default config file.
    """

    # ── Load disease config ────────────────────────────────────────────────────
    cfg = load_disease_config(disease_config_path)

    disease_name  = cfg["disease"]["name"]
    disease_full  = cfg["disease"]["full_name"]
    driver_genes  = cfg.get("driver_genes", [])
    input_path    = ROOT / cfg["data"]["scrnaseq_input"]

    # Config-file defaults, overridable by CLI args
    grn_cfg    = cfg.get("grn", {})
    n_hvg      = n_hvg     if n_hvg     is not None else grn_cfg.get("n_hvg",      3000)
    min_genes  = min_genes if min_genes is not None else grn_cfg.get("min_genes",   200)
    min_cells  = min_cells if min_cells is not None else grn_cfg.get("min_cells",    10)

    print("=" * 65)
    print(f"  GRN-01: Preprocessing scRNA-seq — {disease_name}")
    print(f"  {disease_full}")
    print("=" * 65)
    print(f"\n  Input    : {input_path}")
    print(f"  Drivers  : {len(driver_genes)} genes")
    print(f"  HVGs     : {n_hvg}")

    # ── Handle pre-processed h5ad input ───────────────────────────────────────
    if input_path.suffix == ".h5ad":
        print(f"\n  [SKIP] Input is .h5ad — loading directly, skipping steps 1-6")
        adata = sc.read_h5ad(input_path)
        print(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
        _save_outputs(adata, driver_genes, n_hvg, disease_name, skip_hvg=True)
        return adata

    # ── Step 1: Load CSV ───────────────────────────────────────────────────────
    print(f"\n[1/7] Loading {input_path.name}...")
    if not input_path.exists():
        raise FileNotFoundError(
            f"scRNA-seq input not found: {input_path}\n"
            f"  Check 'data.scrnaseq_input' in your disease_config.yaml"
        )
    df = pd.read_csv(input_path, index_col=0)
    print(f"  Raw matrix: {df.shape[0]} genes × {df.shape[1]} cells")

    # Transpose: scanpy expects cells × genes
    # Auto-detect orientation: if cols > rows, assume genes are columns already
    if df.shape[1] > df.shape[0]:
        print(f"  Matrix appears to be cells × genes already — not transposing")
    else:
        df = df.T
        print(f"  Transposed to: {df.shape[0]} cells × {df.shape[1]} genes")

    # ── Step 2: Build AnnData + extract metadata ───────────────────────────────
    print("\n[2/7] Building AnnData...")
    adata = sc.AnnData(df.astype(np.float32))

    # Try to extract patient/sample IDs from barcodes.
    # Common formats: "P03:1" → "P03",  "SAMPLE1_ACGT" → "SAMPLE1"
    def _extract_sample(bc: str) -> str:
        for sep in (":", "_", "-"):
            if sep in bc:
                return bc.split(sep)[0]
        return bc

    samples = [_extract_sample(bc) for bc in adata.obs_names]
    adata.obs["sample_id"] = samples
    sample_counts = adata.obs["sample_id"].value_counts()
    print(f"  Samples detected: {len(sample_counts)}")
    for sid, cnt in sample_counts.items():
        print(f"    {sid}: {cnt} cells")

    # Subsample if requested
    if max_cells and adata.n_obs > max_cells:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[idx].copy()
        print(f"  Subsampled: → {max_cells} cells")

    # ── Step 3: QC filtering ───────────────────────────────────────────────────
    print(f"\n[3/7] QC filtering "
          f"(min_genes={min_genes}, min_cells={min_cells})...")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"  Post-QC: {adata.n_obs} cells × {adata.n_vars} genes")

    # ── Step 4: Normalise ──────────────────────────────────────────────────────
    print("\n[4/7] Normalising (target sum 1e4)...")
    sc.pp.normalize_total(adata, target_sum=1e4)

    # ── Step 5: Log transform ──────────────────────────────────────────────────
    print("\n[5/7] Log1p transform...")
    sc.pp.log1p(adata)

    # ── Step 6: HVG selection ──────────────────────────────────────────────────
    print(f"\n[6/7] Selecting {n_hvg} highly variable genes...")
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg,
                                    flavor="seurat_v3")
    except Exception:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg,
                                    flavor="seurat")

    hvg_genes = adata.var_names[adata.var["highly_variable"]].tolist()
    print(f"  HVGs selected: {len(hvg_genes)}")

    # ── Step 7: Force-include driver genes ────────────────────────────────────
    print(f"\n[7/7] Force-including {disease_name} driver genes...")
    genes_in_data   = set(adata.var_names)
    present_drivers = [g for g in driver_genes if g in genes_in_data]
    missing_drivers = [g for g in driver_genes if g not in genes_in_data]
    already_hvg     = [g for g in present_drivers if g in set(hvg_genes)]
    rescued         = [g for g in present_drivers if g not in set(hvg_genes)]

    print(f"  Drivers in dataset : {len(present_drivers)}/{len(driver_genes)}")
    print(f"  Already HVG        : {len(already_hvg)}")
    print(f"  Rescued            : {len(rescued)}")
    if rescued:
        print(f"    {rescued}")
    if missing_drivers:
        print(f"  Missing from data  : {missing_drivers}")

    final_genes = list(dict.fromkeys(hvg_genes + present_drivers))
    print(f"  Final gene pool    : {len(final_genes)}")

    adata_sub = adata[:, final_genes].copy()

    # ── Save outputs ───────────────────────────────────────────────────────────
    _save_outputs(adata_sub, driver_genes, n_hvg, disease_name,
                  hvg_genes=hvg_genes,
                  rescued=rescued,
                  present_drivers=present_drivers,
                  sample_counts=sample_counts)

    print("\n" + "=" * 65)
    print(f"  Preprocessing complete — {disease_name}")
    print(f"  Final: {adata_sub.n_obs} cells × {adata_sub.n_vars} genes")
    print(f"  Drivers found: {len(present_drivers)}, rescued: {len(rescued)}")
    print("=" * 65)

    return adata_sub


def _save_outputs(
    adata:           sc.AnnData,
    driver_genes:    list,
    n_hvg:           int,
    disease_name:    str,
    hvg_genes:       list | None  = None,
    rescued:         list | None  = None,
    present_drivers: list | None  = None,
    sample_counts                 = None,
    skip_hvg:        bool         = False,
) -> None:
    """Write h5ad, gene list JSON, and stats JSON to data/grn/intermediate/."""
    adata.write_h5ad(OUT_H5AD)
    print(f"\n  h5ad → {OUT_H5AD}")

    OUT_GENES.write_text(json.dumps({
        "disease":          disease_name,
        "hvg_genes":        hvg_genes or list(adata.var_names),
        "driver_genes":     driver_genes,
        "rescued_drivers":  rescued or [],
        "final_genes":      list(adata.var_names),
        "n_cells":          adata.n_obs,
        "n_genes":          adata.n_vars,
    }, indent=2))
    print(f"  Gene list → {OUT_GENES}")

    stats = {
        "disease":          disease_name,
        "n_cells_final":    int(adata.n_obs),
        "n_genes_final":    int(adata.n_vars),
        "n_hvg_requested":  n_hvg,
        "n_hvg_selected":   len(hvg_genes) if hvg_genes else adata.n_vars,
        "n_drivers_total":  len(driver_genes),
        "n_drivers_found":  len(present_drivers) if present_drivers else 0,
        "n_rescued":        len(rescued) if rescued else 0,
        "samples":          sample_counts.to_dict() if sample_counts is not None else {},
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"  Stats   → {OUT_STATS}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRN-01: Preprocess scRNA-seq for any disease"
    )
    parser.add_argument(
        "--disease-config",
        default=None,
        help=(
            "Path to disease_config.yaml "
            "(default: config/disease_config.yaml)"
        ),
    )
    parser.add_argument("--n-hvg",     type=int, default=None,
                        help="Override HVG count from config")
    parser.add_argument("--max-cells", type=int, default=0,
                        help="Subsample to N cells (0=all)")
    parser.add_argument("--min-genes", type=int, default=None,
                        help="Override min genes per cell from config")
    parser.add_argument("--min-cells", type=int, default=None,
                        help="Override min cells per gene from config")
    args = parser.parse_args()

    config_path = Path(args.disease_config) if args.disease_config else None

    preprocess(
        disease_config_path = config_path,
        n_hvg               = args.n_hvg,
        max_cells           = args.max_cells,
        min_genes           = args.min_genes,
        min_cells           = args.min_cells,
    )