"""
grn/select_targets.py
──────────────────────
GRN → Pipeline Target Selector

This module is the missing link between the GRN analysis and the
protein pipeline. It replaces the hardcoded TARGETS dict in
03_therapy_decision.py with a function that reads directly from
GRN outputs and applies biologically-grounded filtering criteria.

Sources it reads (in priority order):
  1. data/grn/intermediate/tumor_vs_normal.json   — differential expression
  2. data/grn/intermediate/top_regulator_ids.json — pre-mapped UniProt IDs
  3. data/grn/intermediate/genie3_edges.csv       — network centrality fallback

Selection criteria (all must pass for a gene to be included):
  - log2FC >= min_log2fc (default 1.5, ~3x upregulation in tumor)
  - pval < max_pval (default 0.05)
  - UniProt ID resolvable (via top_regulator_ids or live UniProt lookup)
  - Not in the exclusion list (housekeeping genes, unmappable loci)

Output:
  dict mapping gene_symbol → uniprot_id (same format as old TARGETS)
  Also writes data/grn/intermediate/selected_targets.json for traceability.

Usage (standalone):
    python grn/select_targets.py
    python grn/select_targets.py --min-log2fc 2.0 --top-n 20

Usage (from 03_therapy_decision.py):
    from grn.select_targets import load_targets
    TARGETS = load_targets()
"""

from __future__ import annotations

import csv
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

import click

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent
GRN_INTER   = ROOT / "data" / "grn" / "intermediate"
OUT_PATH    = GRN_INTER / "selected_targets.json"

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_MIN_LOG2FC  = 1.5    # ~3x upregulation in tumor vs normal
DEFAULT_MAX_PVAL    = 0.05
DEFAULT_TOP_N       = 25     # max targets to return

# Genes to always exclude: housekeeping / non-druggable / mapping noise
EXCLUSION_LIST = {
    "ACTB", "GAPDH", "B2M", "RPLP0", "HPRT1",   # housekeeping
    "MT-CO1", "MT-CO2", "MT-ND1",                 # mitochondrial genome
    "MALAT1", "NEAT1",                             # lncRNAs
    "RPS2", "RPS3", "RPL4", "RPL5",               # ribosomal
}

# UniProt API endpoint
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"


# ── UniProt ID resolution ──────────────────────────────────────────────────────

def _resolve_uniprot(gene: str, timeout: int = 8) -> Optional[str]:
    """
    Look up the canonical human UniProt accession for a gene symbol.
    Returns None if the gene cannot be mapped.
    """
    url = (
        f"{UNIPROT_SEARCH}?query=gene_exact:{gene}+AND+organism_id:9606"
        f"+AND+reviewed:true&fields=accession&format=json&size=1"
    )
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ProteinFP-pipeline/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        results = data.get("results", [])
        if results:
            return results[0]["primaryAccession"]
    except (urllib.error.URLError, KeyError, json.JSONDecodeError):
        pass
    return None


def _load_precomputed_ids() -> dict[str, str]:
    """
    Load the pre-computed gene → UniProt mapping from top_regulator_ids.json.
    Returns empty dict if file is missing.
    """
    path = GRN_INTER / "top_regulator_ids.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    # Format: { "GENE": { "uniprot_id": "PXXXXX", ... } }
    return {gene: info["uniprot_id"] for gene, info in data.items()
            if info.get("uniprot_id")}


def _load_genie3_centrality(top_n: int = 50) -> dict[str, float]:
    """
    Compute in-degree centrality from GENIE3 edges as a fallback
    importance signal for genes not in tumor_vs_normal.json.
    Returns dict of gene → normalised centrality score (0–1).
    """
    edges_path = GRN_INTER / "genie3_pure_tumor_edges.csv"
    if not edges_path.exists():
        edges_path = GRN_INTER / "genie3_tumor_edges.csv"
    if not edges_path.exists():
        edges_path = GRN_INTER / "genie3_edges.csv"
    if not edges_path.exists():
        return {}

    in_degree: dict[str, float] = {}
    with open(edges_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row.get("Target", row.get("target", ""))
            weight = float(row.get("norm_importance", row.get("weight", 0)))
            in_degree[target] = in_degree.get(target, 0.0) + weight

    if not in_degree:
        return {}

    max_w = max(in_degree.values())
    return {g: w / max_w for g, w in
            sorted(in_degree.items(), key=lambda x: -x[1])[:top_n]}


# ── Main selector ──────────────────────────────────────────────────────────────

def load_targets(
    min_log2fc:    float = DEFAULT_MIN_LOG2FC,
    max_pval:      float = DEFAULT_MAX_PVAL,
    top_n:         int   = DEFAULT_TOP_N,
    resolve_live:  bool  = True,
    verbose:       bool  = True,
) -> dict[str, str]:
    """
    Build a gene → uniprot_id target dict from GRN outputs.

    This is a drop-in replacement for the hardcoded TARGETS dict.

    Args:
        min_log2fc:   Minimum tumor/normal log2 fold-change.
        max_pval:     Maximum adjusted p-value.
        top_n:        Maximum number of targets to return.
        resolve_live: If True, fall back to live UniProt API for unmapped genes.
        verbose:      Print progress to stdout.

    Returns:
        dict[gene_symbol, uniprot_id]
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"\n── Target selector ──────────────────────────────────────────")
    log(f"  Criteria: log2FC ≥ {min_log2fc}, pval < {max_pval}, top {top_n}")

    # 1. Load tumor vs normal expression data
    tn_path = GRN_INTER / "tumor_vs_normal.json"
    if not tn_path.exists():
        raise FileNotFoundError(
            f"tumor_vs_normal.json not found at {tn_path}\n"
            f"  Run grn/02_genie3.py and the differential expression step first."
        )
    tn_data: dict = json.loads(tn_path.read_text())
    log(f"  Loaded {len(tn_data)} genes from tumor_vs_normal.json")

    # 2. Load pre-computed UniProt IDs
    precomputed = _load_precomputed_ids()
    log(f"  Pre-computed UniProt IDs: {len(precomputed)} genes")

    # 3. Filter by expression criteria
    candidates: list[dict] = []
    for gene, stats in tn_data.items():
        if gene in EXCLUSION_LIST:
            continue
        log2fc = stats.get("log2fc", 0.0)
        pval   = stats.get("pval", 1.0)
        if log2fc >= min_log2fc and pval < max_pval:
            candidates.append({
                "gene":    gene,
                "log2fc":  log2fc,
                "pval":    pval,
                "priority": stats.get("priority", "MEDIUM"),
            })

    # Sort by log2FC descending
    candidates.sort(key=lambda x: -x["log2fc"])
    log(f"  Candidates passing filter: {len(candidates)}")

    # 4. Resolve UniProt IDs
    targets: dict[str, str] = {}
    skipped: list[str]      = []
    api_calls = 0

    for cand in candidates:
        if len(targets) >= top_n:
            break

        gene = cand["gene"]

        # Try pre-computed first (fast, no network)
        uid = precomputed.get(gene)

        # Fall back to live UniProt API
        if not uid and resolve_live:
            log(f"  Resolving {gene} via UniProt API...")
            uid = _resolve_uniprot(gene)
            api_calls += 1
            time.sleep(0.3)  # be polite to UniProt

        if uid:
            targets[gene] = uid
            log(f"  ✓ {gene:<12} → {uid}  (log2FC={cand['log2fc']:+.2f})")
        else:
            skipped.append(gene)
            log(f"  ✗ {gene:<12}   no UniProt ID found, skipping")

    log(f"\n  Selected {len(targets)} targets  "
        f"({api_calls} live API calls, {len(skipped)} skipped)")
    if skipped:
        log(f"  Skipped: {', '.join(skipped)}")

    # 5. Persist for traceability
    GRN_INTER.mkdir(parents=True, exist_ok=True)
    record = {
        "generated_by":  "grn/select_targets.py",
        "criteria": {
            "min_log2fc": min_log2fc,
            "max_pval":   max_pval,
            "top_n":      top_n,
        },
        "targets": {
            gene: {
                "uniprot_id": uid,
                "log2fc":     tn_data.get(gene, {}).get("log2fc", 0.0),
                "pval":       tn_data.get(gene, {}).get("pval", 1.0),
                "priority":   tn_data.get(gene, {}).get("priority", "MEDIUM"),
            }
            for gene, uid in targets.items()
        },
    }
    OUT_PATH.write_text(json.dumps(record, indent=2))
    log(f"  Saved → {OUT_PATH}")
    log(f"────────────────────────────────────────────────────────────\n")

    return targets


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--min-log2fc", default=DEFAULT_MIN_LOG2FC, type=float,
              help=f"Minimum log2 fold-change (default: {DEFAULT_MIN_LOG2FC})")
@click.option("--max-pval",   default=DEFAULT_MAX_PVAL,   type=float,
              help=f"Maximum p-value (default: {DEFAULT_MAX_PVAL})")
@click.option("--top-n",      default=DEFAULT_TOP_N,      type=int,
              help=f"Max targets to select (default: {DEFAULT_TOP_N})")
@click.option("--no-live",    is_flag=True, default=False,
              help="Skip live UniProt API (use pre-computed IDs only)")
def main(min_log2fc: float, max_pval: float, top_n: int, no_live: bool) -> None:
    """
    Select drug targets from GRN differential expression data.

    Reads tumor_vs_normal.json + top_regulator_ids.json and applies
    expression + druggability filters to produce a ranked target list.

    Output: data/grn/intermediate/selected_targets.json

    Example:
        python grn/select_targets.py
        python grn/select_targets.py --min-log2fc 2.0 --top-n 15
        python grn/select_targets.py --no-live   (offline, pre-computed IDs only)
    """
    targets = load_targets(
        min_log2fc   = min_log2fc,
        max_pval     = max_pval,
        top_n        = top_n,
        resolve_live = not no_live,
    )

    print("\n  Final target list:")
    print(f"  {'Gene':<12} {'UniProt':<10}")
    print(f"  {'─'*12} {'─'*10}")
    for gene, uid in targets.items():
        print(f"  {gene:<12} {uid}")


if __name__ == "__main__":
    main()