"""
grn/03_therapy_decision.py
───────────────────────────
Therapy modality decision engine.

GAP-1 FIX: The TARGETS dict is no longer hardcoded here.
It is now loaded dynamically from GRN differential expression outputs
via grn/select_targets.load_targets(). This means:

  - Adding a new protein to the GRN analysis automatically includes it here.
  - Changing the log2FC threshold changes which proteins get evaluated.
  - The full target provenance is recorded in selected_targets.json.

To run with the default criteria (log2FC ≥ 1.5, pval < 0.05, top 25):
    python grn/03_therapy_decision.py

To override target selection criteria at runtime:
    python grn/03_therapy_decision.py --min-log2fc 3.0 --top-n 10

To use a pre-built selected_targets.json without re-running selection:
    python grn/03_therapy_decision.py --use-cached-targets
"""

from __future__ import annotations
import json
import click
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
INTER     = ROOT / "data" / "intermediate"
GRN_INTER = ROOT / "data" / "grn" / "intermediate"
REPORTS   = ROOT / "data" / "reports"
OUT_DIR   = ROOT / "data" / "grn" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── GAP-1 FIX: dynamic target loading ─────────────────────────────────────────
# Old code (hardcoded — do not restore):
#   TARGETS = {
#       "CEACAM6": "P40199",
#       "TOP2A":   "P11388",
#       ...
#   }
#
# New code: TARGETS is built at runtime from GRN outputs.
# See grn/select_targets.py for the selection logic.

def _load_targets(
    min_log2fc:          float = 1.5,
    max_pval:            float = 0.05,
    top_n:               int   = 25,
    use_cached:          bool  = False,
    resolve_live:        bool  = True,
) -> dict[str, str]:
    """
    Load targets from GRN outputs. Falls back gracefully at each step:
      1. Cached selected_targets.json (if use_cached=True and file exists)
      2. Dynamic selection from tumor_vs_normal.json + top_regulator_ids.json
      3. Emergency fallback: the original hardcoded list (with a loud warning)
    """
    # Option 1: use cached selection (fast, reproducible)
    cached_path = GRN_INTER / "selected_targets.json"
    if use_cached and cached_path.exists():
        data = json.loads(cached_path.read_text())
        targets = {g: v["uniprot_id"] for g, v in data["targets"].items()}
        print(f"  Loaded {len(targets)} targets from cached selected_targets.json")
        return targets

    # Option 2: dynamic selection (recommended default)
    try:
        from grn.select_targets import load_targets
        return load_targets(
            min_log2fc   = min_log2fc,
            max_pval     = max_pval,
            top_n        = top_n,
            resolve_live = resolve_live,
        )
    except Exception as e:
        print(f"\n  WARNING: Dynamic target selection failed: {e}")
        print(f"  Falling back to hardcoded targets list.\n")

    # Option 3: emergency hardcoded fallback
    # These are the proteins from the original PDAC GRN analysis.
    # Update this list only if you need to test without any GRN data.
    return {
        "CEACAM6": "P40199",
        "TOP2A":   "P11388",
        "CLSPN":   "Q9HAW4",
        "ATAD2":   "Q6PL18",
        "HELLS":   "Q9NRZ9",
        "STMN1":   "P16949",
        "SLC2A1":  "P11166",
        "LCN2":    "P80188",
    }


# ── Known location overrides ───────────────────────────────────────────────────

KNOWN_INTRACELLULAR = {
    "TOP2A", "CLSPN", "ATAD2", "HELLS", "STMN1",
    "MKI67", "TP53", "MYC", "KRAS", "BRCA1", "BRCA2",
}

KNOWN_SURFACE = {
    "CEACAM6", "EGFR", "ERBB2", "MET", "CD274",
    "PDCD1", "CTLA4", "SLC2A1",
}

SURFACE_GO_IDS = {
    "GO:0005886", "GO:0009986", "GO:0005887",
    "GO:0031225", "GO:0005615", "GO:0005576",
}

INTRACELLULAR_GO_IDS = {
    "GO:0005634", "GO:0000785", "GO:0005829",
    "GO:0005737", "GO:0005783", "GO:0005794", "GO:0005739",
}

SURFACE_GO_TERMS_LITERAL = {
    "GO:0005886", "GO:0009986", "GO:0005887",
}


def is_surface_protein(gene: str, report: dict, category: dict) -> bool:
    if gene in KNOWN_INTRACELLULAR:
        return False
    if gene in KNOWN_SURFACE:
        return True

    go_cc_ids = {t.get("go_id", "") for t in report.get("go_terms_cc", [])}
    if go_cc_ids & SURFACE_GO_IDS and not (go_cc_ids & INTRACELLULAR_GO_IDS):
        return True

    cat = category.get("top_category", "")
    if cat in ("transmembrane", "gpcr"):
        return True

    location = report.get("subcellular_location", "").lower()
    if any(kw in location for kw in ["plasma membrane", "cell surface",
                                      "extracellular", "secreted"]):
        if not any(kw in location for kw in ["nucleus", "cytoplasm",
                                              "cytosol", "chromatin"]):
            return True
    return False


def decide_modality(gene: str, report: dict, category: dict,
                    tn_data: dict) -> dict:
    result = {
        "gene": gene, "uniprot": report.get("uniprot_id", ""),
        "decisions": {}, "recommendation": "",
        "rationale": [], "combination": [], "confidence": "",
    }

    surface = is_surface_protein(gene, report, category)
    result["decisions"]["surface"] = surface

    pockets = report.get("binding_pockets", [])
    pocket_vol  = pockets[0].get("volume_A3", 0)  if pockets else 0
    pocket_drug = pockets[0].get("druggability_score", 0) if pockets else 0
    has_good_pocket = pocket_drug >= 0.7 and pocket_vol >= 300
    result["decisions"]["pocket_volume"]       = pocket_vol
    result["decisions"]["pocket_druggability"] = pocket_drug
    result["decisions"]["has_good_pocket"]     = has_good_pocket

    tn             = tn_data.get(gene, {})
    log2fc         = tn.get("log2fc", 0)
    pval           = tn.get("pval", 1)
    tumor_specific = log2fc >= 2 and pval < 0.05
    highly_specific= log2fc >= 4 and pval < 0.05
    result["decisions"]["log2fc"]          = round(log2fc, 2)
    result["decisions"]["tumor_specific"]  = tumor_specific
    result["decisions"]["highly_specific"] = highly_specific

    active_sites = report.get("active_sites", [])
    motif_types  = list({m for s in active_sites for m in s.get("motifs", [])})
    has_atp = any("atp" in m.lower() or "p_loop" in m.lower() or
                  "walker" in m.lower() or "kinase" in m.lower()
                  for m in motif_types)
    has_dna = any("dna" in m.lower() for m in motif_types)
    result["decisions"]["motif_types"] = motif_types[:5]
    result["decisions"]["has_atp"]     = has_atp

    go_mf_names = " ".join(t.get("go_name", "")
                           for t in report.get("go_terms_mf", []))
    is_epigenetic = any(kw in go_mf_names.lower() for kw in
                        ["chromatin", "histone", "bromodomain",
                         "helicase", "remodel", "methyltransfer"])

    ppi     = report.get("ppi_partners", [])
    top_ppi = [p.get("partner_name", "") for p in ppi[:5]]
    known_drug_targets = {
        "KRAS", "TP53", "EGFR", "CDK4", "CDK6", "MDM2",
        "MTOR", "AKT1", "BCL2", "TOP2A", "ATAD2", "CLSPN",
    }
    combo = [p for p in top_ppi if p in known_drug_targets]
    result["combination"] = combo

    modalities = []
    rationale  = []

    if not tumor_specific:
        rationale.append(
            f"log2FC={log2fc:.2f} — not tumor-specific. "
            f"High normal tissue toxicity risk. DEPRIORITIZE."
        )

    if surface:
        modalities.append("ADC (Antibody-Drug Conjugate)")
        rationale.append(
            f"Surface protein with log2FC={log2fc:.2f} — "
            f"antibody delivers toxin specifically to tumor cells"
        )
        if highly_specific:
            modalities.append("Naked antibody / bispecific T-cell engager")
            modalities.append("CAR-T cell therapy")
            rationale.append(
                f"log2FC={log2fc:.2f} — minimal normal tissue exposure, "
                f"naked antibody or CAR-T viable"
            )
    else:
        if has_good_pocket:
            modalities.append("Small molecule inhibitor")
            rationale.append(
                f"Intracellular with druggable pocket "
                f"{pocket_vol:.0f}Å³ (score={pocket_drug:.2f}) "
                f"— small molecule feasible"
            )
        if has_atp:
            rationale.append(
                "ATP-binding site — ATP-competitive inhibitor "
                "is the most direct approach"
            )
        if has_dna:
            rationale.append(
                "DNA-binding motif — consider intercalation or "
                "groove-binding small molecules"
            )
        if is_epigenetic:
            modalities.append("PROTAC / protein degrader")
            rationale.append(
                "Chromatin/epigenetic regulator — PROTAC degradation "
                "removes all protein functions, often more effective "
                "than catalytic inhibition alone for this target class"
            )

    if combo:
        rationale.append(
            f"PPI partners include druggable targets {combo} — "
            f"combination therapy opportunity"
        )

    result["recommendation"] = " + ".join(modalities) if modalities \
                                else "Insufficient data"
    result["rationale"]      = rationale
    n_evidence = sum([tumor_specific, has_good_pocket,
                      bool(active_sites), bool(ppi), bool(modalities)])
    result["confidence"] = "HIGH"   if n_evidence >= 4 else \
                           "MEDIUM" if n_evidence >= 2 else "LOW"
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main(
    min_log2fc:          float = 1.5,
    max_pval:            float = 0.05,
    top_n:               int   = 25,
    use_cached_targets:  bool  = False,
    no_live:             bool  = False,
) -> None:
    # ── Load targets dynamically from GRN ─────────────────────────────────────
    TARGETS = _load_targets(
        min_log2fc   = min_log2fc,
        max_pval     = max_pval,
        top_n        = top_n,
        use_cached   = use_cached_targets,
        resolve_live = not no_live,
    )

    tn_path = GRN_INTER / "tumor_vs_normal.json"
    tn_data = json.loads(tn_path.read_text()) if tn_path.exists() else {}

    all_results = []
    for gene, uid in TARGETS.items():
        rp = REPORTS / f"{uid}_report.json"
        cp = INTER   / f"{uid}_category.json"
        if not rp.exists():
            print(f"  {gene} ({uid}): no consensus report yet — "
                  f"run pipeline/01_fetch_structure.py --uniprot {uid} first")
            continue
        report   = json.loads(rp.read_text())
        category = json.loads(cp.read_text()) if cp.exists() else {}
        all_results.append(decide_modality(gene, report, category, tn_data))

    if not all_results:
        print("\n  No results — run the protein pipeline for these targets first:")
        for gene, uid in TARGETS.items():
            print(f"    python pipeline/01_fetch_structure.py --uniprot {uid}  # {gene}")
        return

    all_results.sort(
        key=lambda r: (
            {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(r["confidence"], 0),
            r["decisions"].get("log2fc", 0)
        ),
        reverse=True
    )

    lines = []
    lines.append("=" * 70)
    lines.append("  THERAPY MODALITY DECISION REPORT")
    lines.append("  PDAC Drug Targets — ProteinFP + GRN + Tumor/Normal")
    lines.append(f"  Targets selected: {len(TARGETS)}  |  "
                 f"Reports found: {len(all_results)}")
    lines.append("=" * 70)

    for i, r in enumerate(all_results, 1):
        fc   = r["decisions"].get("log2fc", 0)
        vol  = r["decisions"].get("pocket_volume", 0)
        drug = r["decisions"].get("pocket_druggability", 0)
        surf = r["decisions"].get("surface", False)
        conf = r["confidence"]

        lines.append(f"\n{'─'*70}")
        lines.append(f"  #{i}  {r['gene']}  ({r['uniprot']})  [{conf}]")
        lines.append(f"{'─'*70}")
        lines.append(f"  Recommendation : {r['recommendation']}")
        lines.append(f"  Tumor/Normal   : log2FC = {fc:+.2f}")
        lines.append(f"  Location       : {'SURFACE' if surf else 'INTRACELLULAR'}")
        lines.append(f"  Best pocket    : {vol:.0f}Å³  druggability={drug:.2f}")
        if r["combination"]:
            lines.append(f"  Combo targets  : {', '.join(r['combination'])}")
        lines.append(f"\n  Rationale:")
        for rat in r["rationale"]:
            lines.append(f"    • {rat}")

    lines.append(f"\n{'='*70}")
    lines.append("  SUMMARY TABLE")
    lines.append(f"{'='*70}")
    lines.append(f"  {'Gene':<10} {'log2FC':>7}  {'Location':<15} "
                 f"{'Modality':<35} {'Conf'}")
    lines.append(f"  {'-'*10} {'-'*7}  {'-'*15} {'-'*35} {'-'*6}")
    for r in all_results:
        fc   = r["decisions"].get("log2fc", 0)
        loc  = "SURFACE" if r["decisions"].get("surface") else "INTRACELLULAR"
        rec  = r["recommendation"][:34]
        conf = r["confidence"]
        lines.append(f"  {r['gene']:<10} {fc:>+7.2f}  {loc:<15} "
                     f"{rec:<35} {conf}")

    text = "\n".join(lines)
    print(text)
    (OUT_DIR / "therapy_recommendations.txt").write_text(text, encoding="utf-8")
    (OUT_DIR / "therapy_recommendations.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n  Saved to data/grn/reports/")


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--min-log2fc",          default=1.5,   type=float,
              help="Min tumor/normal log2FC for target inclusion (default: 1.5)")
@click.option("--max-pval",            default=0.05,  type=float,
              help="Max p-value for target inclusion (default: 0.05)")
@click.option("--top-n",               default=25,    type=int,
              help="Max number of targets to evaluate (default: 25)")
@click.option("--use-cached-targets",  is_flag=True,  default=False,
              help="Use existing selected_targets.json instead of re-running selection")
@click.option("--no-live",             is_flag=True,  default=False,
              help="Skip live UniProt API (use pre-computed IDs only)")
def cli(min_log2fc: float, max_pval: float, top_n: int,
        use_cached_targets: bool, no_live: bool) -> None:
    """
    Therapy modality decision engine — PDAC drug target prioritisation.

    Automatically selects targets from GRN differential expression data,
    then scores each target for therapeutic modality based on protein
    structure, binding pockets, surface accessibility, and tumor specificity.

    Example:
        python grn/03_therapy_decision.py
        python grn/03_therapy_decision.py --min-log2fc 3.0 --top-n 10
        python grn/03_therapy_decision.py --use-cached-targets
    """
    main(
        min_log2fc         = min_log2fc,
        max_pval           = max_pval,
        top_n              = top_n,
        use_cached_targets = use_cached_targets,
        no_live            = no_live,
    )


if __name__ == "__main__":
    cli()