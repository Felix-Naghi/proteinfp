"""
grn/03_therapy_decision.py
───────────────────────────
Fixed version — correct surface vs intracellular detection.
"""

from __future__ import annotations
import json
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
INTER     = ROOT / "data" / "intermediate"
GRN_INTER = ROOT / "data" / "grn" / "intermediate"
REPORTS   = ROOT / "data" / "reports"
OUT_DIR   = ROOT / "data" / "grn" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "CEACAM6": "P40199",
    "TOP2A":   "P11388",
    "CLSPN":   "Q9HAW4",
    "ATAD2":   "Q6PL18",
    "HELLS":   "Q9NRZ9",
    "STMN1":   "P16949",
    "SLC2A1":  "P11166",
    "LCN2":    "P80188",
}

# Known intracellular proteins — override any GO annotation
KNOWN_INTRACELLULAR = {
    "TOP2A", "CLSPN", "ATAD2", "HELLS", "STMN1",
    "MKI67", "TP53", "MYC", "KRAS", "BRCA1", "BRCA2",
}

# Known surface proteins
KNOWN_SURFACE = {
    "CEACAM6", "EGFR", "ERBB2", "MET", "CD274",
    "PDCD1", "CTLA4", "SLC2A1",
}

# GO IDs that genuinely mean cell surface
SURFACE_GO_IDS = {
    "GO:0005886", "GO:0009986", "GO:0005887",
    "GO:0031225", "GO:0005615", "GO:0005576",
}

# GO IDs that mean intracellular
INTRACELLULAR_GO_IDS = {
    "GO:0005634", "GO:0000785", "GO:0005829",
    "GO:0005737", "GO:0005783", "GO:0005794", "GO:0005739",
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
        result["recommendation"] = "Deprioritized — insufficient tumor specificity"
        result["rationale"]      = rationale
        result["confidence"]     = "LOW"
        return result

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
                f"Intracellular with druggable pocket {pocket_vol:.0f}Å³ "
                f"(score={pocket_drug:.2f}) — small molecule feasible"
            )
            if has_atp:
                rationale.append(
                    "ATP-binding site — ATP-competitive inhibitor "
                    "is the most direct approach"
                )
            if has_dna:
                rationale.append(
                    "DNA-binding motif — consider allosteric inhibition "
                    "to avoid non-specific genotoxicity"
                )
        else:
            modalities.append("siRNA / antisense oligonucleotide")
            rationale.append(
                "Intracellular, no druggable pocket — gene silencing "
                "more tractable than small molecule"
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


def main():
    tn_path = GRN_INTER / "tumor_vs_normal.json"
    tn_data = json.loads(tn_path.read_text()) if tn_path.exists() else {}

    all_results = []
    for gene, uid in TARGETS.items():
        rp = REPORTS / f"{uid}_report.json"
        cp = INTER   / f"{uid}_category.json"
        if not rp.exists():
            print(f"  {gene}: no report, skipping")
            continue
        report   = json.loads(rp.read_text())
        category = json.loads(cp.read_text()) if cp.exists() else {}
        all_results.append(decide_modality(gene, report, category, tn_data))

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


if __name__ == "__main__":
    main()