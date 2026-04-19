"""
pipeline/13_consensus.py
─────────────────────────
Module 13 — Consensus scoring + final report generation.

The finale. Aggregates evidence from all 12 upstream modules into a single
ranked, confidence-scored prediction report.

Evidence weighting (from config.yaml):
  experimental_homolog:  3.0  (Swiss-Prot BLAST hit)
  structural_homolog:    2.5  (Foldseek same-fold hit)
  ai_deepfri:            2.0  (DeepFRI GO prediction)
  ai_esm2:               1.8  (ESM-2 embedding)
  sequence_homolog:      1.5  (TrEMBL / inferred)
  string_interaction:    1.2  (STRING DB PPI)
  domain_annotation:     1.0  (InterPro domain)

Output:
  1. Ranked GO term predictions with confidence tiers
  2. Active site summary with HIGH/MEDIUM confidence residues
  3. Binding pocket ranking with druggability scores
  4. Allosteric site catalogue
  5. PPI network summary
  6. EC number classification
  7. Experimental validation suggestions
  8. Full JSON report
  9. Human-readable text report

Usage (standalone):
    python pipeline/13_consensus.py --uniprot P04637

Usage (from orchestrator):
    from pipeline.consensus import build_consensus_report
    report = build_consensus_report("P04637")
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from datetime import datetime

import click

from utils.config import cfg, get_logger

log = get_logger(__name__)

# ── Evidence weights ───────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "experimental_homolog": 3.0,
    "structural_homolog":   2.5,
    "ai_deepfri":           2.0,
    "ai_esm2":              1.8,
    "domain_annotation":    2.0,
    "sequence_homolog":     1.5,
    "string_interaction":   1.2,
    "active_site_motif":    2.0,
    "sequence_baseline":    0.5,
    "sequence_composition": 0.3,
}

# GO term confidence tiers
HIGH_CONF_THRESHOLD   = 5.0
MEDIUM_CONF_THRESHOLD = 2.5


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class RankedGOTerm:
    go_id:          str
    go_name:        str
    namespace:      str
    weighted_score: float
    n_sources:      int
    sources:        list[str]
    confidence:     str        # HIGH / MEDIUM / LOW
    top_evidence:   str        # best single piece of evidence

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConsensusReport:
    """The complete final output of the ProteinFP pipeline."""
    uniprot_id:       str
    gene_name:        str
    protein_name:     str
    organism:         str
    sequence_length:  int
    generated_at:     str

    # Core functional predictions
    top_function:     str
    is_enzyme:        bool
    ec_number:        str
    subcellular_location: str

    # GO terms
    go_terms_mf:      list[RankedGOTerm] = field(default_factory=list)
    go_terms_bp:      list[RankedGOTerm] = field(default_factory=list)
    go_terms_cc:      list[RankedGOTerm] = field(default_factory=list)

    # Sites
    active_sites:     list[dict] = field(default_factory=list)
    binding_pockets:  list[dict] = field(default_factory=list)
    allosteric_sites: list[dict] = field(default_factory=list)

    # Interactions
    ppi_partners:     list[dict] = field(default_factory=list)

    # Quality metrics
    mean_plddt:       float = 0.0
    n_evidence_sources: int = 0
    overall_confidence: str = ""

    # Validation suggestions
    validation_suggestions: list[str] = field(default_factory=list)

    # Module availability flags
    modules_run:      list[str] = field(default_factory=list)
    modules_missing:  list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_text_report(self) -> str:
        """Generate a human-readable text report."""
        lines = [
            "=" * 70,
            f"  ProteinFP Prediction Report",
            f"  Generated: {self.generated_at}",
            "=" * 70,
            "",
            f"  Protein    : {self.protein_name}",
            f"  Gene       : {self.gene_name}",
            f"  UniProt    : {self.uniprot_id}",
            f"  Organism   : {self.organism}",
            f"  Length     : {self.sequence_length} aa",
            f"  Mean pLDDT : {self.mean_plddt:.1f}",
            f"  Confidence : {self.overall_confidence}",
            "",
            "─" * 70,
            "  FUNCTIONAL PREDICTION",
            "─" * 70,
            f"  Top function     : {self.top_function}",
            f"  Enzyme           : {'yes — ' + self.ec_number if self.is_enzyme else 'no (non-enzyme)'}",
            f"  Location         : {self.subcellular_location}",
            f"  Evidence sources : {self.n_evidence_sources}",
            "",
            "─" * 70,
            "  GENE ONTOLOGY PREDICTIONS",
            "─" * 70,
        ]

        for label, terms in [
            ("Molecular Function", self.go_terms_mf[:5]),
            ("Biological Process", self.go_terms_bp[:5]),
            ("Cellular Component", self.go_terms_cc[:3]),
        ]:
            if terms:
                lines.append(f"\n  {label}:")
                for t in terms:
                    lines.append(
                        f"    [{t.confidence:6s}] {t.go_id} {t.go_name[:45]} "
                        f"(score={t.weighted_score:.1f}, {t.n_sources} sources)"
                    )

        if self.active_sites:
            lines += [
                "",
                "─" * 70,
                "  ACTIVE SITES",
                "─" * 70,
            ]
            for site in self.active_sites[:5]:
                res = site.get("residue_number", "?")
                aa  = site.get("one_letter", "?")
                conf = site.get("confidence", "?")
                motifs = ", ".join(site.get("motifs", [])[:2]) or "general"
                lines.append(
                    f"  {aa}{res} [{conf:6s}] motifs: {motifs}"
                )

        if self.binding_pockets:
            lines += [
                "",
                "─" * 70,
                "  BINDING POCKETS",
                "─" * 70,
            ]
            for p in self.binding_pockets[:5]:
                pid   = p.get("pocket_id", "?")
                vol   = p.get("volume_A3", 0)
                drug  = p.get("druggability_score", 0)
                dc    = p.get("druggability_class", "?")
                lines.append(
                    f"  {pid}: vol={vol:.0f}Å³  druggability={drug:.2f} ({dc})"
                )

        if self.allosteric_sites:
            lines += [
                "",
                "─" * 70,
                "  ALLOSTERIC SITES",
                "─" * 70,
            ]
            for s in self.allosteric_sites[:4]:
                sid  = s.get("site_id", "?")
                corr = s.get("mean_correlation", 0)
                conf = s.get("confidence", "?")
                rns  = s.get("residue_numbers", [])[:4]
                lines.append(
                    f"  {sid} [{conf:6s}] corr={corr:.2f} residues={rns}"
                )

        if self.ppi_partners:
            lines += [
                "",
                "─" * 70,
                "  PROTEIN INTERACTIONS (top 8)",
                "─" * 70,
            ]
            for p in self.ppi_partners[:8]:
                name  = p.get("partner_name", "?")
                score = p.get("combined_score", 0)
                itype = p.get("interaction_type", "?")
                lines.append(f"  {name:15s} STRING={score}  type={itype}")

        if self.validation_suggestions:
            lines += [
                "",
                "─" * 70,
                "  VALIDATION SUGGESTIONS",
                "─" * 70,
            ]
            for i, sug in enumerate(self.validation_suggestions, 1):
                lines.append(f"  {i}. {sug}")

        lines += [
            "",
            "─" * 70,
            f"  Modules run: {', '.join(self.modules_run)}",
            "=" * 70,
            "",
        ]
        return "\n".join(lines)


# ── Main function ──────────────────────────────────────────────────────────────

def build_consensus_report(uniprot_id: str) -> ConsensusReport:
    """
    Load all intermediate JSON files and build the consensus report.

    Args:
        uniprot_id: UniProt accession

    Returns:
        ConsensusReport — the complete prediction for this protein.
    """
    log.info(f"── Module 13: Consensus scoring for {uniprot_id} ──")

    inter_dir = Path(cfg.paths["intermediate"])
    uid       = uniprot_id.strip().upper()

    # ── Load all module outputs ───────────────────────────────────────────────
    modules_data = {}
    modules_run  = []
    modules_missing = []

    module_files = {
        "structure":    f"{uid}_structure.json",
        "physico":      f"{uid}_physicochemical.json",
        "active":       f"{uid}_active_sites.json",
        "pockets":      f"{uid}_binding_pockets.json",
        "allosteric":   f"{uid}_allosteric.json",
        "chem_env":     f"{uid}_chemical_env.json",
        "homology":     f"{uid}_homology.json",
        "esm2":         f"{uid}_esm2.json",
        "go":           f"{uid}_go_predictions.json",
        "ec":           f"{uid}_ec_prediction.json",
        "foldseek":     f"{uid}_foldseek.json",
        "ppi":          f"{uid}_ppi.json",
    }

    for key, filename in module_files.items():
        path = inter_dir / filename
        if path.exists():
            with open(path) as f:
                modules_data[key] = json.load(f)
            modules_run.append(key)
            log.info(f"  Loaded: {filename}")
        else:
            modules_missing.append(key)
            log.warning(f"  Missing: {filename}")

    log.info(f"  {len(modules_run)}/{len(module_files)} modules available")

    # ── Extract metadata ──────────────────────────────────────────────────────
    struct      = modules_data.get("structure", {})
    gene_name   = struct.get("gene_name", "unknown")
    prot_name   = struct.get("protein_name", "unknown")
    organism    = struct.get("organism", "unknown")
    seq_length  = struct.get("length", 0)
    mean_plddt  = struct.get("mean_plddt", 0.0)

    # ── Aggregate GO terms ────────────────────────────────────────────────────
    log.info("  [1/5] Aggregating GO evidence...")
    go_mf, go_bp, go_cc = _aggregate_go_terms(modules_data)

    # ── Get top function text ─────────────────────────────────────────────────
    top_function = _extract_top_function(modules_data)

    # ── EC classification ─────────────────────────────────────────────────────
    ec_data    = modules_data.get("ec", {})
    is_enzyme  = ec_data.get("is_enzyme", False)
    ec_number  = ec_data.get("specific_ec", "")
    if not ec_number and ec_data.get("top_prediction"):
        top_pred = ec_data["top_prediction"]
        if isinstance(top_pred, dict):
            ec_number = top_pred.get("ec_full", "")
    # Strip "EC " prefix so scoring can compare first digit directly
    if ec_number.startswith("EC "):
        ec_number = ec_number[3:]

    # ── Subcellular location ──────────────────────────────────────────────────
    location = _extract_location(modules_data, modules_data.get("physico"))

    # ── Active sites ──────────────────────────────────────────────────────────
    log.info("  [2/5] Extracting site predictions...")
    active_data   = modules_data.get("active", {})
    all_active = [
        r for r in active_data.get("active_residues", [])
        if r.get("confidence") in ("HIGH", "MEDIUM")
    ]

    # Annotate residues with domain context from InterPro (ISSUE 4)
    hom_data  = modules_data.get("homology", {})
    domains   = hom_data.get("interpro_domains", [])
    catalytic_domain_keywords = {
        "kinase", "protease", "peptidase", "catalytic", "active", "enzyme",
        "hydrolase", "transferase", "lyase", "oxidoreductase", "isomerase",
        "ligase", "phosphatase", "dehydrogenase", "reductase",
    }
    structural_domain_keywords = {
        "egf", "fibronectin", "immunoglobulin", "zinc finger", "ring finger",
        "cadherin", "lectin", "coiled", "armadillo", "ankyrin", "wd40",
    }

    # Build domain map: residue_number → domain_name
    domain_map: dict[int, str] = {}
    for dom in domains:
        start = dom.get("start", 0)
        end   = dom.get("end", 0)
        name  = dom.get("name", "").lower()
        if start and end:
            for rn in range(start, end + 1):
                domain_map[rn] = name

    for r in all_active:
        rn = r.get("residue_number", 0)
        if rn in domain_map:
            r["domain_context"] = domain_map[rn]

    # Prioritise residues in catalytic domains when domains are annotated
    if domain_map:
        catalytic_res = [
            r for r in all_active
            if any(kw in r.get("domain_context", "") for kw in catalytic_domain_keywords)
        ]
        structural_res = [
            r for r in all_active
            if any(kw in r.get("domain_context", "") for kw in structural_domain_keywords)
        ]
        no_domain_res = [
            r for r in all_active
            if r not in catalytic_res and r not in structural_res
        ]
        # Put catalytic domain residues first, structural last
        ordered = catalytic_res + no_domain_res + structural_res
    else:
        ordered = all_active

    # Further prioritise residues in enzymatic motifs
    motif_priority = {"dfg_loop", "hrd_catalytic_loop", "p_loop_walker_a",
                      "serine_protease_triad", "cysteine_protease_dyad"}
    priority = [r for r in ordered
                if any(m in motif_priority for m in r.get("motifs", []))]
    others   = [r for r in ordered if r not in priority]
    active_sites = (priority + others)[:30]

    # ── Binding pockets ───────────────────────────────────────────────────────
    pocket_data  = modules_data.get("pockets", {})
    pockets      = pocket_data.get("pockets", [])[:10]

    # ── Allosteric sites ──────────────────────────────────────────────────────
    allo_data    = modules_data.get("allosteric", {})
    allo_sites   = allo_data.get("allosteric_sites", [])[:8]

    # ── PPI partners ──────────────────────────────────────────────────────────
    ppi_data     = modules_data.get("ppi", {})
    ppi_partners = ppi_data.get("partners", [])[:15]

    # ── Validation suggestions ────────────────────────────────────────────────
    log.info("  [3/5] Generating validation suggestions...")
    suggestions = _generate_validation_suggestions(
        active_sites, pockets, ppi_partners,
        go_mf, is_enzyme, ec_number
    )

    # ── Overall confidence ────────────────────────────────────────────────────
    log.info("  [4/5] Computing overall confidence...")
    n_sources      = len(modules_run)
    n_exp_hits     = modules_data.get("homology", {}).get("n_experimental_hits", 0)
    overall_conf   = _overall_confidence(n_sources, n_exp_hits, mean_plddt, go_mf)

    # ── Assemble report ───────────────────────────────────────────────────────
    log.info("  [5/5] Assembling final report...")

    report = ConsensusReport(
        uniprot_id=uid,
        gene_name=gene_name,
        protein_name=prot_name,
        organism=organism,
        sequence_length=seq_length,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        top_function=top_function,
        is_enzyme=is_enzyme,
        ec_number=ec_number,
        subcellular_location=location,
        go_terms_mf=go_mf,
        go_terms_bp=go_bp,
        go_terms_cc=go_cc,
        active_sites=active_sites,
        binding_pockets=pockets,
        allosteric_sites=allo_sites,
        ppi_partners=ppi_partners,
        mean_plddt=mean_plddt,
        n_evidence_sources=n_sources,
        overall_confidence=overall_conf,
        validation_suggestions=suggestions,
        modules_run=modules_run,
        modules_missing=modules_missing,
    )

    return report


# ── GO aggregation ─────────────────────────────────────────────────────────────

def _aggregate_go_terms(
    modules_data: dict,
) -> tuple[list[RankedGOTerm], list[RankedGOTerm], list[RankedGOTerm]]:
    """
    Aggregate GO evidence from all modules with weighted scoring.
    Returns (mf_terms, bp_terms, cc_terms) sorted by weighted score.
    """
    weights  = cfg.get("consensus_weights") or DEFAULT_WEIGHTS
    evidence: dict[str, dict] = defaultdict(lambda: {
        "name": "", "namespace": "", "score": 0.0,
        "sources": [], "top_evidence": ""
    })

    def _add(go_id, go_name, ns, weight, source):
        if not go_id:
            return
        ev = evidence[go_id]
        ev["name"]      = go_name or ev["name"]
        ev["namespace"] = ns or ev["namespace"]
        ev["score"]    += weight
        if source not in ev["sources"]:
            ev["sources"].append(source)
        if not ev["top_evidence"] or weight > _source_weight(ev["top_evidence"], weights):
            ev["top_evidence"] = source

    def _source_weight(src, w):
        for key, val in w.items():
            if key in src.lower():
                return val
        return 0.5

    # From homology (experimental hits)
    hom = modules_data.get("homology", {})
    n_exp = hom.get("n_experimental_hits", 0)
    w_hom = weights.get("experimental_homolog", 3.0) if n_exp > 0 \
            else weights.get("sequence_homolog", 1.5)
    for go_id, go_name in zip(
        hom.get("all_go_terms", []), hom.get("all_go_names", [])
    ):
        ns = _infer_ns(go_name)
        src = "experimental_homolog" if n_exp > 0 else "sequence_homolog"
        _add(go_id, go_name, ns, w_hom, src)

    # From InterPro domains
    for dom in hom.get("interpro_domains", []):
        for go_id, go_name in zip(
            dom.get("go_terms", []), dom.get("go_names", [])
        ):
            ns = _infer_ns(go_name)
            _add(go_id, go_name, ns, 3.5, "domain_annotation")

    # From GO predictions (Module 09)
    go_data = modules_data.get("go", {})
    for pred_list, w_key in [
        (go_data.get("mf_predictions", []), "ai_deepfri"),
        (go_data.get("bp_predictions", []), "ai_deepfri"),
        (go_data.get("cc_predictions", []), "ai_deepfri"),
    ]:
        w = weights.get(w_key, 2.0)
        for pred in pred_list:
            go_id   = pred.get("go_id", "")
            go_name = pred.get("go_name", "")
            ns      = pred.get("namespace", _infer_ns(go_name))
            score   = pred.get("score", 0.5)
            for src in pred.get("evidence", ["ai_deepfri"]):
                sw = weights.get(src, w)
                _add(go_id, go_name, ns, sw * score, src)

    # From Foldseek inferred functions (structural)
    fld = modules_data.get("foldseek", {})
    for hit in fld.get("same_fold_hits", [])[:5]:
        desc = hit.get("function_inferred", "") or hit.get("description", "")
        # Map known function descriptions to GO terms
        for go_id, go_name, ns in _desc_to_go(desc):
            _add(go_id, go_name, ns,
                 weights.get("structural_homolog", 2.5), "structural_homolog")

    # Build ranked lists
    mf, bp, cc = [], [], []
    for go_id, ev in evidence.items():
        score = ev["score"]
        ns    = ev["namespace"] or _infer_ns(ev["name"])

        if score >= HIGH_CONF_THRESHOLD:
            conf = "HIGH"
        elif score >= MEDIUM_CONF_THRESHOLD:
            conf = "MEDIUM"
        else:
            conf = "LOW"

        term = RankedGOTerm(
            go_id=go_id,
            go_name=ev["name"],
            namespace=ns,
            weighted_score=round(score, 2),
            n_sources=len(ev["sources"]),
            sources=ev["sources"],
            confidence=conf,
            top_evidence=ev["top_evidence"],
        )

        if ns == "MF":
            mf.append(term)
        elif ns == "BP":
            bp.append(term)
        else:
            cc.append(term)

    for lst in (mf, bp, cc):
        lst.sort(key=lambda t: t.weighted_score, reverse=True)

    return mf[:15], bp[:15], cc[:10]


def _infer_ns(go_name: str) -> str:
    # Strip InterPro namespace prefixes like "F:", "C:", "P:"
    name = (go_name or "").lower()
    if name.startswith(("f:", "c:", "p:")):
        name = name[2:].strip()
    # BP keywords checked first to avoid signaling/phosphorylation landing in MF
    _BP_WORDS = [
        "process", "regulation", "response", "cycle", "repair", "apoptot",
        "transcription", "signaling", "phosphorylation", "ubiquitination",
        "coagulation", "recombination", "folding", "refolding",
        "proliferation", "transport", "transduction",
        "superoxide", "proteolysis", "removal of", "homeostasis",
    ]
    _MF_EXCEPTIONS = {"transporter activity", "transcription factor activity"}
    if any(w in name for w in _BP_WORDS):
        if any(exc in name for exc in _MF_EXCEPTIONS):
            return "MF"
        return "BP"
    if any(w in name for w in ["activity", "binding", "catalytic"]):
        return "MF"
    if any(w in name for w in ["nucleus", "cytoplasm", "membrane", "complex",
                                "organelle", "chromosome", "cytosol"]):
        return "CC"
    return "MF"


def _desc_to_go(desc: str) -> list[tuple[str, str, str]]:
    """Map free-text function descriptions to known GO terms."""
    mappings = [
        ("dna binding",          "GO:0003677", "DNA binding",       "MF"),
        ("transcription factor", "GO:0003700", "transcription factor activity", "MF"),
        ("tumor suppressor",     "GO:0006915", "apoptotic process", "BP"),
        ("kinase",               "GO:0004672", "protein kinase activity", "MF"),
        ("protease",             "GO:0008233", "peptidase activity", "MF"),
        ("nucleus",              "GO:0005634", "nucleus",           "CC"),
    ]
    result = []
    d = desc.lower()
    for keyword, go_id, go_name, ns in mappings:
        if keyword in d:
            result.append((go_id, go_name, ns))
    return result


# ── Helper extractors ──────────────────────────────────────────────────────────

def _extract_top_function(modules_data: dict) -> str:
    """Extract the best function description from all sources."""
    # Priority: Swiss-Prot homolog > InterPro family > GO top term
    hom = modules_data.get("homology", {})
    for hit in hom.get("blast_hits", []):
        if hit.get("reviewed") and hit.get("function_text"):
            return hit["function_text"][:200]

    top_fn = hom.get("top_function", "")
    if top_fn:
        return top_fn[:200]

    go_data = modules_data.get("go", {})
    mf = go_data.get("mf_predictions", [])
    if mf:
        return mf[0].get("go_name", "")

    return "Function not determined"


def _extract_location(modules_data: dict, physico_data: Optional[dict] = None) -> str:
    """
    Extract subcellular location using protein features + GO CC terms.

    Priority order:
      1. Transmembrane signal (high hydrophobicity in top residues) → membrane
      2. Signal peptide (high hydrophobicity in first 30 residues) → extracellular
      3. GO CC terms prioritising membrane/extracellular over nucleus
      4. Fall back to top CC term

    Nucleus tends to accumulate false-positive scores from BLAST homologs
    with nuclear functions. Physical features are a more reliable signal.
    """
    # ── Step 1: Feature-based detection from physicochemical data ────────────
    if physico_data:
        residues = physico_data.get("residues", [])
        if residues:
            hydros = sorted(
                [r.get("hydrophobicity", 0.0) for r in residues], reverse=True
            )
            # Transmembrane signal: mean of top-20 hydrophobic residues > 2.5
            top20_mean = sum(hydros[:20]) / min(20, len(hydros))
            if top20_mean > 2.5:
                # Membrane protein — check GO for more precise term
                go_data  = modules_data.get("go", {})
                cc_terms = go_data.get("cc_predictions", [])
                for term in cc_terms:
                    name = term.get("go_name", "").lower()
                    if any(kw in name for kw in ["membrane", "plasma membrane",
                                                  "cell surface"]):
                        return term.get("go_name", "membrane")
                return "membrane"

            # Signal peptide: first 30 residues with mean hydrophobicity > 1.5
            sorted_res = sorted(residues, key=lambda r: r.get("residue_number", 0))
            first30    = sorted_res[:30]
            if first30:
                sp_mean = sum(r.get("hydrophobicity", 0.0) for r in first30) / len(first30)
                if sp_mean > 1.5:
                    return "extracellular space"

    # ── Step 2: GO CC terms with membrane/extracellular priority ─────────────
    go_data  = modules_data.get("go", {})
    cc_terms = go_data.get("cc_predictions", [])

    if not cc_terms:
        return "unknown"

    priority_keywords = ["membrane", "extracellular", "plasma membrane",
                         "cell surface", "cytoplasm", "mitochondria", "cytosol"]
    for term in cc_terms:
        name = term.get("go_name", "").lower()
        name = name[2:].strip() if name.startswith(("c:", "f:", "p:")) else name
        if any(kw in name for kw in priority_keywords):
            return term.get("go_name", "unknown")

    # Fall back to top CC term
    return cc_terms[0].get("go_name", "unknown")


def _overall_confidence(
    n_sources:   int,
    n_exp_hits:  int,
    mean_plddt:  float,
    go_mf:       list[RankedGOTerm],
) -> str:
    score = 0
    if n_sources >= 10: score += 3
    elif n_sources >= 7: score += 2
    elif n_sources >= 4: score += 1

    if n_exp_hits >= 5:  score += 3
    elif n_exp_hits >= 2: score += 2
    elif n_exp_hits >= 1: score += 1

    if mean_plddt >= 80:  score += 2
    elif mean_plddt >= 70: score += 1

    if go_mf and go_mf[0].confidence == "HIGH": score += 2
    elif go_mf and go_mf[0].confidence == "MEDIUM": score += 1

    if score >= 8:   return "VERY HIGH"
    if score >= 6:   return "HIGH"
    if score >= 4:   return "MEDIUM"
    if score >= 2:   return "LOW"
    return "VERY LOW"


def _generate_validation_suggestions(
    active_sites:  list[dict],
    pockets:       list[dict],
    ppi_partners:  list[dict],
    go_mf:         list[RankedGOTerm],
    is_enzyme:     bool,
    ec_number:     str,
) -> list[str]:
    """Generate concrete experimental validation suggestions."""
    suggestions = []

    # Active site validation
    high_res = [
        r for r in active_sites
        if r.get("confidence") == "HIGH"
    ][:3]
    if high_res:
        res_str = ", ".join(
            f"{r.get('one_letter','?')}{r.get('residue_number','?')}"
            for r in high_res
        )
        suggestions.append(
            f"Alanine scanning mutagenesis of predicted active site residues "
            f"({res_str}) to confirm functional importance"
        )

    # Binding pocket validation
    if pockets:
        top_pocket = pockets[0]
        suggestions.append(
            f"Thermal shift assay or SPR binding assay against "
            f"pocket {top_pocket.get('pocket_id','P1')} "
            f"(vol={top_pocket.get('volume_A3',0):.0f}Å³, "
            f"druggability={top_pocket.get('druggability_score',0):.2f})"
        )

    # PPI validation
    high_conf_partners = [
        p for p in ppi_partners
        if p.get("combined_score", 0) >= 800
    ][:2]
    for partner in high_conf_partners:
        itype = partner.get("interaction_type", "unknown")
        iface = partner.get("interface_residues", [])[:3]
        suggestions.append(
            f"Co-immunoprecipitation with {partner.get('partner_name','?')} "
            f"({itype} interaction, predicted interface: {iface})"
        )

    # Enzyme validation
    if is_enzyme and ec_number:
        suggestions.append(
            f"Enzyme activity assay consistent with EC {ec_number} "
            f"using appropriate substrate panel"
        )

    # GO term validation
    if go_mf:
        top_mf = go_mf[0]
        if "binding" in top_mf.go_name.lower():
            suggestions.append(
                f"EMSA or fluorescence polarisation assay to confirm "
                f"'{top_mf.go_name}' ({top_mf.go_id})"
            )

    # Localisation validation
    suggestions.append(
        "GFP/mCherry fusion protein + confocal microscopy to confirm "
        "predicted subcellular localisation"
    )

    return suggestions[:8]


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637)")
def main(uniprot: str) -> None:
    """
    Module 13 — Consensus scoring + final report.

    Aggregates all upstream module outputs into a single ranked prediction.
    Run this after all other modules have completed.

    Example:
        python pipeline/consensus.py --uniprot P04637
    """
    uniprot   = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    report_dir = Path(cfg.paths["reports"])
    report_dir.mkdir(parents=True, exist_ok=True)

    json_out = report_dir / f"{uniprot}_report.json"
    text_out = report_dir / f"{uniprot}_report.txt"

    report = build_consensus_report(uniprot)

    report.to_json(json_out)
    text = report.to_text_report()
    # Windows console safe output
    safe_text = text.encode('ascii', errors='replace').decode('ascii')
    click.echo(safe_text)
    text_out.write_text(text, encoding="utf-8")
    click.echo(f"\nReports saved to:")
    click.echo(f"  JSON : {json_out}")
    click.echo(f"  Text : {text_out}")


if __name__ == "__main__":
    main()