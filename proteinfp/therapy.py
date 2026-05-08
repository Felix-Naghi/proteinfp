"""
proteinfp/therapy.py
─────────────────────
Therapy mode — automated drug strategy decision and candidate generation.

Given any protein with a completed consensus report, this module:

  1. THERAPY DECISION  (expanded — now scores all 7 modalities)
     Reads the consensus report and ranks ALL viable modalities by score,
     not just the first matching branch of a decision tree.

     Modalities scored (0–1):
       adc             — antibody-drug conjugate
       car_t           — CAR-T cell therapy
       naked_antibody  — naked mAb / bispecific
       small_molecule  — de novo small molecule (active site)
       protac          — PROTAC protein degrader
       allosteric      — allosteric small molecule
       molecular_glue  — molecular glue (no good pocket, needs E3 proximity)

  2. DESIGN TRIGGER
     Automatically calls the right design module(s) based on ranked modalities:
       surface  → adc_design, cart_design, antibody_design
       pocket   → denovo_design (Vina) + admet
       epigenetic+pocket → protac_design
       allosteric only   → allosteric_drug_design

  3. COMBINED REPORT
     data/reports/{UNIPROT}_therapy.json  +  _therapy.txt

SCORING LOGIC (no GRN required — structural signals only)
──────────────────────────────────────────────────────────
Surface signals         : GO CC, subcellular_location, transmembrane patch
Epitope quality         : SASA, immunogenicity, patch size
Pocket quality          : druggability_score, volume, active site overlap
Allosteric quality      : ENM correlation, site size, coupling depth
Epigenetic signal       : GO MF/BP keywords (chromatin, bromodomain, histone…)
PPI druggability        : partner overlap with known drug targets
PTM signal              : phospho-sites near active site → conformational switch
Expression specificity  : NOT available here (use grn/03_therapy_decision.py
                          for expression-aware ADC vs CAR-T discrimination)

ADC vs CAR-T discrimination (without expression data):
  ADC preferred when:
    - epitope SASA moderate (200–1000 Å²) — most ADC targets
    - internalisation signal (GO: "endocytosis", "internalised", "recycled")
    - surface exposure confirmed by GO + physicochemistry
  CAR-T preferred when:
    - epitope SASA very high (>1000 Å²) — large exposed domain
    - NO internalisation signal (CAR-T works on non-internalising antigens too)
    - Protein is a known immune checkpoint / tumour antigen marker
    - cancer_marker GO terms ("tumour antigen", "oncofetal")

USAGE
─────
    proteinfp --uniprot P04637 --therapy
    proteinfp --uniprot P04637 --therapy --antibody --denovo --vina /path/to/vina

    # Test the decision engine on a UniProt ID (no design modules run):
    python proteinfp/therapy.py --uniprot P04637 --test

    # Python API:
    from proteinfp.therapy import run_therapy
    result = run_therapy("P04637")
    print(result.summary())
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

MIN_POCKET_DRUG_SCORE  = 0.6
MIN_POCKET_VOLUME      = 300.0
MIN_ALLO_CORR          = 0.5
MIN_EPITOPE_SASA       = 400.0   # raised: 3-4 residue charged patches are not valid epitopes
MIN_EPITOPE_RESIDUES   = 5       # minimum residues for a reportable epitope

# ADC vs CAR-T discrimination thresholds
ADC_SASA_MIN           = 200.0    # Å² — minimum for ADC
ADC_SASA_MAX           = 1200.0   # Å² — above this prefer CAR-T
CART_SASA_MIN          = 600.0    # Å² — large exposed domain preferred for CAR-T

# ══════════════════════════════════════════════════════════════════════════════
# GO KEYWORD SETS
# ══════════════════════════════════════════════════════════════════════════════

SURFACE_GO_KEYWORDS = {
    "plasma membrane", "cell surface", "extracellular",
    "secreted", "membrane", "extracellular space",
    "extracellular region", "cell wall",
}
INTRACELLULAR_GO_KEYWORDS = {
    "nucleus", "nucleoplasm", "chromatin", "cytoplasm",
    "cytosol", "mitochondria", "endoplasmic reticulum",
    "golgi", "lysosome",
}
EPIGENETIC_GO_KEYWORDS = {
    "chromatin", "histone", "bromodomain", "helicase",
    "chromatin remodeling", "methyltransferase", "demethylase",
    "acetyltransferase", "deacetylase",
}

# ── Curated gene-level overrides (fast, authoritative) ────────────────────────
# These bypass GO-term heuristics entirely. Add new entries as needed.
KNOWN_INTRACELLULAR_GENES = {
    # Tumour suppressors / TFs
    "TP53", "TP63", "TP73", "RB1", "BRCA1", "BRCA2",
    # Oncogenes (cytoplasmic/nuclear)
    "MYC", "MYCN", "KRAS", "NRAS", "HRAS", "BRAF",
    # Epigenetic regulators
    "EZH2", "BRD4", "DNMT1", "DNMT3A", "HDAC1", "HDAC2", "KDM5C",
    # Cell cycle / DNA repair
    "CDK4", "CDK6", "CCND1", "MDM2", "MDM4", "ATM", "ATR",
    "TOP2A", "CLSPN", "ATAD2", "HELLS", "MKI67", "PCNA",
    # Transcription factors
    "STAT3", "STAT1", "NFE2L2", "HIF1A", "SP1",
}
KNOWN_SURFACE_GENES = {
    "EGFR", "ERBB2", "ERBB3", "ERBB4", "MET", "FGFR1", "FGFR2",
    "CD19", "CD20", "CD22", "CD38", "BCMA", "CD274", "PDCD1", "CTLA4",
    "CEACAM5", "CEACAM6", "MSLN", "FOLR1", "HER2",
    "SLC2A1", "SLC7A5", "CLDN18",
}

# GO CC IDs that unambiguously mean surface/extracellular
SURFACE_GO_IDS = {
    "GO:0005886",  # plasma membrane
    "GO:0009986",  # cell surface
    "GO:0005887",  # integral component of plasma membrane
    "GO:0005576",  # extracellular region
    "GO:0005615",  # extracellular space
    "GO:0031225",  # anchored component of membrane
}
# GO CC IDs that unambiguously mean intracellular
INTRACELLULAR_GO_IDS = {
    "GO:0005634",  # nucleus
    "GO:0000785",  # chromatin
    "GO:0005829",  # cytosol
    "GO:0005737",  # cytoplasm
    "GO:0005783",  # endoplasmic reticulum
    "GO:0005739",  # mitochondrion
}

INTERNALISATION_GO_KEYWORDS = {
    "endocytosis", "internalisation", "internalization",
    "receptor-mediated endocytosis", "recycled", "receptor recycling",
    "clathrin", "caveolae",
}
TUMOUR_ANTIGEN_GO_KEYWORDS = {
    "tumor antigen", "tumour antigen", "oncofetal",
    "cancer-testis antigen", "differentiation antigen",
    "mhc", "immune checkpoint",
}
IMMUNOGENIC_AA = {"K", "R", "D", "E", "H", "N", "Q", "S", "T", "Y"}


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModalityScore:
    """Score for a single therapy modality."""
    modality:    str          # "adc" | "car_t" | "naked_antibody" | "small_molecule" |
                              # "protac" | "allosteric" | "molecular_glue"
    score:       float        # 0–1
    viable:      bool         # True if score crosses minimum threshold
    rationale:   List[str]    = field(default_factory=list)
    blockers:    List[str]    = field(default_factory=list)   # why it scored low

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpitopeCandidate:
    epitope_id:           str
    residue_numbers:      List[int]
    residue_letters:      List[str]
    source:               str
    total_sasa:           float
    immunogenicity_score: float
    accessibility:        float
    n_immunogenic_aa:     int
    notes:                str = ""

    @property
    def sequence(self) -> str:
        return "".join(self.residue_letters)

    def summary(self) -> str:
        return (
            f"  {self.epitope_id}  residues={self.residue_numbers[:5]}"
            f"{'...' if len(self.residue_numbers)>5 else ''}  "
            f"SASA={self.total_sasa:.0f}Å²  "
            f"immunogenicity={self.immunogenicity_score:.2f}  "
            f"source={self.source}\n"
            f"    sequence: {self.sequence[:30]}"
            f"{'...' if len(self.sequence)>30 else ''}\n"
            f"    {self.notes}"
        )


@dataclass
class TherapyDecision:
    uniprot_id:         str
    gene_name:          str
    protein_name:       str
    organism:           str

    # Ranked modality list (best first)
    modality_scores:    List[ModalityScore]  = field(default_factory=list)

    # Top picks (convenience)
    primary_modality:   str   = ""
    secondary_modality: str   = ""
    confidence:         str   = "LOW"

    # Evidence flags
    is_surface:         bool  = False
    has_good_pocket:    bool  = False
    has_allosteric:     bool  = False
    is_epigenetic:      bool  = False
    has_internalisation:bool  = False
    has_tumour_antigen: bool  = False

    # Pocket details
    top_pocket_id:      str   = ""
    top_pocket_vol:     float = 0.0
    top_pocket_drug:    float = 0.0

    # Allosteric details
    top_allo_corr:      float = 0.0

    # Top epitope SASA (for ADC/CAR-T split)
    top_epitope_sasa:   float = 0.0

    combination_note:   str   = ""
    n_evidence:         int   = 0

    def summary_line(self) -> str:
        viable = [ms for ms in self.modality_scores if ms.viable]
        lines  = [
            f"  {self.gene_name} ({self.uniprot_id})  [{self.confidence}]",
            f"  Primary   : {self.primary_modality.upper()}",
        ]
        if self.secondary_modality:
            lines.append(f"  Secondary : {self.secondary_modality.upper()}")
        lines.append(f"  Surface   : {'yes' if self.is_surface else 'no'}")
        lines.append(f"  Pocket    : {self.top_pocket_id}  "
                     f"vol={self.top_pocket_vol:.0f}Å³  "
                     f"drug={self.top_pocket_drug:.2f}")
        lines.append(f"\n  All viable modalities (ranked):")
        for ms in viable:
            lines.append(f"    {ms.score:.3f}  {ms.modality:<20}  "
                         + (ms.rationale[0] if ms.rationale else ""))
        return "\n".join(lines)


@dataclass
class TherapyResult:
    uniprot_id:         str
    decision:           TherapyDecision
    epitopes:           List[EpitopeCandidate]  = field(default_factory=list)
    designs_triggered:  List[str]               = field(default_factory=list)
    design_outputs:     Dict[str, str]          = field(default_factory=dict)
    denovo_path:        Optional[str]           = None
    pharm_scores_path:  Optional[str]           = None
    elapsed_sec:        float                   = 0.0

    def summary(self) -> str:
        lines = [
            "",
            "═" * 65,
            "  ProteinFP Therapy Report",
            "═" * 65,
            "",
            f"  Protein  : {self.decision.protein_name}",
            f"  Gene     : {self.decision.gene_name}  ({self.uniprot_id})",
            f"  Organism : {self.decision.organism}",
            "",
            "─" * 65,
            "  THERAPY DECISION",
            "─" * 65,
            self.decision.summary_line(),
        ]

        if self.decision.combination_note:
            lines.append(f"\n  Combination: {self.decision.combination_note}")

        if self.epitopes:
            lines += [
                "",
                "─" * 65,
                f"  ANTIBODY EPITOPE CANDIDATES  ({len(self.epitopes)} found)",
                "─" * 65,
            ]
            for ep in self.epitopes[:3]:
                lines.append(ep.summary())

        if self.design_outputs:
            lines += ["", "─" * 65, "  DESIGN OUTPUTS", "─" * 65]
            for name, path in self.design_outputs.items():
                lines.append(f"  {name:<20}: {path}")

        lines += ["", f"  Wall time : {self.elapsed_sec:.1f}s", "═" * 65, ""]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "uniprot_id":        self.uniprot_id,
            "decision":          asdict(self.decision),
            "epitopes":          [asdict(e) for e in self.epitopes],
            "designs_triggered": self.designs_triggered,
            "design_outputs":    self.design_outputs,
            "denovo_path":       self.denovo_path,
            "pharm_scores_path": self.pharm_scores_path,
            "elapsed_sec":       self.elapsed_sec,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SURFACE / EPIGENETIC / FLAG HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _go_text(report: dict, keys: List[str]) -> str:
    terms = []
    for k in keys:
        terms += [t.get("go_name", "") for t in report.get(k, [])]
    return " ".join(terms).lower()


def _is_surface(report: dict, physico: dict) -> bool:
    gene = report.get("gene_name", "").upper()

    # 1. Curated gene-level overrides — fastest and most reliable
    if gene in KNOWN_INTRACELLULAR_GENES:
        return False
    if gene in KNOWN_SURFACE_GENES:
        return True

    # 2. GO CC ID-based check (precise, not affected by substring matches)
    cc_go_ids = {t.get("go_id", "") for t in report.get("go_terms_cc", [])}
    has_surface_go      = bool(cc_go_ids & SURFACE_GO_IDS)
    has_intracellular_go = bool(cc_go_ids & INTRACELLULAR_GO_IDS)

    if has_surface_go and not has_intracellular_go:
        return True
    if has_intracellular_go:
        return False     # nucleus/chromatin/cytoplasm — definitely not surface

    # 3. Subcellular location text — use SPECIFIC phrases only (not bare "membrane")
    loc_text = report.get("subcellular_location", "").lower()
    specific_surface_phrases = [
        "plasma membrane", "cell surface", "extracellular",
        "secreted", "integral membrane", "transmembrane",
    ]
    intracellular_phrases = [
        "nucleus", "nucleoplasm", "chromatin", "cytoplasm",
        "cytosol", "mitochondria",
    ]
    if any(p in loc_text for p in specific_surface_phrases):
        if not any(p in loc_text for p in intracellular_phrases):
            return True
    if any(p in loc_text for p in intracellular_phrases):
        return False

    # 4. Transmembrane patch from physicochemistry (last resort — high specificity)
    if physico:
        hydros = sorted(
            [r.get("hydrophobicity", 0.0) for r in physico.get("residues", [])],
            reverse=True,
        )
        if hydros and sum(hydros[:20]) / min(20, len(hydros)) > 2.5:
            return True

    return False


def _is_epigenetic(report: dict) -> bool:
    go_text = _go_text(report, ["go_terms_mf", "go_terms_bp"])
    return any(kw in go_text for kw in EPIGENETIC_GO_KEYWORDS)


def _has_internalisation(report: dict) -> bool:
    all_text = _go_text(report, ["go_terms_bp", "go_terms_cc", "go_terms_mf"])
    return any(kw in all_text for kw in INTERNALISATION_GO_KEYWORDS)


def _has_tumour_antigen(report: dict) -> bool:
    all_text = _go_text(report, ["go_terms_bp", "go_terms_cc", "go_terms_mf"])
    return any(kw in all_text for kw in TUMOUR_ANTIGEN_GO_KEYWORDS)


def _top_epitope_sasa(physico: dict, active_sites: List[dict],
                       ppi: List[dict]) -> float:
    """Estimate the best available epitope SASA from physico/active site data."""
    # Try surface patches
    for patch_type in ("hydrophobic_patches", "positive_patches", "negative_patches"):
        patches = physico.get(patch_type, [])
        if patches:
            best = max(patches, key=lambda p: p.get("total_sasa", 0))
            sasa = float(best.get("total_sasa", 0))
            if sasa >= MIN_EPITOPE_SASA:
                return sasa
    # Fallback: estimate from active site count
    n_surf = sum(1 for s in active_sites if s.get("confidence") in ("HIGH", "MEDIUM"))
    return float(min(2000, n_surf * 35))


# ══════════════════════════════════════════════════════════════════════════════
# MODALITY SCORING  (each function returns a ModalityScore)
# ══════════════════════════════════════════════════════════════════════════════

def _score_adc(
    surface: bool, has_internalisation: bool, has_tumour_ag: bool,
    epitope_sasa: float, top_pocket_drug: float,
) -> ModalityScore:
    """
    ADC preferred: surface + moderate SASA + internalisation signal.
    Internalisation is important for ADC — the payload is released intracellularly.
    """
    score = 0.0
    rationale: List[str] = []
    blockers:  List[str] = []

    if not surface:
        return ModalityScore("adc", 0.0, False,
                             blockers=["Protein is not surface-exposed"])

    score += 0.40   # base: surface confirmed
    rationale.append("Surface-exposed — antibody can access target")

    # Internalisation bonus (payload delivery mechanism)
    if has_internalisation:
        score += 0.25
        rationale.append("Internalisation signal present — excellent ADC candidate "
                          "(payload released in endosome/lysosome)")
    else:
        score += 0.08
        blockers.append("No internalisation GO terms — payload release may be limited; "
                        "cleavable linker required")

    # SASA range: ADC works best with moderate-large epitope (200–1200 Å²)
    if ADC_SASA_MIN <= epitope_sasa <= ADC_SASA_MAX:
        score += 0.20
        rationale.append(f"Epitope SASA={epitope_sasa:.0f}Å² — ideal range for ADC")
    elif epitope_sasa > ADC_SASA_MAX:
        score += 0.12
        rationale.append(f"Epitope SASA={epitope_sasa:.0f}Å² — very large; CAR-T may compete")
    elif epitope_sasa > 0:
        score += 0.05
        blockers.append(f"Epitope SASA={epitope_sasa:.0f}Å² — low, may limit ADC access")

    # Tumour antigen bonus
    if has_tumour_ag:
        score += 0.10
        rationale.append("Tumour antigen marker — clinical ADC precedent")

    # Pocket drug score (can dual-purpose ADC + small mol)
    if top_pocket_drug >= MIN_POCKET_DRUG_SCORE:
        score += 0.05
        rationale.append("Also has druggable pocket — ADC + small mol combination viable")

    score = max(0.0, min(1.0, score))
    viable = score >= 0.45
    return ModalityScore("adc", round(score, 3), viable, rationale, blockers)


def _score_cart(
    surface: bool, has_internalisation: bool, has_tumour_ag: bool,
    epitope_sasa: float,
) -> ModalityScore:
    """
    CAR-T preferred: surface + large SASA + tumour antigen + no internalisation needed.
    CAR-T does NOT require internalisation — the T-cell kills by direct contact.
    Very high SASA (large accessible domain) is preferred for stable CAR engagement.
    """
    score = 0.0
    rationale: List[str] = []
    blockers:  List[str] = []

    if not surface:
        return ModalityScore("car_t", 0.0, False,
                             blockers=["Protein is not surface-exposed"])

    score += 0.30
    rationale.append("Surface-exposed — T-cell can form immune synapse")

    # CAR-T benefits from large exposed domain for stable engagement
    if epitope_sasa >= CART_SASA_MIN:
        score += 0.30
        rationale.append(f"Large surface domain (SASA={epitope_sasa:.0f}Å²) — "
                         "excellent CAR-T epitope space")
    elif epitope_sasa >= ADC_SASA_MIN:
        score += 0.15
        rationale.append(f"Adequate SASA={epitope_sasa:.0f}Å² for CAR-T")
    else:
        score += 0.05
        blockers.append(f"Small epitope (SASA={epitope_sasa:.0f}Å²) — may limit CAR-T engagement")

    # Tumour antigen is a major CAR-T positive signal
    if has_tumour_ag:
        score += 0.25
        rationale.append("Tumour antigen marker — strong CAR-T precedent (CD19, CD22, BCMA…)")

    # Internalisation slightly reduces CAR-T score (CAR antigen should persist on surface)
    if has_internalisation:
        score -= 0.10
        blockers.append("Internalisation signal — antigen may be shed/cleared from surface "
                        "under T-cell pressure")

    score = max(0.0, min(1.0, score))
    viable = score >= 0.40
    return ModalityScore("car_t", round(score, 3), viable, rationale, blockers)


def _score_naked_antibody(
    surface: bool, epitope_sasa: float, ppi: List[dict],
) -> ModalityScore:
    """
    Naked mAb / bispecific preferred: surface + PPI blocking potential.
    Works without internalisation. Often combined with immune effector function.
    """
    score = 0.0
    rationale: List[str] = []
    blockers:  List[str] = []

    if not surface:
        return ModalityScore("naked_antibody", 0.0, False,
                             blockers=["Protein is not surface-exposed"])

    score += 0.35
    rationale.append("Surface-exposed — mAb can bind and block")

    if epitope_sasa >= MIN_EPITOPE_SASA:
        score += 0.20
        rationale.append(f"SASA={epitope_sasa:.0f}Å² — accessible for mAb binding")

    # PPI blocking
    known_drug_ppi = {"EGFR", "ERBB2", "VEGF", "TNF", "IL6", "PD1", "PDL1",
                      "CTLA4", "IL17", "IL23", "CD20", "CD38", "RANKL"}
    ppi_names = {p.get("partner_name", "") for p in ppi[:5]}
    ppi_hits  = ppi_names & known_drug_ppi
    if ppi_hits:
        score += 0.30
        rationale.append(f"PPI with {', '.join(ppi_hits)} — "
                         "blocking this interaction is clinically validated")
    else:
        score += 0.05

    score = max(0.0, min(1.0, score))
    viable = score >= 0.40
    return ModalityScore("naked_antibody", round(score, 3), viable, rationale, blockers)


def _score_small_molecule(
    good_pocket: bool, pocket_id: str, pocket_vol: float, pocket_drug: float,
    is_enzyme: bool, ec: str,
) -> ModalityScore:
    score = 0.0
    rationale: List[str] = []
    blockers:  List[str] = []

    if not good_pocket:
        if pocket_drug > 0:
            blockers.append(f"Pocket druggability={pocket_drug:.2f} < {MIN_POCKET_DRUG_SCORE} "
                            f"or vol={pocket_vol:.0f}Å³ < {MIN_POCKET_VOLUME:.0f}Å³")
        else:
            blockers.append("No druggable binding pocket detected")
        return ModalityScore("small_molecule", 0.1, False, rationale, blockers)

    # Base pocket quality
    drug_norm = min(1.0, pocket_drug)
    vol_norm  = min(1.0, pocket_vol / 1000.0)
    score     = 0.35 * drug_norm + 0.25 * vol_norm
    rationale.append(f"Pocket {pocket_id}: vol={pocket_vol:.0f}Å³  "
                     f"druggability={pocket_drug:.2f}")

    if pocket_drug >= 0.80:
        score += 0.20
        rationale.append("Very high druggability pocket — excellent small molecule target")
    elif pocket_drug >= 0.60:
        score += 0.10

    if is_enzyme:
        score += 0.20
        if ec:
            rationale.append(f"Enzyme (EC {ec}) — active site inhibition most direct")
        else:
            rationale.append("Enzyme — active site inhibition feasible")

    score = max(0.0, min(1.0, score))
    viable = score >= 0.35
    return ModalityScore("small_molecule", round(score, 3), viable, rationale, blockers)


def _score_protac(
    good_pocket: bool, pocket_drug: float, is_epigenetic: bool,
    is_surface: bool, pocket_vol: float, ppi: List[dict],
) -> ModalityScore:
    """
    PROTAC preferred: intracellular + epigenetic/hard-to-drug + pocket for warhead.
    Also scores well when protein has a strong PPI with an E3 ligase or E3-adjacent
    partner (e.g. MDM2 for TP53 — the MDM2 interface IS the warhead anchor).
    """
    score = 0.0
    rationale: List[str] = []
    blockers:  List[str] = []

    if is_surface:
        blockers.append("Surface-exposed — PROTAC delivery into cell is difficult; "
                        "consider ADC-PROTAC conjugate instead")
        return ModalityScore("protac", 0.10, False, rationale, blockers)

    # E3 ligase / E3-recruiting PPI partners boost PROTAC score significantly
    # These are the proteins that either ARE E3 ligases or directly recruit them
    e3_recruiting_partners = {
        "MDM2", "MDM4", "MDMX",          # p53 degradation machinery
        "CUL4", "DDB1", "CRBN", "VHL",   # direct E3 ligases used in PROTACs
        "SPOP", "FBXW7", "SKP2",         # substrate recognition F-box proteins
        "KEAP1",                          # Cullin-3 adaptor
    }
    ppi_names = {p.get("partner_name", "") for p in ppi[:10]}
    e3_ppi_hits = ppi_names & e3_recruiting_partners
    if e3_ppi_hits:
        score += 0.35
        hits_str = ", ".join(sorted(e3_ppi_hits))
        rationale.append(f"PPI with {hits_str} — "
                         "this interaction is directly exploitable as a PROTAC warhead anchor")

    if is_epigenetic:
        score += 0.30
        rationale.append("Epigenetic/chromatin regulator — PROTAC removes all protein "
                         "functions, not just catalytic activity")

    if good_pocket:
        score += 0.25
        rationale.append(f"Pocket vol={pocket_vol:.0f}Å³ / drug={pocket_drug:.2f} — "
                         "warhead can bind for ternary complex formation")
    elif pocket_drug > 0.3:
        score += 0.12
        rationale.append(f"Partial pocket (drug={pocket_drug:.2f}) — "
                         "weak warhead binding still viable for PROTAC")
    else:
        score += 0.05
        if not e3_ppi_hits:
            blockers.append("No pocket — warhead binding is speculative")

    # Ideal PROTAC profile: intracellular + (epigenetic OR E3-PPI) + pocket
    if (is_epigenetic or e3_ppi_hits) and (good_pocket or e3_ppi_hits):
        score = min(1.0, score + 0.10)
        rationale.append("Strong PROTAC profile: intracellular + degradation handle + binding site")

    score = max(0.0, min(1.0, score))
    viable = score >= 0.35
    return ModalityScore("protac", round(score, 3), viable, rationale, blockers)


def _score_allosteric(
    has_allo: bool, allo_corr: float, n_allo_residues: int,
    good_pocket: bool,
) -> ModalityScore:
    score = 0.0
    rationale: List[str] = []
    blockers:  List[str] = []

    if not has_allo:
        blockers.append("No allosteric site detected by Module 05")
        return ModalityScore("allosteric", 0.05, False, rationale, blockers)

    corr_score = min(1.0, allo_corr / 0.8)  # normalise, 0.8 = excellent
    # Site size matters — tiny sites are harder to drug
    size_score = min(1.0, n_allo_residues / 8.0)
    if n_allo_residues <= 4:
        size_score *= 0.5
        blockers.append(f"Very small allosteric site ({n_allo_residues} residues) — "
                        "limited surface for small molecule binding")
    elif n_allo_residues <= 6:
        size_score *= 0.75

    score = 0.35 * corr_score + 0.20 * size_score
    rationale.append(f"Allosteric site: corr={allo_corr:.3f}  "
                     f"size={n_allo_residues} residues")

    if allo_corr >= 0.70:
        score += 0.15
        rationale.append("High ENM correlation — strong allosteric-active site coupling")
    elif allo_corr >= 0.50:
        score += 0.08

    # Allosteric is especially attractive when NO orthosteric pocket exists.
    # When a pocket already exists, allosteric is a secondary selectivity option —
    # hard-cap it so it cannot outrank pocket-based or E3-PPI modalities.
    if not good_pocket:
        score += 0.20
        rationale.append("No orthosteric pocket — allosteric is the primary druggable option")
    else:
        score += 0.03
        score = min(score, 0.65)   # hard cap: allosteric alone never beats pocket+PPI
        rationale.append("Allosteric available alongside orthosteric — selectivity advantage")

    score = max(0.0, min(1.0, score))
    viable = score >= 0.30
    return ModalityScore("allosteric", round(score, 3), viable, rationale, blockers)


def _score_molecular_glue(
    good_pocket: bool, has_allo: bool, is_surface: bool, ppi: List[dict],
) -> ModalityScore:
    """
    Molecular glue: preferred when no pocket AND no allosteric site.
    Redirects E3 ligase to degrade target without a warhead-binding pocket.
    """
    score = 0.0
    rationale: List[str] = []
    blockers:  List[str] = []

    if good_pocket:
        blockers.append("Good pocket exists — small molecule or PROTAC preferred")
        return ModalityScore("molecular_glue", 0.05, False, rationale, blockers)
    if is_surface:
        blockers.append("Surface protein — ADC/antibody preferred")
        return ModalityScore("molecular_glue", 0.05, False, rationale, blockers)

    score += 0.30
    rationale.append("No orthosteric pocket — molecular glue avoids needing one")

    if not has_allo:
        score += 0.20
        rationale.append("No allosteric site either — molecular glue is the primary option")

    # PPI with E3 ligases boosts molecular glue score
    e3_partners = {"CUL4", "DDB1", "CRBN", "VHL", "SPOP", "FBXW7", "SKP2", "MDM2"}
    ppi_names = {p.get("partner_name", "") for p in ppi[:10]}
    if ppi_names & e3_partners:
        score += 0.35
        rationale.append(f"PPI with E3 complex ({ppi_names & e3_partners}) — "
                         "redirecting E3 is feasible")
    else:
        score += 0.10
        blockers.append("No E3 complex PPI detected — molecular glue is speculative")

    score = max(0.0, min(1.0, score))
    viable = score >= 0.35
    return ModalityScore("molecular_glue", round(score, 3), viable, rationale, blockers)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DECISION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def make_therapy_decision(report: dict, physico: dict) -> TherapyDecision:
    """
    Score all modalities and return a ranked TherapyDecision.
    No GRN data required — uses structural signals only.
    """
    uid      = report.get("uniprot_id", "?")
    gene     = report.get("gene_name", "?")
    protein  = report.get("protein_name", "?")
    organism = report.get("organism", "?")

    pockets    = report.get("binding_pockets", [])
    allo_sites = report.get("allosteric_sites", [])
    ppi        = report.get("ppi_partners", [])
    active_sites = report.get("active_sites", [])
    ec         = report.get("ec_number", "")
    is_enzyme  = bool(report.get("is_enzyme", ec != ""))

    # Pocket
    top_pocket   = pockets[0] if pockets else {}
    top_vol      = float(top_pocket.get("volume_A3", 0))
    top_drug     = float(top_pocket.get("druggability_score", 0))
    top_pid      = top_pocket.get("pocket_id", "")
    good_pocket  = top_drug >= MIN_POCKET_DRUG_SCORE and top_vol >= MIN_POCKET_VOLUME

    # Allosteric
    top_allo_corr = 0.0
    n_allo_res    = 0
    if allo_sites:
        top_allo_corr = float(allo_sites[0].get("mean_correlation", 0))
        n_allo_res    = int(allo_sites[0].get("size", len(allo_sites[0].get("residue_numbers", []))))
    has_allo = bool(allo_sites) and top_allo_corr >= MIN_ALLO_CORR

    # Surface flags
    surface            = _is_surface(report, physico)
    epigenetic         = _is_epigenetic(report)
    has_internalise    = _has_internalisation(report)
    has_tumour_ag      = _has_tumour_antigen(report)
    epitope_sasa       = _top_epitope_sasa(physico, active_sites, ppi)

    # ── Score every modality ──────────────────────────────────────────────────
    scores: List[ModalityScore] = [
        _score_adc(surface, has_internalise, has_tumour_ag, epitope_sasa, top_drug),
        _score_cart(surface, has_internalise, has_tumour_ag, epitope_sasa),
        _score_naked_antibody(surface, epitope_sasa, ppi),
        _score_small_molecule(good_pocket, top_pid, top_vol, top_drug, is_enzyme, ec),
        _score_protac(good_pocket, top_drug, epigenetic, surface, top_vol, ppi),
        _score_allosteric(has_allo, top_allo_corr, n_allo_res, good_pocket),
        _score_molecular_glue(good_pocket, has_allo, surface, ppi),
    ]

    # Sort viable first, then by score
    scores.sort(key=lambda ms: (ms.viable, ms.score), reverse=True)

    # Primary and secondary
    viable = [ms for ms in scores if ms.viable]
    primary   = viable[0].modality if viable else "undruggable"
    secondary = viable[1].modality if len(viable) > 1 else ""

    # Confidence: driven by top score gap and number of evidence signals
    n_evidence = sum(1 for ms in viable)
    top_score  = scores[0].score if scores else 0.0
    second_score = scores[1].score if len(scores) > 1 else 0.0
    gap = top_score - second_score
    if top_score >= 0.70 and gap >= 0.15:
        confidence = "HIGH"
    elif top_score >= 0.50:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # PPI combo note
    known_drug_targets = {
        "KRAS", "TP53", "EGFR", "CDK4", "CDK6", "MDM2", "BCL2",
        "MTOR", "AKT1", "TOP2A", "ATAD2", "CLSPN", "HSP90",
    }
    combo_partners = [p.get("partner_name","") for p in ppi[:5]
                      if p.get("partner_name","") in known_drug_targets]
    combo_note = (f"PPI partners include {combo_partners} — combination opportunity"
                  if combo_partners else "")

    return TherapyDecision(
        uniprot_id=uid, gene_name=gene, protein_name=protein, organism=organism,
        modality_scores=scores,
        primary_modality=primary, secondary_modality=secondary,
        confidence=confidence,
        is_surface=surface, has_good_pocket=good_pocket, has_allosteric=has_allo,
        is_epigenetic=epigenetic, has_internalisation=has_internalise,
        has_tumour_antigen=has_tumour_ag,
        top_pocket_id=top_pid, top_pocket_vol=top_vol, top_pocket_drug=top_drug,
        top_allo_corr=top_allo_corr, top_epitope_sasa=epitope_sasa,
        combination_note=combo_note, n_evidence=n_evidence,
    )


# ══════════════════════════════════════════════════════════════════════════════
# EPITOPE FINDER (unchanged logic, kept for antibody/ADC/CAR-T paths)
# ══════════════════════════════════════════════════════════════════════════════

def _score_epitope(residue_numbers, residue_letters, physico):
    """Score epitope using SASA from physico where available, fallback to 40Å²/residue."""
    res_map    = {r["residue_number"]: r for r in physico.get("residues", [])}
    # Use physico SASA if we have it, otherwise assume moderate exposure (40 Å²)
    sasa_vals  = [float(res_map.get(rn, {}).get("sasa", 40.0)) for rn in residue_numbers]
    immuno_aa  = sum(1 for aa in residue_letters if aa in IMMUNOGENIC_AA)
    total_sasa = sum(sasa_vals)
    mean_sasa  = total_sasa / max(len(sasa_vals), 1)
    return (
        round(min(1.0, (immuno_aa / max(len(residue_letters), 1)) * 1.5), 3),
        round(min(1.0, mean_sasa / 60.0), 3),
        round(total_sasa, 1),
    )


def find_epitopes(report: dict, physico: dict) -> List[EpitopeCandidate]:
    epitopes = []
    eid = 1

    # Active site surface residues
    surface_active = [s for s in report.get("active_sites", [])
                      if s.get("confidence") in ("HIGH", "MEDIUM")]
    if surface_active:
        nums = [s["residue_number"] for s in surface_active[:15]]
        lets = [s.get("one_letter", "A") for s in surface_active[:15]]
        imm, acc, sasa = _score_epitope(nums, lets, physico)
        if sasa >= MIN_EPITOPE_SASA and len(nums) >= MIN_EPITOPE_RESIDUES:
            epitopes.append(EpitopeCandidate(
                epitope_id=f"E{eid}", residue_numbers=nums, residue_letters=lets,
                source="active_site", total_sasa=sasa,
                immunogenicity_score=imm, accessibility=acc,
                n_immunogenic_aa=sum(1 for a in lets if a in IMMUNOGENIC_AA),
                notes="Active/functional site — highly specific",
            ))
            eid += 1

    # PPI interface — use interface_letters stored in the partner record directly.
    # Do NOT re-lookup from physico; the letters were set at PPI prediction time.
    #
    # Fallback detection: _sequence_interface (fired when no SASA data available)
    # produces identical residue sets for every partner. Detect this by checking
    # whether the first partner's residue numbers appear verbatim in subsequent
    # partners — if so, skip all PPI epitopes (they're not real interface data).
    ppi_partners_filtered = [p for p in report.get("ppi_partners", [])[:5]
                             if p.get("combined_score", 0) >= 400
                             and p.get("interface_residues")
                             and p.get("interface_letters")]

    # Check for the sequence_interface fallback signature:
    # same residue numbers reused across ≥2 partners
    if len(ppi_partners_filtered) >= 2:
        first_nums = tuple(sorted(ppi_partners_filtered[0].get("interface_residues", [])))
        n_identical = sum(
            1 for p in ppi_partners_filtered[1:]
            if tuple(sorted(p.get("interface_residues", []))) == first_nums
        )
        ppi_data_valid = n_identical < len(ppi_partners_filtered) - 1
    else:
        ppi_data_valid = bool(ppi_partners_filtered)

    if ppi_data_valid:
        for partner in ppi_partners_filtered[:3]:
            nums = partner.get("interface_residues", [])
            lets = partner.get("interface_letters", [])
            if len(lets) != len(nums) or not nums:
                continue
            imm, acc, sasa = _score_epitope(nums, lets, physico)
            if sasa >= MIN_EPITOPE_SASA and len(nums) >= MIN_EPITOPE_RESIDUES:
                pname = partner.get("partner_name", "?")
                score_str = partner.get("combined_score", 0)
                epitopes.append(EpitopeCandidate(
                    epitope_id=f"E{eid}", residue_numbers=nums, residue_letters=lets,
                    source="ppi_interface", total_sasa=sasa,
                    immunogenicity_score=imm, accessibility=acc,
                    n_immunogenic_aa=sum(1 for a in lets if a in IMMUNOGENIC_AA),
                    notes=f"PPI interface with {pname} (STRING score={score_str})",
                ))
                eid += 1

    # Surface patches from physico
    for patch_type in ("hydrophobic_patches", "positive_patches", "negative_patches"):
        for patch in sorted(physico.get(patch_type, []),
                            key=lambda p: p.get("total_sasa", 0), reverse=True)[:2]:
            nums = patch.get("residue_numbers", [])
            if not nums:
                continue
            res_map = {r["residue_number"]: r.get("one_letter","A")
                       for r in physico.get("residues", [])}
            lets = [res_map.get(n, "A") for n in nums]
            imm, acc, sasa = _score_epitope(nums, lets, physico)
            if sasa >= MIN_EPITOPE_SASA and len(lets) >= MIN_EPITOPE_RESIDUES:
                epitopes.append(EpitopeCandidate(
                    epitope_id=f"E{eid}", residue_numbers=nums, residue_letters=lets,
                    source="surface_patch", total_sasa=sasa,
                    immunogenicity_score=imm, accessibility=acc,
                    n_immunogenic_aa=sum(1 for a in lets if a in IMMUNOGENIC_AA),
                    notes=f"{patch_type} SASA={sasa:.0f}Å²",
                ))
                eid += 1

    epitopes.sort(key=lambda e: e.immunogenicity_score * e.accessibility, reverse=True)
    return epitopes[:6]


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TRIGGERS
# ══════════════════════════════════════════════════════════════════════════════

def _trigger_adc(uid: str, inter_dir: Path, force: bool) -> Optional[str]:
    try:
        from pipeline.adc_design import run_adc_design
        def _load(f):
            p = inter_dir / f
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        r = run_adc_design(uid, _load(f"{uid}_active_sites.json"),
                           _load(f"{uid}_physicochemical.json"),
                           _load(f"{uid}_ppi.json"),
                           _load(f"{uid}_allosteric.json"), force=force)
        out = inter_dir / f"{uid}_adc.json"
        return str(out) if out.exists() else None
    except Exception as e:
        print(f"  [FAIL] ADC design: {e}")
        return None


def _trigger_cart(uid: str, inter_dir: Path, force: bool) -> Optional[str]:
    try:
        from pipeline.cart_design import run_cart_design
        def _load(f):
            p = inter_dir / f
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        r = run_cart_design(uid, _load(f"{uid}_active_sites.json"),
                            _load(f"{uid}_physicochemical.json"),
                            _load(f"{uid}_ppi.json"),
                            _load(f"{uid}_allosteric.json"), force=force)
        out = inter_dir / f"{uid}_cart.json"
        return str(out) if out.exists() else None
    except Exception as e:
        print(f"  [FAIL] CAR-T design: {e}")
        return None


def _trigger_antibody(uid: str, inter_dir: Path, force: bool) -> Optional[str]:
    try:
        from pipeline.antibody_design import run_antibody_design
        def _load(f):
            p = inter_dir / f
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        out_path = inter_dir / f"{uid}_antibody.json"
        if out_path.exists() and not force:
            return str(out_path)
        r = run_antibody_design(uid, _load(f"{uid}_active_sites.json"),
                                _load(f"{uid}_physicochemical.json"),
                                _load(f"{uid}_ppi.json"),
                                _load(f"{uid}_allosteric.json"))
        r.to_json(out_path)
        return str(out_path) if out_path.exists() else None
    except Exception as e:
        print(f"  [FAIL] Antibody design: {e}")
        return None


def _trigger_protac(uid: str, inter_dir: Path, force: bool) -> Optional[str]:
    try:
        from pipeline.protac_design import run_protac_design
        def _load(f):
            p = inter_dir / f
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        r = run_protac_design(uid, _load(f"{uid}_binding_pockets.json"),
                              _load(f"{uid}_active_sites.json"),
                              _load(f"{uid}_allosteric.json"), force=force)
        out = inter_dir / f"{uid}_protac.json"
        return str(out) if out.exists() else None
    except Exception as e:
        print(f"  [FAIL] PROTAC design: {e}")
        return None


def _trigger_allosteric_drug(uid: str, inter_dir: Path, force: bool) -> Optional[str]:
    try:
        from pipeline.allosteric_drug_design import run_allosteric_drug_design
        def _load(f):
            p = inter_dir / f
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        r = run_allosteric_drug_design(uid, _load(f"{uid}_allosteric.json"),
                                       _load(f"{uid}_physicochemical.json"), force=force)
        out = inter_dir / f"{uid}_allosteric_drug.json"
        return str(out) if out.exists() else None
    except Exception as e:
        print(f"  [FAIL] Allosteric drug design: {e}")
        return None


def _trigger_denovo(uid: str, vina_path: str, receptor_path: str,
                    inter_dir: Path) -> Optional[str]:
    from proteinfp.deps import has_rdkit, has_vina
    if not has_rdkit():
        print("  [SKIP] De novo: RDKit not installed (pip install proteinfp[chem])")
        return None
    if not has_vina(vina_path):
        print(f"  [SKIP] De novo: Vina not found at {vina_path or 'PATH'}")
        return None
    try:
        from pipeline.denovo_design import run_denovo_design
        from pipeline.denovo_design_context import load_consensus_context, load_md_context
        def _load(f):
            p = inter_dir / f
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        resolved_receptor = ""
        if receptor_path and Path(receptor_path).exists():
            resolved_receptor = receptor_path
        run_denovo_design(
            uniprot_id=uid, pocket_data=_load(f"{uid}_binding_pockets.json"),
            active_data=_load(f"{uid}_active_sites.json"),
            allosteric_data=_load(f"{uid}_allosteric.json"),
            chem_env_data=_load(f"{uid}_chemical_env.json"),
            vina_path=vina_path, receptor_path=resolved_receptor,
            consensus_data=load_consensus_context(uid, inter_dir),
            md_data=load_md_context(uid, inter_dir),
        )
        out = inter_dir / f"{uid}_denovo.json"
        return str(out) if out.exists() else None
    except Exception as e:
        print(f"  [FAIL] De novo design: {e}")
        return None


def _trigger_pharm(uid: str) -> Optional[str]:
    try:
        from sim.denovo_to_sim06 import score_denovo_candidates, save_denovo_scores
        scores = score_denovo_candidates(uid, top_n=10, verbose=True)
        if scores:
            return str(save_denovo_scores(scores, uid))
    except Exception as e:
        print(f"  [SKIP] Pharm scoring: {e}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_therapy(
    uniprot_id:    str,
    vina_path:     Optional[str] = None,
    receptor_path: str           = "",
    run_denovo:    bool          = True,
    run_antibody:  bool          = True,
    run_adc:       bool          = True,
    run_cart:      bool          = True,
    run_protac:    bool          = True,
    run_allodrug:  bool          = True,
    force:         bool          = False,
    verbose:       bool          = True,
) -> TherapyResult:
    """
    Run the full therapy workflow.

    Design modules are skipped automatically if:
      - Their output JSON already exists  (use force=True to override)
      - The modality scores below the viable threshold for this protein

    Args:
        uniprot_id:    UniProt accession
        vina_path:     Path to AutoDock Vina (enables de novo small mol)
        receptor_path: PDBQT receptor (auto-prepared if empty)
        run_*:         Enable/disable specific design modules
        force:         Re-run all designs even if cached outputs exist
        verbose:       Print progress

    Returns:
        TherapyResult
    """
    t0  = time.time()
    uid = uniprot_id.strip().upper()

    try:
        from utils.config import cfg
        inter_dir  = Path(cfg.paths["intermediate"])
        report_dir = Path(cfg.paths["reports"])
    except Exception:
        inter_dir  = ROOT / "data" / "intermediate"
        report_dir = ROOT / "data" / "reports"

    # ── Load consensus report ──────────────────────────────────────────────────
    report_path = report_dir / f"{uid}_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Consensus report not found: {report_path}\n"
            f"  Run first: proteinfp --uniprot {uid}"
        )
    report  = json.loads(report_path.read_text(encoding="utf-8"))
    physico_path = inter_dir / f"{uid}_physicochemical.json"
    physico = json.loads(physico_path.read_text(encoding="utf-8")) if physico_path.exists() else {}

    if verbose:
        print(f"\n{'─'*65}")
        print(f"  Therapy: {report.get('gene_name','?')} ({uid})")
        print(f"  {report.get('protein_name','?')}")
        print(f"{'─'*65}")

    # ── Step 1: Decision ───────────────────────────────────────────────────────
    if verbose:
        print("\n  [1] Scoring all therapy modalities...")
    decision = make_therapy_decision(report, physico)

    if verbose:
        print(f"\n  {'Modality':<22} {'Score':>6}  {'Viable':<6}  {'Top reason'}")
        print(f"  {'─'*22} {'─'*6}  {'─'*6}  {'─'*35}")
        for ms in decision.modality_scores:
            reason = ms.rationale[0][:45] if ms.rationale else (ms.blockers[0][:45] if ms.blockers else "")
            print(f"  {ms.modality:<22} {ms.score:>6.3f}  {'✓' if ms.viable else '✗':<6}  {reason}")
        print(f"\n  → Primary  : {decision.primary_modality.upper()}  [{decision.confidence}]")
        if decision.secondary_modality:
            print(f"  → Secondary: {decision.secondary_modality.upper()}")

    # ── Step 2: Epitopes (all surface paths) ───────────────────────────────────
    epitopes: List[EpitopeCandidate] = []
    if decision.is_surface:
        if verbose:
            print("\n  [2] Finding epitope candidates...")
        epitopes = find_epitopes(report, physico)
        if verbose:
            print(f"  → {len(epitopes)} epitopes found")
            for ep in epitopes[:2]:
                print(f"     {ep.epitope_id}: {ep.source}  "
                      f"SASA={ep.total_sasa:.0f}Å²  "
                      f"immunogenicity={ep.immunogenicity_score:.2f}")

    # ── Step 3: Trigger design modules ────────────────────────────────────────
    designs_triggered: List[str]     = []
    design_outputs:    Dict[str,str] = {}
    denovo_path:       Optional[str] = None
    pharm_path:        Optional[str] = None

    viable_names = {ms.modality for ms in decision.modality_scores if ms.viable}

    if verbose:
        print(f"\n  [3] Running design modules for: {', '.join(sorted(viable_names)) or 'none'}")

    # ADC
    if run_adc and "adc" in viable_names:
        if verbose: print("\n  → ADC design (Module 18)...")
        out = _trigger_adc(uid, inter_dir, force)
        if out:
            designs_triggered.append("adc")
            design_outputs["adc"] = out

    # CAR-T
    if run_cart and "car_t" in viable_names:
        if verbose: print("\n  → CAR-T design (Module 19)...")
        out = _trigger_cart(uid, inter_dir, force)
        if out:
            designs_triggered.append("car_t")
            design_outputs["car_t"] = out

    # Naked antibody / bispecific
    if run_antibody and "naked_antibody" in viable_names:
        if verbose: print("\n  → Antibody CDR design (Module 16)...")
        out = _trigger_antibody(uid, inter_dir, force)
        if out:
            designs_triggered.append("antibody")
            design_outputs["antibody"] = out

    # Small molecule (de novo with Vina)
    if run_denovo and "small_molecule" in viable_names:
        if vina_path:
            if verbose: print("\n  → De novo small molecule design (Module 15)...")
            denovo_path = _trigger_denovo(uid, vina_path, receptor_path, inter_dir)
            if denovo_path:
                designs_triggered.append("small_molecule")
                design_outputs["small_molecule"] = denovo_path
                pharm_path = _trigger_pharm(uid)
                if pharm_path:
                    design_outputs["pharm_scores"] = pharm_path
        else:
            if verbose:
                print("  [SKIP] Small molecule design — no --vina path provided")

    # PROTAC
    if run_protac and "protac" in viable_names:
        if verbose: print("\n  → PROTAC design (Module 20)...")
        out = _trigger_protac(uid, inter_dir, force)
        if out:
            designs_triggered.append("protac")
            design_outputs["protac"] = out

    # Allosteric drug
    if run_allodrug and "allosteric" in viable_names:
        if verbose: print("\n  → Allosteric drug design (Module 21)...")
        out = _trigger_allosteric_drug(uid, inter_dir, force)
        if out:
            designs_triggered.append("allosteric")
            design_outputs["allosteric"] = out

    # ── Save ───────────────────────────────────────────────────────────────────
    result = TherapyResult(
        uniprot_id=uid, decision=decision, epitopes=epitopes,
        designs_triggered=designs_triggered, design_outputs=design_outputs,
        denovo_path=denovo_path, pharm_scores_path=pharm_path,
        elapsed_sec=round(time.time() - t0, 1),
    )

    out_json = report_dir / f"{uid}_therapy.json"
    out_txt  = report_dir / f"{uid}_therapy.txt"
    out_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    out_txt.write_text(result.summary(), encoding="utf-8")

    if verbose:
        print(result.summary())
        print(f"  Saved → {out_json}")
        print(f"  Saved → {out_txt}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE DESIGN PICKER
# ══════════════════════════════════════════════════════════════════════════════

# Guidance text shown for each modality — drawn from therapy decision context
_MODALITY_GUIDANCE = {
    "protac": {
        "name": "PROTAC / Protein Degrader",
        "what": (
            "Bifunctional molecule: one end binds your target protein (warhead), "
            "the other recruits an E3 ubiquitin ligase. The target gets ubiquitinated "
            "and degraded by the proteasome. Catalytic mechanism — one PROTAC destroys "
            "many copies of the target protein."
        ),
        "when_best": "Intracellular protein with epigenetic role OR strong MDM2/VHL/CRBN PPI.",
        "parameters": ["E3 ligase choice (CRBN/VHL/IAP/MDM2)", "Linker length/type", "Generations"],
        "module": "Module 20 — protac_design.py",
        "e3_hint_from_ppi": {
            "MDM2": "CRBN or MDM2 (MDM2 PPI detected — MDM2 E3 ligand is a natural choice)",
            "MDM4": "CRBN (pomalidomide — most clinically validated for TP53 restoration)",
            "KEAP1": "CRBN or VHL",
            "CUL4": "CRBN",
            "VHL":  "VHL (VHL is already an E3 — use VHL ligand directly)",
        },
    },
    "allosteric": {
        "name": "Allosteric Small Molecule",
        "what": (
            "Binds a site AWAY from the active site and changes protein conformation "
            "to inhibit or activate function. Higher selectivity than orthosteric drugs "
            "because allosteric sites are less conserved across protein families."
        ),
        "when_best": "High ENM correlation site, especially when active site is undruggable.",
        "parameters": ["Target site (A1/A2/…)", "Mechanism (inhibitor/activator/modulator)", "Generations"],
        "module": "Module 21 — allosteric_drug_design.py",
    },
    "small_molecule": {
        "name": "Small Molecule Inhibitor (De Novo)",
        "what": (
            "Fragment-based evolutionary design of drug-like molecules targeting the "
            "primary druggable pocket. Requires AutoDock Vina for docking scores. "
            "Outputs SMILES with Lipinski filtering and composite fitness."
        ),
        "when_best": "Intracellular enzyme with a deep, hydrophobic pocket (druggability ≥ 0.6).",
        "parameters": ["Vina path (required)", "Pocket (auto-selected from Module 04)"],
        "module": "Module 15 — denovo_design.py",
    },
    "adc": {
        "name": "Antibody-Drug Conjugate (ADC)",
        "what": (
            "Antibody binds a surface epitope → gets internalised → linker cleaves in "
            "endosome → cytotoxic warhead kills the cell. Co-evolves CDR sequences, "
            "linker chemistry, and warhead (MMAE/DM1/SN-38/PBD/calicheamicin)."
        ),
        "when_best": "Surface protein with internalisation signal. Best SASA: 200–1200 Å².",
        "parameters": ["Warhead class (MMAE/DM1/…)", "Epitope mode", "Generations"],
        "module": "Module 18 — adc_design.py",
    },
    "car_t": {
        "name": "CAR-T Cell Therapy",
        "what": (
            "Patient T-cells engineered with a chimeric antigen receptor (CAR) that "
            "binds your surface target. Co-evolves scFv CDR sequences, CAR generation "
            "(1st–4th gen including TRUCK), and hinge region."
        ),
        "when_best": "Surface protein with large exposed domain (SASA > 600 Å²), tumour antigen.",
        "parameters": ["CAR generation (1–5)", "Epitope mode", "Generations"],
        "module": "Module 19 — cart_design.py",
    },
    "naked_antibody": {
        "name": "Naked Antibody / Bispecific",
        "what": (
            "Therapeutic monoclonal antibody that blocks a surface target or its PPI, "
            "recruiting immune effectors (ADCC/CDC). Evolves CDR sequences for "
            "maximum affinity + developability without a payload."
        ),
        "when_best": "Surface protein with a clinically validated PPI (EGFR, PD1, VEGF…).",
        "parameters": ["Epitope mode", "Generations"],
        "module": "Module 16 — antibody_design.py",
    },
    "molecular_glue": {
        "name": "Molecular Glue",
        "what": (
            "Small molecule that creates a new protein-protein interface, typically "
            "between the target and an E3 ligase substrate receptor, leading to "
            "degradation without a defined warhead-binding pocket."
        ),
        "when_best": "No pocket, no allosteric site, but PPI with E3 complex present.",
        "parameters": ["No dedicated module yet — use PROTAC design as closest proxy"],
        "module": "Module 20 — protac_design.py (closest proxy)",
    },
}

_MODALITY_CONTEXT_HINTS = {
    "protac": [
        ("top_pocket_drug",   lambda v: f"Pocket druggability {v:.2f} → warhead binding site identified"),
        ("top_pocket_vol",    lambda v: f"Pocket volume {v:.0f}Å³ → room for warhead (~300 Da)"),
        ("top_allo_corr",     lambda v: f"Allosteric corr {v:.3f} → conformational flexibility aids ternary complex"),
        ("is_epigenetic",     lambda v: "Epigenetic target → removing all function beats inhibiting one site"),
    ],
    "allosteric": [
        ("top_allo_corr",     lambda v: f"ENM correlation {v:.3f} → allosteric signal propagates to active site"),
        ("top_pocket_drug",   lambda v: f"Orthosteric pocket drug={v:.2f} also exists — allosteric offers selectivity"),
    ],
    "small_molecule": [
        ("top_pocket_drug",   lambda v: f"Pocket druggability {v:.2f} — {'excellent' if v >= 0.8 else 'good'} target"),
        ("top_pocket_vol",    lambda v: f"Pocket volume {v:.0f}Å³ — {'large' if v >= 800 else 'standard'} binding site"),
    ],
    "adc": [
        ("top_epitope_sasa",  lambda v: f"Best epitope SASA {v:.0f}Å² — {'ideal' if 200<=v<=1200 else 'marginal'} for ADC"),
        ("has_internalisation", lambda v: ("Internalisation GO terms present — payload delivery confirmed" if v
                                           else "No internalisation GO terms — use cleavable linker")),
    ],
    "car_t": [
        ("top_epitope_sasa",  lambda v: f"Surface domain SASA {v:.0f}Å² — {'excellent' if v>=600 else 'adequate'} for CAR"),
        ("has_tumour_antigen",lambda v: ("Tumour antigen marker detected — strong clinical precedent" if v
                                         else "No tumour antigen GO terms — verify cancer-selective expression")),
    ],
    "naked_antibody": [
        ("top_epitope_sasa",  lambda v: f"Epitope SASA {v:.0f}Å² — mAb binding accessible"),
    ],
}


def _build_guidance(modality: str, decision: "TherapyDecision") -> List[str]:
    """Build context-specific guidance bullets from the therapy decision data."""
    hints = _MODALITY_CONTEXT_HINTS.get(modality, [])
    lines = []
    for attr, fmt_fn in hints:
        val = getattr(decision, attr, None)
        if val is not None:
            try:
                lines.append(fmt_fn(val))
            except Exception:
                pass
    return lines


def _ask_e3_for_protac(decision: "TherapyDecision") -> str:
    """Suggest the best E3 ligase based on PPI partners."""
    ppi_names = {ms.modality for ms in decision.modality_scores}  # placeholder
    # Read from the rationale of the PROTAC modality score
    protac_score = next((ms for ms in decision.modality_scores if ms.modality == "protac"), None)
    if protac_score:
        for r in protac_score.rationale:
            if "MDM" in r:
                return "CRBN"
            if "VHL" in r:
                return "VHL"
            if "KEAP" in r:
                return "CRBN"
    return "CRBN"   # safest default — most clinical precedent


def _run_design_interactive(
    choice: str,
    uid: str,
    decision: "TherapyDecision",
    inter_dir: Path,
    vina_path: Optional[str],
    receptor_path: str,
    force: bool,
) -> Optional[str]:
    """Run one design module with parameters guided by the therapy decision."""

    def _load(f):
        p = inter_dir / f
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    if choice == "protac":
        e3 = _ask_e3_for_protac(decision)
        print(f"\n  Suggested E3 ligase: {e3}")
        override = input(f"  Use {e3}? [Enter to confirm, or type CRBN/VHL/IAP/MDM2]: ").strip().upper()
        e3 = override if override in ("CRBN", "VHL", "IAP", "MDM2") else e3
        gen_str = input("  Generations [50]: ").strip()
        gens = int(gen_str) if gen_str.isdigit() else 50
        from pipeline.protac_design import run_protac_design
        r = run_protac_design(uid, _load(f"{uid}_binding_pockets.json"),
                              _load(f"{uid}_active_sites.json"),
                              _load(f"{uid}_allosteric.json"),
                              preferred_e3=e3, n_generations=gens, force=force)
        out = inter_dir / f"{uid}_protac.json"
        return str(out) if out.exists() else None

    elif choice == "allosteric":
        sites_raw = (_load(f"{uid}_allosteric.json") or {}).get("allosteric_sites", [])
        if sites_raw:
            print(f"\n  Available allosteric sites:")
            for i, s in enumerate(sites_raw[:5], 1):
                print(f"    {i}. {s.get('site_id','?')}  "
                      f"corr={s.get('mean_correlation',0):.3f}  "
                      f"size={s.get('size', len(s.get('residue_numbers',[])))}")
            site_str = input("  Target site [A1]: ").strip() or "A1"
        else:
            site_str = "A1"
        mech = input("  Mechanism — inhibitor / activator / modulator [inhibitor]: ").strip() or "inhibitor"
        mech = mech if mech in ("inhibitor", "activator", "modulator") else "inhibitor"
        gen_str = input("  Generations [50]: ").strip()
        gens = int(gen_str) if gen_str.isdigit() else 50
        from pipeline.allosteric_drug_design import run_allosteric_drug_design
        r = run_allosteric_drug_design(uid, _load(f"{uid}_allosteric.json"),
                                       _load(f"{uid}_physicochemical.json"),
                                       preferred_site=site_str,
                                       preferred_mechanism=mech,
                                       n_generations=gens, force=force)
        out = inter_dir / f"{uid}_allosteric_drug.json"
        return str(out) if out.exists() else None

    elif choice == "small_molecule":
        if not vina_path:
            print("  ✗ Small molecule de novo design requires --vina. "
                  "Re-run with: python proteinfp/therapy.py --uniprot "
                  f"{uid} --interactive --vina /path/to/vina")
            return None
        return _trigger_denovo(uid, vina_path, receptor_path, inter_dir)

    elif choice == "adc":
        warheads = ["MMAE", "DM1", "DM4", "SN38", "Dxd", "CalicheA", "PBD", "MMAF"]
        print(f"\n  Warhead options: {', '.join(warheads)}")
        print(f"  (default: co-evolve — best warhead found automatically)")
        wh = input("  Fix warhead or press Enter to co-evolve: ").strip().upper()
        wh = wh if wh in warheads else None
        epi = input("  Epitope mode — auto/active/ppi/surface/allosteric [auto]: ").strip() or "auto"
        epi = epi if epi in ("auto","active","ppi","surface","allosteric") else "auto"
        gen_str = input("  Generations [50]: ").strip()
        gens = int(gen_str) if gen_str.isdigit() else 50
        from pipeline.adc_design import run_adc_design
        r = run_adc_design(uid, _load(f"{uid}_active_sites.json"),
                           _load(f"{uid}_physicochemical.json"),
                           _load(f"{uid}_ppi.json"),
                           _load(f"{uid}_allosteric.json"),
                           preferred_warhead=wh, epitope_mode=epi,
                           n_generations=gens, force=force)
        out = inter_dir / f"{uid}_adc.json"
        return str(out) if out.exists() else None

    elif choice == "car_t":
        print("\n  CAR generations:")
        print("    1 = CD3ζ only (basic, low persistence)")
        print("    2 = CD28 + CD3ζ  (fast activation — axicabtagene model)")
        print("    3 = 4-1BB + CD3ζ (durable — tisagenlecleucel model)")
        print("    4 = CD28 + 4-1BB + CD3ζ (potent but higher CRS risk)")
        print("    5 = 4th gen TRUCK (+ cytokine payload, solid tumours)")
        gen_car = input("  CAR generation [co-evolve]: ").strip()
        gen_car_int = int(gen_car) if gen_car.isdigit() and 1 <= int(gen_car) <= 5 else None
        epi = input("  Epitope mode — auto/active/ppi/surface/allosteric [auto]: ").strip() or "auto"
        epi = epi if epi in ("auto","active","ppi","surface","allosteric") else "auto"
        gen_str = input("  Generations [50]: ").strip()
        gens = int(gen_str) if gen_str.isdigit() else 50
        from pipeline.cart_design import run_cart_design
        r = run_cart_design(uid, _load(f"{uid}_active_sites.json"),
                            _load(f"{uid}_physicochemical.json"),
                            _load(f"{uid}_ppi.json"),
                            _load(f"{uid}_allosteric.json"),
                            preferred_gen=gen_car_int, epitope_mode=epi,
                            n_generations=gens, force=force)
        out = inter_dir / f"{uid}_cart.json"
        return str(out) if out.exists() else None

    elif choice in ("naked_antibody", "antibody"):
        epi = input("  Epitope mode — auto/active/ppi/surface/allosteric [auto]: ").strip() or "auto"
        epi = epi if epi in ("auto","active","ppi","surface","allosteric") else "auto"
        gen_str = input("  Generations [50]: ").strip()
        gens = int(gen_str) if gen_str.isdigit() else 50
        from pipeline.antibody_design import run_antibody_design
        out_path = inter_dir / f"{uid}_antibody.json"
        r = run_antibody_design(uid, _load(f"{uid}_active_sites.json"),
                                _load(f"{uid}_physicochemical.json"),
                                _load(f"{uid}_ppi.json"),
                                _load(f"{uid}_allosteric.json"),
                                n_generations=gens, epitope_mode=epi)
        r.to_json(out_path)
        return str(out_path) if out_path.exists() else None

    elif choice == "molecular_glue":
        print("  No dedicated molecular glue module yet — running PROTAC design as proxy.")
        return _run_design_interactive("protac", uid, decision, inter_dir,
                                       vina_path, receptor_path, force)

    return None


def interactive_design(
    uniprot_id:    str,
    vina_path:     Optional[str] = None,
    receptor_path: str           = "",
    force:         bool          = False,
) -> None:
    """
    Interactive therapy → design workflow.

    1. Runs the therapy decision engine (fast, ~1s)
    2. Prints a numbered menu of all viable modalities with guidance
    3. User types one or more numbers
    4. Runs the selected design module(s) with parameter prompts pre-filled
       from what the therapy engine found about the protein

    Usage:
        python proteinfp/therapy.py --uniprot P04637 --interactive
        python proteinfp/therapy.py --uniprot P04637 --interactive --vina pipeline/vina.exe
    """
    uid = uniprot_id.strip().upper()

    try:
        from utils.config import cfg
        inter_dir  = Path(cfg.paths["intermediate"])
        report_dir = Path(cfg.paths["reports"])
    except Exception:
        inter_dir  = ROOT / "data" / "intermediate"
        report_dir = ROOT / "data" / "reports"

    # ── Step 1: Run decision engine (no design modules) ───────────────────────
    print(f"\n{'═'*65}")
    print(f"  ProteinFP Interactive Design — {uid}")
    print(f"{'═'*65}")
    print("  Running therapy decision engine...\n")

    result = run_therapy(
        uniprot_id=uid, vina_path=vina_path, receptor_path=receptor_path,
        run_denovo=False, run_antibody=False, run_adc=False, run_cart=False,
        run_protac=False, run_allodrug=False, force=False, verbose=False,
    )
    decision = result.decision
    viable   = [ms for ms in decision.modality_scores if ms.viable]

    if not viable:
        print(f"  ✗ No viable modalities found for {uid}.")
        print(f"    Check that the pipeline has run: proteinfp --uniprot {uid}")
        return

    # ── Step 2: Print the menu ────────────────────────────────────────────────
    print(f"  Protein  : {decision.protein_name}")
    print(f"  Gene     : {decision.gene_name}  ({uid})")
    print(f"  Surface  : {'yes' if decision.is_surface else 'no'}")
    if decision.top_pocket_drug > 0:
        print(f"  Pocket   : {decision.top_pocket_id}  "
              f"vol={decision.top_pocket_vol:.0f}Å³  "
              f"druggability={decision.top_pocket_drug:.2f}")
    if decision.top_allo_corr > 0:
        print(f"  Allosteric corr: {decision.top_allo_corr:.3f}")
    if decision.is_epigenetic:
        print(f"  Epigenetic regulator: yes")
    print()

    print(f"{'─'*65}")
    print(f"  VIABLE THERAPY MODALITIES (ranked by score)")
    print(f"{'─'*65}\n")

    for i, ms in enumerate(viable, 1):
        info = _MODALITY_GUIDANCE.get(ms.modality, {})
        name = info.get("name", ms.modality.replace("_", " ").title())
        print(f"  [{i}] {name}")
        print(f"       Score : {ms.score:.3f}   Module: {info.get('module', '?')}")

        # Show why this modality scored well
        context = _build_guidance(ms.modality, decision)
        for bullet in context[:2]:
            print(f"       ✓ {bullet}")

        # Show the top score rationale
        if ms.rationale:
            print(f"       ✓ {ms.rationale[0]}")
        if ms.blockers:
            print(f"       ⚠ {ms.blockers[0]}")

        # What it is (brief)
        what = info.get("what", "")
        if what:
            # Wrap to 60 chars
            words = what.split()
            line = "       "
            for w in words:
                if len(line) + len(w) > 67:
                    print(line)
                    line = "       " + w + " "
                else:
                    line += w + " "
            if line.strip():
                print(line)

        # Best-used-when
        when = info.get("when_best", "")
        if when:
            print(f"       Best when: {when}")
        print()

    if decision.combination_note:
        print(f"  💊 {decision.combination_note}\n")

    print(f"{'─'*65}")

    # ── Step 3: Get user selection ────────────────────────────────────────────
    print("  Enter one or more numbers separated by spaces, or 'all' to run everything.")
    print("  Example: 1 3   or   all   or   2\n")
    raw = input("  Your choice: ").strip().lower()

    if raw == "all":
        selected_indices = list(range(len(viable)))
    elif raw in ("q", "quit", "exit", ""):
        print("  Cancelled.")
        return
    else:
        selected_indices = []
        for token in raw.split():
            try:
                idx = int(token) - 1
                if 0 <= idx < len(viable):
                    selected_indices.append(idx)
                else:
                    print(f"  ⚠ '{token}' out of range — skipped")
            except ValueError:
                print(f"  ⚠ '{token}' is not a number — skipped")

    if not selected_indices:
        print("  No valid selections. Exiting.")
        return

    selected_modalities = [viable[i].modality for i in selected_indices]
    print(f"\n  Running: {', '.join(selected_modalities)}\n")

    # ── Step 4: Run each selected design module ───────────────────────────────
    outputs: Dict[str, str] = {}
    for modality in selected_modalities:
        info = _MODALITY_GUIDANCE.get(modality, {})
        name = info.get("name", modality)
        print(f"{'─'*65}")
        print(f"  ▶ {name}")
        print(f"{'─'*65}")

        # Print parameter guidance before asking
        context = _build_guidance(modality, decision)
        if context:
            print("\n  Context from your protein:")
            for c in context:
                print(f"    • {c}")
        print()

        out = _run_design_interactive(
            modality, uid, decision, inter_dir, vina_path, receptor_path, force
        )
        if out:
            outputs[modality] = out
            print(f"\n  ✓ Saved → {out}\n")
        else:
            print(f"\n  ✗ {name} did not produce output.\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{'═'*65}")
    print(f"  Interactive design complete")
    print(f"{'═'*65}")
    if outputs:
        print(f"\n  Outputs:")
        for modality, path in outputs.items():
            print(f"    {modality:<20} → {path}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI (--test mode for rapid decision-only testing)
# ══════════════════════════════════════════════════════════════════════════════

import click

@click.command()
@click.option("--uniprot",     "-u", required=True, help="UniProt ID to analyse")
@click.option("--vina",        default=None, help="Path to Vina executable")
@click.option("--receptor",    default="", help="PDBQT receptor path")
@click.option("--test",        is_flag=True, default=False,
              help="Decision only — skip all design modules (fast, ~1s)")
@click.option("--interactive", "-i", is_flag=True, default=False,
              help="Interactive mode — score all modalities, pick one, run it with guided parameters")
@click.option("--no-adc",      is_flag=True, default=False)
@click.option("--no-cart",     is_flag=True, default=False)
@click.option("--no-antibody", is_flag=True, default=False)
@click.option("--no-protac",   is_flag=True, default=False)
@click.option("--no-allodrug", is_flag=True, default=False)
@click.option("--force",       "-f", is_flag=True, default=False,
              help="Re-run even if outputs already exist")
def main(uniprot, vina, receptor, test, interactive, no_adc, no_cart,
         no_antibody, no_protac, no_allodrug, force):
    """
    Therapy modality decision + design trigger.

    \b
    Modes:
      --test         Decision only, no evolution (~1 second)
      --interactive  Ranked menu → pick modality → guided parameter prompts → run
      (default)      Score + run all viable modalities automatically

    \b
    Examples:
        python proteinfp/therapy.py --uniprot P04637 --test
        python proteinfp/therapy.py --uniprot P04637 --interactive
        python proteinfp/therapy.py --uniprot P04637 --interactive --vina pipeline/vina.exe
        python proteinfp/therapy.py --uniprot P04637 --no-cart --no-adc
    """
    if interactive:
        interactive_design(
            uniprot_id    = uniprot.strip().upper(),
            vina_path     = vina,
            receptor_path = receptor,
            force         = force,
        )
    else:
        run_therapy(
            uniprot_id    = uniprot.strip().upper(),
            vina_path     = vina,
            receptor_path = receptor,
            run_denovo    = not test and vina is not None,
            run_antibody  = not test and not no_antibody,
            run_adc       = not test and not no_adc,
            run_cart      = not test and not no_cart,
            run_protac    = not test and not no_protac,
            run_allodrug  = not test and not no_allodrug,
            force         = force,
            verbose       = True,
        )


if __name__ == "__main__":
    main()