"""
proteinfp/therapy.py
─────────────────────
Therapy mode — automated drug strategy decision and candidate generation.

Given any protein with a completed consensus report, this module:

  1. THERAPY DECISION
     Reads the consensus report and decides the best drug modality
     without needing GRN/scRNA-seq data. Works from structure alone.

  2. SMALL MOLECULE PATH  (intracellular proteins with druggable pockets)
     → Triggers de novo molecular design (if Vina + RDKit available)
     → Scores candidates through SIM-06 pharmacological framework
     → Outputs ranked molecules with Kd, pKi, and composite grade

  3. ANTIBODY PATH  (surface-exposed proteins)
     → Selects the best epitope from surface patches, active site, or PPI
     → Scores epitopes by accessibility, conservation, and immunogenicity
     → Outputs CDR-H3 seed sequences and epitope coordinates

  4. COMBINED REPORT
     Saves a unified therapy report:
       data/reports/{UNIPROT}_therapy.json
       data/reports/{UNIPROT}_therapy.txt

USAGE
─────
    # Via CLI (recommended):
    proteinfp --uniprot P28593 --therapy
    proteinfp --uniprot P28593 --therapy --denovo --vina pipeline/vina.exe

    # Python API:
    from proteinfp.therapy import run_therapy
    result = run_therapy("P28593")
    print(result.summary())

DECISION LOGIC
──────────────
The modality decision tree (no GRN required):

  Surface protein?
  ├─ YES → ANTIBODY  (ADC / naked antibody / CAR-T)
  │        + small molecule if good pocket also exists
  └─ NO  → Is there a druggable pocket (score ≥ 0.6, vol ≥ 300Å³)?
           ├─ YES → SMALL MOLECULE (de novo design)
           │        + PROTAC if epigenetic/chromatin regulator
           └─ NO  → Is there an allosteric site?
                   ├─ YES → ALLOSTERIC SMALL MOLECULE
                   └─ NO  → UNDRUGGABLE (flag for further analysis)

Surface determination (without GRN):
  - GO CC terms containing "membrane", "extracellular", "cell surface"
  - High-hydrophobicity surface patches from Module 02
  - Absence of nuclear/cytoplasmic GO CC terms
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

# ── Thresholds ─────────────────────────────────────────────────────────────────

MIN_POCKET_DRUG_SCORE  = 0.6    # minimum druggability for small molecule
MIN_POCKET_VOLUME      = 300.0  # minimum pocket volume (Å³)
MIN_ALLO_CORR          = 0.5    # minimum allosteric correlation
MIN_EPITOPE_SASA       = 200.0  # minimum surface area for antibody epitope (Å²)

# GO CC terms that indicate surface/secreted proteins
SURFACE_GO_KEYWORDS = {
    "plasma membrane", "cell surface", "extracellular",
    "secreted", "membrane", "extracellular space",
    "extracellular region", "cell wall",
}

# GO CC terms that indicate intracellular proteins
INTRACELLULAR_GO_KEYWORDS = {
    "nucleus", "nucleoplasm", "chromatin", "cytoplasm",
    "cytosol", "mitochondria", "endoplasmic reticulum",
    "golgi", "lysosome",
}

# GO MF/BP terms suggesting epigenetic/chromatin regulation → PROTAC candidate
EPIGENETIC_GO_KEYWORDS = {
    "chromatin", "histone", "bromodomain", "helicase",
    "chromatin remodeling", "methyltransferase", "demethylase",
}

# Amino acids preferred in antibody epitopes (surface-exposed, immunogenic)
IMMUNOGENIC_AA = {"K", "R", "D", "E", "H", "N", "Q", "S", "T", "Y"}


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EpitopeCandidate:
    """A candidate antibody epitope region."""
    epitope_id:         str
    residue_numbers:    list[int]
    residue_letters:    list[str]
    source:             str        # "active_site" | "surface_patch" | "ppi_interface"
    total_sasa:         float      # Å² total exposed area
    immunogenicity_score: float    # 0-1 estimate
    accessibility:      float      # 0-1
    n_immunogenic_aa:   int
    notes:              str        = ""

    @property
    def sequence(self) -> str:
        return "".join(self.residue_letters)

    def summary(self) -> str:
        return (
            f"  {self.epitope_id}  residues={self.residue_numbers[:5]}{'...' if len(self.residue_numbers)>5 else ''}  "
            f"SASA={self.total_sasa:.0f}Å²  "
            f"immunogenicity={self.immunogenicity_score:.2f}  "
            f"source={self.source}\n"
            f"    sequence: {self.sequence[:30]}{'...' if len(self.sequence)>30 else ''}\n"
            f"    {self.notes}"
        )


@dataclass
class TherapyDecision:
    """The therapy modality decision for a protein."""
    uniprot_id:         str
    gene_name:          str
    protein_name:       str
    organism:           str

    # Decision outputs
    is_surface:         bool
    primary_modality:   str        # "small_molecule" | "antibody" | "protac" | "allosteric" | "undruggable"
    secondary_modality: str        # additional option if applicable
    confidence:         str        # "HIGH" | "MEDIUM" | "LOW"

    # Evidence
    top_pocket_id:      str        = ""
    top_pocket_vol:     float      = 0.0
    top_pocket_drug:    float      = 0.0
    has_allosteric:     bool       = False
    is_epigenetic:      bool       = False
    n_evidence:         int        = 0

    # Rationale
    rationale:          list[str]  = field(default_factory=list)
    combination_note:   str        = ""

    def summary_line(self) -> str:
        mods = self.primary_modality
        if self.secondary_modality:
            mods += f" + {self.secondary_modality}"
        return (
            f"  {self.gene_name} ({self.uniprot_id})  [{self.confidence}]\n"
            f"  Modality     : {mods}\n"
            f"  Surface      : {'yes' if self.is_surface else 'no'}\n"
            f"  Top pocket   : {self.top_pocket_id}  "
            f"vol={self.top_pocket_vol:.0f}Å³  drug={self.top_pocket_drug:.2f}\n"
        )


@dataclass
class TherapyResult:
    """Full therapy workflow result."""
    uniprot_id:         str
    decision:           TherapyDecision
    epitopes:           list[EpitopeCandidate]  = field(default_factory=list)
    denovo_run:         bool                    = False
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

        if self.decision.rationale:
            lines.append("  Rationale:")
            for r in self.decision.rationale:
                lines.append(f"    • {r}")

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

        if self.denovo_path:
            lines += [
                "",
                "─" * 65,
                "  DE NOVO MOLECULES",
                "─" * 65,
                f"  Design output : {self.denovo_path}",
            ]
            if self.pharm_scores_path:
                lines.append(f"  Pharm scores  : {self.pharm_scores_path}")

        lines += [
            "",
            f"  Wall time : {self.elapsed_sec:.1f}s",
            "═" * 65,
            "",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "uniprot_id":        self.uniprot_id,
            "decision":          asdict(self.decision),
            "epitopes":          [asdict(e) for e in self.epitopes],
            "denovo_run":        self.denovo_run,
            "denovo_path":       self.denovo_path,
            "pharm_scores_path": self.pharm_scores_path,
            "elapsed_sec":       self.elapsed_sec,
        }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — THERAPY DECISION (no GRN required)
# ══════════════════════════════════════════════════════════════════════════════

def _is_surface(report: dict, physico: dict) -> bool:
    """
    Determine surface vs intracellular from GO terms and physicochemistry.
    Works without GRN data.
    """
    # Check GO cellular component terms
    for term in report.get("go_terms_cc", []):
        name = term.get("go_name", "").lower()
        if any(kw in name for kw in SURFACE_GO_KEYWORDS):
            # Double-check it's not also nuclear
            if not any(kw in name for kw in INTRACELLULAR_GO_KEYWORDS):
                return True

    # Check subcellular_location field
    location = report.get("subcellular_location", "").lower()
    if any(kw in location for kw in SURFACE_GO_KEYWORDS):
        if not any(kw in location for kw in INTRACELLULAR_GO_KEYWORDS):
            return True

    # Check physicochemical: transmembrane signal
    if physico:
        residues = physico.get("residues", [])
        if residues:
            hydros = sorted(
                [r.get("hydrophobicity", 0.0) for r in residues],
                reverse=True,
            )
            top20_mean = sum(hydros[:20]) / min(20, len(hydros))
            if top20_mean > 2.5:
                return True

    return False


def _is_epigenetic(report: dict) -> bool:
    """Check if the protein is a chromatin/epigenetic regulator."""
    all_go = (
        [t.get("go_name", "") for t in report.get("go_terms_mf", [])] +
        [t.get("go_name", "") for t in report.get("go_terms_bp", [])]
    )
    go_text = " ".join(all_go).lower()
    return any(kw in go_text for kw in EPIGENETIC_GO_KEYWORDS)


def make_therapy_decision(
    report:  dict,
    physico: dict,
) -> TherapyDecision:
    """
    Make a therapy modality decision from a consensus report.
    No GRN or disease expression data required.
    """
    uid         = report.get("uniprot_id", "?")
    gene        = report.get("gene_name", "?")
    protein     = report.get("protein_name", "?")
    organism    = report.get("organism", "?")
    pockets     = report.get("binding_pockets", [])
    allo_sites  = report.get("allosteric_sites", [])
    ppi         = report.get("ppi_partners", [])

    surface     = _is_surface(report, physico)
    epigenetic  = _is_epigenetic(report)

    # Best pocket
    top_pocket = pockets[0] if pockets else {}
    top_vol    = float(top_pocket.get("volume_A3", 0))
    top_drug   = float(top_pocket.get("druggability_score", 0))
    top_pid    = top_pocket.get("pocket_id", "")
    good_pocket = top_drug >= MIN_POCKET_DRUG_SCORE and top_vol >= MIN_POCKET_VOLUME

    # Allosteric
    has_allo = bool(allo_sites) and float(
        allo_sites[0].get("mean_correlation", 0)
    ) >= MIN_ALLO_CORR

    # PPI combo targets
    known_drug_targets = {
        "KRAS", "TP53", "EGFR", "CDK4", "CDK6", "MDM2", "BCL2",
        "MTOR", "AKT1", "TOP2A", "ATAD2", "CLSPN", "HSP90",
    }
    combo = [
        p.get("partner_name", "")
        for p in ppi[:5]
        if p.get("partner_name", "") in known_drug_targets
    ]

    rationale   = []
    primary     = "undruggable"
    secondary   = ""
    n_evidence  = 0

    # ── Decision tree ──────────────────────────────────────────────────────────
    if surface:
        primary    = "antibody"
        n_evidence += 1
        rationale.append(
            "Surface-exposed protein — antibody-based therapy preferred "
            "(ADC, naked antibody, or bispecific)"
        )
        if good_pocket:
            secondary  = "small_molecule"
            n_evidence += 1
            rationale.append(
                f"Also has druggable pocket {top_pid} "
                f"(vol={top_vol:.0f}Å³, drug={top_drug:.2f}) — "
                "small molecule inhibitor also viable"
            )
    elif good_pocket:
        primary    = "small_molecule"
        n_evidence += 1
        rationale.append(
            f"Intracellular with druggable pocket {top_pid} "
            f"(vol={top_vol:.0f}Å³, drug={top_drug:.2f}) — "
            "small molecule inhibitor feasible"
        )
        if epigenetic:
            secondary  = "protac"
            n_evidence += 1
            rationale.append(
                "Epigenetic/chromatin regulator — PROTAC degrader would "
                "remove all protein functions, often more effective than "
                "catalytic inhibition alone"
            )
    elif has_allo:
        primary    = "allosteric_small_molecule"
        n_evidence += 1
        corr = float(allo_sites[0].get("mean_correlation", 0))
        rationale.append(
            f"No druggable orthosteric pocket but allosteric site A1 "
            f"shows strong coupling (corr={corr:.2f}) — "
            "allosteric small molecule inhibitor recommended"
        )
    else:
        primary = "undruggable"
        rationale.append(
            "No surface exposure, druggable pocket, or allosteric site found. "
            "Consider: targeted protein degradation (molecular glue), "
            "PPI disruption, or nucleic acid-targeting strategies."
        )

    # Combination therapy note
    combo_note = ""
    if combo:
        combo_note = (
            f"PPI partners include known drug targets {combo} — "
            "combination therapy opportunity"
        )
        n_evidence += 1

    # EC-based rationale
    ec = report.get("ec_number", "")
    if ec:
        rationale.append(
            f"Enzyme (EC {ec}) — active site inhibition is the most "
            "direct mechanism; consider transition-state analogues"
        )
        n_evidence += 1

    confidence = (
        "HIGH"   if n_evidence >= 3 else
        "MEDIUM" if n_evidence >= 2 else
        "LOW"
    )

    return TherapyDecision(
        uniprot_id       = uid,
        gene_name        = gene,
        protein_name     = protein,
        organism         = organism,
        is_surface       = surface,
        primary_modality = primary,
        secondary_modality = secondary,
        confidence       = confidence,
        top_pocket_id    = top_pid,
        top_pocket_vol   = top_vol,
        top_pocket_drug  = top_drug,
        has_allosteric   = has_allo,
        is_epigenetic    = epigenetic,
        n_evidence       = n_evidence,
        rationale        = rationale,
        combination_note = combo_note,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ANTIBODY EPITOPE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _score_epitope(
    residue_numbers: list[int],
    residue_letters: list[str],
    physico:         dict,
) -> tuple[float, float, float]:
    """
    Score an epitope candidate.
    Returns (immunogenicity, accessibility, total_sasa).
    """
    res_map = {
        r["residue_number"]: r
        for r in physico.get("residues", [])
    }

    sasa_vals   = []
    immuno_aa   = 0

    for rn, aa in zip(residue_numbers, residue_letters):
        rec = res_map.get(rn, {})
        sasa_vals.append(float(rec.get("sasa", 30.0)))
        if aa in IMMUNOGENIC_AA:
            immuno_aa += 1

    total_sasa     = sum(sasa_vals)
    mean_sasa      = total_sasa / max(len(sasa_vals), 1)
    accessibility  = min(1.0, mean_sasa / 60.0)
    immunogenicity = min(1.0, (immuno_aa / max(len(residue_letters), 1)) * 1.5)

    return round(immunogenicity, 3), round(accessibility, 3), round(total_sasa, 1)


def find_epitopes(
    report:  dict,
    physico: dict,
) -> list[EpitopeCandidate]:
    """
    Find and rank antibody epitope candidates from the consensus report.

    Priority order:
      1. Active site surface residues (functional epitopes — most specific)
      2. PPI interface residues (blocking interfaces)
      3. Largest exposed hydrophobic patch
      4. Largest exposed charged patch
    """
    epitopes = []
    eid      = 1

    # ── Priority 1: Active site surface residues ──────────────────────────────
    active_sites = report.get("active_sites", [])
    surface_active = [
        s for s in active_sites
        if s.get("confidence") in ("HIGH", "MEDIUM")
    ]
    if surface_active:
        nums = [s["residue_number"] for s in surface_active[:15]]
        lets = [s.get("one_letter", "A") for s in surface_active[:15]]
        imm, acc, sasa = _score_epitope(nums, lets, physico)
        if sasa >= MIN_EPITOPE_SASA:
            epitopes.append(EpitopeCandidate(
                epitope_id          = f"E{eid}",
                residue_numbers     = nums,
                residue_letters     = lets,
                source              = "active_site",
                total_sasa          = sasa,
                immunogenicity_score = imm,
                accessibility       = acc,
                n_immunogenic_aa    = sum(1 for a in lets if a in IMMUNOGENIC_AA),
                notes               = "Active/functional site residues — "
                                      "highly specific but may affect normal homologs",
            ))
            eid += 1

    # ── Priority 2: PPI interface ─────────────────────────────────────────────
    for partner in report.get("ppi_partners", [])[:3]:
        if partner.get("combined_score", 0) < 400:
            continue
        nums = partner.get("interface_residues", [])
        lets = partner.get("interface_letters", [])
        if not nums:
            continue
        imm, acc, sasa = _score_epitope(nums, lets, physico)
        if sasa >= MIN_EPITOPE_SASA:
            pname = partner.get("partner_name", "?")
            epitopes.append(EpitopeCandidate(
                epitope_id          = f"E{eid}",
                residue_numbers     = nums,
                residue_letters     = lets,
                source              = "ppi_interface",
                total_sasa          = sasa,
                immunogenicity_score = imm,
                accessibility       = acc,
                n_immunogenic_aa    = sum(1 for a in lets if a in IMMUNOGENIC_AA),
                notes               = f"PPI interface with {pname} "
                                      f"(STRING score={partner.get('combined_score',0)}) — "
                                      "blocking this interface disrupts the interaction",
            ))
            eid += 1

    # ── Priority 3: Surface patches from Module 02 ────────────────────────────
    for patch_type in ("hydrophobic_patches", "positive_patches", "negative_patches"):
        patches = physico.get(patch_type, [])
        if not patches:
            continue
        top = sorted(patches, key=lambda p: p.get("total_sasa", 0), reverse=True)[:2]
        for patch in top:
            nums = patch.get("residue_numbers", [])
            if not nums:
                continue
            res_map = {
                r["residue_number"]: r.get("one_letter", "A")
                for r in physico.get("residues", [])
            }
            lets = [res_map.get(n, "A") for n in nums]
            imm, acc, sasa = _score_epitope(nums, lets, physico)
            if sasa >= MIN_EPITOPE_SASA:
                ptype = patch.get("patch_type", patch_type.replace("_patches",""))
                epitopes.append(EpitopeCandidate(
                    epitope_id          = f"E{eid}",
                    residue_numbers     = nums,
                    residue_letters     = lets,
                    source              = "surface_patch",
                    total_sasa          = sasa,
                    immunogenicity_score = imm,
                    accessibility       = acc,
                    n_immunogenic_aa    = sum(1 for a in lets if a in IMMUNOGENIC_AA),
                    notes               = f"{ptype} surface patch  "
                                          f"SASA={sasa:.0f}Å²",
                ))
                eid += 1

    # Sort by combined score (immunogenicity × accessibility)
    epitopes.sort(
        key=lambda e: e.immunogenicity_score * e.accessibility,
        reverse=True,
    )

    return epitopes[:6]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DE NOVO DESIGN (small molecule path)
# ══════════════════════════════════════════════════════════════════════════════

def _run_denovo_for_therapy(
    uid:          str,
    vina_path:    str,
    inter_dir:    Path,
    receptor_path: str = "",
) -> Optional[str]:
    """
    Trigger de novo molecular design and return the output path.

    receptor_path:
      - ""  (empty)  → auto-convert PDB to PDBQT using built-in converter
      - "/path/to/receptor.pdbqt" → use this pre-prepared file
      - "/path/to/receptor.pdb"   → convert this specific PDB file

    Returns None if design fails or Vina is not available.
    """
    from proteinfp.deps import has_rdkit, has_vina
    if not has_rdkit():
        print("  [SKIP] De novo design — RDKit not installed "
              "(pip install proteinfp[chem])")
        return None
    if not has_vina(vina_path):
        print(f"  [SKIP] De novo design — Vina not found at {vina_path}")
        return None

    # Resolve receptor path — prefer explicit, fallback to auto-convert from PDB
    resolved_receptor = ""
    if receptor_path and Path(receptor_path).exists():
        resolved_receptor = receptor_path
        print(f"  Using provided receptor: {Path(receptor_path).name}")
    else:
        # Check if PDBQT already exists from a previous run
        default_pdbqt = ROOT / "data" / "structures" / f"{uid}.pdbqt"
        if default_pdbqt.exists():
            resolved_receptor = str(default_pdbqt)
            print(f"  Using cached receptor: {default_pdbqt.name}")
        else:
            # Pass empty string — denovo_design._fast_pdb_to_pdbqt will
            # auto-convert from data/structures/{uid}.pdb
            resolved_receptor = ""
            print(f"  No receptor PDBQT found — will auto-convert from PDB")

    try:
        from pipeline.denovo_design import run_denovo_design
        from pipeline.denovo_design_context import (
            load_consensus_context, load_md_context,
        )

        def _load(fname):
            p = inter_dir / fname
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

        print(f"  Running de novo design for {uid}...")
        result = run_denovo_design(
            uniprot_id      = uid,
            pocket_data     = _load(f"{uid}_binding_pockets.json"),
            active_data     = _load(f"{uid}_active_sites.json"),
            allosteric_data = _load(f"{uid}_allosteric.json"),
            chem_env_data   = _load(f"{uid}_chemical_env.json"),
            vina_path       = vina_path,
            receptor_path   = resolved_receptor,   # explicit, never WindowsPath('.')
            consensus_data  = load_consensus_context(uid, inter_dir),
            md_data         = load_md_context(uid, inter_dir),
        )
        out = inter_dir / f"{uid}_denovo.json"
        return str(out) if out.exists() else None

    except Exception as e:
        print(f"  [FAIL] De novo design: {e}")
        import traceback; traceback.print_exc()
        return None


def _run_pharm_scoring(uid: str) -> Optional[str]:
    """Run SIM-06 pharmacological scoring on de novo candidates."""
    try:
        from sim.denovo_to_sim06 import score_denovo_candidates, save_denovo_scores
        scores = score_denovo_candidates(uid, top_n=10, verbose=True)
        if scores:
            out = save_denovo_scores(scores, uid)
            return str(out)
    except Exception as e:
        print(f"  [SKIP] Pharmacological scoring: {e}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_therapy(
    uniprot_id:    str,
    vina_path:     Optional[str] = None,
    receptor_path: str           = "",
    run_denovo:    bool          = True,
    verbose:       bool          = True,
) -> TherapyResult:
    """
    Run the full therapy workflow for a protein.

    Requires a completed consensus report (run `proteinfp --uniprot X` first).

    Args:
        uniprot_id:    UniProt accession
        vina_path:     Path to AutoDock Vina executable (enables de novo)
        receptor_path: Path to receptor PDBQT (auto-prepared from PDB if empty)
        run_denovo:    Whether to run de novo design if Vina is available
        verbose:       Print progress

    Returns:
        TherapyResult with decision, epitopes, and de novo paths
    """
    t0  = time.time()
    uid = uniprot_id.strip().upper()

    # Resolve paths
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
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # Load physicochemical data (optional — improves decisions)
    physico_path = inter_dir / f"{uid}_physicochemical.json"
    physico: dict = {}
    if physico_path.exists():
        physico = json.loads(physico_path.read_text(encoding="utf-8"))

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Therapy analysis: {report.get('gene_name','?')} ({uid})")
        print(f"  {report.get('protein_name','?')}")
        print(f"  Organism: {report.get('organism','?')}")
        print(f"{'─'*60}")

    # ── Step 1: Therapy decision ───────────────────────────────────────────────
    if verbose:
        print("\n  [1/3] Making therapy modality decision...")
    decision = make_therapy_decision(report, physico)

    if verbose:
        print(f"  → Primary modality : {decision.primary_modality.upper()}")
        if decision.secondary_modality:
            print(f"  → Secondary        : {decision.secondary_modality.upper()}")
        print(f"  → Confidence       : {decision.confidence}")
        for r in decision.rationale:
            print(f"     • {r}")

    # ── Step 2: Antibody epitopes (if surface) ────────────────────────────────
    epitopes: list[EpitopeCandidate] = []
    if decision.is_surface or decision.primary_modality == "antibody":
        if verbose:
            print("\n  [2/3] Finding antibody epitope candidates...")
        epitopes = find_epitopes(report, physico)
        if verbose:
            if epitopes:
                print(f"  → Found {len(epitopes)} epitope candidates")
                for ep in epitopes[:2]:
                    print(f"     {ep.epitope_id}: {ep.source}  "
                          f"SASA={ep.total_sasa:.0f}Å²  "
                          f"immunogenicity={ep.immunogenicity_score:.2f}")
            else:
                print("  → No high-quality epitopes found "
                      "(run Module 02 for better surface data)")
    else:
        if verbose:
            print("\n  [2/3] Antibody path skipped (intracellular protein)")

    # ── Step 3: De novo design (if small molecule) ────────────────────────────
    denovo_path     = None
    pharm_path      = None
    denovo_run      = False

    needs_denovo = (
        run_denovo and
        decision.primary_modality in ("small_molecule", "allosteric_small_molecule") or
        (decision.secondary_modality == "small_molecule" and run_denovo)
    )

    if needs_denovo:
        if verbose:
            print("\n  [3/3] Running de novo molecular design...")
        if vina_path:
            denovo_path = _run_denovo_for_therapy(
                uid, vina_path, inter_dir, receptor_path
            )
            if denovo_path:
                denovo_run = True
                if verbose:
                    print(f"  → De novo output: {denovo_path}")
                    print("  Scoring candidates through pharmacological framework...")
                pharm_path = _run_pharm_scoring(uid)
                if pharm_path and verbose:
                    print(f"  → Pharm scores  : {pharm_path}")
        else:
            if verbose:
                print(
                    "  [SKIP] De novo design — no Vina path provided.\n"
                    "  To enable: proteinfp --uniprot {uid} --therapy "
                    "--denovo --vina path/to/vina.exe"
                )
    else:
        if verbose:
            print("\n  [3/3] De novo design skipped "
                  f"(modality={decision.primary_modality})")

    # ── Save therapy report ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    result  = TherapyResult(
        uniprot_id        = uid,
        decision          = decision,
        epitopes          = epitopes,
        denovo_run        = denovo_run,
        denovo_path       = denovo_path,
        pharm_scores_path = pharm_path,
        elapsed_sec       = round(elapsed, 1),
    )

    # Write JSON
    out_json = report_dir / f"{uid}_therapy.json"
    out_json.write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )

    # Write text report
    out_txt = report_dir / f"{uid}_therapy.txt"
    out_txt.write_text(result.summary(), encoding="utf-8")

    if verbose:
        print(result.summary())
        print(f"  Saved → {out_json}")
        print(f"  Saved → {out_txt}")

    return result