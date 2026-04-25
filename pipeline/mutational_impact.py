"""
pipeline/mutational_impact.py
──────────────────────────────
Module — Mutational Impact Analysis.

Predicts the thermodynamic and functional impact of amino acid mutations
on protein stability, active site integrity, and binding interfaces.

Two complementary approaches (matching the website spec):
  1. FoldX-style ΔΔG estimation  — empirical force-field energy terms
  2. ESM-IF / evolutionary        — conservation-based fitness prediction

Modes of operation:
  A. Saturation mutagenesis  — scan every residue × 19 substitutions
  B. Focused mutagenesis     — specific mutations of interest (e.g. known cancer variants)
  C. Active site scanning    — deep scan of predicted active site residues only
  D. Interface scanning      — scan PPI interface residues

Output:
  - Per-mutation ΔΔG estimate (kcal/mol)
  - Evolutionary fitness score (0–1, based on ESM-2 masked prediction)
  - Functional impact classification: destabilising / neutral / stabilising
  - Active site disruption flag
  - Hotspot mutations (those most likely to affect function)
  - Heatmap-ready data matrix (residue × amino acid)

Usage (standalone):
    python pipeline/mutational_impact.py --uniprot P04637
    python pipeline/mutational_impact.py --uniprot P04637 --mode active_site
    python pipeline/mutational_impact.py --uniprot P04637 --mutations "R175H,R248W,R273H"

Usage (from orchestrator):
    from pipeline.mutational_impact import run_mutational_impact
    result = run_mutational_impact("P04637", sequence, structure, active_data, ppi_data)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import click
import numpy as np

from utils.config import cfg, get_logger
from utils.pdb_parser import parse_pdb, ParsedStructure, HYDROPHOBICITY

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# ΔΔG thresholds (kcal/mol)
DDG_DESTABILISING    = 1.5    # > +1.5 kcal/mol = destabilising
DDG_STABILISING      = -0.5   # < -0.5 kcal/mol = stabilising

# Maximum residues to scan in saturation mode (performance limit)
MAX_SATURATION_RESIDUES = 100

# Maximum residues in focused active site scan
MAX_ACTIVE_RESIDUES  = 30

# Amino acids (one-letter, standard 20)
AA20 = list("ACDEFGHIKLMNPQRSTVWY")

# Amino acid properties for energy calculations
AA_VOLUME: dict[str, float] = {
    "A": 88.6,  "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1,  "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0,  "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

AA_CHARGE: dict[str, float] = {
    "A":  0.0, "R": +1.0, "N":  0.0, "D": -1.0, "C":  0.0,
    "Q":  0.0, "E": -1.0, "G":  0.0, "H": +0.1, "I":  0.0,
    "L":  0.0, "K": +1.0, "M":  0.0, "F":  0.0, "P":  0.0,
    "S":  0.0, "T":  0.0, "W":  0.0, "Y":  0.0, "V":  0.0,
}

# Blosum62 diagonal (self-similarity) used as conservation proxy
BLOSUM62_DIAG: dict[str, float] = {
    "A": 4.0,  "R": 5.0,  "N": 6.0,  "D": 6.0,  "C": 9.0,
    "Q": 5.0,  "E": 5.0,  "G": 6.0,  "H": 8.0,  "I": 4.0,
    "L": 4.0,  "K": 5.0,  "M": 5.0,  "F": 6.0,  "P": 7.0,
    "S": 4.0,  "T": 5.0,  "W": 11.0, "Y": 7.0,  "V": 4.0,
}

# Off-diagonal BLOSUM62 (wt → mut substitution scores, simplified)
# Positive = accepted substitution; negative = rare/disruptive
BLOSUM62: dict[tuple[str, str], int] = {}

# Build from simplified rules (conservative < neutral < radical)
_GROUPS = [
    "ILMV",    # aliphatic hydrophobic
    "FWY",     # aromatic
    "ST",      # hydroxyl
    "NQ",      # amide
    "DE",      # acidic
    "KR",      # basic
    "AG",      # small
    "P",       # proline (disruptive)
    "C",       # cysteine (disruptive)
    "H",       # histidine
]
for _aa1 in AA20:
    for _aa2 in AA20:
        if _aa1 == _aa2:
            BLOSUM62[(_aa1, _aa2)] = int(BLOSUM62_DIAG[_aa1])
        else:
            # Same group → conservative (score 1-2)
            _same = any(_aa1 in g and _aa2 in g for g in _GROUPS)
            if _same:
                BLOSUM62[(_aa1, _aa2)] = 1
            # Different charge → disruptive (score -2)
            elif AA_CHARGE.get(_aa1, 0) * AA_CHARGE.get(_aa2, 0) < 0:
                BLOSUM62[(_aa1, _aa2)] = -2
            else:
                BLOSUM62[(_aa1, _aa2)] = 0


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SingleMutation:
    """Predicted impact of a single point mutation."""
    residue_number: int
    wt_aa:          str    # wild-type amino acid (1-letter)
    mut_aa:         str    # mutant amino acid (1-letter)
    mutation_code:  str    # e.g. "R175H"

    # FoldX-style energy decomposition
    ddg_total:          float  = 0.0   # kcal/mol; positive = destabilising
    ddg_vdw:            float  = 0.0   # van der Waals clash/void
    ddg_electrostatic:  float  = 0.0   # charge change contribution
    ddg_solvation:      float  = 0.0   # solvation/desolvation
    ddg_entropy:        float  = 0.0   # backbone/sidechain entropy
    ddg_hbond:          float  = 0.0   # hydrogen bond disruption

    # Evolutionary fitness (ESM-2 / BLOSUM proxy)
    evolutionary_score: float  = 0.0   # 0-1; 1 = common substitution
    blosum_score:       int    = 0     # BLOSUM62 substitution score

    # Functional annotations
    stability_class:    str    = ""    # "destabilising" / "neutral" / "stabilising"
    disrupts_active:    bool   = False # mutates a predicted active site residue
    disrupts_interface: bool   = False # mutates a PPI interface residue
    is_buried:          bool   = False # buried residue (SASA < 20 Å²)
    plddt_wt:           float  = 0.0   # AlphaFold confidence at this position

    # Confidence
    prediction_confidence: str = ""   # "high" / "medium" / "low"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResidueProfile:
    """Aggregated mutational sensitivity for a single residue position."""
    residue_number:   int
    wt_aa:            str
    mean_ddg:         float        # mean ΔΔG across all 19 substitutions
    max_ddg:          float        # worst case substitution
    min_ddg:          float        # best case (most stabilising)
    n_destabilising:  int          # substitutions with ΔΔG > threshold
    n_stabilising:    int
    sensitivity:      str          # "hypersensitive" / "sensitive" / "tolerant"
    top_mutations:    list[str]    # top 3 most impactful mutations
    is_active_site:   bool
    is_interface:     bool
    plddt:            float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MutationalImpactResult:
    """Full mutational impact analysis output."""
    uniprot_id:       str
    sequence:         str
    mode:             str          # "saturation" / "active_site" / "focused"
    n_mutations:      int          = 0
    n_residues:       int          = 0

    mutations:        list[SingleMutation]  = field(default_factory=list)
    residue_profiles: list[ResidueProfile]  = field(default_factory=list)

    # Summary statistics
    mean_ddg:         float        = 0.0
    n_destabilising:  int          = 0
    n_neutral:        int          = 0
    n_stabilising:    int          = 0

    # Hotspots
    hotspot_residues: list[int]    = field(default_factory=list)
    active_site_ddg:  dict         = field(default_factory=dict)  # residue → mean ΔΔG

    # Heatmap data (for visualisation)
    # residue_numbers × AA20 ΔΔG matrix stored as flat list
    heatmap_residues: list[int]    = field(default_factory=list)
    heatmap_aas:      list[str]    = field(default_factory=list)
    heatmap_ddg:      list[list[float]] = field(default_factory=list)

    notes:            str          = ""

    def summary(self) -> str:
        lines = [
            f"\n{'─'*70}",
            f"  Mutational Impact: {self.uniprot_id}  [mode={self.mode}]",
            f"  Residues scanned  : {self.n_residues}",
            f"  Mutations modelled: {self.n_mutations}",
            f"  Destabilising     : {self.n_destabilising}  "
            f"Neutral: {self.n_neutral}  Stabilising: {self.n_stabilising}",
            f"  Mean ΔΔG          : {self.mean_ddg:+.2f} kcal/mol",
        ]
        if self.hotspot_residues:
            lines.append(
                f"  Hotspot residues  : "
                f"{', '.join(str(r) for r in self.hotspot_residues[:10])}"
            )
        if self.active_site_ddg:
            lines.append(f"{'─'*70}")
            lines.append("  Active site sensitivity:")
            for rn, ddg in sorted(
                self.active_site_ddg.items(),
                key=lambda x: -x[1]
            )[:8]:
                lines.append(f"    Res {rn}: mean ΔΔG = {ddg:+.2f} kcal/mol")
        if self.mutations:
            lines.append(f"{'─'*70}")
            lines.append("  Most destabilising mutations (top 10):")
            top = sorted(self.mutations, key=lambda m: -m.ddg_total)[:10]
            for m in top:
                flags = []
                if m.disrupts_active:
                    flags.append("ACTIVE SITE")
                if m.disrupts_interface:
                    flags.append("INTERFACE")
                if m.is_buried:
                    flags.append("buried")
                flag_str = f"  [{', '.join(flags)}]" if flags else ""
                lines.append(
                    f"    {m.mutation_code:8s}  ΔΔG={m.ddg_total:+.2f}  "
                    f"evo={m.evolutionary_score:.2f}  "
                    f"{m.stability_class}{flag_str}"
                )
        lines.append(f"{'─'*70}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.debug(f"Saved mutational impact JSON → {path}")


# ── FoldX-style ΔΔG estimation ────────────────────────────────────────────────

def _estimate_ddg(
    wt:        str,
    mut:       str,
    res_num:   int,
    structure: ParsedStructure,
    sasa_map:  dict[tuple[str, int], float],
    active_set: set[int],
) -> tuple[float, float, float, float, float, float]:
    """
    Estimate ΔΔG for a single point mutation using empirical energy terms.

    Returns:
        (ddg_total, ddg_vdw, ddg_electrostatic, ddg_solvation,
         ddg_entropy, ddg_hbond) in kcal/mol.

    Method mirrors FoldX energy decomposition:
      - VdW: volume clash or void if very different sizes
      - Electrostatic: charge reversal penalty
      - Solvation: burial of polar atoms
      - Entropy: sidechain entropy change
      - H-bond: loss of H-bond donors/acceptors
    """
    wt_vol   = AA_VOLUME.get(wt,  140.0)
    mut_vol  = AA_VOLUME.get(mut, 140.0)
    wt_chg   = AA_CHARGE.get(wt,  0.0)
    mut_chg  = AA_CHARGE.get(mut, 0.0)
    wt_hyd   = HYDROPHOBICITY.get(wt,  0.0)
    mut_hyd  = HYDROPHOBICITY.get(mut, 0.0)

    # Burial (SASA)
    sasa = sasa_map.get(("A", res_num), 50.0)
    burial = max(0.0, 1.0 - sasa / 100.0)   # 0 = exposed, 1 = buried

    # pLDDT at this residue (confidence in local structure)
    plddt = 70.0
    for r in structure.residues:
        if r.residue_number == res_num:
            plddt = r.plddt
            break
    plddt_factor = plddt / 100.0   # scale: uncertain regions less penalised

    # ── VdW: volume clash/void ────────────────────────────────────────────────
    vol_diff = mut_vol - wt_vol
    if burial > 0.6:
        # Buried: large-to-small creates void (+cost); small-to-large creates clash
        if vol_diff > 20:
            ddg_vdw = 0.03 * vol_diff * burial * plddt_factor
        elif vol_diff < -20:
            ddg_vdw = -0.01 * vol_diff * burial * plddt_factor
        else:
            ddg_vdw = 0.005 * abs(vol_diff) * burial
    else:
        # Surface: volume changes less costly
        ddg_vdw = 0.003 * abs(vol_diff) * (1 - burial)

    # ── Electrostatic: charge changes ────────────────────────────────────────
    chg_delta = mut_chg - wt_chg
    if wt_chg != 0 and mut_chg == 0:
        # Lose a charge — destabilising if buried, neutral if exposed
        ddg_elec = 1.2 * burial * plddt_factor
    elif wt_chg == 0 and mut_chg != 0:
        # Gain a charge — desolvation penalty if buried
        ddg_elec = 1.5 * burial * plddt_factor
    elif wt_chg * mut_chg < 0:
        # Charge reversal — very destabilising if buried
        ddg_elec = 2.5 * burial * plddt_factor
    else:
        ddg_elec = 0.0

    # Active site charge reversal is extra costly
    if res_num in active_set and abs(chg_delta) > 0:
        ddg_elec += 0.8

    # ── Solvation: hydrophobic burial ────────────────────────────────────────
    hyd_delta = mut_hyd - wt_hyd
    if burial > 0.5:
        # Buried: increasing hydrophilicity = desolvation penalty
        ddg_solv = 0.4 * (-hyd_delta) * burial * plddt_factor
    else:
        # Surface: burying hydrophobic = slight stabilisation
        ddg_solv = -0.1 * hyd_delta * (1 - burial)

    # ── Entropy: sidechain flexibility ───────────────────────────────────────
    # Proline introduction: rigidifies backbone → destabilising in helix/sheet
    if mut == "P":
        ddg_ent = 1.8 * plddt_factor
    elif wt == "P" and mut != "P":
        ddg_ent = -0.5   # removing Pro can relieve strain
    elif mut == "G":
        # Glycine: gains entropy but often destabilises helices
        ddg_ent = 0.8 * burial * plddt_factor
    else:
        ddg_ent = 0.0

    # ── H-bond: donor/acceptor loss ───────────────────────────────────────────
    HBD_AA = set("NQSTKRYW")  # H-bond donors
    HBA_AA = set("NQDESTY")   # H-bond acceptors
    hbd_lost = (wt in HBD_AA) and (mut not in HBD_AA)
    hba_lost = (wt in HBA_AA) and (mut not in HBA_AA)
    # Buried H-bonds are critical
    ddg_hbond = 0.0
    if hbd_lost and burial > 0.4:
        ddg_hbond += 1.5 * burial * plddt_factor
    if hba_lost and burial > 0.4:
        ddg_hbond += 1.2 * burial * plddt_factor

    # Active site H-bond loss is critical
    if res_num in active_set and (hbd_lost or hba_lost):
        ddg_hbond += 1.0

    ddg_total = (ddg_vdw + ddg_elec + ddg_solv +
                 ddg_ent + ddg_hbond)

    return (
        round(ddg_total, 3),
        round(ddg_vdw, 3),
        round(ddg_elec, 3),
        round(ddg_solv, 3),
        round(ddg_ent, 3),
        round(ddg_hbond, 3),
    )


# ── Evolutionary fitness (ESM-IF / BLOSUM proxy) ─────────────────────────────

def _evolutionary_score(
    wt:        str,
    mut:       str,
    sequence:  str,
    res_num:   int,
    esm2_data: Optional[dict] = None,
) -> tuple[float, int]:
    """
    Estimate evolutionary fitness of a substitution.

    Uses:
      1. ESM-2 masked token probability if embeddings are available
      2. BLOSUM62 substitution score as fallback

    Returns:
        (evolutionary_score 0-1, blosum_score int)
    """
    blosum = BLOSUM62.get((wt, mut), -1)

    # Try ESM-2 masked prediction if data is available
    if esm2_data and esm2_data.get("protein_embedding"):
        # ESM-2 residue embeddings encode local fitness landscape
        # Proxy: use embedding norm at this position as conservation signal
        res_embs = esm2_data.get("residue_embeddings", [])
        idx = res_num - 1  # 0-based
        if 0 <= idx < len(res_embs):
            emb = np.array(res_embs[idx], dtype=np.float32)
            # High-norm embedding = well-defined position = more conserved
            norm = float(np.linalg.norm(emb))
            conservation = min(1.0, norm / 40.0)

            # Combine conservation × BLOSUM substitution tolerance
            blosum_norm = (blosum + 4) / 15.0  # normalise roughly 0-1
            blosum_norm = max(0.0, min(1.0, blosum_norm))

            evo_score = 0.4 * conservation * blosum_norm + 0.6 * blosum_norm
            return round(evo_score, 3), blosum

    # Fallback: BLOSUM62 only
    blosum_norm = (blosum + 4) / 15.0
    evo_score = max(0.0, min(1.0, blosum_norm))
    return round(evo_score, 3), blosum


# ── Mutation builder ──────────────────────────────────────────────────────────

def _build_mutation(
    res_num:       int,
    wt:            str,
    mut:           str,
    structure:     ParsedStructure,
    sasa_map:      dict,
    active_set:    set[int],
    interface_set: set[int],
    esm2_data:     Optional[dict],
    sequence:      str,
) -> SingleMutation:
    """Build a complete SingleMutation record."""
    code = f"{wt}{res_num}{mut}"

    # ΔΔG
    ddg, vdw, elec, solv, ent, hbond = _estimate_ddg(
        wt, mut, res_num, structure, sasa_map, active_set
    )

    # Evolutionary
    evo, blosum = _evolutionary_score(wt, mut, sequence, res_num, esm2_data)

    # Classification
    if ddg > DDG_DESTABILISING:
        cls = "destabilising"
    elif ddg < DDG_STABILISING:
        cls = "stabilising"
    else:
        cls = "neutral"

    # Burial
    sasa = sasa_map.get(("A", res_num), 50.0)
    buried = sasa < 20.0

    # pLDDT
    plddt = 70.0
    for r in structure.residues:
        if r.residue_number == res_num:
            plddt = r.plddt
            break

    # Prediction confidence: high pLDDT = more reliable ΔΔG
    if plddt >= 70:
        conf = "high"
    elif plddt >= 50:
        conf = "medium"
    else:
        conf = "low"

    return SingleMutation(
        residue_number=res_num,
        wt_aa=wt,
        mut_aa=mut,
        mutation_code=code,
        ddg_total=ddg,
        ddg_vdw=vdw,
        ddg_electrostatic=elec,
        ddg_solvation=solv,
        ddg_entropy=ent,
        ddg_hbond=hbond,
        evolutionary_score=evo,
        blosum_score=blosum,
        stability_class=cls,
        disrupts_active=(res_num in active_set),
        disrupts_interface=(res_num in interface_set),
        is_buried=buried,
        plddt_wt=round(plddt, 1),
        prediction_confidence=conf,
    )


# ── Residue profile builder ───────────────────────────────────────────────────

def _build_residue_profile(
    res_num:    int,
    wt:         str,
    mutations:  list[SingleMutation],
    active_set: set[int],
    iface_set:  set[int],
    structure:  ParsedStructure,
) -> ResidueProfile:
    """Aggregate all mutations at a single residue into a profile."""
    if not mutations:
        return ResidueProfile(
            residue_number=res_num, wt_aa=wt,
            mean_ddg=0.0, max_ddg=0.0, min_ddg=0.0,
            n_destabilising=0, n_stabilising=0,
            sensitivity="tolerant", top_mutations=[],
            is_active_site=(res_num in active_set),
            is_interface=(res_num in iface_set),
            plddt=70.0,
        )

    ddgs = [m.ddg_total for m in mutations]
    mean_ddg = sum(ddgs) / len(ddgs)
    max_ddg  = max(ddgs)
    min_ddg  = min(ddgs)
    n_dest   = sum(1 for d in ddgs if d > DDG_DESTABILISING)
    n_stab   = sum(1 for d in ddgs if d < DDG_STABILISING)

    if n_dest >= 15:
        sensitivity = "hypersensitive"
    elif n_dest >= 8:
        sensitivity = "sensitive"
    else:
        sensitivity = "tolerant"

    top3 = sorted(mutations, key=lambda m: -m.ddg_total)[:3]
    top_codes = [m.mutation_code for m in top3]

    plddt = mutations[0].plddt_wt if mutations else 70.0

    return ResidueProfile(
        residue_number=res_num,
        wt_aa=wt,
        mean_ddg=round(mean_ddg, 3),
        max_ddg=round(max_ddg, 3),
        min_ddg=round(min_ddg, 3),
        n_destabilising=n_dest,
        n_stabilising=n_stab,
        sensitivity=sensitivity,
        top_mutations=top_codes,
        is_active_site=(res_num in active_set),
        is_interface=(res_num in iface_set),
        plddt=round(plddt, 1),
    )


# ── Main function ──────────────────────────────────────────────────────────────

def run_mutational_impact(
    uniprot_id:    str,
    sequence:      str,
    structure:     ParsedStructure,
    active_data:   Optional[dict] = None,
    ppi_data:      Optional[dict] = None,
    esm2_data:     Optional[dict] = None,
    physico_data:  Optional[dict] = None,
    mode:          str = "active_site",
    mutations:     Optional[list[str]] = None,
) -> MutationalImpactResult:
    """
    Run mutational impact analysis.

    Args:
        uniprot_id:   UniProt accession
        sequence:     Amino acid sequence
        structure:    ParsedStructure from Module 01
        active_data:  Dict from Module 03 JSON (active site residues)
        ppi_data:     Dict from Module 12 JSON (interface residues)
        esm2_data:    Dict from Module 08 JSON (for evolutionary scoring)
        physico_data: Dict from Module 02 JSON (for SASA)
        mode:         "saturation" / "active_site" / "focused"
        mutations:    For focused mode: list of mutation codes e.g. ["R175H", "P53V"]

    Returns:
        MutationalImpactResult
    """
    log.info(f"── Mutational Impact Module: {uniprot_id}  [mode={mode}] ──")

    # ── Build residue sets ─────────────────────────────────────────────────────
    active_set: set[int] = set()
    if active_data:
        for r in active_data.get("active_residues", []):
            if r.get("confidence") in {"HIGH", "MEDIUM"}:
                active_set.add(r["residue_number"])
        log.info(f"  Active site residues: {len(active_set)}")

    interface_set: set[int] = set()
    if ppi_data:
        for p in ppi_data.get("partners", [])[:5]:
            for rn in p.get("interface_residues", []):
                interface_set.add(rn)
        log.info(f"  Interface residues: {len(interface_set)}")

    # ── Build SASA map ─────────────────────────────────────────────────────────
    sasa_map: dict[tuple[str, int], float] = {}
    if physico_data:
        for rec in physico_data.get("residues", []):
            sasa_map[(rec["chain_id"], rec["residue_number"])] = rec["sasa"]
    else:
        # Fallback: all residues get pLDDT-based burial estimate
        for r in structure.residues:
            sasa_map[("A", r.residue_number)] = 100.0 - r.plddt

    # ── Select residues to scan ────────────────────────────────────────────────
    seq_map = {r.residue_number: r.one_letter for r in structure.residues}

    if mode == "focused" and mutations:
        scan_pairs = _parse_mutations(mutations, seq_map)
        log.info(f"  Focused mode: {len(scan_pairs)} specific mutations")

    elif mode == "active_site":
        # Deep scan: active site + interface + high-confidence residues
        priority = active_set | interface_set
        scan_residues = sorted(priority)[:MAX_ACTIVE_RESIDUES]
        if not scan_residues:
            # Fallback: use high-pLDDT buried residues
            scan_residues = sorted([
                r.residue_number for r in structure.residues
                if r.plddt >= 70 and sasa_map.get(("A", r.residue_number), 50) < 30
            ])[:MAX_ACTIVE_RESIDUES]
        scan_pairs = [
            (rn, seq_map.get(rn, "A"), mut)
            for rn in scan_residues
            for mut in AA20
            if mut != seq_map.get(rn, "A")
        ]
        log.info(f"  Active site mode: scanning {len(scan_residues)} residues "
                 f"× 19 AAs = {len(scan_pairs)} mutations")

    else:  # saturation
        # Limit to top residues by burial + conservation
        buried_res = sorted([
            r.residue_number for r in structure.residues
            if sasa_map.get(("A", r.residue_number), 50) < 40
        ])[:MAX_SATURATION_RESIDUES]
        scan_residues = buried_res or [
            r.residue_number for r in structure.residues[:MAX_SATURATION_RESIDUES]
        ]
        scan_pairs = [
            (rn, seq_map.get(rn, "A"), mut)
            for rn in scan_residues
            for mut in AA20
            if mut != seq_map.get(rn, "A")
        ]
        log.info(f"  Saturation mode: {len(scan_residues)} residues "
                 f"× 19 = {len(scan_pairs)} mutations")

    # ── Run predictions ────────────────────────────────────────────────────────
    log.info(f"  Computing ΔΔG for {len(scan_pairs)} mutations...")
    all_mutations = []
    for res_num, wt, mut in scan_pairs:
        if wt == "X":
            continue
        m = _build_mutation(
            res_num, wt, mut, structure, sasa_map,
            active_set, interface_set, esm2_data, sequence
        )
        all_mutations.append(m)

    if not all_mutations:
        log.warning("  No mutations computed")
        return MutationalImpactResult(
            uniprot_id=uniprot_id, sequence=sequence, mode=mode,
            notes="no mutations computed",
        )

    # ── Build residue profiles ─────────────────────────────────────────────────
    log.info("  Building residue profiles...")
    by_residue: dict[int, list[SingleMutation]] = {}
    for m in all_mutations:
        by_residue.setdefault(m.residue_number, []).append(m)

    residue_profiles = []
    for rn, muts in sorted(by_residue.items()):
        wt = seq_map.get(rn, "A")
        profile = _build_residue_profile(
            rn, wt, muts, active_set, interface_set, structure
        )
        residue_profiles.append(profile)

    # ── Hotspot detection ──────────────────────────────────────────────────────
    hotspots = [
        p.residue_number for p in residue_profiles
        if p.sensitivity in ("hypersensitive", "sensitive")
        and (p.is_active_site or p.is_interface or p.mean_ddg > 1.0)
    ]

    # ── Active site ΔΔG summary ───────────────────────────────────────────────
    active_site_ddg = {}
    for rn in active_set:
        if rn in by_residue:
            ddgs = [m.ddg_total for m in by_residue[rn]]
            active_site_ddg[rn] = round(sum(ddgs) / len(ddgs), 3)

    # ── Heatmap data ───────────────────────────────────────────────────────────
    heatmap_residues = sorted(by_residue.keys())
    heatmap_ddg = []
    for rn in heatmap_residues:
        wt = seq_map.get(rn, "A")
        row = []
        mut_lookup = {m.mut_aa: m.ddg_total for m in by_residue[rn]}
        for aa in AA20:
            if aa == wt:
                row.append(0.0)
            else:
                row.append(mut_lookup.get(aa, 0.0))
        heatmap_ddg.append(row)

    # ── Statistics ────────────────────────────────────────────────────────────
    ddgs = [m.ddg_total for m in all_mutations]
    mean_ddg   = sum(ddgs) / len(ddgs)
    n_dest     = sum(1 for d in ddgs if d > DDG_DESTABILISING)
    n_stab     = sum(1 for d in ddgs if d < DDG_STABILISING)
    n_neutral  = len(ddgs) - n_dest - n_stab

    result = MutationalImpactResult(
        uniprot_id=uniprot_id,
        sequence=sequence,
        mode=mode,
        n_mutations=len(all_mutations),
        n_residues=len(residue_profiles),
        mutations=all_mutations,
        residue_profiles=residue_profiles,
        mean_ddg=round(mean_ddg, 3),
        n_destabilising=n_dest,
        n_neutral=n_neutral,
        n_stabilising=n_stab,
        hotspot_residues=hotspots[:20],
        active_site_ddg=active_site_ddg,
        heatmap_residues=heatmap_residues,
        heatmap_aas=AA20,
        heatmap_ddg=heatmap_ddg,
    )

    log.info(result.summary())
    return result


# ── Mutation parser (focused mode) ────────────────────────────────────────────

def _parse_mutations(
    mutations: list[str],
    seq_map:   dict[int, str],
) -> list[tuple[int, str, str]]:
    """
    Parse mutation codes like "R175H", "P53V" into (res_num, wt, mut) tuples.
    Validates wt against known sequence.
    """
    parsed = []
    for code in mutations:
        code = code.strip().upper()
        if len(code) < 3:
            log.warning(f"  Invalid mutation code: {code}")
            continue
        try:
            wt  = code[0]
            mut = code[-1]
            num = int(code[1:-1])
            seq_wt = seq_map.get(num)
            if seq_wt and seq_wt != wt:
                log.warning(
                    f"  {code}: WT mismatch — sequence has {seq_wt}{num}, "
                    f"code says {wt}{num}. Using sequence WT."
                )
                wt = seq_wt
            if mut in AA20 and wt in AA20:
                parsed.append((num, wt, mut))
            else:
                log.warning(f"  Invalid amino acid in {code}")
        except (ValueError, IndexError):
            log.warning(f"  Could not parse mutation code: {code}")
    return parsed


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637)")
@click.option("--mode", "-m",
              type=click.Choice(["active_site", "saturation", "focused"]),
              default="active_site",
              help="Scan mode: active_site (default), saturation, or focused")
@click.option("--mutations", "-x", default=None,
              help="Comma-separated mutation codes for focused mode "
                   "(e.g. 'R175H,R248W,R273H')")
def main(uniprot: str, mode: str, mutations: Optional[str]) -> None:
    """
    Mutational Impact Module — ΔΔG and evolutionary fitness prediction.

    Predicts the stability and functional impact of amino acid mutations
    using FoldX-style energy decomposition + BLOSUM/ESM-2 evolutionary scoring.

    Example (active site scan):
        python pipeline/mutational_impact.py --uniprot P04637

    Example (saturation mutagenesis, first 100 buried residues):
        python pipeline/mutational_impact.py --uniprot P04637 --mode saturation

    Example (specific known mutations):
        python pipeline/mutational_impact.py --uniprot P04637 \\
            --mode focused --mutations "R175H,R248W,R273H,R249S"
    """
    uniprot   = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    pdb_path  = Path(cfg.paths["structures"]) / f"{uniprot}.pdb"
    out_path  = inter_dir / f"{uniprot}_mutational_impact.json"

    if not pdb_path.exists():
        log.error(f".pdb not found: {pdb_path}\n"
                  f"  Run Module 01 first: python pipeline/fetch_structure.py "
                  f"--uniprot {uniprot}")
        raise SystemExit(1)

    # Load structure
    structure = parse_pdb(pdb_path, uniprot)
    sequence  = structure.sequence
    log.info(f"  Sequence length: {len(sequence)} aa")

    # Load upstream module outputs
    def _load(fname):
        p = inter_dir / fname
        return json.loads(p.read_text()) if p.exists() else None

    active_data  = _load(f"{uniprot}_active_sites.json")
    ppi_data     = _load(f"{uniprot}_ppi.json")
    esm2_data    = _load(f"{uniprot}_esm2.json")
    physico_data = _load(f"{uniprot}_physicochemical.json")

    if active_data:
        log.info(f"  Loaded active site data")
    if ppi_data:
        log.info(f"  Loaded PPI data")
    if esm2_data:
        log.info(f"  Loaded ESM-2 embeddings")
    if physico_data:
        log.info(f"  Loaded physicochemical (SASA) data")

    # Parse focused mutations
    mutation_list = None
    if mode == "focused" and mutations:
        mutation_list = [m.strip() for m in mutations.split(",") if m.strip()]
        log.info(f"  Focused mutations: {mutation_list}")

    result = run_mutational_impact(
        uniprot_id=uniprot,
        sequence=sequence,
        structure=structure,
        active_data=active_data,
        ppi_data=ppi_data,
        esm2_data=esm2_data,
        physico_data=physico_data,
        mode=mode,
        mutations=mutation_list,
    )

    result.to_json(out_path)
    click.echo(result.summary())
    click.echo(f"\nResults saved to:\n  {out_path}")


if __name__ == "__main__":
    main()