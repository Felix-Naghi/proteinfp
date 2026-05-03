"""
sim/denovo_to_sim06.py
───────────────────────
Gap 4 fix — feeds de novo design candidates into SIM-06 pharmacological
scoring so the best evolved molecules receive a full pharmacological score.

WHAT THE GAP WAS
────────────────
SIM-06 loads binding data from:
    data/sim/binding/{drug_name}_binding.json   ← SIM-04 output

De novo candidates live in:
    data/intermediate/{uniprot}_denovo.json     ← de novo designer output

These two paths never crossed.  The molecules that scored best in the
evolutionary design loop never received:
  - A selectivity score (ΔΔS tumor vs normal)
  - A network disruption score (GRN cascade depth)
  - A resistance risk score
  - A safety score (normal tissue)
  - A composite pharmacological grade (A/B/C/D/F)

HOW THIS MODULE FIXES IT
────────────────────────
1. load_denovo_candidates()
   Reads the de novo JSON and converts each top candidate into a
   synthetic "binding score" dict that matches the format SIM-04
   produces.  The Vina docking score is converted to an estimated
   Kd and pKi using the standard ΔG = RT ln(Kd) relationship.

2. score_denovo_candidates()
   Calls SIM-06's compute_pharmacological_scores() with the synthetic
   binding data injected, producing a full PharmacologicalScore for
   each de novo molecule.

3. save_denovo_scores()
   Writes the results to:
     data/sim/scores/{uniprot}_denovo_pharm.json

USAGE
─────
    # After running denovo_design.py:
    from sim.denovo_to_sim06 import score_denovo_candidates
    scores = score_denovo_candidates("Q9HAW4")

    # Or standalone:
    python sim/denovo_to_sim06.py --uniprot Q9HAW4

INTEGRATION WITH SIM-06
────────────────────────
No changes to step06_pharmacological_scoring.py are required.
This module calls SIM-06's scoring functions directly with synthetic
binding data, then saves a separate output file for de novo results.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent.parent
INTER     = ROOT / "data" / "intermediate"
SIM_DIR   = ROOT / "data" / "sim"
GRN_DIR   = ROOT / "data" / "grn"
OUT_DIR   = SIM_DIR / "scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Physical constants ─────────────────────────────────────────────────────────

R = 8.314       # J/mol/K
T = 310.15      # K (37°C)
RT_kJ = R * T / 1000   # kJ/mol

# ── Conversion factors ─────────────────────────────────────────────────────────
# Vina score (kcal/mol) → ΔG (kJ/mol) → Kd (μM)
# ΔG = RT ln(Kd)  →  Kd = exp(ΔG / RT)
KCAL_TO_KJ = 4.184


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DenovoPharmScore:
    """
    Full pharmacological score for a single de novo candidate molecule.
    Extends the PharmacologicalScore from SIM-06 with molecule-specific fields.
    """
    uniprot_id:      str
    smiles:          str
    vina_score:      float      # kcal/mol (negative = better)
    le:              float      # ligand efficiency
    qed:             float      # drug-likeness
    mw:              float
    logp:            float
    tpsa:            float
    hbd:             int
    hba:             int
    generation:      int        # which evolution generation produced it

    # Converted binding properties
    dG_kJ:           float      # ΔG in kJ/mol
    Kd_uM:           float      # estimated Kd in μM
    pKi:             float      # -log10(Kd_M)

    # SIM-06 pharmacological scores (0-100 each)
    efficacy:        float = 0.0
    selectivity:     float = 0.0
    network:         float = 0.0
    resistance:      float = 0.0
    safety:          float = 0.0
    druggability:    float = 0.0
    composite:       float = 0.0
    grade:           str   = "?"

    # Supporting context
    target_gene:     str   = ""
    log2fc:          float = 0.0
    delta_delta_S:   float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"  {self.smiles[:45]:<45}  "
            f"Vina={self.vina_score:>6.2f}  "
            f"LE={self.le:.3f}  "
            f"QED={self.qed:.2f}  "
            f"Kd={self.Kd_uM:.1f}μM  "
            f"Grade={self.grade}  "
            f"Composite={self.composite:.1f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSION: Vina score → SIM-04 binding format
# ══════════════════════════════════════════════════════════════════════════════

def vina_to_binding_entry(
    candidate:    dict,
    uniprot_id:   str,
    gene_name:    str,
    pocket_vol:   float = 500.0,
) -> dict:
    """
    Convert a de novo DenovoCandidate dict into a SIM-04 binding score entry.

    SIM-04 binding score format (from step04_binding_probability.py):
        target_gene:         str
        target_uniprot:      str
        Kd_corrected_uM:     float
        pKi:                 float
        p_binding:           float   (0-1)
        pocket_volume_A3:    float
        fill_ratio:          float   (0-1)
        dG_shape:            float   (negative, kJ/mol)
        dG_corrected_kJ:     float

    Vina → pKi calibration (linear model, from benchmark literature):
        pKi = -1.0 * vina_score - 1.0
        Calibration points: vina=-8 → pKi=7 (Kd=100 nM),
                            vina=-5 → pKi=4 (Kd=100 μM)
        Source: Trott & Olson 2010; Alhossary et al 2015 benchmarks
    """
    vina_kcal = float(candidate.get("score", -5.0))

    # Calibrated linear Vina → pKi mapping
    # pKi = -vina - 1  (vina is negative; more negative = stronger binder)
    pKi   = -1.0 * vina_kcal - 1.0
    pKi   = max(1.0, min(12.0, pKi))     # clamp to physiological range
    Kd_M  = 10.0 ** (-pKi)
    Kd_uM = Kd_M * 1e6

    # ΔG from Kd: ΔG = RT ln(Kd)
    dG_kJ        = R * T * math.log(Kd_M) / 1000   # kJ/mol (negative)
    dG_corrected = dG_kJ

    # Binding probability: logistic function of pKi
    # pKi 9 → P≈0.99, pKi 6 → P≈0.5, pKi 4 → P≈0.05
    p_binding = 1.0 / (1.0 + math.exp(-(pKi - 6.0) * 1.5))

    # Fill ratio: LE-based estimate (higher LE → better fill)
    le = float(candidate.get("le", 0.3))
    fill_ratio = min(0.95, max(0.1, le * 2.5))

    return {
        "target_gene":      gene_name,
        "target_uniprot":   uniprot_id,
        "Kd_corrected_uM":  round(Kd_uM, 4),
        "pKi":              round(pKi, 3),
        "p_binding":        round(p_binding, 6),
        "pocket_volume_A3": pocket_vol,
        "fill_ratio":       round(fill_ratio, 3),
        "dG_shape":         round(dG_kJ, 2),
        "dG_corrected_kJ":  round(dG_corrected, 2),
        # Molecule identity fields (not in SIM-04 format but needed here)
        "_smiles":          candidate.get("smiles", ""),
        "_vina":            vina_kcal,
        "_le":              le,
        "_qed":             float(candidate.get("qed", 0.0)),
        "_mw":              float(candidate.get("mw", 0.0)),
        "_logp":            float(candidate.get("logp", 0.0)),
        "_tpsa":            float(candidate.get("tpsa", 0.0)),
        "_hbd":             int(candidate.get("hbd", 0)),
        "_hba":             int(candidate.get("hba", 0)),
        "_generation":      int(candidate.get("generation", 0)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_denovo_candidates(uniprot_id: str) -> tuple[list[dict], dict]:
    """
    Load top de novo candidates from the design output JSON.

    Returns:
        (candidates_list, denovo_meta)
        candidates_list: list of DenovoCandidate dicts
        denovo_meta: pocket_center, pocket_id, box_size etc.
    """
    denovo_path = INTER / f"{uniprot_id}_denovo.json"
    if not denovo_path.exists():
        raise FileNotFoundError(
            f"De novo output not found: {denovo_path}\n"
            f"  Run pipeline/denovo_design.py --uniprot {uniprot_id} first."
        )

    data = json.loads(denovo_path.read_text(encoding="utf-8"))
    candidates = data.get("top_candidates", [])
    # Filter to only candidates with real docking scores
    valid = [c for c in candidates
             if c.get("smiles") and float(c.get("score", 0)) < -0.5]

    return valid, data


def _load_context(uniprot_id: str) -> tuple[str, float, float, float, float]:
    """
    Load contextual data needed for scoring:
    gene_name, log2fc, delta_delta_S, n_resistance, pocket_volume.
    """
    # Gene name from structure JSON
    struct_path = INTER / f"{uniprot_id}_structure.json"
    gene_name = uniprot_id
    if struct_path.exists():
        s = json.loads(struct_path.read_text(encoding="utf-8"))
        gene_name = s.get("gene_name", uniprot_id)

    # Tumor vs normal log2FC
    tn_path = GRN_DIR / "intermediate" / "tumor_vs_normal.json"
    log2fc = 0.0
    if tn_path.exists():
        tn = json.loads(tn_path.read_text(encoding="utf-8"))
        log2fc = float(tn.get(gene_name, {}).get("log2fc", 0.0))

    # Network perturbation ΔΔS
    # Look for any perturbation file (drug-agnostic for de novo)
    pert_dir = SIM_DIR / "perturbation"
    delta_delta_S = 0.0
    if pert_dir.exists():
        pert_files = list(pert_dir.glob("*_perturbation.json"))
        if pert_files:
            pert = json.loads(pert_files[0].read_text(encoding="utf-8"))
            delta_delta_S = float(pert.get("delta_delta_S", 0.0))

    # Pocket volume from consensus report
    pocket_vol = 500.0
    report_path = ROOT / "data" / "reports" / f"{uniprot_id}_report.json"
    if report_path.exists():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        pockets = rep.get("binding_pockets", [])
        if pockets:
            pocket_vol = float(pockets[0].get("volume_A3", 500.0))

    return gene_name, log2fc, delta_delta_S, pocket_vol


def score_denovo_candidates(
    uniprot_id: str,
    top_n:      int   = 10,
    dose_uM:    float = 10.0,
    verbose:    bool  = True,
) -> list[DenovoPharmScore]:
    """
    Score de novo candidates using SIM-06 pharmacological scoring framework.

    Args:
        uniprot_id: UniProt accession of the target protein
        top_n:      Score only the top N candidates by Vina score
        dose_uM:    Assumed dose for scoring (default 10 μM)
        verbose:    Print progress and results

    Returns:
        list of DenovoPharmScore, sorted by composite score descending
    """
    if verbose:
        print(f"\n── De Novo Pharmacological Scoring: {uniprot_id} ──")

    # Load candidates
    candidates, meta = load_denovo_candidates(uniprot_id)
    if not candidates:
        print(f"  No valid de novo candidates found for {uniprot_id}")
        return []

    # Sort by Vina score and take top N
    candidates = sorted(candidates, key=lambda c: float(c.get("score", 0)))[:top_n]
    if verbose:
        print(f"  Scoring {len(candidates)} candidates "
              f"(best Vina={candidates[0].get('score', 0):.2f} kcal/mol)")

    # Load context
    gene_name, log2fc, delta_delta_S, pocket_vol = _load_context(uniprot_id)
    if verbose:
        print(f"  Target: {gene_name} ({uniprot_id})  "
              f"log2FC={log2fc:+.2f}  ΔΔS={delta_delta_S:+.4f}")

    # Import SIM-06 scoring functions
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from sim.step06_pharmacological_scoring import (
            score_efficacy, score_selectivity, score_network_disruption,
            score_resistance, score_safety, score_druggability,
            WEIGHTS,
        )
    except ImportError as e:
        print(f"  ERROR: Could not import SIM-06 scoring: {e}")
        return []

    # Load perturbation context for network/resistance scores
    pert_dir = SIM_DIR / "perturbation"
    pert_data: dict = {}
    if pert_dir.exists():
        pert_files = list(pert_dir.glob("*_perturbation.json"))
        if pert_files:
            pert_data = json.loads(pert_files[0].read_text(encoding="utf-8"))

    delta_S_tumor  = float(pert_data.get("delta_S_tumor", 0.0))
    n_resistance   = int(pert_data.get("n_upregulated", 0))
    cascade_depth  = max(
        (v for v in pert_data.get("cascade_depth", {}).values()),
        default=3
    )
    n_affected = len(pert_data.get("downregulated", []))
    n_genes_grn = int(pert_data.get("n_genes", 1000))

    # Compute score for each candidate
    results: list[DenovoPharmScore] = []

    for cand in candidates:
        binding = vina_to_binding_entry(
            candidate  = cand,
            uniprot_id = uniprot_id,
            gene_name  = gene_name,
            pocket_vol = pocket_vol,
        )

        Kd_uM   = binding["Kd_corrected_uM"]
        pKi     = binding["pKi"]
        p_bind  = binding["p_binding"]
        pocket  = binding["pocket_volume_A3"]
        fill    = binding["fill_ratio"]

        # Drug concentration at target: assume 10% of dose reaches target compartment
        drug_conc = dose_uM * 0.1

        # Six scoring axes
        eff  = score_efficacy(p_bind, Kd_uM, pKi, drug_conc, dose_uM)
        sel  = score_selectivity(delta_delta_S, log2fc, p_bind)

        try:
            net = score_network_disruption(
                delta_S_tumor, cascade_depth, n_affected, n_genes_grn
            )
        except TypeError:
            # Some versions of score_network_disruption take fewer args
            net = score_network_disruption(delta_S_tumor, cascade_depth, n_affected)

        try:
            res = score_resistance(n_resistance, cascade_depth, p_bind)
        except TypeError:
            res = max(0, 100 - n_resistance * 5)

        saf  = score_safety(delta_delta_S, log2fc)
        drug_score = score_druggability(pocket, fill, pKi)

        # Composite weighted score
        composite = (
            eff  * WEIGHTS["efficacy"]     +
            sel  * WEIGHTS["selectivity"]  +
            net  * WEIGHTS["network"]      +
            res  * WEIGHTS["resistance"]   +
            saf  * WEIGHTS["safety"]       +
            drug_score * WEIGHTS["druggability"]
        )

        grade = (
            "A" if composite >= 75 else
            "B" if composite >= 60 else
            "C" if composite >= 45 else
            "D" if composite >= 30 else "F"
        )

        # Convert Vina score to kJ
        vina = float(cand.get("score", -5.0))
        dG_kJ = vina * KCAL_TO_KJ

        results.append(DenovoPharmScore(
            uniprot_id     = uniprot_id,
            smiles         = binding["_smiles"],
            vina_score     = vina,
            le             = binding["_le"],
            qed            = binding["_qed"],
            mw             = binding["_mw"],
            logp           = binding["_logp"],
            tpsa           = binding["_tpsa"],
            hbd            = binding["_hbd"],
            hba            = binding["_hba"],
            generation     = binding["_generation"],
            dG_kJ          = round(dG_kJ, 2),
            Kd_uM          = round(Kd_uM, 3),
            pKi            = round(pKi, 2),
            efficacy       = round(eff, 1),
            selectivity    = round(sel, 1),
            network        = round(net, 1),
            resistance     = round(res, 1),
            safety         = round(saf, 1),
            druggability   = round(drug_score, 1),
            composite      = round(composite, 1),
            grade          = grade,
            target_gene    = gene_name,
            log2fc         = log2fc,
            delta_delta_S  = delta_delta_S,
        ))

    results.sort(key=lambda r: -r.composite)

    if verbose:
        print(f"\n  {'#':<3} {'SMILES':<45} {'Vina':>6} {'LE':>5} "
              f"{'QED':>5} {'Kd':>8} {'Grade':>5} {'Score':>6}")
        print(f"  {'─'*3} {'─'*45} {'─'*6} {'─'*5} "
              f"{'─'*5} {'─'*8} {'─'*5} {'─'*6}")
        for i, r in enumerate(results, 1):
            print(f"  {i:<3} {r.smiles[:45]:<45} "
                  f"{r.vina_score:>6.2f} "
                  f"{r.le:>5.3f} "
                  f"{r.qed:>5.2f} "
                  f"{r.Kd_uM:>7.2f}μ "
                  f"{r.grade:>5}  "
                  f"{r.composite:>6.1f}")

    return results


def save_denovo_scores(
    scores:     list[DenovoPharmScore],
    uniprot_id: str,
) -> Path:
    """
    Save de novo pharmacological scores to:
        data/sim/scores/{uniprot}_denovo_pharm.json
    """
    out_path = OUT_DIR / f"{uniprot_id}_denovo_pharm.json"
    record = {
        "uniprot_id":    uniprot_id,
        "n_scored":      len(scores),
        "best_composite": scores[0].composite if scores else 0.0,
        "best_grade":    scores[0].grade      if scores else "?",
        "best_smiles":   scores[0].smiles     if scores else "",
        "best_Kd_uM":    scores[0].Kd_uM      if scores else 0.0,
        "scores":        [s.to_dict() for s in scores],
    }
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Gap 4 fix — score de novo candidates with SIM-06 framework.\n"
            "Reads {uniprot}_denovo.json, converts Vina scores to Kd/pKi,\n"
            "and applies the full pharmacological scoring pipeline."
        )
    )
    parser.add_argument("--uniprot", "-u", required=True,
                        help="UniProt ID (e.g. Q9HAW4)")
    parser.add_argument("--top-n",   type=int, default=10,
                        help="Number of top candidates to score (default: 10)")
    parser.add_argument("--dose",    type=float, default=10.0,
                        help="Assumed drug dose in μM (default: 10.0)")
    args = parser.parse_args()

    scores = score_denovo_candidates(
        uniprot_id = args.uniprot,
        top_n      = args.top_n,
        dose_uM    = args.dose,
        verbose    = True,
    )

    if scores:
        out = save_denovo_scores(scores, args.uniprot)
        print(f"\n  Scores saved → {out}")
        print(f"\n  Best candidate:")
        print(f"    {scores[0].summary()}")
        print(f"\n  SIM-06 breakdown:")
        b = scores[0]
        print(f"    Efficacy     {b.efficacy:>5.1f}/100")
        print(f"    Selectivity  {b.selectivity:>5.1f}/100")
        print(f"    Network      {b.network:>5.1f}/100")
        print(f"    Resistance   {b.resistance:>5.1f}/100")
        print(f"    Safety       {b.safety:>5.1f}/100")
        print(f"    Druggability {b.druggability:>5.1f}/100")
        print(f"    ──────────────────")
        print(f"    Composite    {b.composite:>5.1f}/100  Grade: {b.grade}")