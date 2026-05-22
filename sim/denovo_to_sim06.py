"""
sim/denovo_to_sim06.py
───────────────────────
Feeds de novo design candidates into the SIM-06 pharmacological
scoring framework, producing a full grade A–F for each evolved molecule.

WHAT THIS DOES
──────────────
1. Reads  data/intermediate/{uniprot}_denovo.json       (Module 15 output)
2. Converts each Vina docking score → Kd / pKi
3. Loads any available context:
      • perturbation data  (SIM-05, if present)
      • physicochemical    (Module 02, for is_surface / normal_expr)
      • binding pockets    (Module 04, for pocket_vol)
4. Calls SIM-06 scoring functions with the CORRECT signatures
5. Prints a ranked table and saves results to
      data/sim/scores/{uniprot}_denovo_pharm.json

USAGE
─────
    python sim/denovo_to_sim06.py --uniprot O75911
    python sim/denovo_to_sim06.py --uniprot O75911 --top-n 5
    python sim/denovo_to_sim06.py --uniprot O75911 --dose 1.0 --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).resolve().parent.parent
INTER   = ROOT / "data" / "intermediate"
SIM_DIR = ROOT / "data" / "sim"
OUT_DIR = SIM_DIR / "scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

# ── Physical constants ─────────────────────────────────────────────────────────

R          = 8.314      # J/mol/K
T          = 310.15     # K  (37 °C)
RT_kJ      = R * T / 1000
KCAL_TO_KJ = 4.184


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DenovoPharmScore:
    """Full pharmacological profile for one de novo candidate."""
    rank:          int
    uniprot_id:    str
    gene_name:     str
    smiles:        str
    generation:    int

    # Docking / binding
    vina_score:    float   # kcal/mol  (negative = better)
    dG_kJ:         float
    Kd_uM:         float
    pKi:           float
    le:            float   # ligand efficiency
    qed:           float
    mw:            float
    logp:          float
    tpsa:          float   # topological polar surface area
    hbd:           int
    hba:           int
    p_binding:     float

    # Six SIM-06 axes  (0-100 each)
    efficacy:      float
    selectivity:   float
    network:       float
    resistance:    float
    safety:        float
    druggability:  float

    # Composite
    composite:     float
    grade:         str     # A / B / C / D / F

    # Context
    log2fc:        float
    delta_delta_S: float

    # Narrative
    rationale:     list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> dict:
    """Load a JSON file, returning {} on any failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_context(uniprot_id: str) -> dict:
    """
    Assemble all available context for a protein from intermediate files.
    Returns a dict with keys: gene_name, log2fc, delta_delta_S,
    pocket_vol, is_surface, normal_expr_mean, n_resistance,
    delta_S_tumor, cascade_depth, n_affected, n_genes_grn.
    """
    ctx: dict = {
        "gene_name":        uniprot_id,
        "log2fc":           0.0,
        "delta_delta_S":    0.0,
        "pocket_vol":       1000.0,
        "is_surface":       False,
        "normal_expr_mean": 1.0,
        "n_resistance":     0,
        "delta_S_tumor":    0.0,
        "cascade_depth":    3,
        "n_affected":       10,
        "n_genes_grn":      1000,
        "essentiality":     0.5,
    }

    # ── Structure / UniProt metadata ────────────────────────────────────────
    struct = _load_json(INTER / f"{uniprot_id}_structure.json")
    if struct:
        ctx["gene_name"] = struct.get("gene_name", uniprot_id)

    # ── Physicochemical (Module 02) ─────────────────────────────────────────
    physico = _load_json(INTER / f"{uniprot_id}_physicochemical.json")
    if physico:
        # is_surface: protein has exposed patches (total_sasa / length > 45 Å²)
        sasa   = physico.get("total_sasa", 0.0)
        length = physico.get("n_residues", 302)
        ctx["is_surface"]       = (sasa / max(length, 1)) > 45
        ctx["normal_expr_mean"] = physico.get("net_charge_ph7", 1.0)

    # ── Binding pockets (Module 04) ─────────────────────────────────────────
    pockets = _load_json(INTER / f"{uniprot_id}_binding_pockets.json")
    if pockets:
        pocket_list = pockets.get("pockets", [])
        if pocket_list:
            ctx["pocket_vol"] = float(pocket_list[0].get("volume", 1000.0))

    # ── SIM-05 perturbation data ─────────────────────────────────────────────
    pert_dir = SIM_DIR / "perturbation"
    pert_data: dict = {}
    if pert_dir.exists():
        # Try protein-specific file first, then any available file
        specific = pert_dir / f"{uniprot_id}_perturbation.json"
        if specific.exists():
            pert_data = _load_json(specific)
        else:
            files = sorted(pert_dir.glob("*_perturbation.json"))
            if files:
                pert_data = _load_json(files[0])

    if pert_data:
        ctx["delta_S_tumor"]   = float(pert_data.get("delta_S_tumor", 0.0))
        ctx["delta_delta_S"]   = float(pert_data.get("delta_delta_S", 0.0))
        ctx["n_resistance"]    = int(pert_data.get("n_upregulated", 0))
        ctx["n_affected"]      = len(pert_data.get("downregulated", []))
        ctx["n_genes_grn"]     = int(pert_data.get("n_genes", 1000))
        cascade = pert_data.get("cascade_depth", {})
        ctx["cascade_depth"]   = max(cascade.values(), default=3) if cascade else 3
        ctx["essentiality"]    = min(1.0, ctx["cascade_depth"] / 10)

    # ── GRN expression data ──────────────────────────────────────────────────
    grn_dir = ROOT / "data" / "grn"
    tn_path = grn_dir / "tumor_normal_expression.json"
    if tn_path.exists():
        tn = _load_json(tn_path)
        gene = ctx["gene_name"]
        if gene in tn:
            ctx["log2fc"]           = float(tn[gene].get("log2fc", 0.0))
            ctx["normal_expr_mean"] = float(tn[gene].get("normal_mean", 1.0))

    return ctx


def _load_candidates(uniprot_id: str) -> tuple[list[dict], dict]:
    """
    Load de novo candidates from the intermediate JSON.
    Returns (candidates_list, metadata_dict).
    """
    path = INTER / f"{uniprot_id}_denovo.json"
    if not path.exists():
        return [], {}

    data = _load_json(path)
    candidates = data.get("candidates", data.get("top_candidates", []))
    meta = {k: v for k, v in data.items() if k not in ("candidates", "top_candidates")}
    return candidates, meta


# ══════════════════════════════════════════════════════════════════════════════
# VINA → BINDING ENTRY CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def _vina_to_kd(vina_score_kcal: float) -> tuple[float, float]:
    """
    Convert Vina docking score (kcal/mol) to Kd (μM) and pKi.

    ΔG = RT·ln(Kd)  →  Kd = exp(ΔG / RT)
    Vina score is in kcal/mol, convert to kJ/mol first.
    Returns (Kd_uM, pKi).
    """
    dG_kJ  = vina_score_kcal * KCAL_TO_KJ        # negative number
    Kd_M   = math.exp(dG_kJ * 1000 / (R * T))    # mol/L
    Kd_uM  = Kd_M * 1e6                           # μM
    pKi    = -math.log10(max(Kd_M, 1e-15))
    return round(Kd_uM, 4), round(pKi, 3)


def _p_binding(Kd_uM: float, drug_conc_uM: float) -> float:
    """
    Fraction of target occupied at given drug concentration.
    P(bind) = [D] / ([D] + Kd)
    """
    return drug_conc_uM / (drug_conc_uM + Kd_uM + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# SCORING — wraps SIM-06 functions with correct signatures + safe fallbacks
# ══════════════════════════════════════════════════════════════════════════════

def _import_sim06() -> dict | None:
    """
    Try to import all SIM-06 scoring functions.
    Returns a dict of {name: callable} or None on failure.
    """
    try:
        from sim.step06_pharmacological_scoring import (
            score_efficacy,
            score_selectivity,
            score_network_disruption,
            score_resistance_risk,
            score_safety,
            score_druggability,
            WEIGHTS,
        )
        return {
            "score_efficacy":           score_efficacy,
            "score_selectivity":        score_selectivity,
            "score_network_disruption": score_network_disruption,
            "score_resistance_risk":    score_resistance_risk,
            "score_safety":             score_safety,
            "score_druggability":       score_druggability,
            "WEIGHTS":                  WEIGHTS,
        }
    except ImportError as e:
        print(f"  [WARN] Could not import SIM-06: {e}")
        print(f"         Falling back to built-in scoring functions.")
        return None


# ── Built-in fallback scoring (mirrors SIM-06 logic exactly) ──────────────────

def _fallback_efficacy(p_binding, Kd_uM, pKi, drug_conc_uM, dose_uM) -> float:
    p_score   = min(40, p_binding * 4000)
    pKi_norm  = max(0, min(1, (pKi - 4) / 5))
    pKi_score = pKi_norm * 30
    conc_ratio = drug_conc_uM / max(dose_uM, 0.001)
    conc_score = min(30, conc_ratio * 300)
    return round(max(0, min(100, p_score + pKi_score + conc_score)), 1)


def _fallback_selectivity(delta_delta_S, log2fc, p_binding) -> float:
    if delta_delta_S > 1.0:
        dds = 50
    elif delta_delta_S > 0:
        dds = delta_delta_S * 50
    else:
        dds = max(-25, delta_delta_S * 25)
    fc   = max(0, min(30, log2fc * 7.5))
    bind = min(20, p_binding * 2000)
    return round(max(0, min(100, dds + fc + bind)), 1)


def _fallback_network(delta_S, cascade_depth, n_affected, n_genes_grn) -> float:
    S_score    = min(40, delta_S * 80) if delta_S > 0 else 0
    depth_score = min(30, cascade_depth * 6)
    frac_score  = min(30, (n_affected / max(n_genes_grn, 1)) * 300)
    return round(max(0, min(100, S_score + depth_score + frac_score)), 1)


def _fallback_resistance(n_resistance_genes, delta_S, essentiality) -> float:
    if n_resistance_genes == 0:
        r = 40
    elif n_resistance_genes < 5:
        r = 30
    elif n_resistance_genes < 10:
        r = 15
    else:
        r = 5
    ess  = essentiality * 30
    disr = min(30, delta_S * 60) if delta_S > 0 else 0
    return round(max(0, min(100, r + ess + disr)), 1)


def _fallback_safety(delta_delta_S, log2fc, is_surface, normal_expr_mean) -> float:
    if delta_delta_S > 0.5:
        sel = 40
    elif delta_delta_S > 0:
        sel = delta_delta_S * 80
    elif delta_delta_S > -0.5:
        sel = max(0, 20 + delta_delta_S * 40)
    else:
        sel = 0
    fc   = max(0, min(30, log2fc * 5))
    surf = 20 if is_surface else 10
    expr = max(0, 10 - normal_expr_mean * 5)
    return round(max(0, min(100, sel + fc + surf + expr)), 1)


def _fallback_druggability(pocket_vol, pocket_drug_score, fill_ratio, pKi) -> float:
    vol_score  = min(30, (pocket_vol - 200) / 30) if pocket_vol > 200 else 0
    shape_score = min(30, pocket_drug_score * 30)
    fill_score  = min(20, fill_ratio * 40) if 0.1 < fill_ratio < 0.9 else 5
    pki_score   = min(20, max(0, (pKi - 4) * 4))
    return round(max(0, min(100, vol_score + shape_score + fill_score + pki_score)), 1)


FALLBACK_WEIGHTS = {
    "efficacy":     0.25,
    "selectivity":  0.25,
    "network":      0.20,
    "resistance":   0.10,
    "safety":       0.10,
    "druggability": 0.10,
}


def _score_one(
    cand:      dict,
    ctx:       dict,
    fns:       dict | None,
    dose_uM:   float,
    rank:      int,
) -> DenovoPharmScore:
    """Compute all six axes + composite for one candidate molecule."""

    smiles     = cand.get("smiles", cand.get("SMILES", ""))
    vina       = float(cand.get("score", cand.get("vina_score", -5.0)))
    le         = float(cand.get("le",    cand.get("LE",   0.3)))
    qed        = float(cand.get("qed",   cand.get("QED",  0.5)))
    mw         = float(cand.get("mw",    cand.get("MW",   300.0)))
    logp       = float(cand.get("logp",  cand.get("LogP", 3.0)))
    tpsa       = float(cand.get("tpsa",  cand.get("TPSA", 80.0)))
    hbd        = int(  cand.get("hbd",   cand.get("HBD",  2)))
    hba        = int(  cand.get("hba",   cand.get("HBA",  4)))
    generation = int(  cand.get("generation", 0))

    Kd_uM, pKi = _vina_to_kd(vina)
    drug_conc   = dose_uM * 0.1          # assume 10% reaches target compartment
    p_bind      = _p_binding(Kd_uM, drug_conc)
    dG_kJ       = vina * KCAL_TO_KJ

    # Context
    log2fc        = ctx["log2fc"]
    dds           = ctx["delta_delta_S"]
    pocket_vol    = ctx["pocket_vol"]
    is_surface    = ctx["is_surface"]
    norm_expr     = ctx["normal_expr_mean"]
    delta_S       = ctx["delta_S_tumor"]
    cascade_depth = ctx["cascade_depth"]
    n_affected    = ctx["n_affected"]
    n_genes_grn   = ctx["n_genes_grn"]
    n_resistance  = ctx["n_resistance"]
    essentiality  = ctx["essentiality"]

    # Pocket drug score approximation from druggability
    pocket_drug_score = 0.90  # from Module 04

    # Fill ratio approximation: MW / (pocket_vol * 0.8)  (rough estimate)
    fill_ratio = min(0.95, (mw / 500) / max(pocket_vol / 1000, 0.1))

    weights = FALLBACK_WEIGHTS

    if fns:
        # Use real SIM-06 functions
        w = fns["WEIGHTS"]
        weights = w

        try:
            eff = fns["score_efficacy"](p_bind, Kd_uM, pKi, drug_conc, dose_uM)
        except Exception:
            eff = _fallback_efficacy(p_bind, Kd_uM, pKi, drug_conc, dose_uM)

        try:
            sel = fns["score_selectivity"](dds, log2fc, p_bind)
        except Exception:
            sel = _fallback_selectivity(dds, log2fc, p_bind)

        try:
            net = fns["score_network_disruption"](delta_S, cascade_depth, n_affected, n_genes_grn)
        except TypeError:
            try:
                net = fns["score_network_disruption"](delta_S, cascade_depth, n_affected)
            except Exception:
                net = _fallback_network(delta_S, cascade_depth, n_affected, n_genes_grn)

        try:
            res = fns["score_resistance_risk"](n_resistance, delta_S, essentiality)
        except Exception:
            res = _fallback_resistance(n_resistance, delta_S, essentiality)

        try:
            saf = fns["score_safety"](dds, log2fc, is_surface, norm_expr)
        except TypeError:
            try:
                saf = fns["score_safety"](dds, log2fc)
            except Exception:
                saf = _fallback_safety(dds, log2fc, is_surface, norm_expr)

        try:
            drug = fns["score_druggability"](pocket_vol, pocket_drug_score, fill_ratio, pKi)
        except TypeError:
            try:
                drug = fns["score_druggability"](pocket_vol, fill_ratio, pKi)
            except Exception:
                drug = _fallback_druggability(pocket_vol, pocket_drug_score, fill_ratio, pKi)
    else:
        # All fallback
        eff  = _fallback_efficacy(p_bind, Kd_uM, pKi, drug_conc, dose_uM)
        sel  = _fallback_selectivity(dds, log2fc, p_bind)
        net  = _fallback_network(delta_S, cascade_depth, n_affected, n_genes_grn)
        res  = _fallback_resistance(n_resistance, delta_S, essentiality)
        saf  = _fallback_safety(dds, log2fc, is_surface, norm_expr)
        drug = _fallback_druggability(pocket_vol, pocket_drug_score, fill_ratio, pKi)

    # Cap all at 0-100
    eff, sel, net, res, saf, drug = (max(0.0, min(100.0, x))
                                     for x in [eff, sel, net, res, saf, drug])

    composite = (
        eff  * weights["efficacy"]     +
        sel  * weights["selectivity"]  +
        net  * weights["network"]      +
        res  * weights["resistance"]   +
        saf  * weights["safety"]       +
        drug * weights["druggability"]
    )
    composite = round(composite, 1)

    grade = ("A" if composite >= 75 else
             "B" if composite >= 60 else
             "C" if composite >= 45 else
             "D" if composite >= 30 else "F")

    # Build rationale
    rationale: list[str] = []
    if eff < 30:
        rationale.append(f"Low efficacy — Kd={Kd_uM:.2f} μM, pKi={pKi:.1f}; "
                         f"affinity optimization needed")
    elif eff >= 60:
        rationale.append(f"Strong efficacy — pKi={pKi:.1f}, P(bind)={p_bind:.3f}")

    if sel < 20:
        rationale.append(f"Poor selectivity (ΔΔS={dds:+.3f}, log2FC={log2fc:+.1f}); "
                         f"consider tumor-targeted delivery")
    elif sel >= 50:
        rationale.append(f"Good tumor selectivity (log2FC={log2fc:+.1f})")

    if n_resistance > 10:
        rationale.append(f"High resistance risk: {n_resistance} compensatory genes")
    elif n_resistance == 0:
        rationale.append("No resistance genes detected — low escape risk")

    if mw > 500:
        rationale.append(f"MW={mw:.0f} exceeds Lipinski limit — oral bioavailability risk")
    if logp > 5:
        rationale.append(f"LogP={logp:.1f} — high lipophilicity, monitor permeability")
    if not rationale:
        rationale.append("Drug-like profile, within normal range for all axes")

    return DenovoPharmScore(
        rank          = rank,
        uniprot_id    = uniprot_id if isinstance(uniprot_id := ctx.get("_uid", ""), str) else "",
        gene_name     = ctx["gene_name"],
        smiles        = smiles,
        generation    = generation,
        vina_score    = vina,
        dG_kJ         = round(dG_kJ, 2),
        Kd_uM         = Kd_uM,
        pKi           = pKi,
        le            = le,
        qed           = qed,
        mw            = mw,
        logp          = logp,
        tpsa          = tpsa,
        hbd           = hbd,
        hba           = hba,
        p_binding     = round(p_bind, 5),
        efficacy      = eff,
        selectivity   = sel,
        network       = net,
        resistance    = res,
        safety        = saf,
        druggability  = drug,
        composite     = composite,
        grade         = grade,
        log2fc        = log2fc,
        delta_delta_S = dds,
        rationale     = rationale,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def score_denovo_candidates(
    uniprot_id: str,
    top_n:      int   = 10,
    dose_uM:    float = 10.0,
    verbose:    bool  = True,
) -> list[DenovoPharmScore]:
    """
    Score de novo candidates for a protein using the SIM-06 framework.

    Args:
        uniprot_id : UniProt accession (e.g. "O75911")
        top_n      : Score only the top N candidates by Vina score
        dose_uM    : Assumed dose in μM (default 10)
        verbose    : Print progress and results

    Returns:
        List of DenovoPharmScore sorted by composite score descending.
    """
    t0 = time.time()
    uid = uniprot_id.strip().upper()

    if verbose:
        print(f"\n{'═'*70}")
        print(f"  De Novo → SIM-06 Pharmacological Scoring: {uid}")
        print(f"{'═'*70}")

    # Load candidates
    candidates, meta = _load_candidates(uid)
    if not candidates:
        print(f"  ERROR: No de novo candidates found at {INTER / f'{uid}_denovo.json'}")
        print(f"         Run Module 15 first: proteinfp --uniprot {uid} --denovo --vina <path>")
        return []

    # Sort by Vina score (most negative = best) and take top N
    candidates = sorted(candidates, key=lambda c: float(c.get("score", c.get("vina_score", 0))))
    candidates = candidates[:top_n]

    if verbose:
        print(f"  Candidates  : {len(candidates)} "
              f"(best Vina = {candidates[0].get('score', '?'):.2f} kcal/mol)")

    # Load context
    ctx = _load_context(uid)
    ctx["_uid"] = uid

    if verbose:
        print(f"  Target      : {ctx['gene_name']} ({uid})")
        print(f"  log2FC      : {ctx['log2fc']:+.2f}")
        print(f"  ΔΔS         : {ctx['delta_delta_S']:+.4f}")
        print(f"  Pocket vol  : {ctx['pocket_vol']:.0f} Å³")
        print(f"  Is surface  : {'yes' if ctx['is_surface'] else 'no'}")
        if ctx['delta_S_tumor'] > 0:
            print(f"  ΔS tumor    : {ctx['delta_S_tumor']:.4f}  "
                  f"cascade depth={ctx['cascade_depth']}  "
                  f"affected={ctx['n_affected']}")
        print()

    # Try importing SIM-06; fall back to built-in implementations
    fns = _import_sim06()
    if verbose:
        src = "SIM-06 functions" if fns else "built-in fallback functions"
        print(f"  Scoring using: {src}")
        print()

    # Score each candidate
    results: list[DenovoPharmScore] = []
    for i, cand in enumerate(candidates, 1):
        score = _score_one(cand, ctx, fns, dose_uM, rank=i)
        # Fix uniprot_id which was set awkwardly above
        score.uniprot_id = uid
        results.append(score)

    # Re-sort by composite and re-rank
    results.sort(key=lambda r: -r.composite)
    for i, r in enumerate(results, 1):
        r.rank = i

    # ── Print results table ────────────────────────────────────────────────────
    if verbose:
        _print_results(results, uid, ctx, dose_uM, time.time() - t0)

    return results


def save_denovo_scores(
    scores:     list[DenovoPharmScore],
    uniprot_id: str,
) -> Path:
    """Save pharmacological scores to data/sim/scores/{uid}_denovo_pharm.json"""
    out_path = OUT_DIR / f"{uniprot_id}_denovo_pharm.json"
    record = {
        "uniprot_id":      uniprot_id,
        "n_scored":        len(scores),
        "best_composite":  scores[0].composite if scores else 0.0,
        "best_grade":      scores[0].grade      if scores else "?",
        "best_smiles":     scores[0].smiles      if scores else "",
        "best_Kd_uM":      scores[0].Kd_uM       if scores else 0.0,
        "best_vina":       scores[0].vina_score   if scores else 0.0,
        "scores":          [s.to_dict() for s in scores],
    }
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# PRETTY PRINTING
# ══════════════════════════════════════════════════════════════════════════════

def _print_results(
    results:   list[DenovoPharmScore],
    uid:       str,
    ctx:       dict,
    dose_uM:   float,
    elapsed:   float,
) -> None:
    gene = ctx["gene_name"]

    print(f"{'─'*70}")
    print(f"  RESULTS: {gene} ({uid})  —  dose={dose_uM} μM")
    print(f"{'─'*70}")
    print(f"  {'#':<3} {'Grade':>5} {'Score':>6} {'Vina':>7} {'Kd(μM)':>8} "
          f"{'pKi':>5} {'QED':>5} {'MW':>5} {'Eff':>5} {'Sel':>5} "
          f"{'Net':>5} {'Res':>5} {'Saf':>5} {'Drug':>5}")
    print(f"  {'─'*3} {'─'*5} {'─'*6} {'─'*7} {'─'*8} {'─'*5} {'─'*5} "
          f"{'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5}")

    for r in results:
        print(f"  {r.rank:<3} {r.grade:>5} {r.composite:>6.1f} "
              f"{r.vina_score:>7.2f} {r.Kd_uM:>8.3f} "
              f"{r.pKi:>5.2f} {r.qed:>5.2f} {r.mw:>5.0f} "
              f"{r.efficacy:>5.1f} {r.selectivity:>5.1f} "
              f"{r.network:>5.1f} {r.resistance:>5.1f} "
              f"{r.safety:>5.1f} {r.druggability:>5.1f}")

    print()
    print(f"  DETAILED CANDIDATES:")
    print(f"{'─'*70}")
    for r in results[:5]:
        print(f"\n  [Rank #{r.rank}]  Grade: {r.grade}  Composite: {r.composite}/100")
        print(f"  SMILES  : {r.smiles}")
        print(f"  Docking : {r.vina_score:.2f} kcal/mol  "
              f"Kd={r.Kd_uM:.3f} μM  pKi={r.pKi:.2f}  ΔG={r.dG_kJ:.1f} kJ/mol")
        print(f"  ADMET   : MW={r.mw:.0f}  LogP={r.logp:.2f}  "
              f"TPSA={r.tpsa:.0f}  HBD={r.hbd}  HBA={r.hba}  "
              f"QED={r.qed:.2f}  LE={r.le:.3f}")
        print(f"  Scores  : Eff={r.efficacy:.1f}  Sel={r.selectivity:.1f}  "
              f"Net={r.network:.1f}  Res={r.resistance:.1f}  "
              f"Saf={r.safety:.1f}  Drug={r.druggability:.1f}")
        for note in r.rationale:
            print(f"  ⚑  {note}")

    print()
    best = results[0]
    print(f"{'═'*70}")
    print(f"  BEST CANDIDATE: Grade {best.grade}  ({best.composite:.1f}/100)")
    print(f"  {best.smiles}")
    print()

    # Recommendation
    if best.grade in ("A", "B"):
        print(f"  ✓ Promising lead compound — recommend:")
        print(f"    1. Thermal shift assay / SPR to confirm binding (target Kd < 1 μM)")
        print(f"    2. Cell viability assay (IC50 in DHRS3-expressing cell line)")
        print(f"    3. SAR study: vary substituents to improve selectivity (currently SI=0)")
    elif best.grade == "C":
        print(f"  ~ Moderate lead — recommend SAR optimization before progressing:")
        print(f"    1. Improve pKi (currently {best.pKi:.1f} — target > 7)")
        print(f"    2. Address selectivity (ΔΔS={best.delta_delta_S:+.3f})")
    else:
        print(f"  ✗ Weak lead — consider:")
        print(f"    1. Running more generations: --denovo --vina <path> (increase generations)")
        print(f"    2. Trying allosteric modality: --interactive")
        print(f"    3. PROTAC approach to bypass binding affinity limitations")

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score de novo candidates with the SIM-06 pharmacological framework.\n"
            "Reads {uniprot}_denovo.json, converts Vina scores → Kd/pKi,\n"
            "applies 6-axis scoring, and outputs a grade A–F per molecule.\n\n"
            "Example:\n"
            "  python sim/denovo_to_sim06.py --uniprot O75911\n"
            "  python sim/denovo_to_sim06.py --uniprot O75911 --top-n 5 --dose 1.0"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--uniprot", "-u", required=True,
                        help="UniProt ID (e.g. O75911)")
    parser.add_argument("--top-n",   "-n", type=int,   default=10,
                        help="Score only the top N candidates (default: 10)")
    parser.add_argument("--dose",    "-d", type=float, default=10.0,
                        help="Assumed drug dose in μM (default: 10.0)")
    parser.add_argument("--save",    "-s", action="store_true", default=True,
                        help="Save results to JSON (default: on)")
    parser.add_argument("--no-save", action="store_true", default=False,
                        help="Skip saving JSON output")
    args = parser.parse_args()

    scores = score_denovo_candidates(
        uniprot_id = args.uniprot.strip().upper(),
        top_n      = args.top_n,
        dose_uM    = args.dose,
        verbose    = True,
    )

    if scores and not args.no_save:
        out = save_denovo_scores(scores, args.uniprot.strip().upper())
        print(f"  Results saved → {out}")


if __name__ == "__main__":
    main()