"""
sim/step06_pharmacological_scoring.py
───────────────────────────────────────
Module SIM-06 — Pharmacological Scoring (Final Integration)

Integrates all five upstream modules into a single unified
pharmacological score for each drug-target-cell combination.

This is the final output layer of the whole-cell simulation.

Architecture:
    Module 1: Cell environment          → compartment concentrations
    Module 2: Protein ensemble          → conformational probabilities
    Module 3: Drug distribution         → cellular pharmacokinetics
    Module 4: Binding probability       → ΔG decomposition, Kd
    Module 5: Network perturbation      → ΔS, cascade, resistance

    Module 6: Pharmacological scoring   → unified efficacy prediction

Scoring framework:
    Six orthogonal axes, each 0-100:

    1. EFFICACY SCORE
       How strongly does the drug hit its intended targets?
       = f(p_binding, Kd, pKi, drug_concentration_at_target)

    2. SELECTIVITY SCORE
       Does it hit tumor more than normal?
       = f(ΔΔS, tumor_vs_normal_fold_change, compartment_specificity)

    3. NETWORK DISRUPTION SCORE
       How much does it perturb the cancer gene program?
       = f(ΔS_tumor, cascade_depth, n_affected_genes, hub_centrality)

    4. RESISTANCE RISK SCORE
       How likely is the tumor to develop resistance?
       = f(n_resistance_genes, escape_pathway_score, target_essentiality)
       (inverted: high score = LOW resistance risk)

    5. SAFETY SCORE
       How toxic is it to normal tissue?
       = f(ΔΔS, normal_expression_fold_change, off_target_binding)
       (inverted: high score = LOW toxicity)

    6. DRUGGABILITY SCORE
       How tractable is this target for further optimization?
       = f(pocket_volume, pocket_shape, fill_ratio, pKi)

    COMPOSITE SCORE = weighted sum of all six axes
    Default weights: efficacy=0.25, selectivity=0.25, network=0.20,
                     resistance=0.10, safety=0.10, druggability=0.10

    COMBINATION THERAPY PREDICTOR:
    For each pair of targets, compute synergy score:
    Synergy = overlap_of_cascade_nodes / union_of_cascade_nodes
    Non-overlapping cascades = additive/synergistic
    Overlapping cascades = potentially redundant

Usage:
    python sim/step06_pharmacological_scoring.py --drug gemcitabine
    python sim/step06_pharmacological_scoring.py --drug gemcitabine --compare
    python sim/step06_pharmacological_scoring.py --drug gemcitabine --report
"""

from __future__ import annotations

import json
import math
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).resolve().parent.parent
SIM_DIR = ROOT / "data" / "sim"
GRN_DIR = ROOT / "data" / "grn"
OUT_DIR = SIM_DIR / "scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Scoring weights ───────────────────────────────────────────────────────────

WEIGHTS = {
    "efficacy":     0.25,
    "selectivity":  0.25,
    "network":      0.20,
    "resistance":   0.10,
    "safety":       0.10,
    "druggability": 0.10,
}

# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class PharmacologicalScore:
    """Complete pharmacological profile for a drug-target pair."""
    drug_name:      str
    target_gene:    str
    target_uniprot: str
    dose_uM:        float

    # Six scoring axes (0-100)
    efficacy:       float
    selectivity:    float
    network:        float
    resistance:     float   # high = LOW resistance risk
    safety:         float   # high = LOW toxicity
    druggability:   float

    # Composite
    composite:      float
    grade:          str     # A/B/C/D/F

    # Supporting data
    Kd_uM:          float
    pKi:            float
    p_binding:      float
    delta_S_tumor:  float
    delta_delta_S:  float
    n_resistance:   int
    pocket_vol:     float
    fill_ratio:     float
    log2fc_tumor_normal: float

    # Recommendation
    modality:       str
    priority:       str
    rationale:      list

    def to_dict(self) -> dict:
        return asdict(self)


# ── Individual score calculators ──────────────────────────────────────────────

def score_efficacy(
    p_binding:    float,
    Kd_uM:        float,
    pKi:          float,
    drug_conc_uM: float,
    dose_uM:      float,
) -> float:
    """
    Efficacy score (0-100).

    Components:
    - Binding probability at clinical dose: P(bind) → 0-40 pts
    - Intrinsic affinity (pKi):             pKi  → 0-30 pts
    - Drug reaches target compartment:       conc/dose → 0-30 pts

    Clinical benchmarks:
    - pKi > 8 (Kd < 10 nM)  = excellent (100%)
    - pKi 6-8 (10 nM-1 μM)  = good (70-90%)
    - pKi 4-6 (1-100 μM)    = moderate (40-70%)
    - pKi < 4 (> 100 μM)    = poor (<40%)
    """
    # Binding probability component (0-40)
    p_score = min(40, p_binding * 4000)

    # pKi component (0-30)
    # pKi 9 → 30 pts, pKi 4 → 0 pts
    pKi_norm = max(0, min(1, (pKi - 4) / 5))
    pKi_score = pKi_norm * 30

    # Drug concentration at target (0-30)
    # Full score if drug_conc >= 10% of dose
    conc_ratio = drug_conc_uM / max(dose_uM, 0.001)
    conc_score = min(30, conc_ratio * 300)

    return round(p_score + pKi_score + conc_score, 1)


def score_selectivity(
    delta_delta_S: float,
    log2fc:        float,
    p_binding:     float,
) -> float:
    """
    Selectivity score (0-100).

    Measures tumor vs normal preference.

    Components:
    - ΔΔS (network entropy differential):  → 0-50 pts
    - log2FC (tumor vs normal expression): → 0-30 pts
    - Binding probability (signal):         → 0-20 pts

    ΔΔS benchmarks:
    > 1.0  = highly selective   → 50 pts
    0.1-1  = moderately selective → 25 pts
    ~0     = non-selective      → 0 pts
    < 0    = normal-preferring  → -25 pts (penalty)
    """
    # ΔΔS component (0-50)
    if delta_delta_S > 1.0:
        dds_score = 50
    elif delta_delta_S > 0:
        dds_score = delta_delta_S * 50
    else:
        dds_score = max(-25, delta_delta_S * 25)

    # log2FC component (0-30)
    # log2FC > 4 → 30 pts, log2FC < 0 → 0 pts
    fc_score = max(0, min(30, log2fc * 7.5))

    # Binding signal (0-20)
    bind_score = min(20, p_binding * 2000)

    return round(max(0, dds_score + fc_score + bind_score), 1)


def score_network_disruption(
    delta_S:      float,
    cascade_depth: int,
    n_affected:   int,
    n_genes_grn:  int,
) -> float:
    """
    Network disruption score (0-100).

    Measures how broadly the drug disrupts the cancer gene program.
    Broader disruption = harder for cancer to adapt = better drug.

    Components:
    - ΔS (entropy change):         → 0-40 pts
    - Cascade depth:               → 0-30 pts
    - Fraction of GRN affected:    → 0-30 pts

    Benchmarks:
    ΔS > 0.5 nats  → excellent disruption
    ΔS 0.05-0.5    → moderate
    ΔS < 0.05      → minimal
    """
    # Entropy change (0-40)
    if delta_S > 0.5:
        S_score = 40
    elif delta_S > 0:
        S_score = delta_S * 80
    else:
        S_score = 0

    # Cascade depth (0-30)
    # Depth 5+ → 30 pts, depth 0 → 0 pts
    depth_score = min(30, cascade_depth * 6)

    # Fraction affected (0-30)
    frac_affected = n_affected / max(n_genes_grn, 1)
    frac_score    = min(30, frac_affected * 300)

    return round(S_score + depth_score + frac_score, 1)


def score_resistance_risk(
    n_resistance_genes: int,
    delta_S:            float,
    target_essentiality: float,  # 0-1, from GRN centrality
) -> float:
    """
    Resistance score (0-100) — HIGH = LOW resistance risk.

    Resistance is less likely when:
    - Few compensatory genes are upregulated
    - The target is essential (high GRN centrality)
    - Network disruption is large (hard to compensate)

    Inverted: 100 = definitely won't develop resistance
              0   = will definitely develop resistance
    """
    # Resistance gene penalty (0-40 points deducted from 40)
    if n_resistance_genes == 0:
        resist_score = 40
    elif n_resistance_genes < 5:
        resist_score = 30
    elif n_resistance_genes < 10:
        resist_score = 15
    else:
        resist_score = 5

    # Essentiality bonus (0-30)
    ess_score = target_essentiality * 30

    # Network disruption bonus (0-30)
    # Larger disruption → harder to compensate
    if delta_S > 0.5:
        disr_score = 30
    elif delta_S > 0:
        disr_score = delta_S * 60
    else:
        disr_score = 0

    return round(resist_score + ess_score + disr_score, 1)


def score_safety(
    delta_delta_S:    float,
    log2fc:           float,
    is_surface:       bool,
    normal_expr_mean: float,
) -> float:
    """
    Safety score (0-100) — HIGH = LOW toxicity.

    Safety is higher when:
    - Drug is more tumor-selective (positive ΔΔS)
    - Target has high tumor vs normal expression (log2FC)
    - Surface proteins can be targeted with ADCs (localized delivery)
    - Low normal cell expression

    """
    # ΔΔS selectivity (0-40)
    if delta_delta_S > 0.5:
        sel_score = 40
    elif delta_delta_S > 0:
        sel_score = delta_delta_S * 80
    elif delta_delta_S > -0.5:
        sel_score = max(0, 20 + delta_delta_S * 40)
    else:
        sel_score = 0

    # log2FC safety margin (0-30)
    # High FC means drug barely affects normal cells
    fc_score = max(0, min(30, log2fc * 5))

    # Surface/intracellular bonus (0-20)
    surf_score = 20 if is_surface else 10

    # Normal expression penalty (0-10)
    # Low normal expression = safer to target
    expr_score = max(0, 10 - normal_expr_mean * 5)

    return round(sel_score + fc_score + surf_score + expr_score, 1)


def score_druggability(
    pocket_vol:  float,
    pocket_drug: float,
    fill_ratio:  float,
    pKi:         float,
) -> float:
    """
    Druggability score (0-100).

    Measures how tractable this target-drug combination is
    for medicinal chemistry optimization.

    Components:
    - Pocket volume & shape:  → 0-40 pts
    - Fill ratio quality:     → 0-30 pts
    - Current affinity (pKi): → 0-30 pts
    """
    # Pocket quality (0-40)
    pocket_score = min(40, (pocket_vol / 1500) * 20 + pocket_drug * 20)

    # Fill ratio quality (0-30)
    # Optimal fill 0.3-0.6: good starting point for optimization
    fill_dist    = abs(fill_ratio - 0.45)
    fill_score   = max(0, 30 - fill_dist * 60)

    # Current affinity (0-30)
    pKi_score = max(0, min(30, (pKi - 3) * 6))

    return round(pocket_score + fill_score + pKi_score, 1)


def assign_grade(composite: float) -> str:
    if composite >= 80:  return "A"
    if composite >= 65:  return "B"
    if composite >= 50:  return "C"
    if composite >= 35:  return "D"
    return "F"


def assign_priority(composite: float, selectivity: float,
                    safety: float) -> str:
    if composite >= 70 and selectivity >= 50 and safety >= 50:
        return "HIGH PRIORITY — advance to lead optimization"
    if composite >= 55 and (selectivity >= 40 or safety >= 40):
        return "MEDIUM PRIORITY — investigate further"
    if composite >= 40:
        return "LOW PRIORITY — structural modifications needed"
    return "DEPRIORITIZED — insufficient profile"


# ── Combination synergy ───────────────────────────────────────────────────────

def compute_combination_synergy(
    scores: list[PharmacologicalScore],
    perturbation_data: dict,
) -> dict:
    """
    Compute pairwise combination synergy scores.

    Synergy model based on cascade overlap:
    - Non-overlapping cascades: additive/synergistic
      (drug A hits genes drug B doesn't reach)
    - Overlapping cascades: potentially redundant
      (both drugs hit the same downstream genes)

    Bliss independence model:
    P(kill | A+B) = P(kill|A) + P(kill|B) - P(kill|A) * P(kill|B)
    Synergy = actual_kill - predicted_kill (Bliss)
    """
    synergy_matrix = {}

    downreg_sets = {}
    for s in scores:
        gene = s.target_gene
        down = set(g for g, _, _, d in
                   perturbation_data.get("downregulated", [])
                   if d < -0.01)
        downreg_sets[gene] = down

    for i, s1 in enumerate(scores):
        for j, s2 in enumerate(scores):
            if i >= j:
                continue

            g1 = s1.target_gene
            g2 = s2.target_gene

            set1 = downreg_sets.get(g1, set())
            set2 = downreg_sets.get(g2, set())

            union        = len(set1 | set2)
            intersection = len(set1 & set2)
            overlap      = intersection / max(union, 1)

            # Bliss independence
            p1      = s1.p_binding
            p2      = s2.p_binding
            bliss   = p1 + p2 - p1 * p2
            # Synergy = 1 - overlap (non-overlap = synergistic)
            synergy = (1 - overlap) * 100

            synergy_matrix[f"{g1}+{g2}"] = {
                "gene1":         g1,
                "gene2":         g2,
                "overlap":       round(overlap, 3),
                "synergy_score": round(synergy, 1),
                "bliss_p":       round(bliss, 6),
                "interpretation": (
                    "SYNERGISTIC"  if synergy > 70 else
                    "ADDITIVE"     if synergy > 40 else
                    "REDUNDANT"
                ),
            }

    return synergy_matrix


# ── Load all upstream data ────────────────────────────────────────────────────

def load_all_data(drug_name: str) -> tuple[dict, dict, dict, dict]:
    """Load outputs from all five upstream modules."""

    def _load(path, label):
        if path.exists():
            return json.loads(path.read_text())
        print(f"  WARNING: {label} not found at {path}")
        return {}

    env_data   = _load(SIM_DIR / "cell_environment.json",
                       "Module 1 (cell environment)")
    dist_data  = _load(SIM_DIR / "distribution" /
                       f"{drug_name.lower()}_distribution.json",
                       "Module 3 (drug distribution)")
    bind_data  = _load(SIM_DIR / "binding" /
                       f"{drug_name.lower()}_binding.json",
                       "Module 4 (binding scores)")
    pert_data  = _load(SIM_DIR / "perturbation" /
                       f"{drug_name.lower()}_perturbation.json",
                       "Module 5 (perturbation)")

    # Load tumor vs normal data
    tn_data    = _load(ROOT / "data" / "grn" / "intermediate" /
                       "tumor_vs_normal.json",
                       "Tumor vs normal expression")

    # Load therapy recommendations
    therapy    = _load(ROOT / "data" / "grn" / "reports" /
                       "therapy_recommendations.json",
                       "Therapy recommendations")

    return env_data, dist_data, bind_data, pert_data, tn_data, therapy


# ── Main scoring ──────────────────────────────────────────────────────────────

def compute_pharmacological_scores(
    drug_name: str,
    dose_uM:   float = 10.0,
) -> list[PharmacologicalScore]:
    """
    Compute full pharmacological score for each target.
    """
    print(f"\n  Loading data from all modules...")
    env_data, dist_data, bind_data, pert_data, tn_data, therapy = \
        load_all_data(drug_name)

    if not bind_data:
        print("ERROR: No binding data. Run modules 1-4 first.")
        return []

    binding_scores = bind_data.get("scores", [])
    dist_concs     = dist_data.get("binding_events", [])
    delta_S_tumor  = pert_data.get("delta_S_tumor", 0.0)
    delta_delta_S  = pert_data.get("delta_delta_S", 0.0)
    n_resistance   = pert_data.get("n_upregulated", 0)
    cascade_depth  = max(
        (v for v in pert_data.get("cascade_depth", {}).values()),
        default=0
    )
    n_affected     = len(pert_data.get("downregulated", []))
    n_genes_grn    = pert_data.get("n_genes", 1246)

    # Build drug concentration lookup
    conc_lookup = {}
    for e in dist_concs:
        gene = e.get("target_gene", "")
        conc_lookup[gene] = e.get("drug_conc_uM", 0.0)

    # Build therapy recommendation lookup
    therapy_lookup = {}
    for t in therapy:
        therapy_lookup[t.get("gene", "")] = t

    all_scores = []

    for bs in binding_scores:
        gene    = bs.get("target_gene", "")
        uid     = bs.get("target_uniprot", "")
        Kd_uM   = float(bs.get("Kd_corrected_uM", 100))
        pKi     = float(bs.get("pKi", 4.0))
        p_bind  = float(bs.get("p_binding", 0))
        pocket  = float(bs.get("pocket_volume_A3", 500))
        fill    = float(bs.get("fill_ratio", 0.5))
        drug_c  = conc_lookup.get(gene, 0.1)
        pocket_drug = float(bs.get("dG_shape", -10)) / -15  # normalize

        # Tumor vs normal
        tn      = tn_data.get(gene, {})
        log2fc  = float(tn.get("log2fc", 0))
        norm_mean = float(tn.get("normal_mean", 1.0))

        # Therapy recommendation
        th      = therapy_lookup.get(gene, {})
        is_surface = th.get("decisions", {}).get("surface", False)
        modality   = th.get("recommendation", "unknown")

        # Target essentiality from GRN centrality
        # Approximate: genes in cascade depth 1 are more essential
        cascade = pert_data.get("cascade_depth", {})
        depth   = cascade.get(gene, 5)
        essentiality = max(0, 1 - depth * 0.2)

        # ── Compute six scores ────────────────────────────────────────────
        eff  = score_efficacy(p_bind, Kd_uM, pKi, drug_c, dose_uM)
        sel  = score_selectivity(delta_delta_S, log2fc, p_bind)
        net  = score_network_disruption(
            delta_S_tumor, cascade_depth, n_affected, n_genes_grn
        )
        res  = score_resistance_risk(n_resistance, delta_S_tumor, essentiality)
        saf  = score_safety(delta_delta_S, log2fc, is_surface, norm_mean)
        drug_score = score_druggability(pocket, pocket_drug, fill, pKi)

        # Cap all scores at 0-100
        eff, sel, net, res, saf, drug_score = (
            max(0, min(100, x))
            for x in [eff, sel, net, res, saf, drug_score]
        )

        # Composite
        composite = (
            WEIGHTS["efficacy"]     * eff  +
            WEIGHTS["selectivity"]  * sel  +
            WEIGHTS["network"]      * net  +
            WEIGHTS["resistance"]   * res  +
            WEIGHTS["safety"]       * saf  +
            WEIGHTS["druggability"] * drug_score
        )
        composite = round(composite, 1)

        grade    = assign_grade(composite)
        priority = assign_priority(composite, sel, saf)

        # Build rationale
        rationale = []
        if eff < 30:
            rationale.append(f"Low efficacy: P(bind)={p_bind:.4f}, "
                             f"Kd={Kd_uM:.1f} μM — needs affinity optimization")
        if sel > 50:
            rationale.append(f"Good selectivity: log2FC={log2fc:.1f}, "
                             f"ΔΔS={delta_delta_S:+.3f}")
        if n_resistance > 10:
            rationale.append(f"High resistance risk: {n_resistance} "
                             f"compensatory genes activated")
        if log2fc > 4:
            rationale.append(f"Tumor-specific target: "
                             f"{log2fc:.1f} log2FC tumor/normal")
        if pocket > 800:
            rationale.append(f"Large druggable pocket: {pocket:.0f} Å³ — "
                             f"good for optimization")
        if not rationale:
            rationale.append("Profile within normal range")

        ps = PharmacologicalScore(
            drug_name      = drug_name,
            target_gene    = gene,
            target_uniprot = uid,
            dose_uM        = dose_uM,
            efficacy       = eff,
            selectivity    = sel,
            network        = net,
            resistance     = res,
            safety         = saf,
            druggability   = drug_score,
            composite      = composite,
            grade          = grade,
            Kd_uM          = Kd_uM,
            pKi            = pKi,
            p_binding      = p_bind,
            delta_S_tumor  = delta_S_tumor,
            delta_delta_S  = delta_delta_S,
            n_resistance   = n_resistance,
            pocket_vol     = pocket,
            fill_ratio     = fill,
            log2fc_tumor_normal = log2fc,
            modality       = modality,
            priority       = priority,
            rationale      = rationale,
        )
        all_scores.append(ps)

    return all_scores


# ── Report generation ─────────────────────────────────────────────────────────

def print_final_report(
    scores:       list[PharmacologicalScore],
    drug_name:    str,
    dose_uM:      float,
    synergy:      dict,
    pert_data:    dict,
):
    print(f"\n{'='*70}")
    print(f"  PHARMACOLOGICAL SCORING REPORT")
    print(f"  Drug: {drug_name}  |  Dose: {dose_uM} μM")
    print(f"  PDAC Tumor Cell — Whole-Cell Simulation")
    print(f"{'='*70}")

    scores_sorted = sorted(scores, key=lambda s: -s.composite)

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n  {'Gene':<10} {'Eff':>5} {'Sel':>5} {'Net':>5} "
          f"{'Res':>5} {'Saf':>5} {'Drug':>5} {'Comp':>6} {'Gr'}")
    print(f"  {'-'*10} {'-'*5} {'-'*5} {'-'*5} "
          f"{'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*2}")
    for s in scores_sorted:
        print(f"  {s.target_gene:<10} "
              f"{s.efficacy:>5.1f} {s.selectivity:>5.1f} "
              f"{s.network:>5.1f} {s.resistance:>5.1f} "
              f"{s.safety:>5.1f} {s.druggability:>5.1f} "
              f"{s.composite:>6.1f}  {s.grade}")

    # ── Detailed profiles ──────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  DETAILED TARGET PROFILES")
    print(f"{'─'*70}")

    for s in scores_sorted:
        bar_width = int(s.composite / 2)
        bar = "█" * bar_width + "░" * (50 - bar_width)
        print(f"\n  {s.target_gene} [{s.grade}]  {s.composite:.1f}/100")
        print(f"  {bar}")
        print(f"  Kd={s.Kd_uM:.1f}μM  pKi={s.pKi:.2f}  "
              f"P(bind)={s.p_binding:.4f}  "
              f"log2FC={s.log2fc_tumor_normal:+.1f}")
        print(f"  Modality: {s.modality[:60]}")
        print(f"  Priority: {s.priority}")
        for r in s.rationale:
            print(f"    → {r}")

    # ── Network context ────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  NETWORK PERTURBATION CONTEXT")
    print(f"{'─'*70}")
    print(f"  GRN disruption (ΔS)     : {pert_data.get('delta_S_tumor', 0):+.4f} nats")
    print(f"  Tumor selectivity (ΔΔS) : {pert_data.get('delta_delta_S', 0):+.4f} nats")
    print(f"  Cascade depth           : {max(pert_data.get('cascade_depth', {}).values(), default=0)}")
    print(f"  Resistance genes        : {pert_data.get('n_upregulated', 0)}")

    if pert_data.get("upregulated"):
        top_resist = pert_data["upregulated"][:5]
        print(f"\n  Top resistance genes (upregulated by drug):")
        for entry in top_resist:
            if isinstance(entry, (list, tuple)) and len(entry) >= 4:
                gene, base, pert, delta = entry
                print(f"    {gene:<12} baseline={base:.3f} → "
                      f"perturbed={pert:.3f} (Δ={delta:+.3f})")

    # ── Combination therapy ────────────────────────────────────────────────
    if synergy:
        print(f"\n{'─'*70}")
        print(f"  COMBINATION THERAPY ANALYSIS")
        print(f"{'─'*70}")
        for combo, data in sorted(synergy.items(),
                                   key=lambda x: -x[1]["synergy_score"]):
            interp = data["interpretation"]
            score  = data["synergy_score"]
            print(f"  {combo:<20} synergy={score:.1f}  {interp}")

    # ── Final recommendation ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL RECOMMENDATION")
    print(f"{'='*70}")

    best = scores_sorted[0]
    print(f"\n  Best target for {drug_name}: {best.target_gene} "
          f"(composite={best.composite:.1f}, grade={best.grade})")
    print(f"  Recommended modality: {best.modality[:65]}")
    print(f"\n  Clinical translation path:")

    if best.composite >= 65:
        print(f"  1. Confirm binding in biochemical assay (target Kd < 1 μM)")
        print(f"  2. Test in PDAC cell line panel (PANC-1, MIAPaCa-2, BxPC-3)")
        print(f"  3. Validate network disruption by RNA-seq after treatment")
        print(f"  4. Check combination with gemcitabine (standard of care)")
        print(f"  5. Mouse xenograft model for in vivo validation")
    else:
        print(f"  1. Optimize binding affinity — current Kd {best.Kd_uM:.1f} μM "
              f"needs improvement to < 1 μM")
        print(f"  2. Structure-activity relationship (SAR) study needed")
        print(f"  3. Consider PROTAC degrader approach to bypass binding "
              f"affinity limitations")

    # Selectivity warning
    if best.delta_delta_S < 0.05:
        print(f"\n  ⚠ SELECTIVITY WARNING:")
        print(f"  ΔΔS = {best.delta_delta_S:+.4f} — drug affects normal "
              f"cells similarly to tumor cells.")
        print(f"  Consider tumor-targeted delivery (ADC, nanoparticle) "
              f"to improve therapeutic index.")

    print(f"\n{'='*70}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    drug_name: str   = "gemcitabine",
    dose_uM:   float = 10.0,
    report:    bool  = True,
):
    print("=" * 70)
    print("  SIM-06: Pharmacological Scoring — Final Integration")
    print("  Modules 1-5 → Unified Drug Efficacy Prediction")
    print("=" * 70)

    # Compute scores
    scores = compute_pharmacological_scores(drug_name, dose_uM)
    if not scores:
        return

    # Load perturbation data for network context
    pert_path  = SIM_DIR / "perturbation" / f"{drug_name.lower()}_perturbation.json"
    pert_data  = json.loads(pert_path.read_text()) if pert_path.exists() else {}

    # Combination synergy
    synergy = compute_combination_synergy(scores, pert_data)

    if report:
        print_final_report(scores, drug_name, dose_uM, synergy, pert_data)

    # Save
    output = {
        "drug_name":  drug_name,
        "dose_uM":    dose_uM,
        "scores":     [s.to_dict() for s in scores],
        "synergy":    synergy,
        "summary": {
            "best_target":    scores[0].target_gene if scores else None,
            "best_composite": scores[0].composite   if scores else None,
            "best_grade":     scores[0].grade        if scores else None,
            "n_targets":      len(scores),
            "weights_used":   WEIGHTS,
        },
    }

    def _serialize(obj):
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    out_path = OUT_DIR / f"{drug_name.lower()}_pharmacological_score.json"
    out_path.write_text(json.dumps(_serialize(output), indent=2))
    print(f"\n  Full results saved to {out_path}")

    # Also save a human-readable summary
    summary_lines = [
        f"PHARMACOLOGICAL SCORING SUMMARY",
        f"Drug: {drug_name}  Dose: {dose_uM} μM",
        f"",
        f"{'Gene':<10} {'Composite':>10} {'Grade':>6} {'Priority'}",
        f"{'-'*65}",
    ]
    for s in sorted(scores, key=lambda x: -x.composite):
        summary_lines.append(
            f"{s.target_gene:<10} {s.composite:>10.1f} {s.grade:>6}  "
            f"{s.priority[:40]}"
        )
    summary_path = OUT_DIR / f"{drug_name.lower()}_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"  Summary saved to {summary_path}")

    print(f"\n{'='*70}")
    print(f"  SIM-06 complete — Whole-cell simulation pipeline finished")
    print(f"{'='*70}")
    print(f"\n  Pipeline summary:")
    print(f"    SIM-01: Cell environment     → 7 compartments parameterized")
    print(f"    SIM-02: Protein ensembles    → {len(scores)} targets characterized")
    print(f"    SIM-03: Drug distribution    → cellular pharmacokinetics")
    print(f"    SIM-04: Binding probability  → ΔG decomposition per target")
    print(f"    SIM-05: Network perturbation → GRN entropy + cascade")
    print(f"    SIM-06: Pharmacological score → unified efficacy prediction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SIM-06: Pharmacological Scoring"
    )
    parser.add_argument("--drug",   default="gemcitabine")
    parser.add_argument("--dose",   type=float, default=10.0)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    main(args.drug, args.dose, not args.no_report)