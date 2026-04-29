"""
sim/03_drug_distribution.py
────────────────────────────
Module SIM-03 — Drug Distribution Model

Computes the probability that a drug molecule encounters each
target protein in a binding-competent conformational state at
therapeutically relevant concentration.

This is the first truly integrated calculation — it combines:
    Module 1: Cell environment (compartment concentrations)
    Module 2: Protein ensembles (conformational state probabilities)
    Physics:  Diffusion, binding kinetics, thermodynamics

Core equation:
    P(binding) = P(drug_present) * P(protein_accessible) * P(encounter)

Where:
    P(drug_present)     = drug concentration in compartment / Kd_apparent
    P(protein_accessible) = sum of P(state_i) for binding-competent states
    P(encounter)        = diffusion-limited encounter rate * residence time

The encounter rate uses the Smoluchowski equation:
    k_on = 4π * D * r_contact * N_A
    where D = D_drug + D_protein (relative diffusion)
    and r_contact = sum of molecular radii

Diffusion coefficients from Stokes-Einstein:
    D = kB * T / (6π * η * r)
    where η = viscosity of compartment
    and r = hydrodynamic radius from molecular weight

For transporter-dependent drugs (gemcitabine):
    The rate-limiting step is transporter-mediated entry,
    not passive diffusion. We model this explicitly.

Output:
    - Binding probability per protein per compartment
    - Effective on-rate and off-rate estimates
    - Selectivity index (tumor vs normal)
    - Competition between targets (if multiple targets in same compartment)
    - Predicted IC50 under cellular conditions

Usage:
    python sim/03_drug_distribution.py --drug gemcitabine
    python sim/03_drug_distribution.py --smiles "CCO" --name ethanol
    python sim/03_drug_distribution.py --drug gemcitabine --dose 1.0
"""

from __future__ import annotations

import json
import math
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parent.parent
SIM_DIR  = ROOT / "data" / "sim"
OUT_DIR  = SIM_DIR / "distribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Physical constants ────────────────────────────────────────────────────────

kB   = 1.380649e-23   # J/K
NA   = 6.02214076e23  # /mol
R    = 8.314          # J/mol/K
T    = 310.15         # K (37°C)
kT   = kB * T         # J
kT_kJ = kT * NA / 1000  # kJ/mol = 2.578

# ── Known drugs database ──────────────────────────────────────────────────────

KNOWN_DRUGS = {
    "gemcitabine": {
        "name":             "Gemcitabine",
        "smiles":           "O=C1N=C(N)C=CN1[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2F",
        "molecular_weight": 263.20,
        "logP":             -1.99,
        "pKa_basic":        3.6,
        "pKa_acidic":       13.0,
        "hbd":              4,
        "hba":              7,
        "psa":              103.5,
        "charge_at_pH74":   0.0,
        "permeability":     "transporter",
        "transporter":      "SLC29A1",
        "known_targets":    ["DCK", "RRM1", "TOP2A"],
        "mechanism":        "DNA antimetabolite — incorporates into replicating DNA",
        "clinical_status":  "FDA approved — first-line PDAC",
    },
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BindingEvent:
    """Predicted drug-protein binding event."""
    drug_name:          str
    target_gene:        str
    target_uniprot:     str
    compartment:        str

    # Concentrations
    drug_conc_uM:       float   # drug concentration in compartment
    protein_conc_nM:    float   # protein concentration estimate

    # Kinetics
    k_on_M_s:           float   # association rate constant M-1 s-1
    k_off_s:            float   # dissociation rate constant s-1
    Kd_apparent_uM:     float   # apparent dissociation constant

    # Probabilities
    p_drug_present:     float   # P(drug reaches this compartment)
    p_protein_open:     float   # P(protein in binding-competent state)
    p_binding:          float   # combined binding probability

    # Thermodynamics
    delta_G_binding:    float   # estimated ΔG of binding kJ/mol
    selectivity_index:  float   # tumor vs normal binding ratio

    # Ensemble context
    dominant_state:     str     # most populated state
    conf_entropy:       float   # conformational entropy of target

    def to_dict(self) -> dict:
        return asdict(self)


# ── Diffusion coefficient computation ─────────────────────────────────────────

def stokes_einstein_D(
    molecular_weight: float,
    viscosity_mPas:   float = 1.0,
) -> float:
    """
    Diffusion coefficient from Stokes-Einstein equation.
    D = kB * T / (6π * η * r)

    Hydrodynamic radius estimated from MW using empirical scaling:
    r ≈ 0.066 * MW^(1/3) nm  (for globular proteins)
    r ≈ 0.052 * MW^(0.4) nm  (for small molecules, more compact)

    Returns D in m²/s
    """
    # Hydrodynamic radius (m)
    if molecular_weight < 1000:
        # Small molecule
        r_nm = 0.052 * (molecular_weight ** 0.4)
    else:
        # Protein
        r_nm = 0.066 * (molecular_weight ** (1/3))

    r_m  = r_nm * 1e-9
    eta  = viscosity_mPas * 1e-3  # mPa·s → Pa·s

    D = kB * T / (6 * math.pi * eta * r_m)
    return D


def compartment_diffusion_coefficient(
    molecular_weight: float,
    compartment_name: str,
    cell_env:         dict,
) -> float:
    """
    Effective diffusion coefficient in a cellular compartment.
    Accounts for viscosity and molecular crowding.

    Crowding reduces diffusion: D_eff = D_free * exp(-α * φ)
    where α ≈ 1.5-2.0 and φ is volume fraction of crowding agents.
    """
    comp       = cell_env.get(compartment_name, cell_env.get("cytoplasm", {}))
    viscosity  = comp.get("viscosity_mPas", 3.0)
    crowding   = comp.get("crowding_factor", 1.8)

    D_free  = stokes_einstein_D(molecular_weight, viscosity)

    # Crowding correction (Phillies equation)
    phi     = min(0.4, crowding * 0.15)
    alpha   = 1.8  # empirical
    D_eff   = D_free * math.exp(-alpha * phi)

    return D_eff


# ── Encounter rate computation ────────────────────────────────────────────────

def smoluchowski_kon(
    D_drug:       float,    # m²/s
    D_protein:    float,    # m²/s
    r_drug_nm:    float,    # nm
    r_protein_nm: float,    # nm
) -> float:
    """
    Diffusion-limited association rate from Smoluchowski equation.
    k_on = 4π * (D_drug + D_protein) * r_contact * N_A

    Returns k_on in M-1 s-1
    """
    D_rel     = D_drug + D_protein
    r_contact = (r_drug_nm + r_protein_nm) * 1e-9  # m

    k_on = 4 * math.pi * D_rel * r_contact * NA
    return k_on


def estimate_koff(
    Kd_uM:  float,
    k_on:   float,
) -> float:
    """
    k_off = Kd * k_on
    Returns k_off in s-1
    """
    Kd_M = Kd_uM * 1e-6
    return Kd_M * k_on


# ── Apparent Kd under cellular conditions ─────────────────────────────────────

def compute_apparent_Kd(
    drug:         dict,
    ensemble:     dict,
    cell_env:     dict,
    compartment:  str,
) -> float:
    """
    Compute apparent Kd under PDAC cellular conditions.

    Kd_apparent = Kd_biochemical * correction_factors

    Corrections:
    1. pH effect: ionization of drug changes effective concentration
    2. Protein conformational ensemble: only binding-competent states
       contribute → Kd_app = Kd_intrinsic / P(binding_competent)
    3. Molecular crowding: increases effective concentration
       (excluded volume pushes drug toward binding site)
    4. Competing ligands: ATP competes with ATP-competitive inhibitors

    Kd_biochemical estimated from druggability score:
    log(Kd) ≈ -2 * druggability + 3  (empirical, in μM)
    High druggability (1.0) → Kd ~1 μM
    Low druggability (0.5) → Kd ~100 μM
    """
    # Base Kd from druggability
    mean_drug = ensemble.get("mean_druggability", 0.5)
    log_Kd    = -2 * mean_drug + 3  # in μM
    Kd_base   = 10 ** log_Kd

    # Correction 1: conformational ensemble
    # Only binding-competent states (active, apo, allosteric_open)
    states = ensemble.get("states", [])
    binding_states = {"active", "apo", "allosteric_open"}
    p_competent = sum(
        s.get("probability", 0)
        for s in states
        if s.get("name") in binding_states
    )
    p_competent = max(0.01, p_competent)
    Kd_ensemble = Kd_base / p_competent  # effective Kd accounting for ensemble

    # Correction 2: pH effect on drug ionization
    comp     = cell_env.get(compartment, cell_env.get("cytoplasm", {}))
    comp_pH  = comp.get("pH", 7.2)
    pKa_b    = drug.get("pKa_basic", 0)
    if pKa_b > 0:
        # Fraction ionized at compartment pH
        f_ion = 1 / (1 + 10 ** (comp_pH - pKa_b))
        # Ionized drug generally binds better to charged active sites
        pH_corr = 1 / (1 + f_ion * 0.5)
    else:
        pH_corr = 1.0
    Kd_pH = Kd_ensemble * pH_corr

    # Correction 3: crowding
    crowding    = comp.get("crowding_factor", 1.8)
    crowd_corr  = 1 / (1 + (crowding - 1) * 0.2)
    Kd_crowding = Kd_pH * crowd_corr

    # Correction 4: ATP competition (for ATP-competitive drugs)
    atp_comp = cell_env.get(compartment, {}).get("atp_conc", 3.0)
    Kd_atp   = comp.get("atp_conc", 3.0) / 1000  # mM → M ... not needed
    # If drug has ATP-like scaffold (nucleoside like gemcitabine)
    if drug.get("psa", 100) > 90 and drug.get("hba", 0) > 5:
        atp_corr    = 1 + atp_comp / 0.1  # Kd(ATP) ≈ 0.1 mM
        Kd_atp_corr = Kd_crowding * atp_corr
    else:
        Kd_atp_corr = Kd_crowding

    return round(Kd_atp_corr, 4)


# ── Protein concentration estimation ─────────────────────────────────────────

def estimate_protein_concentration(
    uniprot_id:  str,
    compartment: str,
    cell_env:    dict,
) -> float:
    """
    Estimate protein concentration in compartment (nM).

    Uses:
    1. scRNA-seq expression level (relative)
    2. Compartment volume to convert to concentration
    3. Typical protein copy number scaling

    Typical nuclear protein: 10,000-100,000 copies/cell
    Cell volume ~2700 fL, nucleus ~800 fL
    10,000 copies / (800e-15 L * 6.022e23) = ~20 nM
    """
    # Try to load expression from GRN data
    try:
        grn_path = ROOT / "data" / "grn" / "intermediate" / "preprocessed.h5ad"
        if grn_path.exists():
            import scanpy as sc
            import numpy as np

            adata  = sc.read_h5ad(grn_path)
            tumor  = adata[adata.obs["leiden"] == "6"]
            report = json.loads(
                (ROOT / "data" / "reports" / f"{uniprot_id}_report.json")
                .read_text()
            )
            gene = report.get("gene_name", "")

            if gene in tumor.var_names:
                if hasattr(tumor.X, "toarray"):
                    X = tumor.X.toarray()
                else:
                    X = np.array(tumor.X)
                idx        = list(tumor.var_names).index(gene)
                mean_expr  = float(X[:, idx].mean())
                # Scale: mean expression 1.0 ≈ ~10,000 copies
                n_copies   = mean_expr * 10000
                comp_vol_L = cell_env.get(compartment, {}).get(
                    "volume_fL", 800) * 1e-15
                conc_M     = n_copies / (comp_vol_L * NA)
                return round(conc_M * 1e9, 2)  # nM
    except Exception:
        pass

    # Fallback: typical nuclear protein
    comp_vol = cell_env.get(compartment, {}).get("volume_fL", 800)
    n_copies = 50000  # typical expressed protein
    vol_L    = comp_vol * 1e-15
    conc_nM  = (n_copies / (vol_L * NA)) * 1e9
    return round(conc_nM, 2)


# ── Binding probability ────────────────────────────────────────────────────────

def compute_binding_probability(
    drug_conc_uM:  float,
    Kd_uM:         float,
    p_competent:   float,
) -> float:
    """
    P(binding) = P(drug_occupies_site) * P(protein_accessible)

    P(drug_occupies_site) = [Drug] / ([Drug] + Kd)  (from receptor theory)
    P(protein_accessible) = p_competent (from ensemble)

    Combined:
    P(binding) = ([Drug] / ([Drug] + Kd)) * p_competent
    """
    p_occupancy = drug_conc_uM / (drug_conc_uM + Kd_uM)
    return round(p_occupancy * p_competent, 6)


# ── Selectivity computation ───────────────────────────────────────────────────

def compute_selectivity(
    p_binding_tumor:  float,
    p_binding_normal: float,
) -> float:
    """
    Selectivity index = P(binding in tumor) / P(binding in normal)
    > 1: tumor-selective
    = 1: no selectivity
    < 1: normal-tissue preferring (bad)
    """
    if p_binding_normal < 1e-10:
        return float('inf') if p_binding_tumor > 0 else 1.0
    return round(p_binding_tumor / p_binding_normal, 3)


# ── Free energy of binding ────────────────────────────────────────────────────

def delta_G_binding(Kd_uM: float) -> float:
    """
    ΔG = RT * ln(Kd)  in kJ/mol
    Kd in M: ΔG = 8.314 * 310.15 * ln(Kd_M) / 1000

    More negative = tighter binding = better drug
    Typical drug: -30 to -50 kJ/mol
    """
    Kd_M = Kd_uM * 1e-6
    dG   = R * T * math.log(Kd_M) / 1000  # kJ/mol
    return round(dG, 2)


# ── Normal cell comparison ────────────────────────────────────────────────────

NORMAL_CELL_ENV = {
    "cytoplasm": {
        "pH": 7.4, "crowding_factor": 1.2,
        "mg_conc": 0.8, "ca_conc": 0.0001,
        "atp_conc": 2.0, "viscosity_mPas": 2.0,
        "volume_fL": 1500,
    },
    "nucleus": {
        "pH": 7.5, "crowding_factor": 2.0,
        "mg_conc": 1.0, "ca_conc": 0.001,
        "atp_conc": 2.0, "viscosity_mPas": 30.0,
        "volume_fL": 300,
    },
    "plasma_membrane": {
        "pH": 7.4, "crowding_factor": 1.5,
        "viscosity_mPas": 80.0, "volume_fL": 0.3,
    },
}

NORMAL_DRUG_CONC = {
    "extracellular":         10.0,
    "cytoplasm":             8.0,    # normal cell — better transporter expression
    "nucleus":               4.0,
    "plasma_membrane":       0.0,
    "endoplasmic_reticulum": 1.0,
    "mitochondria":          1.5,
    "lysosome":              0.1,
}


# ── Main distribution analysis ────────────────────────────────────────────────

def analyze_drug_distribution(
    drug:      dict,
    cell_env:  dict,
    sim_data:  dict,
    ensembles: dict,
    dose_uM:   float = 10.0,
    verbose:   bool  = True,
) -> list[BindingEvent]:
    """
    Full drug distribution analysis across all target proteins
    in all compartments.
    """
    events     = []
    drug_concs = sim_data.get("steady_state", {})

    # Molecular radii for encounter rate
    mw_drug    = drug.get("molecular_weight", 263)
    r_drug_nm  = 0.052 * (mw_drug ** 0.4)

    for uid, ensemble in ensembles.items():
        gene        = ensemble.get("gene_name", uid)
        compartment = ensemble.get("compartment", "nucleus")
        mw_protein  = 50000  # typical ~50 kDa

        # Drug concentration in this compartment
        drug_conc = drug_concs.get(compartment, 0.0)
        if drug_conc < 1e-6:
            continue

        # Diffusion coefficients
        D_drug    = compartment_diffusion_coefficient(
            mw_drug, compartment, cell_env
        )
        D_protein = compartment_diffusion_coefficient(
            mw_protein, compartment, cell_env
        )

        # Encounter rate (Smoluchowski)
        r_prot_nm = 0.066 * (mw_protein ** (1/3))
        k_on      = smoluchowski_kon(D_drug, D_protein, r_drug_nm, r_prot_nm)

        # Apparent Kd under cellular conditions
        Kd_app  = compute_apparent_Kd(drug, ensemble, cell_env, compartment)
        k_off   = estimate_koff(Kd_app, k_on)

        # Fraction of protein in binding-competent states
        states  = ensemble.get("states", [])
        binding_states = {"active", "apo", "allosteric_open"}
        p_open  = sum(
            s.get("probability", 0)
            for s in states
            if s.get("name") in binding_states
        )
        p_open  = max(0.01, p_open)

        # Dominant conformational state
        dominant = max(states, key=lambda s: s.get("probability", 0),
                       default={"name": "unknown"})

        # Binding probability in PDAC tumor
        p_bind_tumor = compute_binding_probability(drug_conc, Kd_app, p_open)

        # Binding probability in normal cell (for selectivity)
        normal_conc    = NORMAL_DRUG_CONC.get(compartment, drug_conc * 0.8)
        normal_comp    = NORMAL_CELL_ENV.get(
            compartment, NORMAL_CELL_ENV.get("cytoplasm", {}))
        normal_crowd   = normal_comp.get("crowding_factor", 1.2)
        Kd_normal      = Kd_app * (1 + (normal_crowd - 1) * 0.1)
        p_bind_normal  = compute_binding_probability(
            normal_conc, Kd_normal, p_open * 0.9
        )

        selectivity = compute_selectivity(p_bind_tumor, p_bind_normal)

        # Protein concentration
        prot_conc_nM = estimate_protein_concentration(
            uid, compartment, cell_env
        )

        # Free energy of binding
        dG = delta_G_binding(Kd_app)

        event = BindingEvent(
            drug_name         = drug.get("name", "unknown"),
            target_gene       = gene,
            target_uniprot    = uid,
            compartment       = compartment,
            drug_conc_uM      = round(drug_conc, 4),
            protein_conc_nM   = prot_conc_nM,
            k_on_M_s          = round(k_on, 2),
            k_off_s           = round(k_off, 6),
            Kd_apparent_uM    = Kd_app,
            p_drug_present    = round(drug_conc / dose_uM, 4),
            p_protein_open    = round(p_open, 4),
            p_binding         = p_bind_tumor,
            delta_G_binding   = dG,
            selectivity_index = selectivity,
            dominant_state    = dominant.get("name", "unknown"),
            conf_entropy      = ensemble.get("conformational_entropy", 0),
        )
        events.append(event)

    return events


def print_distribution_report(
    events:   list[BindingEvent],
    drug:     dict,
    dose_uM:  float,
):
    print(f"\n{'='*70}")
    print(f"  SIM-03: Drug Distribution Report — {drug.get('name')}")
    print(f"  Dose: {dose_uM} μM  |  PDAC tumor cell (cluster 6)")
    print(f"{'='*70}")

    if not events:
        print("  No binding events above threshold.")
        return

    # Sort by binding probability
    events_sorted = sorted(events, key=lambda e: -e.p_binding)

    print(f"\n  {'Target':<10} {'Compart':<22} {'[Drug]':>8} "
          f"{'Kd_app':>8} {'P(bind)':>8} {'ΔG':>7} {'Sel.':>6}")
    print(f"  {'-'*10} {'-'*22} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*7} {'-'*6}")

    for e in events_sorted:
        print(f"  {e.target_gene:<10} {e.compartment:<22} "
              f"{e.drug_conc_uM:>7.3f}μ  "
              f"{e.Kd_apparent_uM:>7.2f}μ  "
              f"{e.p_binding:>8.4f}  "
              f"{e.delta_G_binding:>+6.1f}  "
              f"{e.selectivity_index:>6.2f}x")

    print(f"\n{'─'*70}")
    print(f"  DETAILED ANALYSIS")
    print(f"{'─'*70}")

    for e in events_sorted:
        sel_str = (f"TUMOR SELECTIVE ({e.selectivity_index:.1f}x)"
                   if e.selectivity_index > 2
                   else f"Non-selective ({e.selectivity_index:.1f}x)")
        print(f"\n  {e.target_gene} ({e.target_uniprot})")
        print(f"    Compartment    : {e.compartment}")
        print(f"    Drug conc      : {e.drug_conc_uM:.4f} μM")
        print(f"    Protein conc   : {e.protein_conc_nM:.1f} nM")
        print(f"    k_on           : {e.k_on_M_s:.2e} M⁻¹s⁻¹")
        print(f"    k_off          : {e.k_off_s:.2e} s⁻¹")
        print(f"    Kd (apparent)  : {e.Kd_apparent_uM:.2f} μM")
        print(f"    ΔG (binding)   : {e.delta_G_binding:+.1f} kJ/mol")
        print(f"    P(binding)     : {e.p_binding:.4f}")
        print(f"    P(open state)  : {e.p_protein_open:.4f} "
              f"[dominant: {e.dominant_state}]")
        print(f"    Conf. entropy  : {e.conf_entropy:.4f} nats")
        print(f"    Selectivity    : {sel_str}")

    # Summary statistics
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    best  = events_sorted[0]
    worst = events_sorted[-1]
    mean_sel = np.mean([e.selectivity_index for e in events
                        if e.selectivity_index < 1000])
    total_p  = sum(e.p_binding for e in events)

    print(f"  Most likely binding target : {best.target_gene} "
          f"(P={best.p_binding:.4f})")
    print(f"  Least likely target        : {worst.target_gene} "
          f"(P={worst.p_binding:.4f})")
    print(f"  Mean selectivity index     : {mean_sel:.2f}x")
    print(f"  Total binding probability  : {min(1.0, total_p):.4f}")
    print(f"\n  Pharmacological interpretation:")
    high_sel = [e for e in events if e.selectivity_index > 2]
    if high_sel:
        print(f"  → {len(high_sel)} targets show tumor-selective binding:")
        for e in sorted(high_sel, key=lambda x: -x.selectivity_index):
            print(f"    {e.target_gene}: {e.selectivity_index:.1f}x "
                  f"tumor/normal selectivity")
    else:
        print(f"  → No strongly tumor-selective binding detected at this dose")
        print(f"  → Consider dose optimization or structural modification")


# ── Dose-response curve ───────────────────────────────────────────────────────

def compute_dose_response(
    drug:      dict,
    cell_env:  dict,
    sim_data:  dict,
    ensembles: dict,
    doses_uM:  list = None,
) -> dict:
    """
    Compute binding probability across a range of doses.
    Used to estimate IC50 under cellular conditions.
    """
    if doses_uM is None:
        doses_uM = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    response = {uid: {"doses": doses_uM, "p_binding": []}
                for uid in ensembles}

    for dose in doses_uM:
        # Scale drug concentrations proportionally
        base_conc  = sim_data.get("steady_state", {})
        base_dose  = sim_data.get("dose_uM", 10.0)
        scale      = dose / max(base_dose, 0.001)
        scaled_conc = {k: v * scale for k, v in base_conc.items()}

        sim_scaled = {**sim_data, "steady_state": scaled_conc,
                      "dose_uM": dose}
        events     = analyze_drug_distribution(
            drug, cell_env, sim_scaled, ensembles, dose, verbose=False
        )

        for e in events:
            if e.target_uniprot in response:
                response[e.target_uniprot]["p_binding"].append(e.p_binding)
            else:
                response[e.target_uniprot] = {
                    "doses": doses_uM,
                    "p_binding": [e.p_binding],
                }

    # Estimate IC50 for each target
    for uid, data in response.items():
        doses = data["doses"]
        probs = data["p_binding"]
        if len(probs) < 2:
            data["IC50_uM"] = None
            continue
        # Find dose where P(binding) ~ 0.5 * max(P)
        max_p    = max(probs) if probs else 0
        half_max = max_p * 0.5
        IC50     = None
        for i in range(len(probs) - 1):
            if probs[i] <= half_max <= probs[i+1]:
                # Linear interpolation
                t   = (half_max - probs[i]) / (probs[i+1] - probs[i] + 1e-10)
                IC50 = doses[i] + t * (doses[i+1] - doses[i])
                break
        data["IC50_uM"] = round(IC50, 3) if IC50 else None

    return response


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    drug_name:   str   = "gemcitabine",
    smiles:      str   = None,
    mol_name:    str   = None,
    dose_uM:     float = 10.0,
    dose_response: bool = True,
):
    print("=" * 70)
    print("  SIM-03: Drug Distribution Model")
    print("=" * 70)

    # ── Load drug ─────────────────────────────────────────────────────────
    if smiles:
        print(f"\nComputing properties for custom molecule ({mol_name})...")
        # drug props loaded from saved JSON file
        # Try to load from saved props
        prop_path = SIM_DIR / f"molecule_{mol_name}_props.json"
        if prop_path.exists():
            drug = json.loads(prop_path.read_text())
        else:
            drug = {
                "name":   mol_name or "custom",
                "smiles": smiles,
                "molecular_weight": 300,
                "logP": 2.0,
                "pKa_basic": 0,
                "pKa_acidic": 0,
                "hbd": 2,
                "hba": 4,
                "psa": 60,
                "charge_at_pH74": 0,
                "permeability": "medium",
            }
    elif drug_name in KNOWN_DRUGS:
        drug = KNOWN_DRUGS[drug_name]
        print(f"\nUsing {drug['name']} (pre-defined)")
    else:
        print(f"Drug '{drug_name}' not found. Use --smiles to provide custom molecule.")
        return

    # ── Load cell environment ─────────────────────────────────────────────
    env_path = SIM_DIR / "cell_environment.json"
    if not env_path.exists():
        print("ERROR: Run sim/01_cell_environment.py first")
        return
    env_data  = json.loads(env_path.read_text())
    cell_env  = env_data["cell_environment"]
    sim_data  = env_data["simulation"]

    print(f"  Cell environment loaded: {len(cell_env)} compartments")
    print(f"  Drug steady-state concentrations:")
    for comp, conc in sorted(sim_data["steady_state"].items(),
                              key=lambda x: -x[1]):
        print(f"    {comp:<25} {conc:.4f} μM")

    # ── Load protein ensembles ────────────────────────────────────────────
    ensemble_dir = SIM_DIR / "ensembles"
    ensembles    = {}
    for f in ensemble_dir.glob("*_ensemble.json"):
        uid  = f.stem.replace("_ensemble", "")
        data = json.loads(f.read_text())
        ensembles[uid] = data
    print(f"\n  Protein ensembles loaded: {len(ensembles)}")

    if not ensembles:
        print("ERROR: Run sim/02_protein_ensemble.py --all-targets first")
        return

    # ── Analyze distribution ──────────────────────────────────────────────
    print(f"\nAnalyzing drug distribution at {dose_uM} μM dose...")
    events = analyze_drug_distribution(
        drug, cell_env, sim_data, ensembles, dose_uM
    )

    print_distribution_report(events, drug, dose_uM)

    # ── Dose-response ─────────────────────────────────────────────────────
    if dose_response and events:
        print(f"\n{'='*70}")
        print(f"  DOSE-RESPONSE ANALYSIS")
        print(f"{'='*70}")
        dr = compute_dose_response(drug, cell_env, sim_data, ensembles)
        print(f"\n  Estimated IC50 under PDAC cellular conditions:")
        for uid, data in dr.items():
            gene = ensembles.get(uid, {}).get("gene_name", uid)
            ic50 = data.get("IC50_uM")
            if ic50:
                print(f"    {gene:<12} IC50 ~ {ic50:.3f} μM")
            else:
                print(f"    {gene:<12} IC50 not determinable in this range")

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        "drug":          drug,
        "dose_uM":       dose_uM,
        "binding_events":[e.to_dict() for e in events],
        "n_events":      len(events),
    }
    out_path = OUT_DIR / f"{drug.get('name','drug').lower()}_distribution.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved to {out_path}")

    print(f"\n{'='*70}")
    print(f"  SIM-03 complete. Ready for SIM-04 (Binding Probability Model)")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SIM-03: Drug Distribution and Binding Probability"
    )
    parser.add_argument("--drug",    default="gemcitabine",
                        help="Drug name (default: gemcitabine)")
    parser.add_argument("--smiles",  help="SMILES for custom molecule")
    parser.add_argument("--name",    help="Name for custom molecule")
    parser.add_argument("--dose",    type=float, default=10.0,
                        help="Dose in μM (default: 10.0)")
    parser.add_argument("--no-dr",   action="store_true",
                        help="Skip dose-response analysis")
    args = parser.parse_args()

    main(
        drug_name    = args.drug,
        smiles       = args.smiles,
        mol_name     = args.name,
        dose_uM      = args.dose,
        dose_response= not args.no_dr,
    )