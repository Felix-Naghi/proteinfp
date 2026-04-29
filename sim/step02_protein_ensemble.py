"""
sim/02_protein_ensemble.py
───────────────────────────
Module SIM-02 — Protein Ensemble Model

Represents each protein not as a single static structure but as a
probability distribution over conformational states, weighted by
the PDAC cell environment from Module SIM-01.

Core concept:
    Standard docking uses one rigid structure. Reality: proteins
    exist in ensembles of conformations. The population of each
    conformation depends on:
        - Temperature (always 37°C here)
        - pH of the compartment the protein lives in
        - Molecular crowding (shifts equilibria)
        - Binding partner occupancy (from GRN)
        - Post-translational modification state
        - Ion concentrations (especially Mg2+, Ca2+, Zn2+)

    Boltzmann statistics govern conformation populations:
        P(state_i) = exp(-ΔGi / RT) / Z
        where Z = sum(exp(-ΔGj / RT)) for all j
        and ΔGi is the free energy of state i

    ΔGi is decomposed into:
        ΔGi = ΔG_folding + ΔG_pH + ΔG_crowding + ΔG_ions

    Each term is computed from your ProteinFP ESM-2 embeddings,
    active site data, and the cell environment model.

Output per protein:
    - Conformational ensemble (N states with probabilities)
    - Effective binding pocket geometry (ensemble-averaged)
    - Allosteric communication network (from ESM-2 attention)
    - Ligandability score under PDAC conditions
    - Comparison: PDAC cell vs normal cell conformational shift

Usage:
    python sim/02_protein_ensemble.py --uniprot Q9HAW4
    python sim/02_protein_ensemble.py --uniprot Q9HAW4 --drug gemcitabine
    python sim/02_protein_ensemble.py --all-targets
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

ROOT      = Path(__file__).resolve().parent.parent
INTER     = ROOT / "data" / "intermediate"
SIM_DIR   = ROOT / "data" / "sim"
OUT_DIR   = SIM_DIR / "ensembles"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Physical constants ────────────────────────────────────────────────────────

R   = 8.314      # J/mol/K
T   = 310.15     # K (37°C)
RT  = R * T      # J/mol
kT  = RT / 1000  # kJ/mol = 2.578 kJ/mol at 37°C

# ── Conformational state definitions ─────────────────────────────────────────

# Every protein is modeled with these conformational macro-states.
# Fine-grained MD would have millions of microstates — we use
# biologically meaningful macrostates that affect drug binding.

CONF_STATES = [
    "active",          # catalytically active, open pocket
    "inactive",        # catalytically inactive, closed/collapsed pocket
    "apo",             # ligand-free, intermediate pocket
    "partially_open",  # partially open — intermediate binding competence
    "allosteric_open", # allosteric site occupied, pocket geometry changed
]


@dataclass
class ConformationalState:
    """A single conformational macrostate of a protein."""
    name:              str
    delta_G_kJ_mol:    float      # free energy relative to apo state
    pocket_volume_A3:  float      # binding pocket volume in this state
    pocket_shape:      float      # shape complementarity score 0-1
    druggability:      float      # druggability in this state
    probability:       float = 0.0  # Boltzmann probability (computed)
    accessible:        bool  = True # is this state accessible under PDAC conditions


@dataclass
class ProteinEnsemble:
    """
    Full conformational ensemble for a protein under PDAC conditions.
    """
    uniprot_id:          str
    gene_name:           str
    compartment:         str       # where this protein lives
    length:              int
    mean_plddt:          float

    # Conformational states
    states:              list[ConformationalState] = field(default_factory=list)

    # Ensemble-averaged properties
    mean_pocket_volume:  float = 0.0
    mean_druggability:   float = 0.0
    conformational_entropy: float = 0.0  # Shannon entropy of state distribution

    # Cell environment effects
    pH_effect_kJ:        float = 0.0   # ΔΔG from PDAC vs normal pH
    crowding_effect_kJ:  float = 0.0   # ΔΔG from molecular crowding
    ion_effect_kJ:       float = 0.0   # ΔΔG from altered ion concentrations

    # ESM-2 derived properties
    embedding_norm:      float = 0.0
    n_functional_res:    int   = 0
    allostery_score:     float = 0.0   # allosteric communication strength

    # Comparison to normal cell
    pdac_vs_normal_ddG:  float = 0.0   # ΔΔG PDAC - normal (+ = more stable in PDAC)
    ligandability_shift: float = 0.0   # change in druggability PDAC vs normal

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ── Free energy computation ───────────────────────────────────────────────────

def compute_pH_effect(
    protein_data:  dict,
    compartment_pH: float,
    normal_pH:     float = 7.4,
) -> float:
    """
    Compute ΔΔG from pH change using electrostatic model.

    In the PDAC cell environment, pH differs from normal:
    - Cytoplasm: 7.2 vs 7.4 normal (0.2 unit drop)
    - Extracellular: 6.7 vs 7.4 (0.7 unit drop)
    - Nucleus: 7.35 vs 7.5 (0.15 unit drop)

    Effect on protein stability depends on:
    - Number of titratable residues (His, Asp, Glu, Lys, Arg)
    - Their pKa values
    - Whether they're in the active site

    We use a simplified model: ΔΔG ≈ n_titratable * ΔpH * f_active
    where f_active is fraction of titratable residues in/near active site
    """
    sequence = protein_data.get("sequence", "")
    if not sequence:
        return 0.0

    # Count titratable residues
    titratable = {"H": 6.0, "D": 3.9, "E": 4.3, "K": 10.5, "R": 12.5}
    n_titratable = sum(sequence.count(aa) for aa in titratable)

    # Fraction in active site (from active site predictions)
    active_residues = set(
        r.get("residue_number", 0)
        for r in protein_data.get("active_residues", [])
        if r.get("confidence") in ("HIGH", "MEDIUM")
    )
    total_res = len(sequence)
    f_active  = len(active_residues) / max(total_res, 1)

    # ΔpH effect
    delta_pH = compartment_pH - normal_pH

    # ΔΔG ≈ n_titratable * |ΔpH| * RT * ln(10) * f_active_weight
    # RT * ln(10) = 5.94 kJ/mol at 37°C
    RTln10 = RT * math.log(10) / 1000  # kJ/mol
    ddG = n_titratable * abs(delta_pH) * RTln10 * (1 + f_active * 2)

    # Sign: pH drop generally destabilizes proteins (positive ΔΔG)
    # unless protein is adapted to acidic conditions
    return round(ddG if delta_pH < 0 else -ddG * 0.5, 3)


def compute_crowding_effect(
    protein_data:   dict,
    crowding_factor: float,
    normal_crowding: float = 1.0,
) -> float:
    """
    Compute ΔΔG from molecular crowding using excluded volume theory.

    Crowding has two opposing effects:
    1. Stabilizes compact folded states (excluded volume effect)
       ΔG_crowd ≈ -kT * N * φ^(2/3) where φ is volume fraction
    2. Destabilizes binding by competing for surface area

    Net effect depends on protein size:
    - Large proteins: crowding generally stabilizes
    - Small proteins: less effect
    - Intrinsically disordered regions: crowding compacts them

    PDAC cells have higher crowding (factor 1.8 cytoplasm vs 1.0 normal)
    which shifts conformational equilibria toward more compact states.
    """
    length = protein_data.get("length", 300)

    # Volume fraction approximation
    phi_pdac   = min(0.4, crowding_factor * 0.15)
    phi_normal = min(0.4, normal_crowding * 0.15)

    # Stabilization energy from excluded volume
    # ΔG_excl ≈ -kT * (length/100)^0.6 * phi
    size_factor = (length / 100) ** 0.6
    dG_pdac    = -kT * size_factor * phi_pdac
    dG_normal  = -kT * size_factor * phi_normal

    ddG = dG_pdac - dG_normal  # negative = more stable in PDAC (good for pocket)

    return round(ddG, 3)


def compute_ion_effect(
    protein_data: dict,
    comp_mg:      float,
    comp_ca:      float,
    normal_mg:    float = 0.8,
    normal_ca:    float = 0.0001,
) -> float:
    """
    Compute ΔΔG from altered ion concentrations.

    Key ions for protein stability:
    - Mg2+: stabilizes ATP-binding proteins and nucleic acid binding
      PDAC has reduced Mg2+ (0.5 vs 0.8 mM normal)
    - Ca2+: elevated in PDAC cytoplasm (signaling)
      Can destabilize or activate specific proteins

    Uses simplified electrostatic binding model:
    ΔG_ion = -RT * ln(1 + [Ion]/Kd_ion)
    where Kd_ion is the metal binding dissociation constant
    """
    # Check if protein has metal binding (from motifs)
    active_residues  = protein_data.get("active_residues", [])
    motif_types      = []
    for r in active_residues:
        motif_types.extend(r.get("motifs", []))

    has_metal  = any("zinc" in m.lower() or "metal" in m.lower()
                     or "ghkl" in m.lower()
                     for m in motif_types)
    has_atp    = any("atp" in m.lower() or "p_loop" in m.lower() or
                     "walker" in m.lower()
                     for m in motif_types)

    ddG = 0.0

    if has_metal or has_atp:
        # Mg2+ effect on ATP-binding proteins
        # Kd(Mg-ATP) ≈ 0.1 mM
        Kd_mg = 0.1
        dG_mg_pdac   = -kT * math.log(1 + comp_mg   / Kd_mg)
        dG_mg_normal = -kT * math.log(1 + normal_mg  / Kd_mg)
        ddG += (dG_mg_pdac - dG_mg_normal)

    if has_metal:
        # Ca2+ effect on metal-binding proteins
        Kd_ca = 0.001  # mM (high affinity Ca binding)
        dG_ca_pdac   = -kT * math.log(1 + comp_ca   / Kd_ca)
        dG_ca_normal = -kT * math.log(1 + normal_ca  / Kd_ca)
        ddG += (dG_ca_pdac - dG_ca_normal) * 0.5

    return round(ddG, 3)


# ── Conformational state builder ──────────────────────────────────────────────

def build_conformational_states(
    protein_data:   dict,
    cell_env_data:  dict,
    compartment:    str,
) -> list[ConformationalState]:
    """
    Build conformational state ensemble using:
    1. ProteinFP binding pocket data for pocket volumes/druggability
    2. Allosteric site data for communication networks
    3. ESM-2 embedding variance for disorder prediction
    4. Cell environment for Boltzmann weighting

    The free energies are estimated from:
    - Active state: baseline from druggability score
    - Inactive state: +ΔG from pocket collapse
    - Apo state: set to 0 (reference)
    - Partially open: intermediate
    - Allosteric open: depends on allosteric site presence
    """
    pockets    = protein_data.get("binding_pockets", [])
    allosteric = protein_data.get("allosteric_sites", [])
    plddt      = protein_data.get("mean_plddt", 70.0)

    # Base pocket properties
    if pockets:
        base_vol  = pockets[0].get("volume_A3", 500)
        base_drug = pockets[0].get("druggability_score", 0.5)
    else:
        base_vol  = 300.0
        base_drug = 0.3

    # pLDDT-based conformational flexibility
    # Low pLDDT → more disordered → more conformational states accessible
    flexibility = max(0.1, (100 - plddt) / 100)

    # Get environment properties for this compartment
    comp_data   = cell_env_data.get(compartment, cell_env_data.get("cytoplasm", {}))
    comp_pH     = comp_data.get("pH", 7.2)
    crowding    = comp_data.get("crowding_factor", 1.8)
    comp_mg     = comp_data.get("mg_conc", 0.5)
    comp_ca     = comp_data.get("ca_conc", 0.0001)

    # Build states
    states = []

    # ── Active state ──────────────────────────────────────────────────────
    # High druggability → active state more populated
    dG_active = -kT * math.log(max(0.01, base_drug)) * 2
    # pH destabilizes active state if acidic
    if comp_pH < 7.0:
        dG_active += abs(7.0 - comp_pH) * 0.5 * kT

    states.append(ConformationalState(
        name            = "active",
        delta_G_kJ_mol  = round(dG_active, 3),
        pocket_volume_A3= base_vol,
        pocket_shape    = base_drug,
        druggability    = base_drug,
    ))

    # ── Apo state (reference, ΔG = 0) ────────────────────────────────────
    states.append(ConformationalState(
        name             = "apo",
        delta_G_kJ_mol   = 0.0,
        pocket_volume_A3 = base_vol * 0.85,
        pocket_shape     = base_drug * 0.85,
        druggability     = base_drug * 0.85,
    ))

    # ── Partially open state ──────────────────────────────────────────────
    states.append(ConformationalState(
        name             = "partially_open",
        delta_G_kJ_mol   = round(kT * 0.5, 3),
        pocket_volume_A3 = base_vol * 0.7,
        pocket_shape     = base_drug * 0.7,
        druggability     = base_drug * 0.7,
    ))

    # ── Inactive state ────────────────────────────────────────────────────
    # Penalized by crowding (compact states favored)
    dG_inactive = kT * 2.0 - crowding_effect(crowding)
    states.append(ConformationalState(
        name             = "inactive",
        delta_G_kJ_mol   = round(dG_inactive, 3),
        pocket_volume_A3 = base_vol * 0.3,
        pocket_shape     = base_drug * 0.2,
        druggability     = base_drug * 0.2,
        accessible       = flexibility > 0.2,
    ))

    # ── Allosteric open state ─────────────────────────────────────────────
    if allosteric:
        allosteric_score = allosteric[0].get("correlation", 0.9)
        dG_allosteric    = -kT * math.log(max(0.01, allosteric_score))
        # Allosteric state has modified pocket
        states.append(ConformationalState(
            name             = "allosteric_open",
            delta_G_kJ_mol   = round(dG_allosteric, 3),
            pocket_volume_A3 = base_vol * 1.15,  # slightly larger
            pocket_shape     = base_drug * 0.9,
            druggability     = base_drug * 0.95,
        ))

    return states


def crowding_effect(crowding_factor: float) -> float:
    """Free energy stabilization from crowding (kJ/mol)"""
    return kT * math.log(max(1.0, crowding_factor)) * 0.5


# ── Boltzmann weighting ───────────────────────────────────────────────────────

def apply_boltzmann_weights(
    states:       list[ConformationalState],
    ddG_pH:       float,
    ddG_crowding: float,
    ddG_ions:     float,
) -> list[ConformationalState]:
    """
    Apply Boltzmann statistics to compute state probabilities.

    P(i) = exp(-ΔGi_eff / kT) / Z
    ΔGi_eff = ΔGi + environment_corrections

    Environment corrections affect all states equally (shifts baseline)
    but state-specific modifiers account for differential effects.
    """
    # Environment correction to total free energy
    env_correction = ddG_pH * 0.3 + ddG_crowding + ddG_ions * 0.5

    # Compute Boltzmann factors
    boltzmann = []
    for state in states:
        if not state.accessible:
            boltzmann.append(0.0)
            continue
        dG_eff = state.delta_G_kJ_mol + env_correction
        bf     = math.exp(-dG_eff / kT)
        boltzmann.append(bf)

    # Partition function
    Z = sum(boltzmann)
    if Z <= 0:
        Z = 1.0

    # Assign probabilities
    for state, bf in zip(states, boltzmann):
        state.probability = round(bf / Z, 4)

    return states


# ── Ensemble averaging ────────────────────────────────────────────────────────

def compute_ensemble_averages(
    states: list[ConformationalState],
) -> tuple[float, float, float]:
    """
    Compute ensemble-averaged pocket volume, druggability,
    and conformational entropy.

    <X> = sum_i P(i) * X(i)     (ensemble average)
    S   = -sum_i P(i) * ln(P(i)) (Shannon entropy of state distribution)
    """
    mean_vol  = sum(s.probability * s.pocket_volume_A3 for s in states)
    mean_drug = sum(s.probability * s.druggability     for s in states)

    # Conformational entropy (nats)
    entropy   = -sum(
        s.probability * math.log(s.probability + 1e-10)
        for s in states
        if s.probability > 0
    )

    return round(mean_vol, 2), round(mean_drug, 4), round(entropy, 4)


# ── ESM-2 derived properties ──────────────────────────────────────────────────

def extract_esm2_properties(uid: str) -> dict:
    """
    Extract ESM-2 embedding properties for allosteric communication
    and disorder prediction.
    """
    esm_path = INTER / f"{uid}_esm2.json"
    if not esm_path.exists():
        return {"norm": 0.0, "n_functional": 0, "allostery": 0.0}

    esm = json.loads(esm_path.read_text())

    norm         = esm.get("embedding_norm", 0.0)
    n_functional = len(esm.get("predicted_functional_residues", []))
    length       = esm.get("sequence_length", 300)

    # Allostery score: ratio of functional residues to total
    # Higher ratio = more distributed functional network = more allosteric
    allostery = min(1.0, n_functional / max(length, 1) * 3)

    return {
        "norm":        norm,
        "n_functional": n_functional,
        "allostery":   round(allostery, 4),
    }


# ── Normal cell baseline ──────────────────────────────────────────────────────

def build_normal_cell_env() -> dict:
    """
    Simplified normal pancreatic ductal cell environment
    for comparison with PDAC.
    """
    return {
        "cytoplasm": {
            "pH":             7.4,
            "crowding_factor": 1.2,
            "mg_conc":        0.8,
            "ca_conc":        0.0001,
        },
        "nucleus": {
            "pH":             7.5,
            "crowding_factor": 2.0,
            "mg_conc":        1.0,
            "ca_conc":        0.001,
        },
    }


# ── Main ensemble builder ─────────────────────────────────────────────────────

def build_protein_ensemble(
    uid:          str,
    cell_env:     dict,
    normal_env:   dict,
    verbose:      bool = True,
) -> Optional[ProteinEnsemble]:
    """
    Build full conformational ensemble for a protein.
    """
    # Load ProteinFP report
    report_path = ROOT / "data" / "reports" / f"{uid}_report.json"
    if not report_path.exists():
        print(f"  {uid}: no report found — run full pipeline first")
        return None

    report = json.loads(report_path.read_text())

    # Load active sites
    active_path = INTER / f"{uid}_active_sites.json"
    active_data = json.loads(active_path.read_text()) \
                  if active_path.exists() else {}

    # Merge into single protein_data dict
    protein_data = {
        **report,
        "active_residues": active_data.get("active_residues", []),
        "length":          report.get("length", 300),
        "mean_plddt":      report.get("mean_plddt", 70.0),
        "sequence":        active_data.get("sequence", ""),
    }

    gene   = report.get("gene_name", uid)
    length = protein_data["length"]
    plddt  = protein_data["mean_plddt"]

    # Determine compartment
    compartment = _infer_compartment(report)

    # Get compartment-specific environment
    comp_data    = cell_env.get(compartment, cell_env.get("cytoplasm", {}))
    comp_pH      = comp_data.get("pH", 7.2)
    crowding     = comp_data.get("crowding_factor", 1.8)
    comp_mg      = comp_data.get("mg_conc", 0.5)
    comp_ca      = comp_data.get("ca_conc", 0.0001)

    normal_comp  = normal_env.get(
        compartment, normal_env.get("cytoplasm", {})
    )
    normal_pH    = normal_comp.get("pH", 7.4)
    normal_crowd = normal_comp.get("crowding_factor", 1.2)
    normal_mg    = normal_comp.get("mg_conc", 0.8)
    normal_ca    = normal_comp.get("ca_conc", 0.0001)

    # Compute environment effects
    ddG_pH      = compute_pH_effect(protein_data, comp_pH, normal_pH)
    ddG_crowd   = compute_crowding_effect(protein_data, crowding, normal_crowd)
    ddG_ions    = compute_ion_effect(protein_data, comp_mg, comp_ca,
                                      normal_mg, normal_ca)

    # Build conformational states
    states = build_conformational_states(protein_data, cell_env, compartment)

    # Apply Boltzmann weights under PDAC conditions
    states = apply_boltzmann_weights(states, ddG_pH, ddG_crowd, ddG_ions)

    # Ensemble averages under PDAC conditions
    mean_vol, mean_drug, conf_entropy = compute_ensemble_averages(states)

    # Build normal cell baseline for comparison
    states_normal = build_conformational_states(protein_data, normal_env,
                                                 compartment)
    states_normal = apply_boltzmann_weights(states_normal, 0, 0, 0)
    _, mean_drug_normal, _ = compute_ensemble_averages(states_normal)

    # ESM-2 properties
    esm2 = extract_esm2_properties(uid)

    # ΔΔG total PDAC vs normal
    ddG_total = ddG_pH + ddG_crowd + ddG_ions

    # Ligandability shift
    ligandability_shift = mean_drug - mean_drug_normal

    ensemble = ProteinEnsemble(
        uniprot_id           = uid,
        gene_name            = gene,
        compartment          = compartment,
        length               = length,
        mean_plddt           = plddt,
        states               = states,
        mean_pocket_volume   = mean_vol,
        mean_druggability    = mean_drug,
        conformational_entropy = conf_entropy,
        pH_effect_kJ         = ddG_pH,
        crowding_effect_kJ   = ddG_crowd,
        ion_effect_kJ        = ddG_ions,
        embedding_norm       = esm2["norm"],
        n_functional_res     = esm2["n_functional"],
        allostery_score      = esm2["allostery"],
        pdac_vs_normal_ddG   = round(ddG_total, 3),
        ligandability_shift  = round(ligandability_shift, 4),
    )

    if verbose:
        _print_ensemble(ensemble)

    # Save
    out_path = OUT_DIR / f"{uid}_ensemble.json"
    out_path.write_text(json.dumps(ensemble.to_dict(), indent=2))

    return ensemble


def _infer_compartment(report: dict) -> str:
    """Infer primary compartment from GO cellular component terms."""
    go_cc = [t.get("go_name", "").lower()
             for t in report.get("go_terms_cc", [])]

    if any("nucleus" in g or "chromatin" in g for g in go_cc):
        return "nucleus"
    if any("mitochondri" in g for g in go_cc):
        return "mitochondria"
    if any("endoplasmic" in g or "er " in g for g in go_cc):
        return "endoplasmic_reticulum"
    if any("lysosom" in g for g in go_cc):
        return "lysosome"
    if any("membrane" in g or "surface" in g for g in go_cc):
        return "plasma_membrane"
    return "cytoplasm"


def _print_ensemble(e: ProteinEnsemble):
    print(f"\n  {'─'*60}")
    print(f"  {e.gene_name} ({e.uniprot_id}) — {e.compartment}")
    print(f"  {'─'*60}")
    print(f"  Length: {e.length} aa  pLDDT: {e.mean_plddt:.1f}  "
          f"ESM-2 norm: {e.embedding_norm:.2f}")
    print(f"\n  Conformational states (Boltzmann weighted):")
    print(f"  {'State':<20} {'P':>6}  {'Volume':>8}  {'Drugg':>6}")
    for s in sorted(e.states, key=lambda x: -x.probability):
        print(f"  {s.name:<20} {s.probability:>6.3f}  "
              f"{s.pocket_volume_A3:>7.0f}Å³  {s.druggability:>6.3f}")
    print(f"\n  Ensemble averages:")
    print(f"    Mean pocket volume : {e.mean_pocket_volume:.0f} Å³")
    print(f"    Mean druggability  : {e.mean_druggability:.4f}")
    print(f"    Conformational entropy: {e.conformational_entropy:.4f} nats")
    print(f"\n  Cell environment effects (vs normal ductal):")
    print(f"    ΔΔG(pH)      : {e.pH_effect_kJ:+.3f} kJ/mol")
    print(f"    ΔΔG(crowding): {e.crowding_effect_kJ:+.3f} kJ/mol")
    print(f"    ΔΔG(ions)    : {e.ion_effect_kJ:+.3f} kJ/mol")
    print(f"    ΔΔG(total)   : {e.pdac_vs_normal_ddG:+.3f} kJ/mol")
    print(f"\n  Ligandability shift PDAC vs normal: "
          f"{e.ligandability_shift:+.4f}")
    if e.ligandability_shift > 0:
        print(f"    → Protein MORE druggable in PDAC tumor cell")
    elif e.ligandability_shift < -0.05:
        print(f"    → Protein LESS druggable in PDAC tumor cell")
    else:
        print(f"    → Similar druggability in PDAC vs normal")


# ── Molecule property computation (from SMILES) ───────────────────────────────

def compute_molecule_properties(smiles: str, name: str = "unknown") -> dict:
    """
    Compute physicochemical properties from SMILES string.
    Uses RDKit if available, falls back to rule-based estimation.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        props = {
            "name":             name,
            "smiles":           smiles,
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logP":             round(Crippen.MolLogP(mol), 3),
            "hbd":              rdMolDescriptors.CalcNumHBD(mol),
            "hba":              rdMolDescriptors.CalcNumHBA(mol),
            "psa":              round(Descriptors.TPSA(mol), 2),
            "rotatable_bonds":  rdMolDescriptors.CalcNumRotatableBonds(mol),
            "aromatic_rings":   rdMolDescriptors.CalcNumAromaticRings(mol),
            "heavy_atoms":      mol.GetNumHeavyAtoms(),
            "lipinski_ok":      (
                Descriptors.MolWt(mol) <= 500 and
                Crippen.MolLogP(mol) <= 5 and
                rdMolDescriptors.CalcNumHBD(mol) <= 5 and
                rdMolDescriptors.CalcNumHBA(mol) <= 10
            ),
            "method": "rdkit",
        }
        return props

    except ImportError:
        # RDKit not available — use rule-based estimation
        return _estimate_properties_from_smiles(smiles, name)


def _estimate_properties_from_smiles(smiles: str, name: str) -> dict:
    """
    Rule-based property estimation when RDKit is unavailable.
    Counts atoms and functional groups from SMILES string.
    """
    # Count heavy atoms (rough MW estimate)
    atom_weights = {'C': 12, 'N': 14, 'O': 16, 'S': 32,
                    'F': 19, 'Cl': 35, 'Br': 80, 'P': 31}
    mw = sum(smiles.upper().count(sym) * w
             for sym, w in atom_weights.items()) + \
         smiles.count('H') * 1

    # Count HBD/HBA roughly
    hbd = smiles.count('N') + smiles.count('O') + smiles.count('n')
    hba = smiles.count('N') + smiles.count('O') * 2

    # logP estimation (very rough)
    n_c    = smiles.count('C') + smiles.count('c')
    n_o    = smiles.count('O') + smiles.count('o')
    n_n    = smiles.count('N') + smiles.count('n')
    n_f    = smiles.count('F')
    logP   = 0.5 * n_c - 1.0 * n_o - 0.7 * n_n - 0.5 * n_f

    return {
        "name":             name,
        "smiles":           smiles,
        "molecular_weight": mw,
        "logP":             round(logP, 2),
        "hbd":              min(hbd, 10),
        "hba":              min(hba, 15),
        "psa":              min(hbd * 20 + hba * 10, 200),
        "rotatable_bonds":  smiles.count('-') // 2,
        "aromatic_rings":   smiles.count('c') // 5,
        "heavy_atoms":      n_c + n_o + n_n + n_f,
        "lipinski_ok":      mw <= 500 and logP <= 5,
        "method":           "rule_based_estimate",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

PDAC_TARGETS = {
    "CEACAM6": "P40199",
    "TOP2A":   "P11388",
    "CLSPN":   "Q9HAW4",
    "ATAD2":   "Q6PL18",
    "HELLS":   "Q9NRZ9",
}


def main(uniprot: str = None, all_targets: bool = False,
         drug_smiles: str = None, drug_name: str = None):

    print("=" * 65)
    print("  SIM-02: Protein Ensemble Model")
    print("=" * 65)

    # Load cell environment from Module 1
    env_path = SIM_DIR / "cell_environment.json"
    if not env_path.exists():
        print("ERROR: Run sim/01_cell_environment.py first")
        return

    env_data    = json.loads(env_path.read_text())
    cell_env    = env_data["cell_environment"]
    normal_env  = build_normal_cell_env()

    # Compute drug properties if SMILES provided
    if drug_smiles:
        print(f"\nComputing properties for custom molecule...")
        mol_props = compute_molecule_properties(drug_smiles, drug_name or "custom")
        print(f"  MW={mol_props['molecular_weight']} g/mol  "
              f"logP={mol_props['logP']}  "
              f"PSA={mol_props['psa']} Å²  "
              f"Lipinski: {'✓' if mol_props['lipinski_ok'] else '✗'}")
        print(f"  Method: {mol_props['method']}")
        out = SIM_DIR / f"molecule_{drug_name or 'custom'}_props.json"
        out.write_text(json.dumps(mol_props, indent=2))
        print(f"  Saved to {out}")

    # Build ensembles
    targets = {}
    if all_targets:
        targets = PDAC_TARGETS
    elif uniprot:
        # Find gene name
        gene = next((g for g, u in PDAC_TARGETS.items()
                     if u == uniprot), uniprot)
        targets = {gene: uniprot}
    else:
        targets = PDAC_TARGETS

    print(f"\nBuilding protein ensembles for {len(targets)} targets...")

    ensembles = {}
    for gene, uid in targets.items():
        print(f"\n  Processing {gene} ({uid})...")
        ens = build_protein_ensemble(uid, cell_env, normal_env)
        if ens:
            ensembles[uid] = ens

    # Summary table
    if len(ensembles) > 1:
        print(f"\n{'='*65}")
        print(f"  ENSEMBLE SUMMARY — PDAC Conformational Landscape")
        print(f"{'='*65}")
        print(f"  {'Gene':<10} {'Compart':<22} {'<Vol>':>7} "
              f"{'<Drug>':>7} {'ΔΔG':>7} {'ΔLig':>7} {'S_conf':>7}")
        print(f"  {'-'*10} {'-'*22} {'-'*7} {'-'*7} {'-'*7} "
              f"{'-'*7} {'-'*7}")
        for uid, ens in ensembles.items():
            print(f"  {ens.gene_name:<10} {ens.compartment:<22} "
                  f"{ens.mean_pocket_volume:>7.0f} "
                  f"{ens.mean_druggability:>7.4f} "
                  f"{ens.pdac_vs_normal_ddG:>+7.2f} "
                  f"{ens.ligandability_shift:>+7.4f} "
                  f"{ens.conformational_entropy:>7.4f}")

    print(f"\n  Ensembles saved to data/sim/ensembles/")
    print(f"\n{'='*65}")
    print(f"  SIM-02 complete. Ready for SIM-03 (Drug Distribution)")
    print(f"{'='*65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SIM-02: Protein Ensemble Model"
    )
    parser.add_argument("--uniprot",    help="Single UniProt ID")
    parser.add_argument("--all-targets",action="store_true",
                        help="Run all PDAC targets")
    parser.add_argument("--drug-smiles",help="SMILES of custom drug molecule")
    parser.add_argument("--drug-name",  help="Name of custom drug molecule")
    args = parser.parse_args()

    main(
        uniprot     = args.uniprot,
        all_targets = args.all_targets,
        drug_smiles = args.drug_smiles,
        drug_name   = args.drug_name,
    )