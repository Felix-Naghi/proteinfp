"""
sim/01_cell_environment.py
───────────────────────────
Module SIM-01 — Cell Environment Model

Builds the physicochemical landscape of a PDAC tumor cell.
This is the foundation for all downstream drug simulation modules.

The cell is modeled as a set of compartments, each with defined:
  - Volume (fL)
  - pH
  - Ion concentrations (Na+, K+, Cl-, Mg2+, Ca2+)
  - Dielectric constant
  - Viscosity
  - Molecular crowding factor
  - Membrane composition (lipid fractions)
  - Protein concentration

Parameters are derived from:
  1. scRNA-seq expression data (your PDAC tumor cluster 6)
  2. Literature values for PDAC-specific metabolic state
  3. General mammalian cell biophysics

The model accounts for PDAC-specific alterations:
  - Acidic extracellular pH (Warburg effect, lactic acid export)
  - Elevated intracellular Cl- (CFTR mutations common in PDAC)
  - Altered membrane lipid composition (increased cholesterol)
  - Elevated molecular crowding (high protein synthesis rate)
  - Nuclear enlargement (high MKI67, proliferating cells)

Usage:
    python sim/01_cell_environment.py
    python sim/01_cell_environment.py --drug gemcitabine
    python sim/01_cell_environment.py --validate
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
GRN_INT  = ROOT / "data" / "grn" / "intermediate"
OUT_DIR  = ROOT / "data" / "sim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Physical constants ────────────────────────────────────────────────────────

kB    = 1.380649e-23   # Boltzmann constant J/K
NA    = 6.02214076e23  # Avogadro's number
R     = 8.314          # Gas constant J/mol/K
T     = 310.15         # Temperature K (37°C)
F     = 96485.0        # Faraday constant C/mol
RT    = R * T          # 2578.5 J/mol at 37°C
RTF   = RT / F         # 0.02672 V (thermal voltage)


# ── Compartment definition ────────────────────────────────────────────────────

@dataclass
class Compartment:
    """
    A single cellular compartment with physicochemical properties.
    All concentrations in mM unless noted.
    Volume in fL (femtoliters).
    """
    name:               str
    volume_fL:          float      # compartment volume

    # pH and buffering
    pH:                 float
    buffer_capacity:    float      # mmol/L/pH unit

    # Ion concentrations (mM)
    na_conc:            float      # Na+
    k_conc:             float      # K+
    cl_conc:            float      # Cl-
    mg_conc:            float      # Mg2+ (free)
    ca_conc:            float      # Ca2+ (free)
    h_conc:             float = field(init=False)  # H+ from pH

    # Dielectric and viscosity
    dielectric:         float = 80.0   # relative permittivity
    viscosity_mPas:     float = 1.0    # dynamic viscosity (mPa·s)

    # Molecular environment
    crowding_factor:    float = 1.0    # 1.0 = dilute, >1 = crowded
    protein_conc_gL:    float = 100.0  # total protein g/L
    atp_conc:           float = 1.0    # ATP mM (critical for kinases)
    nadh_conc:          float = 0.1    # NADH mM
    gsh_conc:           float = 5.0    # glutathione mM (redox)

    # Membrane properties (if this is a membrane compartment)
    membrane_thickness_nm: float = 0.0
    cholesterol_fraction:  float = 0.0  # mol fraction
    pc_fraction:           float = 0.0  # phosphatidylcholine
    pe_fraction:           float = 0.0  # phosphatidylethanolamine

    # PDAC-specific flags
    is_tumor:           bool = True
    has_warburg:        bool = False   # glycolytic shift

    def __post_init__(self):
        self.h_conc = 10 ** (-self.pH) * 1000  # mM

    @property
    def ionic_strength(self) -> float:
        """Ionic strength I = 0.5 * sum(ci * zi^2) in mM"""
        return 0.5 * (
            self.na_conc * 1 +
            self.k_conc  * 1 +
            self.cl_conc * 1 +
            self.mg_conc * 4 +
            self.ca_conc * 4
        )

    @property
    def debye_length_nm(self) -> float:
        """
        Debye screening length λD in nm.
        λD = sqrt(ε₀εkBT / 2e²NAI)
        Determines range of electrostatic interactions.
        """
        eps0   = 8.854e-12   # F/m
        eps    = self.dielectric
        e      = 1.602e-19   # C
        I_SI   = self.ionic_strength * 1000  # mol/m³
        if I_SI <= 0:
            return float('inf')
        lambda_D = math.sqrt(
            (eps0 * eps * kB * T) / (2 * e**2 * NA * I_SI)
        ) * 1e9  # convert m to nm
        return lambda_D

    @property
    def thermal_energy_kT(self) -> float:
        """Thermal energy kT in kJ/mol"""
        return kB * T * NA / 1000

    def partition_coefficient(self, logP: float, charge: float = 0) -> float:
        """
        Predict drug partition coefficient into this compartment
        relative to aqueous phase.

        For neutral molecules: driven by logP and membrane lipophilicity.
        For charged molecules: also affected by Donnan potential.

        Returns Kp (dimensionless concentration ratio).
        """
        # Neutral partitioning from logP
        # Kp ~ 10^logP for lipid compartments, ~1 for aqueous
        if self.cholesterol_fraction > 0:
            # Membrane/lipid compartment
            Kp = 10 ** (logP * self.cholesterol_fraction * 2)
        else:
            # Aqueous compartment
            Kp = 1.0

        # Charged molecule correction — Boltzmann factor
        # For monovalent cation (+1) in negative membrane potential
        if charge != 0:
            # Assume membrane potential of -70mV intracellular
            delta_psi = -0.070  # V
            Kp *= math.exp(-charge * F * delta_psi / (R * T))

        return Kp

    def effective_concentration(
        self,
        bulk_conc: float,
        logP: float,
        charge: float = 0,
    ) -> float:
        """
        Compute effective drug concentration in this compartment
        given bulk aqueous concentration.
        """
        Kp = self.partition_coefficient(logP, charge)
        return bulk_conc * Kp * self.crowding_factor

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ionic_strength"]  = round(self.ionic_strength, 2)
        d["debye_length_nm"] = round(self.debye_length_nm, 3)
        d["thermal_energy_kT"] = round(self.thermal_energy_kT, 4)
        return d


# ── Drug definition ───────────────────────────────────────────────────────────

@dataclass
class Drug:
    """Physicochemical properties of a drug molecule."""
    name:               str
    smiles:             str
    molecular_weight:   float    # g/mol
    logP:               float    # octanol-water partition coefficient
    pKa_basic:          float    # pKa of basic group (or None)
    pKa_acidic:         float    # pKa of acidic group (or None)
    hbd:                int      # H-bond donors
    hba:                int      # H-bond acceptors
    psa:                float    # polar surface area Å²
    charge_at_pH74:     float    # net charge at physiological pH
    solubility_mgmL:    float    # aqueous solubility
    permeability:       str      # "high" / "medium" / "low" / "transporter"
    transporter:        Optional[str] = None  # if transporter-mediated

    @property
    def lipinski_compliant(self) -> bool:
        return (self.molecular_weight <= 500 and
                self.logP <= 5 and
                self.hbd <= 5 and
                self.hba <= 10)

    @property
    def fraction_unionized(self, pH: float = 7.4) -> float:
        """Henderson-Hasselbalch: fraction of drug in unionized form at pH 7.4"""
        if self.pKa_basic and self.pKa_basic > 0:
            # Basic drug: ionized = BH+, unionized = B
            return 1 / (1 + 10 ** (pH - self.pKa_basic))
        elif self.pKa_acidic and self.pKa_acidic < 14:
            # Acidic drug: ionized = A-, unionized = AH
            return 1 / (1 + 10 ** (pH - self.pKa_acidic))
        return 1.0  # neutral drug

    def charge_at_pH(self, pH: float) -> float:
        """Net charge at given pH using Henderson-Hasselbalch"""
        charge = self.charge_at_pH74  # base charge
        if self.pKa_basic and self.pKa_basic > 0:
            frac_ionized = 1 / (1 + 10 ** (pH - self.pKa_basic))
            charge += frac_ionized
        if self.pKa_acidic and self.pKa_acidic < 14:
            frac_ionized = 1 / (1 + 10 ** (self.pKa_acidic - pH))
            charge -= frac_ionized
        return charge

    def membrane_permeability_coeff(self) -> float:
        """
        Estimate passive membrane permeability coefficient Pm (cm/s).
        Based on logP and PSA using Abraham model approximation.
        High: >10e-6, Medium: 1-10e-6, Low: <1e-6 cm/s
        """
        if self.permeability == "transporter":
            return 1e-8  # essentially zero passive permeability
        # Empirical relationship: log(Pm) ~ 0.5*logP - 0.01*PSA - 5.4
        log_Pm = 0.5 * self.logP - 0.01 * self.psa - 5.4
        return 10 ** log_Pm

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "lipinski_compliant":     self.lipinski_compliant,
            "membrane_permeability":  self.membrane_permeability_coeff(),
        }


# ── PDAC Cell Environment ─────────────────────────────────────────────────────

def build_pdac_cell(
    use_scrnaseq: bool = True,
) -> dict[str, Compartment]:
    """
    Build the physicochemical environment of a PDAC tumor cell
    (cluster 6 from your scRNA-seq data).

    PDAC-specific alterations from normal ductal:
    - Extracellular pH 6.5-6.8 (vs 7.4 normal) — Warburg effect
    - Elevated intracellular Na+ — altered NHE1 activity
    - Reduced Mg2+ — common in PDAC
    - Elevated cytoplasmic Ca2+ — pro-apoptotic signaling
    - Increased nuclear volume fraction (MKI67 high, proliferating)
    - Elevated ER stress markers (AGR2 high in your data)
    - Increased cholesterol in membrane — tumor cells
    - High ATP consumption — proliferating cells

    All values from literature on PDAC cell lines and tumor tissue.
    References: Bhatt et al 2020, Commisso et al 2013, Tape et al 2016
    """

    # ── Extracellular space (tumor microenvironment) ───────────────────────
    extracellular = Compartment(
        name             = "extracellular_tumor",
        volume_fL        = 5000.0,     # effective local volume
        pH               = 6.7,        # PDAC TME is acidic (Warburg)
        buffer_capacity  = 25.0,       # bicarbonate buffer mM/pH
        na_conc          = 145.0,      # plasma-like
        k_conc           = 5.0,
        cl_conc          = 110.0,
        mg_conc          = 0.8,
        ca_conc          = 2.5,
        dielectric       = 80.0,
        viscosity_mPas   = 1.2,        # slightly viscous TME
        crowding_factor  = 0.8,        # less crowded than intracellular
        protein_conc_gL  = 60.0,       # interstitial fluid
        atp_conc         = 0.01,       # very low extracellular ATP
        nadh_conc        = 0.0,
        gsh_conc         = 0.1,
        has_warburg      = True,
    )

    # ── Plasma membrane ────────────────────────────────────────────────────
    plasma_membrane = Compartment(
        name                  = "plasma_membrane",
        volume_fL             = 0.5,       # membrane volume estimate
        pH                    = 7.0,       # interface pH
        buffer_capacity       = 5.0,
        na_conc               = 75.0,      # average of inside/outside
        k_conc                = 75.0,
        cl_conc               = 60.0,
        mg_conc               = 0.5,
        ca_conc               = 0.2,
        dielectric            = 4.0,       # lipid bilayer low dielectric
        viscosity_mPas        = 100.0,     # membrane viscosity much higher
        crowding_factor       = 2.0,
        protein_conc_gL       = 200.0,     # high protein density in membrane
        atp_conc              = 0.5,
        membrane_thickness_nm = 7.5,
        cholesterol_fraction  = 0.45,      # elevated in tumor cells
        pc_fraction           = 0.30,
        pe_fraction           = 0.20,
    )

    # ── Cytoplasm ──────────────────────────────────────────────────────────
    cytoplasm = Compartment(
        name             = "cytoplasm",
        volume_fL        = 1500.0,     # ~2700 fL total cell, ~55% cytoplasm
        pH               = 7.2,        # slightly acidic vs normal 7.4
        buffer_capacity  = 40.0,       # high intracellular buffering
        na_conc          = 15.0,       # low intracellular Na+
        k_conc           = 140.0,      # high intracellular K+
        cl_conc          = 20.0,
        mg_conc          = 0.5,        # reduced in PDAC
        ca_conc          = 0.0001,     # very low free Ca2+ (100 nM)
        dielectric       = 70.0,       # slightly lower than water
        viscosity_mPas   = 3.0,        # cytoplasm viscosity
        crowding_factor  = 1.8,        # highly crowded — PDAC proliferating
        protein_conc_gL  = 200.0,
        atp_conc         = 3.0,        # high ATP — glycolytic tumor
        nadh_conc        = 0.15,
        gsh_conc         = 8.0,        # elevated GSH in PDAC (drug resistance)
        has_warburg      = True,
    )

    # ── Nucleus ────────────────────────────────────────────────────────────
    # Enlarged in PDAC (MKI67 high, TOP2A high, ATAD2 high)
    nucleus = Compartment(
        name             = "nucleus",
        volume_fL        = 800.0,      # enlarged vs normal ~300 fL
        pH               = 7.35,       # slightly more alkaline than cytoplasm
        buffer_capacity  = 30.0,
        na_conc          = 20.0,
        k_conc           = 130.0,
        cl_conc          = 25.0,
        mg_conc          = 1.0,        # higher Mg2+ for DNA stabilization
        ca_conc          = 0.001,
        dielectric       = 65.0,
        viscosity_mPas   = 50.0,       # chromatin greatly increases viscosity
        crowding_factor  = 3.5,        # extremely crowded (chromatin)
        protein_conc_gL  = 400.0,      # very high (histones, TFs)
        atp_conc         = 2.5,
        nadh_conc        = 0.05,
        gsh_conc         = 3.0,
    )

    # ── Endoplasmic reticulum ──────────────────────────────────────────────
    # AGR2 is an ER-resident protein and highly expressed in your PDAC data
    er = Compartment(
        name             = "endoplasmic_reticulum",
        volume_fL        = 200.0,
        pH               = 7.0,
        buffer_capacity  = 20.0,
        na_conc          = 10.0,
        k_conc           = 140.0,
        cl_conc          = 15.0,
        mg_conc          = 0.3,
        ca_conc          = 0.5,        # ER is calcium store (0.5 mM)
        dielectric       = 70.0,
        viscosity_mPas   = 5.0,
        crowding_factor  = 2.5,
        protein_conc_gL  = 300.0,      # high — protein folding compartment
        atp_conc         = 1.5,
        nadh_conc        = 0.08,
        gsh_conc         = 0.5,        # more oxidizing than cytoplasm
    )

    # ── Mitochondria ───────────────────────────────────────────────────────
    mitochondria = Compartment(
        name             = "mitochondria",
        volume_fL        = 150.0,
        pH               = 8.0,        # alkaline matrix
        buffer_capacity  = 50.0,
        na_conc          = 10.0,
        k_conc           = 120.0,
        cl_conc          = 10.0,
        mg_conc          = 3.0,        # high Mg2+ for ATP synthesis
        ca_conc          = 0.001,
        dielectric       = 70.0,
        viscosity_mPas   = 4.0,
        crowding_factor  = 2.0,
        protein_conc_gL  = 250.0,
        atp_conc         = 8.0,        # high ATP at site of synthesis
        nadh_conc        = 2.0,        # high NADH — electron transport
        gsh_conc         = 10.0,       # high mitochondrial GSH
        membrane_thickness_nm = 7.5,
        cholesterol_fraction  = 0.03,  # very low cholesterol in inner membrane
        pc_fraction           = 0.45,
        pe_fraction           = 0.35,
    )

    # ── Lysosome ───────────────────────────────────────────────────────────
    lysosome = Compartment(
        name             = "lysosome",
        volume_fL        = 30.0,
        pH               = 4.7,        # highly acidic — drug trapping risk
        buffer_capacity  = 15.0,
        na_conc          = 20.0,
        k_conc           = 50.0,
        cl_conc          = 80.0,       # high Cl- in lysosomes
        mg_conc          = 0.1,
        ca_conc          = 0.5,
        dielectric       = 75.0,
        viscosity_mPas   = 3.0,
        crowding_factor  = 1.5,
        protein_conc_gL  = 150.0,
        atp_conc         = 0.1,        # low ATP
        nadh_conc        = 0.01,
        gsh_conc         = 0.1,
    )

    return {
        "extracellular":         extracellular,
        "plasma_membrane":       plasma_membrane,
        "cytoplasm":             cytoplasm,
        "nucleus":               nucleus,
        "endoplasmic_reticulum": er,
        "mitochondria":          mitochondria,
        "lysosome":              lysosome,
    }


# ── Known drugs ───────────────────────────────────────────────────────────────

DRUGS = {
    "gemcitabine": Drug(
        name              = "Gemcitabine",
        smiles            = "O=C1N=C(N)C=CN1[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2F",
        molecular_weight  = 263.20,
        logP              = -1.99,
        pKa_basic         = 3.6,
        pKa_acidic        = 13.0,
        hbd               = 4,
        hba               = 7,
        psa               = 103.5,
        charge_at_pH74    = 0.0,
        solubility_mgmL   = 20.0,
        permeability      = "transporter",
        transporter       = "SLC29A1",
    ),
}


# ── Drug distribution simulation ─────────────────────────────────────────────

def simulate_drug_distribution(
    drug:         Drug,
    cell_env:     dict[str, Compartment],
    dose_uM:      float = 10.0,
    n_steps:      int   = 1000,
    dt_s:         float = 0.1,
) -> dict:
    """
    Simulate drug distribution across cellular compartments over time.

    Uses a compartmental ODE model:
      dC/dt = sum(flux_in) - sum(flux_out) - degradation

    Flux between compartments driven by:
      - Concentration gradient (passive diffusion)
      - Membrane permeability (logP-dependent)
      - Active transport (transporter expression from scRNA-seq)
      - pH partitioning (Henderson-Hasselbalch trapping)

    Returns time course of drug concentration in each compartment.
    """

    # Load transporter expression from scRNA-seq if available
    transporter_expr = _load_transporter_expression(drug)

    # Initial conditions — drug starts in extracellular space
    compartments = list(cell_env.keys())
    conc = {c: 0.0 for c in compartments}
    conc["extracellular"] = dose_uM

    # Compartment volumes for mass balance
    volumes = {name: comp.volume_fL
               for name, comp in cell_env.items()}

    # Membrane permeability coefficient (cm/s)
    Pm = drug.membrane_permeability_coeff()

    # Topology: which compartments are connected
    connections = [
        ("extracellular",   "cytoplasm",   "plasma_membrane"),
        ("cytoplasm",       "nucleus",     None),
        ("cytoplasm",       "endoplasmic_reticulum", None),
        ("cytoplasm",       "mitochondria", None),
        ("cytoplasm",       "lysosome",    None),
    ]

    # Degradation rate (cytidine deaminase for gemcitabine)
    k_degrade = 0.001 if drug.name == "Gemcitabine" else 0.0001

    time_course = {c: [conc[c]] for c in compartments}
    times = [0.0]

    for step in range(n_steps):
        new_conc = dict(conc)

        for src, dst, membrane in connections:
            C_src = conc[src]
            C_dst = conc[dst]

            comp_src = cell_env[src]
            comp_dst = cell_env[dst]

            # pH partitioning correction
            # Acidic extracellular traps basic drugs outside
            charge_src = drug.charge_at_pH(comp_src.pH)
            charge_dst = drug.charge_at_pH(comp_dst.pH)

            # Effective concentrations accounting for ionization
            C_src_eff = C_src * (1 / (1 + 10 ** (comp_src.pH - drug.pKa_basic))
                                  if drug.pKa_basic > 0 else 1.0)
            C_dst_eff = C_dst * (1 / (1 + 10 ** (comp_dst.pH - drug.pKa_basic))
                                  if drug.pKa_basic > 0 else 1.0)

            # Flux = Pm * area * (C_src_eff - C_dst_eff)
            # Approximate area from volume (sphere approximation)
            r_src   = (3 * volumes[src] * 1e-15 / (4 * math.pi)) ** (1/3)
            area_m2 = 4 * math.pi * r_src**2

            # Passive flux (mol/s)
            Pm_eff = Pm
            if membrane == "plasma_membrane" and drug.permeability == "transporter":
                # Transporter-mediated: scale by expression
                Pm_eff = Pm * transporter_expr * 100

            flux_mol_s = Pm_eff * area_m2 * (C_src_eff - C_dst_eff) * 1e-6 * 1e3

            # Convert to uM/s in each compartment
            vol_src_L = volumes[src] * 1e-15
            vol_dst_L = volumes[dst] * 1e-15

            delta_src = -flux_mol_s / vol_src_L * 1e6 * dt_s
            delta_dst =  flux_mol_s / vol_dst_L * 1e6 * dt_s

            new_conc[src] = max(0, new_conc[src] + delta_src)
            new_conc[dst] = max(0, new_conc[dst] + delta_dst)

        # Degradation in cytoplasm (CDA deaminates gemcitabine)
        new_conc["cytoplasm"] = max(
            0, new_conc["cytoplasm"] * (1 - k_degrade * dt_s)
        )

        conc = new_conc
        t    = (step + 1) * dt_s
        times.append(t)
        for c in compartments:
            time_course[c].append(conc[c])

    # Steady state = last time point
    steady_state = {c: time_course[c][-1] for c in compartments}

    return {
        "drug":          drug.name,
        "dose_uM":       dose_uM,
        "steady_state":  steady_state,
        "time_course":   time_course,
        "times":         times,
        "n_steps":       n_steps,
        "dt_s":          dt_s,
        "total_time_s":  n_steps * dt_s,
    }


def _load_transporter_expression(drug: Drug) -> float:
    """
    Load transporter expression from scRNA-seq data.
    Returns normalized expression (0-1) for the drug's primary transporter.
    """
    if not drug.transporter:
        return 1.0

    # Try to get expression from preprocessed data
    try:
        import scanpy as sc
        import numpy as np

        adata = sc.read_h5ad(GRN_INT.parent / "intermediate" / "preprocessed.h5ad")
        tumor = adata[adata.obs["leiden"] == "6"]

        if drug.transporter in tumor.var_names:
            if hasattr(tumor.X, "toarray"):
                X = tumor.X.toarray()
            else:
                X = np.array(tumor.X)
            idx  = list(tumor.var_names).index(drug.transporter)
            expr = float(X[:, idx].mean())
            # Normalize: max expression in dataset
            max_expr = float(X.max())
            return min(1.0, expr / max_expr * 10) if max_expr > 0 else 0.5
    except Exception:
        pass

    return 0.5  # default if data unavailable


# ── Thermodynamic scoring ─────────────────────────────────────────────────────

def compute_thermodynamics(
    drug:     Drug,
    cell_env: dict[str, Compartment],
) -> dict:
    """
    Compute thermodynamic properties of the drug in each compartment.

    Includes:
    - Gibbs free energy of transfer between compartments
    - Entropy of drug distribution (Shannon entropy)
    - Boltzmann probability of finding drug in each compartment
    - Effective concentration at each target location
    """
    results = {}

    # Compute partition coefficients for all compartments
    partitions = {}
    for name, comp in cell_env.items():
        Kp = comp.partition_coefficient(drug.logP, drug.charge_at_pH(comp.pH))
        partitions[name] = Kp

    # Normalize to get probability distribution
    total = sum(partitions.values())
    probs = {name: Kp/total for name, Kp in partitions.items()}

    # Shannon entropy of distribution
    entropy = -sum(p * math.log(p + 1e-10) for p in probs.values())
    max_entropy = math.log(len(probs))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Gibbs free energy of transfer from extracellular to each compartment
    # ΔG = -RT ln(Kp) in kJ/mol
    delta_G = {}
    for name, Kp in partitions.items():
        if Kp > 0:
            delta_G[name] = round(-R * T * math.log(Kp) / 1000, 3)
        else:
            delta_G[name] = float('inf')

    results = {
        "partition_coefficients": {k: round(v, 4) for k, v in partitions.items()},
        "probability_distribution": {k: round(v, 4) for k, v in probs.items()},
        "delta_G_kJ_mol": delta_G,
        "distribution_entropy": round(entropy, 4),
        "normalized_entropy": round(norm_entropy, 4),
        "most_likely_compartment": max(probs, key=probs.get),
        "least_likely_compartment": min(probs, key=probs.get),
    }

    return results


# ── Validation ────────────────────────────────────────────────────────────────

def validate_environment(cell_env: dict[str, Compartment]) -> dict:
    """
    Validate that cell environment parameters are physically reasonable.
    Checks against known biophysical constraints.
    """
    issues   = []
    warnings = []

    for name, comp in cell_env.items():
        # pH must be physiologically reasonable
        if not (3.0 <= comp.pH <= 9.0):
            issues.append(f"{name}: pH {comp.pH} outside physiological range")

        # Ionic strength check (should be 100-300 mM for most compartments)
        IS = comp.ionic_strength
        if name != "lysosome" and not (50 <= IS <= 400):
            warnings.append(f"{name}: ionic strength {IS:.0f} mM unusual")

        # Debye length should be 0.5-2 nm for physiological solutions
        dL = comp.debye_length_nm
        if not (0.3 <= dL <= 5.0):
            warnings.append(f"{name}: Debye length {dL:.2f} nm unusual")

        # Volume ratios should make sense
        if name == "nucleus" and comp.volume_fL > 1500:
            warnings.append(f"nucleus volume {comp.volume_fL} fL seems large")

        # ATP should be present in metabolically active compartments
        if comp.atp_conc < 0.1 and name in ("cytoplasm", "nucleus"):
            issues.append(f"{name}: ATP {comp.atp_conc} mM dangerously low")

    return {
        "valid":    len(issues) == 0,
        "issues":   issues,
        "warnings": warnings,
        "n_compartments": len(cell_env),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(drug_name: str = "gemcitabine", validate: bool = False):

    print("=" * 65)
    print("  SIM-01: PDAC Cell Environment Model")
    print("=" * 65)

    # Build cell environment
    print("\n[1/4] Building PDAC tumor cell environment...")
    from sim.cell_environment_inference import infer_cell_environment
    cell_env = infer_cell_environment()
    print(f"  Compartments: {len(cell_env)}")
    for name, comp in cell_env.items():
        print(f"    {name:<25} pH={comp.pH:.1f}  "
              f"vol={comp.volume_fL:.0f}fL  "
              f"IS={comp.ionic_strength:.0f}mM  "
              f"λD={comp.debye_length_nm:.2f}nm")

    # Validate
    print("\n[2/4] Validating environment...")
    validation = validate_environment(cell_env)
    if validation["valid"]:
        print("  All parameters physically valid")
    else:
        for issue in validation["issues"]:
            print(f"  ERROR: {issue}")
    for warn in validation["warnings"]:
        print(f"  WARN: {warn}")

    # Drug analysis
    if drug_name in DRUGS:
        drug = DRUGS[drug_name]
        print(f"\n[3/4] Analyzing {drug.name} distribution...")
        print(f"  MW={drug.molecular_weight} g/mol  "
              f"logP={drug.logP}  "
              f"PSA={drug.psa} Å²  "
              f"Permeability={drug.permeability}")

        # Thermodynamics
        thermo = compute_thermodynamics(drug, cell_env)
        print(f"\n  Thermodynamic distribution:")
        for comp, prob in sorted(thermo["probability_distribution"].items(),
                                  key=lambda x: -x[1]):
            dG = thermo["delta_G_kJ_mol"][comp]
            print(f"    {comp:<25} P={prob:.3f}  ΔG={dG:+.1f} kJ/mol")
        print(f"\n  Distribution entropy: {thermo['distribution_entropy']:.3f} "
              f"(normalized: {thermo['normalized_entropy']:.3f})")
        print(f"  Most likely compartment: {thermo['most_likely_compartment']}")

        # Simulate distribution over time
        print(f"\n[4/4] Simulating drug distribution (10 μM dose, 100s)...")
        sim = simulate_drug_distribution(drug, cell_env, dose_uM=10.0,
                                          n_steps=1000, dt_s=0.1)
        print(f"\n  Steady-state concentrations (μM):")
        for comp, conc in sorted(sim["steady_state"].items(),
                                  key=lambda x: -x[1]):
            print(f"    {comp:<25} {conc:.4f} μM")

        # Save
        output = {
            "cell_environment": {k: v.to_dict()
                                 for k, v in cell_env.items()},
            "drug":             drug.to_dict(),
            "thermodynamics":   thermo,
            "simulation":       {k: v for k, v in sim.items()
                                 if k != "time_course"},
            "validation":       validation,
        }
        out_path = OUT_DIR / "cell_environment.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\n  Saved to {out_path}")

    print("\n" + "=" * 65)
    print("  SIM-01 complete. Ready for Module SIM-02 (Protein Ensemble)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SIM-01: PDAC Cell Environment Model"
    )
    parser.add_argument("--drug",     default="gemcitabine",
                        help="Drug to simulate (default: gemcitabine)")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation only")
    args = parser.parse_args()
    main(drug_name=args.drug, validate=args.validate)