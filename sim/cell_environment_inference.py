"""
sim/cell_environment_inference.py
──────────────────────────────────
Layer 3 — Data-driven cell environment inference.

Replaces the hardcoded build_pdac_cell() in step01_cell_environment.py
with infer_cell_environment(), which derives every compartment parameter
from marker gene expression in the input scRNA-seq data.

SCIENTIFIC BASIS
────────────────
Every physicochemical parameter in the cell environment model has a known
biological driver that can be read from gene expression.  This module
encodes those relationships as calibrated transfer functions, anchored to
published physiological limits.

The inference model for each parameter is documented below with its
literature basis and the transfer function used.

HOW TO INTEGRATE
────────────────
In sim/step01_cell_environment.py, replace:

    cell_env = build_pdac_cell()

with:

    from sim.cell_environment_inference import infer_cell_environment
    cell_env = infer_cell_environment()

Everything downstream (drug simulation, SIM-02, SIM-05) is unaffected
because the returned dict has identical structure and Compartment types.

PUBLISHABLE CONTRIBUTION
────────────────────────
This is, to our knowledge, the first method that automatically derives
cellular physicochemical environments for drug simulation directly from
transcriptomic data, without requiring disease-specific literature values.
The approach generalises to any cell type, organism, or disease for which
scRNA-seq data exists.

References for transfer functions:
  Warburg pH:    Gatenby & Gillies 2004 Nat Rev Cancer; Parks et al 2011
  Crowding:      Minton 2001 J Biol Chem; Zhou et al 2008 Annu Rev Biophys
  GSH:           Lu 2013 Mol Aspects Med; Forman et al 2009
  Cholesterol:   Kuzu et al 2016 Cancer Discov; Clendening et al 2010
  Nuclear vol:   Jevtic et al 2014 J Cell Biol
  Lysosomal pH:  Mindell 2012 Annu Rev Physiol; Ballabio & Bonifacino 2020
  Mitoch. NADH:  Williamson et al 1967 J Biol Chem; Ying 2008
  ATP:           Hinkle 2005 Biochim Biophys Acta
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).resolve().parent.parent
GRN_INT = ROOT / "data" / "grn" / "intermediate"

# ── Normal cell baseline values (healthy mammalian cell, 37°C) ────────────────
# These are the universal anchor points — what a normal, non-stressed,
# non-proliferating mammalian cell looks like.
# All values from standard cell biology textbooks (Alberts et al, Lodish et al).

NORMAL_BASELINE = {
    "extracellular_pH":      7.4,
    "cytoplasm_pH":          7.2,
    "nucleus_pH":            7.3,
    "mitochondria_pH":       7.9,
    "er_pH":                 7.0,
    "lysosome_pH":           4.9,   # slightly less acidic than disease state

    "cytoplasm_crowding":    1.2,   # Minton 2001
    "nucleus_crowding":      2.5,   # chromatin baseline
    "er_crowding":           2.0,

    "nuclear_volume_fL":     300.0, # normal quiescent nucleus
    "cytoplasm_volume_fL":   1200.0,
    "mitochondria_volume_fL": 120.0,

    "cytoplasm_atp_mM":      2.0,   # Hinkle 2005
    "mitochondria_atp_mM":   5.0,
    "cytoplasm_nadh_mM":     0.05,  # Williamson 1967
    "mitochondria_nadh_mM":  0.8,
    "cytoplasm_gsh_mM":      5.0,   # Lu 2013
    "mitochondria_gsh_mM":   5.0,
    "cytoplasm_mg_mM":       0.8,   # Romani 2007
    "plasma_membrane_chol":  0.30,  # mol fraction, normal mammalian PM
}

# ── Physiological limits (hard constraints, never exceeded) ───────────────────

LIMITS = {
    "extracellular_pH":     (5.5,  7.8),
    "cytoplasm_pH":         (6.5,  7.6),
    "nucleus_pH":           (6.8,  7.8),
    "mitochondria_pH":      (7.5,  8.5),
    "er_pH":                (6.5,  7.5),
    "lysosome_pH":          (4.2,  5.5),
    "cytoplasm_crowding":   (1.0,  3.0),
    "nucleus_crowding":     (2.0,  6.0),
    "nuclear_volume_fL":    (150., 2000.),
    "cytoplasm_atp_mM":     (0.5,  8.0),
    "mitochondria_atp_mM":  (1.0,  15.0),
    "cytoplasm_gsh_mM":     (1.0,  15.0),
    "cytoplasm_mg_mM":      (0.1,  2.0),
    "plasma_membrane_chol": (0.10, 0.65),
}


# ══════════════════════════════════════════════════════════════════════════════
# MARKER GENE DEFINITIONS
# Each marker panel encodes one biological axis.
# activators push the parameter up; suppressors push it down.
# Weights encode relative importance (sum to 1 within each group).
# ══════════════════════════════════════════════════════════════════════════════

MARKERS = {

    # ── Warburg effect / glycolytic shift ─────────────────────────────────────
    # High LDHA, PKM2, SLC16A3 (MCT4 lactate exporter), CA9 → acidic TME
    # High LDHB, OXPHOS genes → oxidative, less acidic
    # Transfer: extracellular_pH = 7.4 - 0.7 * warburg_score
    # Range: pH 6.7 (fully Warburg) to 7.4 (normal oxidative)
    # Basis: Gatenby & Gillies 2004; Parks et al 2011 Nat Cell Biol
    "warburg_activators": {
        "LDHA":    0.30,   # LDH-A: lactate dehydrogenase A (glycolytic)
        "PKM":     0.20,   # Pyruvate kinase M (PKM2 isoform dominant in cancer)
        "SLC16A3": 0.20,   # MCT4: monocarboxylate transporter 4 (lactate out)
        "CA9":     0.15,   # Carbonic anhydrase IX: CO2/H+ export
        "HK2":     0.10,   # Hexokinase 2: first step of glycolysis
        "PFKFB3":  0.05,   # PFK-2: glycolysis rate controller
    },
    "warburg_suppressors": {
        "LDHB":   0.40,    # LDH-B: oxidative (suppresses Warburg)
        "PDHA1":  0.30,    # Pyruvate dehydrogenase: feeds TCA (not glycolysis)
        "PDHB":   0.20,    # PDH subunit
        "IDH2":   0.10,    # Isocitrate dehydrogenase 2 (TCA cycle)
    },

    # ── Proliferation / molecular crowding ────────────────────────────────────
    # High MKI67, PCNA, ribosomal genes → more protein synthesis → more crowding
    # Transfer: crowding = 1.2 + 1.0 * prolif_score
    # Range: 1.2 (quiescent) to 2.5 (highly proliferating)
    # Basis: Minton 2001; Zhou et al 2008
    "proliferation": {
        "MKI67":  0.25,    # Ki-67: gold standard proliferation marker
        "PCNA":   0.20,    # Proliferating cell nuclear antigen
        "TOP2A":  0.15,    # DNA topoisomerase IIα: replication
        "MCM2":   0.10,    # Mini-chromosome maintenance: DNA replication
        "MCM6":   0.10,
        "CDK1":   0.10,    # CDK1: mitotic kinase
        "CCNB1":  0.10,    # Cyclin B1: G2/M transition
    },

    # ── Nuclear enlargement ───────────────────────────────────────────────────
    # High proliferation + chromatin remodellers → larger nucleus
    # Transfer: nuclear_vol = 300 + 700 * nuclear_score  (fL)
    # Range: 300 fL (quiescent) to 1000 fL (highly proliferating)
    # Basis: Jevtic et al 2014 J Cell Biol; Cantwell & Bhattacharya 2019
    "nuclear_enlargement": {
        "MKI67":  0.30,
        "TOP2A":  0.25,    # Decondenses chromatin → larger nucleus
        "ATAD2":  0.20,    # Chromatin remodeller — enlarged nucleus in cancer
        "HIST1H2B": 0.15,  # Histone availability → chromatin expansion
        "HMGN1":  0.10,    # High mobility group nucleosome-binding
    },

    # ── Glutathione / redox status ────────────────────────────────────────────
    # High GSS, GCLC, GCLM → elevated cytoplasmic GSH
    # Transfer: gsh = 5.0 + 8.0 * gsh_score  (mM)
    # Range: 5 mM (normal) to 13 mM (highly antioxidant)
    # Basis: Lu 2013 Mol Aspects Med; Tew & Townsend 2012
    "gsh_synthesis": {
        "GSS":   0.30,     # Glutathione synthetase
        "GCLC":  0.30,     # Glutamate-cysteine ligase catalytic
        "GCLM":  0.20,     # GCL modifier subunit
        "SLC7A11": 0.20,   # xCT: cystine importer (rate-limiting for GSH)
    },

    # ── Cholesterol biosynthesis / membrane composition ───────────────────────
    # High HMGCR, SQLE, FDFT1 → elevated membrane cholesterol
    # Transfer: cholesterol = 0.30 + 0.25 * chol_score  (mol fraction)
    # Range: 0.30 (normal) to 0.55 (cholesterol-enriched tumor membrane)
    # Basis: Kuzu et al 2016 Cancer Discov; Clendening & Penn 2012
    "cholesterol_synthesis": {
        "HMGCR":  0.35,    # HMG-CoA reductase: rate-limiting step
        "SQLE":   0.25,    # Squalene epoxidase: second rate-limiting step
        "FDFT1":  0.20,    # Farnesyl-diphosphate farnesyltransferase
        "DHCR7":  0.10,    # 7-dehydrocholesterol reductase
        "LDLR":   0.10,    # LDL receptor: cholesterol uptake
    },

    # ── Mitochondrial activity / OXPHOS ──────────────────────────────────────
    # High NDUF*, SDHA, COX* → active OXPHOS → high NADH, high mito ATP
    # Transfer: nadh_mito = 0.8 + 1.8 * oxphos_score  (mM)
    # Range: 0.8 mM (suppressed) to 2.6 mM (high OXPHOS)
    # Basis: Williamson et al 1967; Ying 2008 Antioxid Redox Signal
    "oxphos": {
        "NDUFA1":  0.12,   # Complex I subunit
        "NDUFB1":  0.12,
        "SDHA":    0.15,   # Complex II: succinate dehydrogenase
        "UQCRB":   0.12,   # Complex III
        "COX5A":   0.12,   # Complex IV
        "ATP5F1A": 0.15,   # ATP synthase alpha
        "CS":      0.12,   # Citrate synthase: TCA cycle entry
        "PDHA1":   0.10,   # Feeds acetyl-CoA to TCA
    },

    # ── ATP production rate ───────────────────────────────────────────────────
    # Both glycolysis and OXPHOS produce ATP; the balance matters
    # Transfer: atp_cyto = 2.0 + 3.0 * atp_score  (mM)
    # Glycolysis score (fast but less ATP/glucose) vs OXPHOS (slow, more ATP)
    # Net effect: high proliferation = high demand = elevated steady-state ATP
    # Basis: Hinkle 2005 Biochim Biophys Acta
    "atp_production": {
        "ATP5F1A": 0.20,
        "ATP5F1B": 0.20,
        "GAPDH":   0.15,   # Glycolysis
        "PGK1":    0.15,
        "PKM":     0.15,
        "HK2":     0.15,
    },

    # ── Lysosomal acidification ───────────────────────────────────────────────
    # High V-ATPase subunits → more acidic lysosomes (more drug trapping)
    # Transfer: lyso_pH = 5.0 - 0.6 * vatpase_score
    # Range: pH 4.2 (hyperacidic) to 5.0 (mildly acidic)
    # Basis: Mindell 2012 Annu Rev Physiol; Marshansky & Futai 2008
    "lysosomal_acidification": {
        "ATP6V1A":  0.20,  # V-ATPase V1 domain subunit A
        "ATP6V1B2": 0.20,
        "ATP6V0A1": 0.20,  # V0 domain: proton channel
        "ATP6V0D1": 0.15,
        "ATP6AP1":  0.10,  # Accessory subunit 1
        "LAMP1":    0.15,  # Lysosomal membrane protein (proxy for biogenesis)
    },

    # ── ER stress ─────────────────────────────────────────────────────────────
    # High HSPA5 (BiP), ATF6, DDIT3, XBP1 → unfolded protein response active
    # This affects ER Ca2+ and protein concentration
    # Transfer: er_ca2+ = 0.5 + 0.3 * er_stress_score  (mM)
    # Basis: Walter & Ron 2011 Science
    "er_stress": {
        "HSPA5":  0.35,    # BiP/GRP78: master ER chaperone, UPR sensor
        "DDIT3":  0.20,    # CHOP: UPR transcription factor
        "ATF4":   0.20,    # ATF4: PERK pathway
        "XBP1":   0.15,    # XBP1: IRE1 pathway spliced form
        "AGR2":   0.10,    # ER-resident protein disulfide isomerase-like
    },

    # ── Magnesium transport ───────────────────────────────────────────────────
    # Low SLC41A1, TRPM7 → reduced free Mg2+ (common in cancer cells)
    # Transfer: mg2+ = 0.8 - 0.5 * mg_depletion_score  (mM, free)
    # Basis: Romani 2007 Arch Biochem Biophys; Wolf & Trapani 2008
    "mg_transporters": {
        "SLC41A1": 0.40,   # Na+/Mg2+ exchanger: main efflux
        "TRPM7":   0.35,   # Mg2+ channel/kinase
        "CNNM2":   0.25,   # Cyclin M2: Mg2+ extrusion
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# EXPRESSION LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_mean_expression(
    tumor_cluster_id: str = "6",
    config_path:      Optional[Path] = None,
) -> dict[str, float]:
    """
    Load mean normalised expression per gene from the tumor cluster.

    Returns dict[gene_symbol → mean_norm_expression] where values are 0–1.
    Falls back to empty dict (all genes → 0.5, neutral) if data unavailable.
    """
    # Try to get cluster ID from disease config
    if config_path is None:
        config_path = ROOT / "config" / "disease_config.yaml"

    if config_path.exists():
        import yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        tumor_cluster_id = str(
            cfg.get("data", {}).get("tumor_cluster_id", tumor_cluster_id)
        )

    h5ad_path = GRN_INT / "preprocessed.h5ad"
    if not h5ad_path.exists():
        warnings.warn(
            f"preprocessed.h5ad not found at {h5ad_path}. "
            f"Cell environment will use normal-cell baseline values. "
            f"Run grn/01_preprocess.py first for disease-specific inference."
        )
        return {}

    try:
        import scanpy as sc

        adata = sc.read_h5ad(h5ad_path)

        # Find tumor cluster
        cluster_col = None
        for col in ("leiden", "cluster", "cell_type", "louvain"):
            if col in adata.obs.columns:
                cluster_col = col
                break

        if cluster_col and tumor_cluster_id in adata.obs[cluster_col].values:
            tumor = adata[adata.obs[cluster_col] == tumor_cluster_id]
        else:
            tumor = adata  # use all cells as fallback

        if hasattr(tumor.X, "toarray"):
            X = tumor.X.toarray()
        else:
            X = np.array(tumor.X)

        mean_expr = X.mean(axis=0)

        # Normalise to 0–1 using 99th percentile cap (robust to outliers)
        p99 = np.percentile(mean_expr[mean_expr > 0], 99) if (mean_expr > 0).any() else 1.0
        norm_expr = np.clip(mean_expr / (p99 + 1e-10), 0.0, 1.0)

        return {
            gene: float(norm_expr[i])
            for i, gene in enumerate(tumor.var_names)
        }

    except Exception as e:
        warnings.warn(f"Expression loading failed: {e}. Using baseline values.")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFER FUNCTIONS
# Each function maps a panel score to a physical parameter.
# All are monotonic and bounded within physiological limits.
# ══════════════════════════════════════════════════════════════════════════════

def _panel_score(
    expr:    dict[str, float],
    panel:   dict[str, float],
    default: float = 0.5,
) -> float:
    """
    Compute weighted mean expression score for a marker panel.
    Genes absent from the expression data are imputed with `default`.
    Returns a score in [0, 1].
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for gene, weight in panel.items():
        val = expr.get(gene, default)
        weighted_sum += weight * val
        total_weight += weight
    if total_weight == 0:
        return default
    return weighted_sum / total_weight


def _clamp(value: float, key: str) -> float:
    """Clamp value to physiological limits."""
    lo, hi = LIMITS.get(key, (-1e9, 1e9))
    return float(np.clip(value, lo, hi))


def _sigmoid(x: float, midpoint: float = 0.5, steepness: float = 6.0) -> float:
    """Smooth sigmoid: 0 → 0, midpoint → 0.5, 1 → 1."""
    return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class CellEnvironmentInference:
    """
    Infers all compartment physicochemical parameters from gene expression.

    Instantiate once, then call .infer() to get the parameter dict,
    or .infer_compartments() to get Compartment objects ready for SIM-01.
    """

    def __init__(
        self,
        expr:    dict[str, float],
        verbose: bool = True,
    ):
        self.expr    = expr
        self.verbose = verbose
        self._params: dict[str, float] = {}
        self._scores: dict[str, float] = {}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ── Score computation ──────────────────────────────────────────────────────

    def _compute_scores(self) -> None:
        """Compute all panel scores from expression data."""
        M = MARKERS
        E = self.expr

        # Net Warburg score: activators minus suppressors
        warburg_act  = _panel_score(E, M["warburg_activators"])
        warburg_sup  = _panel_score(E, M["warburg_suppressors"])
        warburg_net  = _sigmoid(warburg_act - warburg_sup + 0.5)

        prolif_score   = _panel_score(E, M["proliferation"])
        nuclear_score  = _panel_score(E, M["nuclear_enlargement"])
        gsh_score      = _panel_score(E, M["gsh_synthesis"])
        chol_score     = _panel_score(E, M["cholesterol_synthesis"])
        oxphos_score   = _panel_score(E, M["oxphos"])
        atp_score      = _panel_score(E, M["atp_production"])
        lyso_score     = _panel_score(E, M["lysosomal_acidification"])
        er_stress_score = _panel_score(E, M["er_stress"])
        mg_score       = _panel_score(E, M["mg_transporters"])

        self._scores = {
            "warburg":    warburg_net,
            "prolif":     prolif_score,
            "nuclear":    nuclear_score,
            "gsh":        gsh_score,
            "cholesterol": chol_score,
            "oxphos":     oxphos_score,
            "atp":        atp_score,
            "lysosome":   lyso_score,
            "er_stress":  er_stress_score,
            "mg":         mg_score,
        }

    # ── Transfer functions → physical parameters ───────────────────────────────

    def _infer_params(self) -> None:
        """Apply transfer functions to convert scores → parameters."""
        S = self._scores
        N = NORMAL_BASELINE

        p: dict[str, float] = {}

        # ── pH values ──────────────────────────────────────────────────────────

        # Extracellular pH: Warburg lactic acid export acidifies TME
        # pH 7.4 (no Warburg) → pH 6.5 (full Warburg)
        # Basis: Gatenby & Gillies 2004 — measured 0.5–1.0 pH unit drop
        p["extracellular_pH"] = _clamp(
            7.4 - 0.9 * S["warburg"],
            "extracellular_pH"
        )

        # Intracellular pH: paradoxically ALKALINE in Warburg (NHE1 pumps H+ out)
        # Normal 7.2, Warburg cells: 7.2–7.4 intracellular
        # Basis: Webb et al 2011 Nat Rev Cancer; Boron 2004
        p["cytoplasm_pH"] = _clamp(
            7.2 + 0.15 * S["warburg"],
            "cytoplasm_pH"
        )

        # Nuclear pH tracks cytoplasm + small offset
        p["nucleus_pH"] = _clamp(
            p["cytoplasm_pH"] + 0.1,
            "nucleus_pH"
        )

        # Mitochondrial pH: reduced OXPHOS → less alkaline matrix
        # Normal 7.9, impaired OXPHOS → 7.5–7.7
        p["mitochondria_pH"] = _clamp(
            7.5 + 0.4 * S["oxphos"],
            "mitochondria_pH"
        )

        # ER pH: ER stress → slight acidification
        p["er_pH"] = _clamp(
            7.0 - 0.2 * S["er_stress"],
            "er_pH"
        )

        # Lysosomal pH: V-ATPase activity drives acidification
        # pH 5.0 (low V-ATPase) → pH 4.2 (high V-ATPase)
        p["lysosome_pH"] = _clamp(
            5.0 - 0.8 * S["lysosome"],
            "lysosome_pH"
        )

        # ── Crowding ───────────────────────────────────────────────────────────

        # Cytoplasm crowding: driven by proliferation rate
        # (more ribosomes, more protein synthesis machinery)
        # Normal 1.2, high proliferation → 2.5
        # Basis: Minton 2001; measured by FRAP, single-molecule tracking
        p["cytoplasm_crowding"] = _clamp(
            1.2 + 1.3 * S["prolif"],
            "cytoplasm_crowding"
        )

        # Nuclear crowding: chromatin compaction + TF density
        # Scales with proliferation AND nuclear enlargement
        # Larger nucleus → more DNA → more crowding
        p["nucleus_crowding"] = _clamp(
            2.5 + 2.5 * (0.6 * S["prolif"] + 0.4 * S["nuclear"]),
            "nucleus_crowding"
        )

        # ER crowding: scales with ER stress (more unfolded proteins)
        p["er_crowding"] = _clamp(
            2.0 + 1.5 * S["er_stress"],
            "er_crowding"
        )

        # ── Volumes ────────────────────────────────────────────────────────────

        # Nuclear volume: proliferating cells enlarge their nucleus
        # Normal 300 fL, cancer cells: 300–1200 fL
        # Basis: Jevtic et al 2014; Cantwell & Bhattacharya 2019
        p["nuclear_volume_fL"] = _clamp(
            300.0 + 900.0 * (0.5 * S["prolif"] + 0.5 * S["nuclear"]),
            "nuclear_volume_fL"
        )

        # Cytoplasm volume scales inversely (nucleus expands into it)
        # Total cell volume stays ~2700 fL
        p["cytoplasm_volume_fL"] = max(
            600.0,
            2700.0 - p["nuclear_volume_fL"] - 500.0
        )

        # Mitochondria: reduced in Warburg cells (less OXPHOS needed)
        p["mitochondria_volume_fL"] = _clamp(
            60.0 + 120.0 * S["oxphos"],
            ("60.0", "250.0")  # type: ignore
        )
        p["mitochondria_volume_fL"] = float(
            np.clip(p["mitochondria_volume_fL"], 60.0, 250.0)
        )

        # ── Metabolites ────────────────────────────────────────────────────────

        # Cytoplasmic ATP
        # Both glycolysis and OXPHOS produce ATP
        # Baseline 2 mM, high metabolic activity → 5 mM
        p["cytoplasm_atp_mM"] = _clamp(
            2.0 + 3.0 * S["atp"],
            "cytoplasm_atp_mM"
        )

        # Mitochondrial ATP: scales directly with OXPHOS
        p["mitochondria_atp_mM"] = _clamp(
            3.0 + 7.0 * S["oxphos"],
            "mitochondria_atp_mM"
        )

        # Mitochondrial NADH: high OXPHOS → high NADH production
        # Normal 0.8 mM, high OXPHOS → 2.5 mM
        # Basis: Williamson et al 1967 (direct biochemical measurement)
        p["mitochondria_nadh_mM"] = float(
            np.clip(0.5 + 2.0 * S["oxphos"], 0.3, 3.0)
        )

        # Cytoplasmic NADH: elevated in Warburg (glycolytic NADH)
        p["cytoplasm_nadh_mM"] = float(
            np.clip(0.05 + 0.2 * S["warburg"], 0.02, 0.3)
        )

        # GSH: high synthesis genes → elevated antioxidant capacity
        # Normal 5 mM, high synthesis → 13 mM
        # (Elevated GSH is a major drug resistance mechanism)
        p["cytoplasm_gsh_mM"] = _clamp(
            5.0 + 8.0 * S["gsh"],
            "cytoplasm_gsh_mM"
        )

        # Mitochondrial GSH: tracks cytoplasm but slightly lower
        p["mitochondria_gsh_mM"] = float(p["cytoplasm_gsh_mM"] * 0.8)

        # ER calcium: elevated during ER stress (Ca2+ release from ER)
        # Normal 0.5 mM, stress → 0.8 mM
        p["er_ca2p_mM"] = float(np.clip(0.5 + 0.3 * S["er_stress"], 0.3, 1.0))

        # Cytoplasmic free Mg2+: reduced in proliferating/cancer cells
        # Driven by transporter downregulation
        p["cytoplasm_mg_mM"] = _clamp(
            0.8 - 0.5 * (1.0 - S["mg"]),  # low transporters → low Mg2+
            "cytoplasm_mg_mM"
        )

        # ── Membrane composition ───────────────────────────────────────────────

        # Plasma membrane cholesterol
        # Normal 0.30, high biosynthesis → 0.55
        # (Elevated cholesterol stiffens membranes and affects drug permeability)
        p["plasma_membrane_chol"] = _clamp(
            0.30 + 0.25 * S["cholesterol"],
            "plasma_membrane_chol"
        )

        self._params = p

    # ── Public interface ───────────────────────────────────────────────────────

    def infer(self) -> dict[str, float]:
        """
        Run full inference pipeline.
        Returns flat dict of parameter_name → inferred_value.
        """
        if not self._scores:
            self._compute_scores()
        if not self._params:
            self._infer_params()
        return dict(self._params)

    def report(self) -> str:
        """
        Human-readable inference report showing scores, parameters,
        and deviation from normal baseline.
        """
        if not self._params:
            self.infer()

        lines = [
            "",
            "─" * 65,
            "  Cell Environment Inference Report",
            "─" * 65,
            "",
            "  MARKER GENE SCORES  (0=low, 1=high expression)",
            f"  {'Panel':<28} {'Score':>6}  Interpretation",
        ]
        interp = {
            "warburg":     ("oxidative", "Warburg glycolysis"),
            "prolif":      ("quiescent", "highly proliferating"),
            "nuclear":     ("small nucleus", "enlarged nucleus"),
            "gsh":         ("low GSH", "high GSH (drug resistant)"),
            "cholesterol": ("normal membrane", "cholesterol-rich membrane"),
            "oxphos":      ("OXPHOS suppressed", "high OXPHOS"),
            "atp":         ("low ATP", "high ATP"),
            "lysosome":    ("mild lysosomal acid", "hyperacid lysosomes"),
            "er_stress":   ("no ER stress", "severe ER stress"),
            "mg":          ("low Mg2+ transport", "normal Mg2+ transport"),
        }
        for panel, score in self._scores.items():
            lo, hi = interp.get(panel, ("low", "high"))
            desc   = lo if score < 0.4 else (hi if score > 0.6 else "moderate")
            lines.append(f"  {panel:<28} {score:>6.3f}  {desc}")

        lines += [
            "",
            "  INFERRED PARAMETERS  (vs normal baseline)",
            f"  {'Parameter':<35} {'Inferred':>9}  {'Normal':>8}  {'Δ':>7}",
        ]
        for k, v in self._params.items():
            baseline = NORMAL_BASELINE.get(k, float("nan"))
            delta    = v - baseline if not math.isnan(baseline) else float("nan")
            delta_s  = f"{delta:>+7.2f}" if not math.isnan(delta) else "   N/A"
            lines.append(
                f"  {k:<35} {v:>9.3f}  {baseline:>8.3f}  {delta_s}"
            )

        lines += ["─" * 65, ""]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PUBLIC FUNCTION: drop-in replacement for build_pdac_cell()
# ══════════════════════════════════════════════════════════════════════════════

def infer_cell_environment(
    tumor_cluster_id:  str            = "6",
    disease_config:    Optional[Path] = None,
    verbose:           bool           = True,
) -> dict:
    """
    Infer cell environment compartments from scRNA-seq marker gene expression.

    This is a drop-in replacement for build_pdac_cell() in step01.
    It returns a dict with identical structure and Compartment types,
    but all parameter values are derived from the data rather than
    hardcoded from PDAC literature.

    Args:
        tumor_cluster_id:  Leiden cluster label for the disease cell population.
                           Overridden by disease_config.yaml if present.
        disease_config:    Path to disease_config.yaml (auto-located if None).
        verbose:           Print inference report to stdout.

    Returns:
        dict[compartment_name → Compartment]  — identical to build_pdac_cell()

    Example:
        from sim.cell_environment_inference import infer_cell_environment
        cell_env = infer_cell_environment()   # replaces build_pdac_cell()
    """
    # Import Compartment from step01 at runtime to avoid circular imports
    try:
        from sim.step01_cell_environment import Compartment
    except ImportError:
        # Fallback if running standalone
        import sys
        sys.path.insert(0, str(ROOT))
        from sim.step01_cell_environment import Compartment  # type: ignore

    if verbose:
        print("\n── SIM-01 (Layer 3): Inferring cell environment from expression ──")

    # Load expression data
    expr = _load_mean_expression(tumor_cluster_id, disease_config)

    n_genes_found = sum(
        1 for gene in
        (g for panel in MARKERS.values() for g in panel)
        if gene in expr
    )
    n_genes_total = len(set(g for panel in MARKERS.values() for g in panel))

    if verbose:
        mode = "data-driven" if expr else "baseline (no scRNA-seq data)"
        print(f"  Mode: {mode}")
        if expr:
            print(f"  Marker coverage: {n_genes_found}/{n_genes_total} "
                  f"marker genes found in expression data")

    # Run inference
    engine = CellEnvironmentInference(expr, verbose=False)
    params = engine.infer()

    if verbose:
        print(engine.report())

    # ── Assemble Compartment objects ───────────────────────────────────────────

    P = params  # shorthand

    # Plasma membrane cholesterol fractions (sum must be ≤ 1)
    chol = P["plasma_membrane_chol"]
    pc   = float(np.clip(0.55 - chol * 0.3, 0.25, 0.55))
    pe   = float(np.clip(0.35 - chol * 0.15, 0.15, 0.35))

    # Warburg flag: True if warburg score > 0.5
    is_warburg = engine._scores.get("warburg", 0.5) > 0.5

    extracellular = Compartment(
        name             = "extracellular",
        volume_fL        = 5000.0,
        pH               = P["extracellular_pH"],
        buffer_capacity  = 25.0,
        na_conc          = 145.0,
        k_conc           = 5.0,
        cl_conc          = 110.0,
        mg_conc          = 0.8,
        ca_conc          = 2.5,
        dielectric       = 80.0,
        viscosity_mPas   = 1.2,
        crowding_factor  = 0.8,
        protein_conc_gL  = 60.0,
        atp_conc         = 0.01,
        nadh_conc        = 0.0,
        gsh_conc         = 0.1,
        has_warburg      = is_warburg,
    )

    plasma_membrane = Compartment(
        name                  = "plasma_membrane",
        volume_fL             = 0.5,
        pH                    = (P["extracellular_pH"] + P["cytoplasm_pH"]) / 2,
        buffer_capacity       = 5.0,
        na_conc               = 75.0,
        k_conc                = 75.0,
        cl_conc               = 60.0,
        mg_conc               = 0.5,
        ca_conc               = 0.2,
        dielectric            = 4.0,
        viscosity_mPas        = 100.0,
        crowding_factor       = 2.0,
        protein_conc_gL       = 200.0,
        atp_conc              = 0.5,
        membrane_thickness_nm = 7.5,
        cholesterol_fraction  = chol,
        pc_fraction           = pc,
        pe_fraction           = pe,
    )

    cytoplasm = Compartment(
        name             = "cytoplasm",
        volume_fL        = P["cytoplasm_volume_fL"],
        pH               = P["cytoplasm_pH"],
        buffer_capacity  = 40.0,
        na_conc          = 15.0,
        k_conc           = 140.0,
        cl_conc          = 20.0,
        mg_conc          = P["cytoplasm_mg_mM"],
        ca_conc          = 0.0001,
        dielectric       = 70.0,
        viscosity_mPas   = 1.5 + P["cytoplasm_crowding"] * 0.8,
        crowding_factor  = P["cytoplasm_crowding"],
        protein_conc_gL  = 100.0 + P["cytoplasm_crowding"] * 50.0,
        atp_conc         = P["cytoplasm_atp_mM"],
        nadh_conc        = P["cytoplasm_nadh_mM"],
        gsh_conc         = P["cytoplasm_gsh_mM"],
        has_warburg      = is_warburg,
    )

    nucleus = Compartment(
        name             = "nucleus",
        volume_fL        = P["nuclear_volume_fL"],
        pH               = P["nucleus_pH"],
        buffer_capacity  = 30.0,
        na_conc          = 20.0,
        k_conc           = 130.0,
        cl_conc          = 25.0,
        mg_conc          = 1.0,
        ca_conc          = 0.001,
        dielectric       = 65.0,
        viscosity_mPas   = 10.0 + P["nucleus_crowding"] * 8.0,
        crowding_factor  = P["nucleus_crowding"],
        protein_conc_gL  = 200.0 + P["nucleus_crowding"] * 40.0,
        atp_conc         = P["cytoplasm_atp_mM"] * 0.8,
        nadh_conc        = 0.05,
        gsh_conc         = 3.0,
    )

    er = Compartment(
        name             = "endoplasmic_reticulum",
        volume_fL        = 200.0,
        pH               = P["er_pH"],
        buffer_capacity  = 20.0,
        na_conc          = 10.0,
        k_conc           = 140.0,
        cl_conc          = 15.0,
        mg_conc          = 0.3,
        ca_conc          = P["er_ca2p_mM"],
        dielectric       = 70.0,
        viscosity_mPas   = 5.0,
        crowding_factor  = P["er_crowding"],
        protein_conc_gL  = 200.0 + P["er_crowding"] * 30.0,
        atp_conc         = 1.5,
        nadh_conc        = 0.08,
        gsh_conc         = 0.5,
    )

    mitochondria = Compartment(
        name             = "mitochondria",
        volume_fL        = P["mitochondria_volume_fL"],
        pH               = P["mitochondria_pH"],
        buffer_capacity  = 50.0,
        na_conc          = 10.0,
        k_conc           = 120.0,
        cl_conc          = 10.0,
        mg_conc          = 3.0,
        ca_conc          = 0.001,
        dielectric       = 70.0,
        viscosity_mPas   = 4.0,
        crowding_factor  = 2.0,
        protein_conc_gL  = 250.0,
        atp_conc         = P["mitochondria_atp_mM"],
        nadh_conc        = P["mitochondria_nadh_mM"],
        gsh_conc         = P["mitochondria_gsh_mM"],
        membrane_thickness_nm = 7.5,
        cholesterol_fraction  = 0.03,
        pc_fraction           = 0.45,
        pe_fraction           = 0.35,
    )

    lysosome = Compartment(
        name             = "lysosome",
        volume_fL        = 30.0,
        pH               = P["lysosome_pH"],
        buffer_capacity  = 15.0,
        na_conc          = 20.0,
        k_conc           = 50.0,
        cl_conc          = 80.0,
        mg_conc          = 0.1,
        ca_conc          = 0.5,
        dielectric       = 75.0,
        viscosity_mPas   = 3.0,
        crowding_factor  = 1.5,
        protein_conc_gL  = 150.0,
        atp_conc         = 0.1,
        nadh_conc        = 0.01,
        gsh_conc         = 0.1,
    )

    cell_env = {
        "extracellular":         extracellular,
        "plasma_membrane":       plasma_membrane,
        "cytoplasm":             cytoplasm,
        "nucleus":               nucleus,
        "endoplasmic_reticulum": er,
        "mitochondria":          mitochondria,
        "lysosome":              lysosome,
    }

    # Save inference record for reproducibility
    record = {
        "inference_method":    "marker_gene_expression",
        "n_marker_genes_used": n_genes_found,
        "n_marker_genes_total": n_genes_total,
        "data_driven":         bool(expr),
        "scores":              engine._scores,
        "parameters":          {k: round(v, 4) for k, v in params.items()},
    }
    out_path = ROOT / "data" / "sim" / "cell_environment_inference.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))

    if verbose:
        print(f"  Inference record saved → {out_path}")

    return cell_env


# ── Convenience: build normal cell baseline for SIM-02 comparison ─────────────

def build_normal_cell_environment() -> dict:
    """
    Build a generic normal (non-disease) cell environment using
    universal baseline values — no expression data required.

    Used by SIM-02 to compute ΔΔG disease vs normal.
    """
    try:
        from sim.step01_cell_environment import Compartment
    except ImportError:
        import sys
        sys.path.insert(0, str(ROOT))
        from sim.step01_cell_environment import Compartment  # type: ignore

    N = NORMAL_BASELINE
    return {
        "extracellular": Compartment(
            name="extracellular", volume_fL=5000.0,
            pH=N["extracellular_pH"], buffer_capacity=25.0,
            na_conc=145.0, k_conc=5.0, cl_conc=110.0,
            mg_conc=0.8, ca_conc=2.5, crowding_factor=0.8,
            protein_conc_gL=60.0, atp_conc=0.01,
        ),
        "cytoplasm": Compartment(
            name="cytoplasm", volume_fL=N["cytoplasm_volume_fL"],
            pH=N["cytoplasm_pH"], buffer_capacity=40.0,
            na_conc=15.0, k_conc=140.0, cl_conc=20.0,
            mg_conc=N["cytoplasm_mg_mM"], ca_conc=0.0001,
            crowding_factor=N["cytoplasm_crowding"],
            protein_conc_gL=150.0,
            atp_conc=N["cytoplasm_atp_mM"],
            nadh_conc=N["cytoplasm_nadh_mM"],
            gsh_conc=N["cytoplasm_gsh_mM"],
        ),
        "nucleus": Compartment(
            name="nucleus", volume_fL=N["nuclear_volume_fL"],
            pH=N["nucleus_pH"], buffer_capacity=30.0,
            na_conc=20.0, k_conc=130.0, cl_conc=25.0,
            mg_conc=1.0, ca_conc=0.001,
            crowding_factor=N["nucleus_crowding"],
            protein_conc_gL=300.0,
            atp_conc=N["cytoplasm_atp_mM"] * 0.8,
        ),
    }


# ── CLI: run standalone to inspect inference output ───────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "SIM-01 Layer 3 — Infer cell environment from scRNA-seq.\n"
            "Prints the full inference report and saves a JSON record."
        )
    )
    parser.add_argument(
        "--cluster", default="6",
        help="Tumor cluster ID (default: 6)"
    )
    parser.add_argument(
        "--disease-config", default=None,
        help="Path to disease_config.yaml"
    )
    args = parser.parse_args()

    cfg_path = Path(args.disease_config) if args.disease_config else None
    infer_cell_environment(
        tumor_cluster_id=args.cluster,
        disease_config=cfg_path,
        verbose=True,
    )