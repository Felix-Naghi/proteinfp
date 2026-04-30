"""
sim/step05_network_perturbation.py
───────────────────────────────────
Module SIM-05 — Network Perturbation Model

Takes binding probabilities from Module 4 and propagates them
through the PDAC Gene Regulatory Network to predict:

    1. How does drug binding change the network state?
    2. Which downstream genes are affected?
    3. How much does the tumor GRN entropy change vs normal?
    4. What escape/resistance pathways activate?

Mathematics:
    The GRN is modeled as a system of ODEs:
    
    dxi/dt = fi(x) - γi * xi
    
    Where:
        xi    = expression/activity of gene i (normalized 0-1)
        fi(x) = regulatory input function (Hill kinetics)
        γi    = degradation rate

    Hill kinetics for gene i regulated by j:
        fi(x) = Σj [ wij * xj^n / (Kij^n + xj^n) ]
    
    Where:
        wij = edge weight from GRN (GENIE3 importance score)
        n   = Hill coefficient (cooperativity, default 2)
        Kij = half-saturation constant (default 0.5)

    Drug perturbation:
        When drug binds protein i with probability P(bind),
        the activity of node i is reduced:
        xi_perturbed = xi_baseline * (1 - P(bind) * efficacy)
    
    We simulate:
        1. Baseline steady state (no drug)
        2. Perturbed steady state (with drug)
        3. Compute ΔS = Shannon entropy change
        4. Propagate to downstream targets (cascade analysis)

    Differential entropy (selectivity):
        ΔΔS = ΔS(tumor) - ΔS(normal)
        Positive = drug disrupts tumor more than normal = GOOD

Output:
    - Network state vector (baseline vs perturbed)
    - Entropy change per node
    - Total ΔS (network disruption score)
    - ΔΔS (tumor selectivity)
    - Top 20 most affected downstream genes
    - Predicted resistance genes (upregulated compensators)

Usage:
    python sim/step05_network_perturbation.py --drug gemcitabine
    python sim/step05_network_perturbation.py --drug gemcitabine --dose 100
"""

from __future__ import annotations

import json
import math
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parent.parent
SIM_DIR  = ROOT / "data" / "sim"
GRN_DIR  = ROOT / "data" / "grn" / "intermediate"
OUT_DIR  = SIM_DIR / "perturbation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── ODE parameters ────────────────────────────────────────────────────────────

HILL_N         = 2.0    # Hill coefficient (cooperativity)
HILL_K         = 0.5    # Half-saturation constant
DEGRADATION    = 0.1    # Default degradation rate γ (per time unit)
DT             = 0.01   # Integration timestep
N_STEPS        = 5000   # Steps to reach steady state
CONVERGENCE    = 1e-6   # Convergence criterion

# Efficacy: how much does binding reduce protein activity?
# 1.0 = complete inhibition, 0.5 = 50% inhibition at Kd
# We use a Hill-like efficacy: eff = P(bind)^0.5 (sublinear)
BINDING_EFFICACY = 0.8  # maximum efficacy at saturation


# ── GRN loading ───────────────────────────────────────────────────────────────

def load_grn(max_edges: int = 5000) -> tuple[list, dict, np.ndarray]:
    """
    Load PDAC tumor GRN from GENIE3 output.
    Returns: (genes, gene_index, weight_matrix)
    """
    edges_path = GRN_DIR / "genie3_pure_tumor_edges.csv"
    if not edges_path.exists():
        edges_path = GRN_DIR / "genie3_tumor_edges.csv"
    if not edges_path.exists():
        edges_path = GRN_DIR / "genie3_edges.csv"

    print(f"  Loading GRN from {edges_path.name}...")

    import csv
    edges = []
    with open(edges_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append({
                "regulator":      row["Regulator"],
                "target":         row["Target"],
                "norm_importance": float(row["norm_importance"]),
            })

    # Sort by importance and take top edges
    edges.sort(key=lambda e: -e["norm_importance"])
    edges = edges[:max_edges]

    # Build gene set
    genes_set = set()
    for e in edges:
        genes_set.add(e["regulator"])
        genes_set.add(e["target"])
    genes     = sorted(genes_set)
    gene_idx  = {g: i for i, g in enumerate(genes)}
    n         = len(genes)

    print(f"  Genes: {n}  Edges: {len(edges)}")

    # Build weight matrix W[i,j] = weight of edge j → i
    W = np.zeros((n, n), dtype=np.float32)
    for e in edges:
        i = gene_idx.get(e["target"])
        j = gene_idx.get(e["regulator"])
        if i is not None and j is not None:
            W[i, j] = float(e["norm_importance"])

    # Normalize rows so total input to each gene sums to 1
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = W / row_sums

    return genes, gene_idx, W


def load_baseline_expression(
    genes:    list,
    gene_idx: dict,
) -> np.ndarray:
    """
    Load baseline gene expression from scRNA-seq data (tumor cluster 6).
    Normalized to 0-1 range.
    """
    try:
        import scanpy as sc

        adata    = sc.read_h5ad(
            ROOT / "data" / "grn" / "intermediate" / "preprocessed.h5ad"
        )
        tumor    = adata[adata.obs["leiden"] == "6"]

        if hasattr(tumor.X, "toarray"):
            X = tumor.X.toarray()
        else:
            X = np.array(tumor.X)

        mean_expr  = X.mean(axis=0)
        max_expr   = mean_expr.max()
        norm_expr  = mean_expr / (max_expr + 1e-10)

        x0 = np.full(len(genes), 0.3, dtype=np.float32)  # default
        for i, gene in enumerate(genes):
            if gene in tumor.var_names:
                idx      = list(tumor.var_names).index(gene)
                x0[i]   = float(norm_expr[idx])

        print(f"  Loaded expression for {len(genes)} genes from scRNA-seq")
        return x0

    except Exception as e:
        print(f"  scRNA-seq load failed ({e}) — using uniform baseline")
        return np.full(len(genes), 0.3, dtype=np.float32)


# ── ODE system ────────────────────────────────────────────────────────────────

def hill_activation(x: np.ndarray, n: float = HILL_N,
                    K: float = HILL_K) -> np.ndarray:
    """
    Hill activation function: f(x) = x^n / (K^n + x^n)
    Smooth sigmoid-like response.
    """
    xn = np.power(np.maximum(x, 0), n)
    Kn = K ** n
    return xn / (Kn + xn)


def grn_ode(
    x:    np.ndarray,
    W:    np.ndarray,
    gamma: np.ndarray,
    basal: np.ndarray,
) -> np.ndarray:
    """
    dx/dt = W @ f(x) + basal - gamma * x
    
    W     : weight matrix [n x n]
    f(x)  : Hill activation of each node
    basal : basal expression rate (prevents complete silencing)
    gamma : degradation rate
    """
    fx    = hill_activation(x)
    input_signal = W @ fx
    dxdt  = input_signal + basal - gamma * x
    return dxdt


def simulate_steady_state(
    x0:    np.ndarray,
    W:     np.ndarray,
    gamma: np.ndarray,
    basal: np.ndarray,
    perturbation: Optional[np.ndarray] = None,
    n_steps: int = N_STEPS,
    dt: float = DT,
) -> tuple[np.ndarray, int]:
    """
    Integrate ODE system to steady state using RK4 method.
    RK4 is more accurate than Euler and handles stiff systems better.

    The perturbation is applied as a continuous activity reduction:
        dx/dt = f(x) - gamma*x - perturbation*x

    This means the perturbation scales with current activity level,
    making the system sensitive to perturbation magnitude.

    Returns steady-state x and number of steps taken.
    """
    x = x0.copy()

    def _dxdt(x_):
        dx = grn_ode(x_, W, gamma, basal)
        if perturbation is not None:
            # Perturbation as continuous degradation term
            # Larger perturbation = faster degradation of target activity
            dx -= perturbation * x_ * (1 + perturbation * 2)
        return dx

    for step in range(n_steps):
        # RK4 integration
        k1 = _dxdt(x)
        k2 = _dxdt(np.clip(x + 0.5 * dt * k1, 0, 1))
        k3 = _dxdt(np.clip(x + 0.5 * dt * k2, 0, 1))
        k4 = _dxdt(np.clip(x + dt * k3, 0, 1))

        x_new = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        x_new = np.clip(x_new, 0, 1)

        if np.max(np.abs(x_new - x)) < CONVERGENCE:
            return x_new, step

        x = x_new

    return x, n_steps


# ── Entropy computation ───────────────────────────────────────────────────────

def shannon_entropy(x: np.ndarray) -> float:
    """
    Shannon entropy of normalized expression state vector.
    
    S = -Σi pi * log(pi)
    where pi = xi / Σ xi  (normalized to probability distribution)
    
    High entropy = many genes expressed at similar levels (disordered)
    Low entropy  = few genes dominating (ordered/specific state)
    """
    total = x.sum()
    if total <= 0:
        return 0.0
    p    = x / total
    p    = p[p > 1e-10]  # remove zeros
    S    = -np.sum(p * np.log(p))
    return float(S)


def differential_entropy(
    x_baseline:  np.ndarray,
    x_perturbed: np.ndarray,
) -> float:
    """
    ΔS = S(perturbed) - S(baseline)
    
    Positive: drug increases entropy (disrupts ordered state)
              → good for cancer (disrupts tumor gene program)
    Negative: drug decreases entropy (orders state further)
              → may reinforce tumor state (bad)
    """
    S_base = shannon_entropy(x_baseline)
    S_pert = shannon_entropy(x_perturbed)
    return S_pert - S_base


# ── Normal cell model ─────────────────────────────────────────────────────────

def build_normal_cell_state(
    genes:    list,
    gene_idx: dict,
) -> np.ndarray:
    """
    Build normal ductal cell baseline state.
    Uses lower expression of PDAC driver genes.
    """
    try:
        import scanpy as sc
        import pandas as pd

        # Load normal ductal cells from GSE84133
        base = ROOT / "data" / "grn" / "input"
        normal_files = (
            list(base.glob("*human*/*human*umifm*.csv")) or
            list(base.glob("*human*umifm*.csv"))
        )
        if normal_files:
            dfs = []
            for f in normal_files[:4]:
                df = pd.read_csv(f, index_col=0)
                ductal = df[df["assigned_cluster"] == "ductal"].drop(
                    columns=["barcode", "assigned_cluster"], errors="ignore"
                )
                dfs.append(ductal)
            normal_df  = pd.concat(dfs)
            mean_normal = normal_df.mean()
            max_expr    = mean_normal.max()
            norm_normal = mean_normal / (max_expr + 1e-10)

            x_normal = np.full(len(genes), 0.2, dtype=np.float32)
            for i, gene in enumerate(genes):
                if gene in norm_normal.index:
                    x_normal[i] = float(norm_normal[gene])
            print(f"  Loaded normal ductal expression for comparison")
            return x_normal
    except Exception as e:
        print(f"  Normal cell load failed ({e}) — using scaled baseline")

    # Fallback: normal cells have ~70% of tumor expression
    return np.full(len(genes), 0.2, dtype=np.float32)


# ── Cascade analysis ──────────────────────────────────────────────────────────

def find_affected_genes(
    x_baseline:  np.ndarray,
    x_perturbed: np.ndarray,
    genes:       list,
    top_n:       int = 20,
) -> tuple[list, list]:
    """
    Find genes most affected by drug perturbation.
    
    Returns:
        downregulated: genes with largest decrease in activity
        upregulated:   genes with largest increase (resistance candidates)
    """
    delta = x_perturbed - x_baseline

    # Downregulated (suppressed by drug)
    down_idx = np.argsort(delta)[:top_n]
    down = [(genes[i], float(x_baseline[i]),
             float(x_perturbed[i]), float(delta[i]))
            for i in down_idx if delta[i] < -0.01]

    # Upregulated (potential resistance genes)
    up_idx = np.argsort(delta)[-top_n:][::-1]
    up = [(genes[i], float(x_baseline[i]),
           float(x_perturbed[i]), float(delta[i]))
          for i in up_idx if delta[i] > 0.01]

    return down, up


def compute_cascade_depth(
    perturbed_genes: set,
    W:               np.ndarray,
    gene_idx:        dict,
    genes:           list,
    max_depth:       int = 5,
) -> dict:
    """
    BFS to find cascade of affected genes.
    Returns dict: gene → shortest path length from perturbed set.
    """
    from collections import deque

    visited = {}
    queue   = deque()

    for gene in perturbed_genes:
        if gene in gene_idx:
            queue.append((gene_idx[gene], 0))
            visited[gene] = 0

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue

        # Find downstream genes (rows where W[i, node] > 0)
        downstream = np.where(W[:, node] > 0.01)[0]
        for ds in downstream:
            g = genes[ds]
            if g not in visited:
                visited[g] = depth + 1
                queue.append((ds, depth + 1))

    return visited


# ── Perturbation vector from binding scores ───────────────────────────────────

def build_perturbation_vector(
    binding_scores: list,
    genes:          list,
    gene_idx:       dict,
) -> np.ndarray:
    """
    Convert binding probabilities to perturbation vector.

    Uses a sigmoid amplification so moderate binding (P=0.3)
    produces meaningful network perturbation:
        pert[i] = sigmoid(P(bind) * 5) * BINDING_EFFICACY

    This ensures dose-dependent response is visible in entropy.
    """
    import math
    pert = np.zeros(len(genes), dtype=np.float32)

    for score in binding_scores:
        gene   = score.get("target_gene", "")
        if gene in gene_idx:
            p_bind = float(score.get("p_binding", 0))
            # Sigmoid amplification: small P gives small pert,
            # P=0.3 gives ~0.52, P=0.8 gives ~0.90
            amplified = 1 / (1 + math.exp(-p_bind * 8 + 2))
            pert[gene_idx[gene]] = float(amplified * BINDING_EFFICACY)

    return pert


# ── Main perturbation analysis ────────────────────────────────────────────────

def run_perturbation_analysis(
    drug_name:      str,
    binding_scores: list,
    dose_uM:        float = 10.0,
    verbose:        bool  = True,
) -> dict:
    """
    Full network perturbation analysis.
    """
    if verbose:
        print(f"\n{'='*65}")
        print(f"  SIM-05: Network Perturbation — {drug_name}")
        print(f"  Dose: {dose_uM} μM")
        print(f"{'='*65}")

    # ── Load GRN ──────────────────────────────────────────────────────────
    genes, gene_idx, W = load_grn(max_edges=3000)
    n = len(genes)

    # ── Baseline expression ───────────────────────────────────────────────
    print("\n  [1/6] Loading baseline expression...")
    x0_tumor  = load_baseline_expression(genes, gene_idx)
    x0_normal = build_normal_cell_state(genes, gene_idx)

    # ODE parameters
    gamma = np.full(n, DEGRADATION, dtype=np.float32)
    basal = x0_tumor * 0.05  # small basal production rate

    # ── Simulate tumor baseline ───────────────────────────────────────────
    print("  [2/6] Simulating tumor baseline steady state...")
    x_tumor_base, steps = simulate_steady_state(x0_tumor, W, gamma, basal)
    print(f"    Converged in {steps} steps")
    S_tumor_base = shannon_entropy(x_tumor_base)
    print(f"    Baseline entropy: {S_tumor_base:.4f} nats")

    # ── Build perturbation vector ─────────────────────────────────────────
    print("  [3/6] Building drug perturbation vector...")
    pert = build_perturbation_vector(binding_scores, genes, gene_idx)
    n_perturbed = (pert > 0).sum()
    print(f"    Directly perturbed genes: {n_perturbed}")
    if n_perturbed > 0:
        for score in binding_scores:
            gene = score.get("target_gene", "")
            if gene in gene_idx and pert[gene_idx[gene]] > 0:
                print(f"      {gene}: perturbation = "
                      f"{pert[gene_idx[gene]]:.4f}")

    # ── Simulate tumor perturbed ──────────────────────────────────────────
    print("  [4/6] Simulating tumor perturbed steady state...")
    x_tumor_pert, steps = simulate_steady_state(
        x_tumor_base, W, gamma, basal, perturbation=pert
    )
    print(f"    Converged in {steps} steps")
    S_tumor_pert = shannon_entropy(x_tumor_pert)
    delta_S_tumor = differential_entropy(x_tumor_base, x_tumor_pert)
    print(f"    Perturbed entropy: {S_tumor_pert:.4f} nats")
    print(f"    ΔS (tumor):        {delta_S_tumor:+.4f} nats")

    # ── Simulate normal cell perturbed ────────────────────────────────────
    print("  [5/6] Simulating normal cell perturbation...")
    basal_normal = x0_normal * 0.05
    x_normal_base, _ = simulate_steady_state(
        x0_normal, W, gamma, basal_normal
    )
    x_normal_pert, steps = simulate_steady_state(
        x_normal_base, W, gamma, basal_normal, perturbation=pert
    )
    S_normal_base  = shannon_entropy(x_normal_base)
    delta_S_normal = differential_entropy(x_normal_base, x_normal_pert)
    print(f"    ΔS (normal):       {delta_S_normal:+.4f} nats")

    # Differential entropy
    delta_delta_S = delta_S_tumor - delta_S_normal
    print(f"    ΔΔS (tumor - normal): {delta_delta_S:+.4f} nats")

    # ── Cascade analysis ──────────────────────────────────────────────────
    print("  [6/6] Cascade analysis...")
    down_genes, up_genes = find_affected_genes(
        x_tumor_base, x_tumor_pert, genes, top_n=20
    )

    # Cascade depth
    direct_perturbed = {score["target_gene"] for score in binding_scores
                        if score.get("p_binding", 0) > 0.001}
    cascade = compute_cascade_depth(direct_perturbed, W, gene_idx, genes)

    if verbose:
        _print_perturbation_report(
            drug_name, dose_uM, genes,
            x_tumor_base, x_tumor_pert,
            x_normal_base, x_normal_pert,
            S_tumor_base, S_tumor_pert,
            S_normal_base, delta_S_tumor, delta_S_normal,
            delta_delta_S, down_genes, up_genes, cascade,
            binding_scores,
        )

    return {
        "drug_name":       drug_name,
        "dose_uM":         dose_uM,
        "n_genes":         n,
        "n_perturbed":     int(n_perturbed),
        "S_tumor_base":    round(float(S_tumor_base), 4),
        "S_tumor_pert":    round(float(S_tumor_pert), 4),
        "S_normal_base":   round(float(S_normal_base), 4),
        "delta_S_tumor":   round(float(delta_S_tumor), 4),
        "delta_S_normal":  round(float(delta_S_normal), 4),
        "delta_delta_S":   round(float(delta_delta_S), 4),
        "n_downregulated": len(down_genes),
        "n_upregulated":   len(up_genes),
        "downregulated":   [(g, round(b,4), round(p,4), round(d,4))
                            for g,b,p,d in down_genes],
        "upregulated":     [(g, round(b,4), round(p,4), round(d,4))
                            for g,b,p,d in up_genes],
        "cascade_depth":   {k: v for k, v in
                            sorted(cascade.items(), key=lambda x: x[1])[:30]},
        "selectivity_interpretation": _interpret_selectivity(delta_delta_S),
    }


def _interpret_selectivity(ddS: float) -> str:
    if ddS > 0.5:
        return "HIGHLY TUMOR SELECTIVE — drug disrupts tumor GRN much more than normal"
    elif ddS > 0.1:
        return "MODERATELY TUMOR SELECTIVE — some tumor preference"
    elif ddS > -0.1:
        return "NON-SELECTIVE — similar effect on tumor and normal GRN"
    elif ddS > -0.5:
        return "MILDLY NORMAL-PREFERRING — slightly more disruption in normal cells"
    else:
        return "NORMAL-CELL TOXIC — drug disrupts normal GRN more than tumor"


def _print_perturbation_report(
    drug_name, dose_uM, genes,
    x_tb, x_tp, x_nb, x_np,
    S_tb, S_tp, S_nb,
    dS_tumor, dS_normal, ddS,
    down_genes, up_genes, cascade,
    binding_scores,
):
    print(f"\n{'='*65}")
    print(f"  NETWORK PERTURBATION REPORT — {drug_name} @ {dose_uM} μM")
    print(f"{'='*65}")

    print(f"\n  ENTROPY ANALYSIS:")
    print(f"  {'Metric':<35} {'Tumor':>10} {'Normal':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}")
    print(f"  {'Baseline entropy (nats)':<35} {S_tb:>10.4f} {S_nb:>10.4f}")
    print(f"  {'Perturbed entropy (nats)':<35} {S_tp:>10.4f} {'N/A':>10}")
    print(f"  {'ΔS (drug effect)':<35} {dS_tumor:>+10.4f} {dS_normal:>+10.4f}")
    print(f"  {'ΔΔS (tumor - normal)':<35} {ddS:>+10.4f}")
    print(f"\n  Interpretation: {_interpret_selectivity(ddS)}")

    print(f"\n  DIRECTLY PERTURBED TARGETS:")
    for score in binding_scores:
        gene   = score.get("target_gene", "")
        p_bind = score.get("p_binding", 0)
        pert_v = p_bind * BINDING_EFFICACY
        if p_bind > 0.0001:
            print(f"    {gene:<12} P(bind)={p_bind:.4f}  "
                  f"activity reduction={pert_v:.4f}")

    if down_genes:
        print(f"\n  TOP DOWNREGULATED GENES (drug-suppressed):")
        print(f"  {'Gene':<12} {'Baseline':>10} {'Perturbed':>10} {'Delta':>8}")
        for gene, base, pert, delta in down_genes[:10]:
            print(f"  {gene:<12} {base:>10.4f} {pert:>10.4f} {delta:>+8.4f}")

    if up_genes:
        print(f"\n  TOP UPREGULATED GENES (potential resistance):")
        print(f"  {'Gene':<12} {'Baseline':>10} {'Perturbed':>10} {'Delta':>8}")
        for gene, base, pert, delta in up_genes[:10]:
            print(f"  {gene:<12} {base:>10.4f} {pert:>10.4f} {delta:>+8.4f}")

    if cascade:
        print(f"\n  CASCADE PROPAGATION (depth from direct targets):")
        by_depth = {}
        for gene, depth in cascade.items():
            by_depth.setdefault(depth, []).append(gene)
        for depth in sorted(by_depth.keys())[:4]:
            genes_at_depth = by_depth[depth][:8]
            print(f"    Depth {depth}: {', '.join(genes_at_depth)}"
                  + (" ..." if len(by_depth[depth]) > 8 else ""))

    print(f"\n{'='*65}")
    print(f"  PHARMACOLOGICAL SUMMARY")
    print(f"{'='*65}")
    print(f"  Drug                : {drug_name}")
    print(f"  Dose                : {dose_uM} μM")
    print(f"  GRN disruption (ΔS) : {dS_tumor:+.4f} nats")
    print(f"  Tumor selectivity   : ΔΔS = {ddS:+.4f} nats")
    print(f"  Genes downregulated : {len(down_genes)}")
    print(f"  Resistance genes    : {len(up_genes)}")
    print(f"  Cascade depth reach : {max(cascade.values()) if cascade else 0}")


# ── Dose scan ─────────────────────────────────────────────────────────────────

def dose_scan(
    drug_name:       str,
    binding_path:    Path,
    doses:           list = None,
) -> dict:
    """
    Run perturbation analysis across a range of doses.
    Find the dose where ΔΔS is maximized (optimal selectivity).
    """
    if doses is None:
        doses = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]

    base_scores = json.loads(binding_path.read_text())["scores"]
    base_dose   = json.loads(binding_path.read_text())["dose_uM"]

    results = {}
    for dose in doses:
        # Scale binding probabilities linearly with dose
        scale  = dose / base_dose
        scaled = []
        for s in base_scores:
            s2 = dict(s)
            # P(binding) scales non-linearly with dose via receptor occupancy
            Kd   = s.get("Kd_corrected_uM", 100)
            p    = dose / (dose + Kd)  # receptor occupancy at this dose
            p   *= s.get("p_protein_open", 0.7)
            s2["p_binding"] = float(p)
            scaled.append(s2)

        result = run_perturbation_analysis(
            drug_name, scaled, dose, verbose=False
        )
        results[dose] = result
        print(f"    Dose {dose:>7.1f} μM:  "
              f"ΔS={result['delta_S_tumor']:+.4f}  "
              f"ΔΔS={result['delta_delta_S']:+.4f}  "
              f"({result['selectivity_interpretation'][:30]})")

    # Find optimal dose
    best_dose = max(results, key=lambda d: results[d]["delta_delta_S"])
    print(f"\n  Optimal dose for tumor selectivity: {best_dose} μM")
    print(f"  ΔΔS at optimal dose: "
          f"{results[best_dose]['delta_delta_S']:+.4f} nats")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    drug_name:  str   = "gemcitabine",
    dose_uM:    float = 10.0,
    scan:       bool  = False,
):
    print("=" * 65)
    print("  SIM-05: Network Perturbation Model")
    print("  ODE-based GRN dynamics + Shannon entropy scoring")
    print("=" * 65)

    # Load binding scores from Module 4
    binding_path = (SIM_DIR / "binding" /
                    f"{drug_name.lower()}_binding.json")
    if not binding_path.exists():
        print(f"ERROR: {binding_path} not found")
        print("Run sim/step04_binding_probability.py first")
        return

    binding_data   = json.loads(binding_path.read_text())
    binding_scores = binding_data["scores"]

    print(f"\n  Loaded {len(binding_scores)} binding scores for {drug_name}")
    for s in binding_scores:
        print(f"    {s['target_gene']:<10} P(bind)={s['p_binding']:.6f}  "
              f"Kd={s['Kd_corrected_uM']:.1f} μM")

    # Run perturbation analysis
    result = run_perturbation_analysis(
        drug_name, binding_scores, dose_uM
    )

    # Optional dose scan
    if scan:
        print(f"\n{'='*65}")
        print(f"  DOSE SCAN ANALYSIS")
        print(f"{'='*65}")
        scan_results = dose_scan(drug_name, binding_path)
        result["dose_scan"] = {
            str(d): {
                "delta_S_tumor":  r["delta_S_tumor"],
                "delta_delta_S":  r["delta_delta_S"],
            }
            for d, r in scan_results.items()
        }

    # Save
    out_path = OUT_DIR / f"{drug_name.lower()}_perturbation.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Results saved to {out_path}")

    print(f"\n{'='*65}")
    print(f"  SIM-05 complete. Ready for SIM-06 (Pharmacological Scoring)")
    print(f"{'='*65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SIM-05: Network Perturbation Model"
    )
    parser.add_argument("--drug",  default="gemcitabine")
    parser.add_argument("--dose",  type=float, default=10.0)
    parser.add_argument("--scan",  action="store_true",
                        help="Run dose scan to find optimal selectivity")
    args = parser.parse_args()
    main(args.drug, args.dose, args.scan)