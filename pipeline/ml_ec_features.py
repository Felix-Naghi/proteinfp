"""
pipeline/ml_ec_features.py
──────────────────────────
Feature Engineering for ML-based EC Classification.

Produces a rich, high-dimensional feature vector from protein data:

  Block A — Sequence composition (660 dims)
    • AAC:  20-dim amino acid composition
    • DPC:  400-dim dipeptide composition
    • TPC:  8000-dim tripeptide composition (PCA-reduced to 64)
    • CTD:  21-dim physicochemical property CTD features
    • PAAC: 50-dim pseudo amino acid composition
    • APAAC: 50-dim amphiphilic PAAC

  Block B — ESM-2 embedding features (128 dims)
    • Projected/pooled from 1280-dim per-protein embedding
    • 8 statistical moments per cluster of dims

  Block C — Structural/topological (86 dims)
    • pLDDT statistics (8 dims)
    • Secondary structure fractions (3 dims)
    • Buried/exposed residue counts from SASA (6 dims)
    • Active site geometry (12 dims)
    • Pocket druggability (8 dims)
    • Contact map graph features (12 dims)
    • ENM flexibility statistics (8 dims)
    • Disulphide bond count / Cys-pair topology (6 dims)
    • Cofactor binding fingerprint (23 dims — metal types)

  Block D — Evidence signals (42 dims)
    • Homology hits (BLAST identity, E-value bins) (10 dims)
    • GO term evidence (12 dims, one-hot broad MF categories)
    • Motif hit indicators (20 dims)

Total raw feature vector: 916 dims (TPC PCA reduces to ~800 effective)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

# ── Amino acid constants ───────────────────────────────────────────────────────

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX  = {aa: i for i, aa in enumerate(AA_LIST)}

# Physicochemical scales used in CTD & PAAC
_HYDROPHOBICITY = {
    "A": 0.62, "C": 0.29, "D":-0.90, "E":-0.74, "F": 1.19,
    "G": 0.48, "H":-0.40, "I": 1.38, "K":-1.50, "L": 1.06,
    "M": 0.64, "N":-0.78, "P": 0.12, "Q":-0.85, "R":-2.53,
    "S":-0.18, "T":-0.05, "V": 1.08, "W": 0.81, "Y": 0.26,
}
_POLARITY = {
    "A": 0.0, "C": 1.0, "D": 3.0, "E": 3.0, "F": 2.0,
    "G": 0.0, "H": 2.0, "I": 1.0, "K": 3.0, "L": 1.0,
    "M": 1.0, "N": 2.0, "P": 0.0, "Q": 2.0, "R": 3.0,
    "S": 1.0, "T": 1.0, "V": 1.0, "W": 2.0, "Y": 2.0,
}
_VOLUME = {
    "A": 1.0, "C": 2.43,"D": 2.78,"E": 3.78,"F": 5.89,
    "G": 0.0, "H": 4.66,"I": 4.0, "K": 4.77,"L": 4.0,
    "M": 4.43,"N": 2.95,"P": 2.72,"Q": 3.95,"R": 6.13,
    "S": 1.60,"T": 2.60,"V": 3.0, "W": 8.08,"Y": 6.47,
}
_CHARGE = {
    "A": 0, "C": 0, "D":-1, "E":-1, "F": 0,
    "G": 0, "H": 1, "I": 0, "K": 1, "L": 0,
    "M": 0, "N": 0, "P": 0, "Q": 0, "R": 1,
    "S": 0, "T": 0, "V": 0, "W": 0, "Y": 0,
}

SCALES = [_HYDROPHOBICITY, _POLARITY, _VOLUME, _CHARGE]

# Metal-binding residue types for cofactor fingerprint
METAL_AAS = {"H", "C", "D", "E", "M", "N", "Q", "S", "T", "Y"}
METAL_TYPES_MOTIFS = [
    ("zinc",      {"C", "H"}, 4),
    ("iron",      {"C", "H"}, 4),
    ("calcium",   {"D", "E"}, 2),
    ("magnesium", {"D", "E", "N"}, 3),
    ("manganese", {"D", "E", "H"}, 3),
    ("copper",    {"H", "C", "M"}, 2),
    ("nickel",    {"H", "C", "D"}, 2),
    ("potassium", {"D", "E", "S"}, 1),
]

# ── Feature extraction functions ───────────────────────────────────────────────

def _aac(seq: str) -> np.ndarray:
    """20-dim amino acid composition."""
    v = np.zeros(20, dtype=np.float32)
    for aa in seq:
        if aa in AA_IDX:
            v[AA_IDX[aa]] += 1
    if len(seq) > 0:
        v /= len(seq)
    return v


def _dpc(seq: str) -> np.ndarray:
    """400-dim dipeptide composition."""
    v = np.zeros(400, dtype=np.float32)
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i+1]
        if a in AA_IDX and b in AA_IDX:
            v[AA_IDX[a] * 20 + AA_IDX[b]] += 1
    total = max(1, len(seq) - 1)
    return v / total


def _tpc_compressed(seq: str, n_components: int = 64) -> np.ndarray:
    """
    Tripeptide composition compressed by random projection to 64 dims.
    Full TPC is 8000-dim; random projection preserves pairwise distances.
    """
    counts: dict[tuple, float] = {}
    for i in range(len(seq) - 2):
        key = (seq[i], seq[i+1], seq[i+2])
        if all(aa in AA_IDX for aa in key):
            counts[key] = counts.get(key, 0.0) + 1.0
    total = max(1.0, float(sum(counts.values())))
    # Build sparse vector then project
    sparse = np.zeros(8000, dtype=np.float32)
    for (a, b, c), cnt in counts.items():
        idx = AA_IDX[a] * 400 + AA_IDX[b] * 20 + AA_IDX[c]
        sparse[idx] = cnt / total
    # Deterministic random projection (fixed seed)
    rng = np.random.RandomState(42)
    proj_matrix = rng.randn(8000, n_components).astype(np.float32) / np.sqrt(n_components)
    return sparse @ proj_matrix


def _ctd_features(seq: str) -> np.ndarray:
    """
    21-dim CTD (Composition, Transition, Distribution) features.
    Uses hydrophobicity, polarity, and volume scales.
    """
    feats = []
    for scale in [_HYDROPHOBICITY, _POLARITY, _VOLUME]:
        vals = np.array([scale.get(aa, 0.0) for aa in seq], dtype=np.float32)
        if len(vals) == 0:
            feats.extend([0.0] * 7)
            continue
        # Composition: mean, std
        # Transition: fraction of sign changes
        # Distribution: position of 25/50/75/100th percentile residue
        feats.append(float(vals.mean()))
        feats.append(float(vals.std()))
        sign_changes = sum(
            1 for i in range(len(vals)-1)
            if (vals[i] > 0) != (vals[i+1] > 0)
        ) / max(1, len(vals)-1)
        feats.append(sign_changes)
        sorted_idx = np.argsort(vals)
        for q in [0.25, 0.50, 0.75, 1.0]:
            idx = int(q * (len(vals)-1))
            feats.append(float(sorted_idx[idx]) / max(1, len(vals)))
    return np.array(feats, dtype=np.float32)


def _paac(seq: str, lag: int = 10, weight: float = 0.05) -> np.ndarray:
    """
    50-dim pseudo amino acid composition (Chou 2001).
    Captures long-range sequence order information via correlation factors.
    """
    n = len(seq)
    if n == 0:
        return np.zeros(20 + lag, dtype=np.float32)

    # AAC part
    aac = _aac(seq)

    # Sequence order correlation factors
    h_vals = np.array([_HYDROPHOBICITY.get(aa, 0.0) for aa in seq], dtype=np.float64)
    p_vals = np.array([_POLARITY.get(aa, 0.0) for aa in seq], dtype=np.float64)

    # Normalize scales
    h_vals = (h_vals - h_vals.mean()) / (h_vals.std() + 1e-8)
    p_vals = (p_vals - p_vals.mean()) / (p_vals.std() + 1e-8)

    theta = []
    for k in range(1, min(lag + 1, n)):
        rh = float(np.mean((h_vals[:-k] - h_vals[k:])**2)) if n > k else 0.0
        rp = float(np.mean((p_vals[:-k] - p_vals[k:])**2)) if n > k else 0.0
        theta.append((rh + rp) / 2.0)

    # Pad theta if sequence too short
    while len(theta) < lag:
        theta.append(0.0)
    theta = np.array(theta[:lag], dtype=np.float32)

    theta_sum = float(np.sum(theta)) * weight
    denom = 1.0 + theta_sum

    paac_aac = aac / denom
    paac_theta = (weight * theta) / denom

    return np.concatenate([paac_aac, paac_theta])


def _apaac(seq: str, lag: int = 10, weight: float = 0.05) -> np.ndarray:
    """
    50-dim amphiphilic PAAC.
    Separates hydrophobic from hydrophilic correlation components.
    """
    n = len(seq)
    if n == 0:
        return np.zeros(20 + 2 * lag, dtype=np.float32)

    aac = _aac(seq)
    h_vals = np.array([_HYDROPHOBICITY.get(aa, 0.0) for aa in seq], dtype=np.float64)
    p_vals = np.array([_POLARITY.get(aa, 0.0) for aa in seq], dtype=np.float64)

    h_vals = (h_vals - h_vals.mean()) / (h_vals.std() + 1e-8)
    p_vals = (p_vals - p_vals.mean()) / (p_vals.std() + 1e-8)

    tau_h, tau_p = [], []
    for k in range(1, min(lag + 1, n)):
        tau_h.append(float(np.mean(h_vals[:-k] * h_vals[k:])) if n > k else 0.0)
        tau_p.append(float(np.mean(p_vals[:-k] * p_vals[k:])) if n > k else 0.0)

    while len(tau_h) < lag:
        tau_h.append(0.0)
        tau_p.append(0.0)

    tau_h = np.array(tau_h[:lag], dtype=np.float32)
    tau_p = np.array(tau_p[:lag], dtype=np.float32)
    tau_sum = weight * (float(np.sum(np.abs(tau_h))) + float(np.sum(np.abs(tau_p))))
    denom = 1.0 + tau_sum

    return np.concatenate([aac / denom, (weight * tau_h) / denom, (weight * tau_p) / denom])


def _esm2_features(esm2_result: Optional[dict]) -> np.ndarray:
    """
    128-dim projected ESM-2 features.
    Uses 8 statistical summaries of the 1280-dim protein embedding,
    clustered into 16 groups of 80 dims → 16×8 = 128.
    """
    if not esm2_result:
        return np.zeros(128, dtype=np.float32)

    raw_emb = esm2_result.get("protein_embedding") or esm2_result.get("mean_embedding", [])
    if raw_emb is None:
        return np.zeros(128, dtype=np.float32)
    emb = np.array(raw_emb, dtype=np.float32)
    if emb.ndim == 0 or len(emb) == 0:
        return np.zeros(128, dtype=np.float32)
    if len(emb) != 1280:
        if len(emb) > 0:
            # Resize by interpolation
            emb = np.interp(np.linspace(0, len(emb)-1, 1280),
                             np.arange(len(emb)), emb).astype(np.float32)
        else:
            return np.zeros(128, dtype=np.float32)

    # Split into 16 groups of 80 dims, compute 8 stats each
    groups = emb.reshape(16, 80)
    feats = []
    for g in groups:
        feats.extend([
            float(g.mean()),
            float(g.std()),
            float(g.min()),
            float(g.max()),
            float(np.percentile(g, 25)),
            float(np.percentile(g, 75)),
            float(np.sum(g > 0)) / 80.0,   # fraction positive
            float(np.linalg.norm(g)),        # L2 norm
        ])
    return np.array(feats, dtype=np.float32)


def _structural_features(
    pdb_result:    Optional[dict],
    active_result: Optional[dict],
    pocket_result: Optional[dict],
    enm_result:    Optional[dict],
    physico_result: Optional[dict],
) -> np.ndarray:
    """86-dim structural and topological feature block."""
    feats = []

    # ── pLDDT statistics (8 dims) ─────────────────────────────────────────────
    if pdb_result:
        plddt_vals = [r.get("plddt", 0.0) for r in pdb_result.get("residues", [])]
        if plddt_vals:
            arr = np.array(plddt_vals, dtype=np.float32)
            feats.extend([
                float(arr.mean()),
                float(arr.std()),
                float(np.percentile(arr, 10)),
                float(np.percentile(arr, 25)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 75)),
                float(np.sum(arr > 90)) / len(arr),   # high-conf fraction
                float(np.sum(arr < 50)) / len(arr),   # disordered fraction
            ])
        else:
            feats.extend([0.0] * 8)
    else:
        feats.extend([0.0] * 8)

    # ── Secondary structure fractions (3 dims) ────────────────────────────────
    if pdb_result:
        ss = pdb_result.get("secondary_structure_fractions", {})
        feats.extend([
            ss.get("helix", 0.0),
            ss.get("strand", 0.0),
            ss.get("coil",  0.0),
        ])
    else:
        feats.extend([0.0] * 3)

    # ── SASA burial (6 dims) ──────────────────────────────────────────────────
    if physico_result:
        residues = physico_result.get("residues", [])
        if residues:
            sasa_vals = [r.get("sasa", 0.0) for r in residues]
            arr = np.array(sasa_vals, dtype=np.float32)
            feats.extend([
                float(arr.mean()),
                float(arr.std()),
                float(np.sum(arr < 10)) / len(arr),   # fully buried
                float(np.sum(arr > 40)) / len(arr),   # fully exposed
                float(physico_result.get("hydrophobic_moment", 0.0)),
                float(physico_result.get("net_charge", 0.0)),
            ])
        else:
            feats.extend([0.0] * 6)
    else:
        feats.extend([0.0] * 6)

    # ── Active site geometry (12 dims) ────────────────────────────────────────
    if active_result:
        motifs = active_result.get("catalytic_motifs", [])
        feats.extend([
            float(len(motifs)),
            float(active_result.get("n_high_confidence", 0)),
            float(active_result.get("n_medium_confidence", 0)),
            float(sum(1 for m in motifs if m.get("motif_type") == "serine_protease_triad")),
            float(sum(1 for m in motifs if m.get("motif_type") == "cysteine_protease_dyad")),
            float(sum(1 for m in motifs if m.get("motif_type") == "zinc_binding_cluster")),
            float(sum(1 for m in motifs if m.get("motif_type") == "p_loop_walker_a")),
            float(sum(1 for m in motifs if m.get("zinc_type") == "catalytic")),
            float(sum(1 for m in motifs if m.get("zinc_type") == "structural")),
            float(np.mean([m.get("mean_distance", 0.0) for m in motifs]) if motifs else 0.0),
            float(len(active_result.get("active_residues", []))),
            float(len([r for r in active_result.get("active_residues", [])
                       if r.get("type") == "metal_binding"])),
        ])
    else:
        feats.extend([0.0] * 12)

    # ── Pocket druggability (8 dims) ──────────────────────────────────────────
    if pocket_result:
        pockets = pocket_result.get("pockets", [])[:3]  # top-3 pockets
        for i in range(3):
            if i < len(pockets):
                p = pockets[i]
                feats.extend([
                    float(p.get("volume", 0.0)) / 1000.0,  # normalised
                    float(p.get("druggability_score", 0.0)),
                ])
            else:
                feats.extend([0.0, 0.0])
        feats.append(float(len(pockets)))
        feats.append(float(pocket_result.get("best_druggability", 0.0)))
    else:
        feats.extend([0.0] * 8)

    # ── ENM flexibility (8 dims) ──────────────────────────────────────────────
    if enm_result:
        bfactors = enm_result.get("bfactors", [])
        if bfactors:
            arr = np.array(bfactors, dtype=np.float32)
            feats.extend([
                float(arr.mean()),
                float(arr.std()),
                float(arr.max()),
                float(np.percentile(arr, 90)),
                float(np.sum(arr > arr.mean() + arr.std())) / len(arr),
                float(enm_result.get("n_flexible_regions", 0)),
                float(enm_result.get("n_rigid_regions", 0)),
                float(enm_result.get("allosteric_signal_strength", 0.0)),
            ])
        else:
            feats.extend([0.0] * 8)
    else:
        feats.extend([0.0] * 8)

    # ── Contact map graph features (12 dims) ──────────────────────────────────
    # Computed from ESM-2 contact predictions if available
    feats.extend([0.0] * 12)  # placeholder — filled in by build_feature_vector

    # ── Disulphide / Cys topology (6 dims) ────────────────────────────────────
    feats.extend([0.0] * 6)   # placeholder — filled in by build_feature_vector

    # ── Cofactor metal fingerprint (23 dims) ──────────────────────────────────
    feats.extend([0.0] * 23)  # placeholder — filled in by build_feature_vector

    return np.array(feats[:86], dtype=np.float32)


def _contact_graph_features(esm2_result: Optional[dict], seq: str) -> np.ndarray:
    """
    12-dim contact map graph topology features.
    Treats the contact map as an adjacency matrix and computes graph stats.
    """
    if not esm2_result:
        return np.zeros(12, dtype=np.float32)
    raw = esm2_result.get("contact_map", [])
    if not raw:
        return np.zeros(12, dtype=np.float32)

    try:
        cm = np.array(raw, dtype=np.float32)
        thresh = 0.5
        adj = (cm > thresh).astype(np.float32)
        # Remove diagonal
        np.fill_diagonal(adj, 0)

        degrees = adj.sum(axis=1)
        n = len(degrees)
        if n == 0:
            return np.zeros(12, dtype=np.float32)

        # Clustering coefficient approximation
        triangles = float(np.trace(np.linalg.matrix_power(adj.astype(np.float64), 3)))
        triplets  = float(np.sum(degrees * (degrees - 1)))
        clustering = (triangles / triplets) if triplets > 0 else 0.0

        feats = [
            float(degrees.mean()),
            float(degrees.std()),
            float(degrees.max()),
            float(np.percentile(degrees, 75)),
            float(np.sum(degrees > degrees.mean() + degrees.std())) / n,
            clustering,
            float(adj.sum()) / (n * (n-1) + 1),   # density
            # Long-range contacts (|i-j| > 12)
            float(np.sum(
                [adj[i, j] for i in range(n) for j in range(i+13, n) if j < n]
            )) / max(1, n),
            float(np.sum(
                [adj[i, j] for i in range(n) for j in range(max(0,i-6), min(n,i+7))]
            )) / max(1, n),   # short-range contacts
            float(np.linalg.norm(adj, ord='fro')) / n,  # Frobenius norm / n
            float(np.sum(degrees == 0)) / n,            # isolated nodes
            float(np.percentile(degrees, 25)),
        ]
        return np.array(feats, dtype=np.float32)
    except Exception:
        return np.zeros(12, dtype=np.float32)


def _disulfide_features(seq: str, active_result: Optional[dict]) -> np.ndarray:
    """6-dim disulphide / cysteine topology features."""
    cys_positions = [i for i, aa in enumerate(seq) if aa == "C"]
    n_cys = len(cys_positions)
    n_disulf = n_cys // 2  # rough estimate

    # Cys fraction
    cys_frac = n_cys / max(1, len(seq))

    # Cys clustering: mean nearest-neighbour distance between Cys residues
    if len(cys_positions) > 1:
        dists = []
        for i in range(len(cys_positions)-1):
            dists.append(float(cys_positions[i+1] - cys_positions[i]))
        nn_dist = float(np.mean(dists))
        nn_std  = float(np.std(dists))
    else:
        nn_dist, nn_std = 0.0, 0.0

    # C4H motif (structural zinc finger) vs CCHH (catalytic)
    seq_blocks = [seq[max(0,p-2):min(len(seq),p+3)] for p in cys_positions]
    n_cchh = sum(1 for b in seq_blocks if "H" in b)

    return np.array([
        float(n_cys),
        cys_frac,
        float(n_disulf),
        nn_dist,
        nn_std,
        float(n_cchh),
    ], dtype=np.float32)


def _cofactor_fingerprint(seq: str, active_result: Optional[dict]) -> np.ndarray:
    """
    23-dim cofactor/metal binding fingerprint.
    Encodes potential binding of 8 metal types + FAD/NAD/ATP indicators.
    """
    feats = []

    # Metal binding indicators (8 metals × 2 features = 16)
    for metal_name, binding_aas, min_count in METAL_TYPES_MOTIFS:
        binding_count = sum(1 for aa in seq if aa in binding_aas)
        has_motif     = 1.0 if binding_count >= min_count else 0.0
        feats.extend([float(binding_count) / max(1, len(seq)), has_motif])

    # Nucleotide cofactor indicators
    # FAD/FMN: Rossman fold → Gly-rich motif in first 30 residues
    gly_early = sum(1 for aa in seq[:50] if aa == "G") / max(1, len(seq[:50]))
    feats.append(gly_early)

    # NAD+/NADH: Gly-Xaa-Gly-Xaa-Xaa-Gly motif
    nad_score = 0.0
    for i in range(len(seq) - 5):
        if seq[i] == "G" and seq[i+2] == "G" and seq[i+5] == "G":
            nad_score += 1.0
    feats.append(min(1.0, nad_score / 3.0))

    # ATP: P-loop / Walker A motif (GxxxxGK[TS])
    atp_score = 0.0
    for i in range(len(seq) - 7):
        if seq[i] == "G" and seq[i+5] == "G" and seq[i+6] == "K":
            if seq[i+7] in ("T", "S"):
                atp_score += 1.0
    feats.append(min(1.0, atp_score))

    # Heme: CXXCH motif
    heme_score = 0.0
    for i in range(len(seq) - 5):
        if seq[i] == "C" and seq[i+4] == "C" and seq[i+5] == "H":
            heme_score += 1.0
    feats.append(min(1.0, heme_score))

    # PLP (pyridoxal phosphate): Lys as Schiff base, GxxS motif
    plp_score = float("K" in seq and "G" in seq) * 0.5
    feats.append(plp_score)

    # TPP (thiamine): pyrimidine-binding motif
    tpp_score = 0.0
    for i in range(len(seq) - 3):
        if seq[i] == "G" and seq[i+3] in ("D", "E"):
            tpp_score += 0.2
    feats.append(min(1.0, tpp_score))

    # CoA-binding: beta-mercaptoethylamine moiety → Cys-rich
    coa_score = float(seq.count("C") > 2 and "P" in seq) * 0.5
    feats.append(coa_score)

    return np.array(feats[:23], dtype=np.float32)


def _evidence_features(
    go_result:       Optional[dict],
    homology_result: Optional[dict],
    active_result:   Optional[dict],
) -> np.ndarray:
    """42-dim evidence feature block from external signals."""
    feats = []

    # ── Homology evidence (10 dims) ───────────────────────────────────────────
    if homology_result:
        hits = homology_result.get("blast_hits", [])[:10]
        identities  = [h.get("identity", 0.0)  for h in hits]
        evalues_log = [-math.log10(max(1e-200, h.get("evalue", 1.0))) for h in hits]
        reviewed    = [1.0 if h.get("reviewed") else 0.0 for h in hits]

        feats.extend([
            float(np.mean(identities))  if identities  else 0.0,
            float(np.max(identities))   if identities  else 0.0,
            float(np.mean(evalues_log)) if evalues_log else 0.0,
            float(np.max(evalues_log))  if evalues_log else 0.0,
            float(sum(1 for i in identities if i > 0.90)),   # near-identical
            float(sum(1 for i in identities if 0.60 < i <= 0.90)),
            float(sum(1 for i in identities if 0.30 < i <= 0.60)),
            float(sum(reviewed)),
            float(len(hits)),
            float(homology_result.get("n_interpro_domains", 0)),
        ])
    else:
        feats.extend([0.0] * 10)

    # ── GO term evidence (12 dims, broad MF categories) ───────────────────────
    GO_CATEGORIES = [
        ("GO:0016491", "oxidoreductase"),
        ("GO:0016301", "kinase"),
        ("GO:0016787", "hydrolase"),
        ("GO:0016829", "lyase"),
        ("GO:0016853", "isomerase"),
        ("GO:0016874", "ligase"),
        ("GO:0005488", "binding"),
        ("GO:0003700", "transcription_factor"),
        ("GO:0008565", "transporter"),
        ("GO:0005198", "structural"),
        ("GO:0003723", "RNA_binding"),
        ("GO:0003677", "DNA_binding"),
    ]
    if go_result:
        all_go_ids = set()
        for pred in (go_result.get("mf_predictions", []) +
                     go_result.get("bp_predictions", [])):
            all_go_ids.add(pred.get("go_id", ""))
        for go_id, _ in GO_CATEGORIES:
            feats.append(1.0 if go_id in all_go_ids else 0.0)
    else:
        feats.extend([0.0] * 12)

    # ── Active site motif indicators (20 dims) ────────────────────────────────
    MOTIF_TYPES = [
        "serine_protease_triad",
        "cysteine_protease_dyad",
        "zinc_binding_cluster",
        "p_loop_walker_a",
        "his_dyad",
        "asp_triad",
        "glu_glu_motif",
        "iron_sulfur",
        "heme_binding",
        "flavin_binding",
    ]
    if active_result:
        found = {m.get("motif_type", "") for m in active_result.get("catalytic_motifs", [])}
        for mtype in MOTIF_TYPES:
            feats.append(1.0 if mtype in found else 0.0)
        feats.extend([0.0] * (10 - len(MOTIF_TYPES[:10])))
        # confidence summary
        n_high   = float(active_result.get("n_high_confidence", 0))
        n_medium = float(active_result.get("n_medium_confidence", 0))
        feats.extend([
            n_high,
            n_medium,
            float(n_high + n_medium),
            1.0 if n_high > 0 else 0.0,
            1.0 if n_medium > 0 else 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,   # reserved
        ])
    else:
        feats.extend([0.0] * 20)

    return np.array(feats[:42], dtype=np.float32)


# ── Public API ─────────────────────────────────────────────────────────────────

def build_feature_vector(
    sequence:        str,
    esm2_result:     Optional[dict] = None,
    pdb_result:      Optional[dict] = None,
    active_result:   Optional[dict] = None,
    pocket_result:   Optional[dict] = None,
    enm_result:      Optional[dict] = None,
    physico_result:  Optional[dict] = None,
    go_result:       Optional[dict] = None,
    homology_result: Optional[dict] = None,
) -> np.ndarray:
    """
    Build the complete feature vector for EC classification.

    Returns:
        np.ndarray of shape (N_FEATURES,) — typically ~800 dims.
        All features are float32 and bounded (no infinities/NaNs).
    """
    seq = sequence.upper().replace(" ", "")

    # Block A — Sequence composition
    aac_feat   = _aac(seq)                         # 20
    dpc_feat   = _dpc(seq)                         # 400
    tpc_feat   = _tpc_compressed(seq, n_components=64)  # 64
    ctd_feat   = _ctd_features(seq)                # 21
    paac_feat  = _paac(seq, lag=30)                # 50   (20+30)
    apaac_feat = _apaac(seq, lag=30)               # 80   (20+60)

    # Block B — ESM-2
    esm2_feat = _esm2_features(esm2_result)        # 128

    # Block C — Structural
    struct_feat = _structural_features(
        pdb_result, active_result, pocket_result, enm_result, physico_result
    )                                              # 86
    # Fill contact map slot (dims 49-60 in struct)
    contact_feat = _contact_graph_features(esm2_result, seq)   # 12
    disulf_feat  = _disulfide_features(seq, active_result)     # 6
    cofact_feat  = _cofactor_fingerprint(seq, active_result)   # 23

    # Block D — Evidence
    evidence_feat = _evidence_features(go_result, homology_result, active_result)  # 42

    # Concatenate all blocks
    feat_vec = np.concatenate([
        aac_feat,      # 20
        dpc_feat,      # 400
        tpc_feat,      # 64
        ctd_feat,      # 21
        paac_feat,     # 50
        apaac_feat,    # 80
        esm2_feat,     # 128
        struct_feat,   # 86 (with placeholder zeros for contact/disulf/cofact)
        contact_feat,  # 12
        disulf_feat,   # 6
        cofact_feat,   # 23
        evidence_feat, # 42
    ]).astype(np.float32)

    # Replace NaN/Inf with 0
    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)

    return feat_vec


def get_feature_names() -> list[str]:
    """Return human-readable feature names for interpretability."""
    names = []
    names += [f"AAC_{aa}"  for aa in AA_LIST]
    names += [f"DPC_{a}{b}" for a in AA_LIST for b in AA_LIST]
    names += [f"TPC_proj_{i}" for i in range(64)]
    names += [f"CTD_{i}" for i in range(21)]
    names += [f"PAAC_{i}" for i in range(50)]
    names += [f"APAAC_{i}" for i in range(80)]
    names += [f"ESM2_feat_{i}" for i in range(128)]
    names += [f"struct_{i}" for i in range(86)]
    names += [f"contact_{i}" for i in range(12)]
    names += [f"disulf_{i}" for i in range(6)]
    names += [f"cofact_{i}" for i in range(23)]
    names += [f"evidence_{i}" for i in range(42)]
    return names


FEATURE_DIM = (20 + 400 + 64 + 21 + 50 + 80 + 128 + 86 + 12 + 6 + 23 + 42)
# = 932 dims total

class MLECFeatures:
    """
    Thin wrapper around build_feature_vector for use in ml_ec_train.py.
    Provides a sklearn-style transform interface.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def transform(self, sequence: str, **result_kwargs) -> np.ndarray:
        return build_feature_vector(sequence, **result_kwargs)

    def fit(self, *args, **kwargs):
        return self   # stateless — nothing to fit

    @staticmethod
    def feature_names() -> list[str]:
        return get_feature_names()

    @staticmethod
    def feature_dim() -> int:
        return FEATURE_DIM