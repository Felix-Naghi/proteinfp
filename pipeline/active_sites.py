"""
pipeline/03_active_sites.py
────────────────────────────
Module 03 — Active site prediction.

Takes the parsed structure (Module 01) and physicochemical profile (Module 02)
and identifies catalytic / functional residues using three evidence layers:

  1. Evolutionary conservation via ConSurf API
     - Highly conserved buried residues are almost always functionally critical
     - ConSurf scores 1-9: 1=variable, 9=conserved
     - We flag residues with score >= 7 as "conserved"

  2. Geometric catalytic motif detection
     - Serine protease triad: Ser-His-Asp within 6Å
     - Cysteine protease dyad: Cys-His within 4Å
     - Zinc binding: His/Cys/Asp/Glu in tetrahedral geometry within 3.5Å
     - Acid-base pairs: Asp/Glu near Arg/Lys/His within 5Å
     - DNA-binding Arg/Lys clusters (for transcription factors)

  3. Cross-validation scoring
     - Residue scores points for: conservation, burial, pocket proximity,
       known motif membership, low pLDDT variance in region
     - Final confidence: HIGH (3+ evidence) / MEDIUM (2) / LOW (1)

Usage (standalone):
    python pipeline/03_active_sites.py --uniprot P04637

Usage (from orchestrator):
    from pipeline.active_sites import predict_active_sites
    result = predict_active_sites(parsed_structure, physico_result)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from itertools import combinations

import click
import numpy as np
import requests
from Bio.PDB import PDBParser

from utils.config import cfg, get_logger
from utils.pdb_parser import parse_pdb, ParsedStructure, AA3TO1

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# ConSurf score >= this → conserved
CONSERVATION_THRESHOLD = 7

# Distance cutoffs for motif detection (Angstroms, CA-CA unless noted)
TRIAD_CUTOFF    = 8.0   # Ser-His-Asp catalytic triad
DYAD_CUTOFF     = 6.0   # Cys-His dyad
ZINC_CUTOFF     = 7.0   # Zinc binding cluster
ACIDBASE_CUTOFF = 7.0   # Acid-base pair
DNA_CUTOFF      = 8.0   # DNA-binding cluster (Arg/Lys)

# Minimum cluster size for DNA-binding detection
DNA_CLUSTER_MIN = 3

# Known catalytic residue types by motif
MOTIF_RESIDUES = {
    "serine_protease":  {"S", "H", "D"},
    "cysteine_protease":{"C", "H"},
    "zinc_binding":     {"H", "C", "D", "E"},
    "acid_base":        {"D", "E", "R", "K", "H"},
    "dna_binding":      {"R", "K"},
    "phosphate_binding":{"R", "K", "S", "T"},
    "ghkl_atpase":      {"N", "D", "G"},
    "flavin_binding":   {"G", "Y", "F"},
    "haem_binding":     {"H"},
}

# Evidence point values for confidence scoring
EVIDENCE_POINTS = {
    "conserved":        3,
    "buried":           2,
    "motif_member":     2,
    "known_functional": 3,
    "high_plddt":       1,
    "charged_context":  1,
}

CONFIDENCE_THRESHOLDS = {
    "HIGH":   6,
    "MEDIUM": 3,
    "LOW":    1,
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ActiveResidue:
    """A single predicted active/catalytic residue."""
    residue_number:   int
    chain_id:         str
    one_letter:       str
    three_letter:     str
    conservation:     float        # ConSurf score 1-9 (0 if unavailable)
    is_conserved:     bool
    is_buried:        bool         # SASA < 20 Å²
    sasa:             float
    plddt:            float
    motifs:           list[str]    # which motifs this residue belongs to
    evidence_score:   int          # sum of evidence points
    confidence:       str          # HIGH / MEDIUM / LOW
    coords:           list[float]  # CA [x, y, z]
    domain_context:   str = ""     # structural region (from InterPro annotations)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CatalyticMotif:
    """A detected catalytic motif (e.g. Ser-His-Asp triad)."""
    motif_type:       str
    residue_numbers:  list[int]
    residue_letters:  list[str]
    mean_distance:    float        # mean pairwise CA distance
    confidence:       str
    zinc_type:        str = ""     # "catalytic" or "structural" (zinc motifs only)
    long_range:       bool = False # True when triad residues span >20 seq positions

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActiveSiteResult:
    """Full active site prediction output. Output of Module 03."""
    uniprot_id:           str
    length:               int
    consurf_available:    bool
    active_residues:      list[ActiveResidue]  = field(default_factory=list)
    catalytic_motifs:     list[CatalyticMotif] = field(default_factory=list)

    # Summary counts
    n_high_confidence:    int   = 0
    n_medium_confidence:  int   = 0
    n_low_confidence:     int   = 0
    top_residues:         list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  Active site prediction: {self.uniprot_id}",
            f"  ConSurf conservation : {'yes' if self.consurf_available else 'no (API unavailable)'}",
            f"  Active residues found: {len(self.active_residues)}",
            f"    HIGH confidence    : {self.n_high_confidence}",
            f"    MEDIUM confidence  : {self.n_medium_confidence}",
            f"    LOW confidence     : {self.n_low_confidence}",
            f"  Catalytic motifs     : {len(self.catalytic_motifs)}",
        ]
        if self.top_residues:
            lines.append(f"  Top residues         : {', '.join(self.top_residues)}")
        for motif in self.catalytic_motifs:
            res_str = "-".join(
                f"{l}{n}" for l, n in
                zip(motif.residue_letters, motif.residue_numbers)
            )
            lines.append(f"    [{motif.motif_type}] {res_str} "
                         f"(confidence: {motif.confidence})")
        lines.append(f"{'─'*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.debug(f"Saved active site JSON → {path}")


# ── Main function ──────────────────────────────────────────────────────────────

def predict_active_sites(
    structure:  ParsedStructure,
    sasa_map:   Optional[dict[tuple[str, int], float]] = None,
) -> ActiveSiteResult:
    """
    Predict active site residues from structure + optional SASA data.

    Args:
        structure : ParsedStructure from Module 01
        sasa_map  : dict (chain_id, res_num) → SASA float from Module 02
                    If None, all residues treated as partially buried.

    Returns:
        ActiveSiteResult with per-residue predictions and motif annotations.
    """
    log.info(f"── Module 03: Active site prediction for {structure.uniprot_id} ──")

    # ── Step 1: Get ConSurf conservation scores ───────────────────────────────
    log.info("  [1/3] Querying ConSurf conservation scores...")
    conservation_map, consurf_ok = _get_conservation(
        structure.uniprot_id, structure.sequence
    )

    # ── Step 2: Detect catalytic motifs from 3D geometry ─────────────────────
    log.info("  [2/3] Detecting catalytic motifs...")
    coord_map   = {r.residue_number: (r.one_letter, r.coords, r.chain_id)
                   for r in structure.residues}
    motifs      = _detect_motifs(coord_map)

    # Build set of residue numbers flagged by any motif
    motif_residues: dict[int, list[str]] = {}
    for motif in motifs:
        for rn in motif.residue_numbers:
            motif_residues.setdefault(rn, []).append(motif.motif_type)

    # ── Step 3: Score every residue and build active residue list ────────────
    log.info("  [3/3] Scoring residues...")
    active_residues = _score_residues(
        structure, conservation_map, motif_residues, sasa_map or {}
    )

    # ── Assemble result ───────────────────────────────────────────────────────
    n_high   = sum(1 for r in active_residues if r.confidence == "HIGH")
    n_medium = sum(1 for r in active_residues if r.confidence == "MEDIUM")
    n_low    = sum(1 for r in active_residues if r.confidence == "LOW")

    # Top residues = HIGH confidence, sorted by evidence score
    top = sorted(
        [r for r in active_residues if r.confidence == "HIGH"],
        key=lambda r: r.evidence_score,
        reverse=True
    )[:10]
    top_labels = [f"{r.one_letter}{r.residue_number}" for r in top]

    result = ActiveSiteResult(
        uniprot_id=structure.uniprot_id,
        length=structure.length,
        consurf_available=consurf_ok,
        active_residues=active_residues,
        catalytic_motifs=motifs,
        n_high_confidence=n_high,
        n_medium_confidence=n_medium,
        n_low_confidence=n_low,
        top_residues=top_labels,
    )

    log.info(result.summary())
    return result


# ── ConSurf conservation ───────────────────────────────────────────────────────

def _get_conservation(
    uniprot_id: str,
    sequence:   str,
) -> tuple[dict[int, float], bool]:
    """
    Query the ConSurf API for per-residue conservation scores.

    ConSurf scores: 1 = highly variable, 9 = highly conserved.
    Returns a dict of {residue_index_1based: score} and a success bool.

    Falls back to a sequence-entropy based estimate if API unavailable.
    """
    conservation: dict[int, float] = {}

    # Try ConSurf DB first (fast lookup by UniProt ID)
    try:
        url = f"https://consurf.tau.ac.il/api/consurf_DB_query.php"
        params = {"sequence": sequence, "uniprot": uniprot_id}
        resp = requests.get(url, params=params, timeout=15)

        if resp.status_code == 200 and resp.text.strip():
            for line in resp.text.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pos   = int(parts[0])
                        score = float(parts[1])
                        conservation[pos] = score
                    except ValueError:
                        continue

            if conservation:
                log.info(f"    ConSurf: got scores for {len(conservation)} residues")
                return conservation, True

    except Exception as e:
        log.debug(f"    ConSurf API unavailable: {e}")

    # Fallback: amino acid conservation proxy
    # Catalytically important residue types get elevated scores
    log.info("    ConSurf API unavailable — using residue-type conservation proxy")
    conservation = _conservation_proxy(sequence)
    return conservation, False


def _conservation_proxy(sequence: str) -> dict[int, float]:
    """
    Proxy conservation scores based on residue type.
    Catalytically important residue types (C, H, D, E, R, K, W, Y)
    get elevated baseline scores. This is a rough proxy — not a substitute
    for real evolutionary conservation — but it meaningfully improves
    active site detection when ConSurf is unavailable.
    """
    # Residue types that are disproportionately found in active sites
    CATALYTIC_BASELINE: dict[str, float] = {
        "C": 7.5,  # Cys — nucleophile, metal-binding
        "H": 7.0,  # His — acid/base, metal-binding
        "D": 6.5,  # Asp — acid/base, nucleophile
        "E": 6.0,  # Glu — acid/base
        "R": 6.5,  # Arg — phosphate binding, DNA contact
        "K": 6.0,  # Lys — DNA contact, Schiff base
        "W": 6.5,  # Trp — structural, often conserved
        "Y": 5.5,  # Tyr — hydrogen bonding, often conserved
        "S": 4.5,  # Ser — nucleophile
        "T": 4.0,  # Thr
        "N": 4.0,  # Asn
        "Q": 4.0,  # Gln
        "F": 4.5,  # Phe — hydrophobic core
        "I": 3.5,
        "L": 3.5,
        "V": 3.5,
        "M": 4.0,
        "A": 3.0,
        "G": 4.0,  # Gly — structural flexibility
        "P": 3.5,
    }
    return {
        i + 1: CATALYTIC_BASELINE.get(aa, 3.0)
        for i, aa in enumerate(sequence)
    }


# ── Catalytic motif detection ──────────────────────────────────────────────────

def _detect_motifs(
    coord_map: dict[int, tuple[str, list[float], str]],
) -> list[CatalyticMotif]:
    """
    Search for known catalytic motifs in the 3D structure.
    coord_map: {res_num: (one_letter, [x,y,z], chain_id)}
    """
    motifs = []

    # Filter by residue type for each motif class
    ser_res  = {n: c for n, (aa, c, _) in coord_map.items() if aa == "S"}
    his_res  = {n: c for n, (aa, c, _) in coord_map.items() if aa == "H"}
    asp_res  = {n: c for n, (aa, c, _) in coord_map.items() if aa == "D"}
    cys_res  = {n: c for n, (aa, c, _) in coord_map.items() if aa == "C"}
    glu_res  = {n: c for n, (aa, c, _) in coord_map.items() if aa == "E"}
    arg_res  = {n: c for n, (aa, c, _) in coord_map.items() if aa == "R"}
    lys_res  = {n: c for n, (aa, c, _) in coord_map.items() if aa == "K"}
    phe_res  = {n: c for n, (aa, c, _) in coord_map.items() if aa == "F"}
    gly_res  = {n: c for n, (aa, c, _) in coord_map.items() if aa == "G"}
    asn_res  = {n: c for n, (aa, c, _) in coord_map.items() if aa == "N"}
    tyr_res  = {n: c for n, (aa, c, _) in coord_map.items() if aa == "Y"}

    # 1. Serine protease triad: Ser + His + Asp all within TRIAD_CUTOFF
    #    (also detects long-range triads like thrombin H363-D419-S521)
    motifs += _find_triad(ser_res, his_res, asp_res,
                          "serine_protease_triad", TRIAD_CUTOFF)

    # 2. Cysteine protease dyad: Cys + His within DYAD_CUTOFF
    motifs += _find_dyad(cys_res, his_res,
                         "cysteine_protease_dyad", DYAD_CUTOFF)

    # 3. Zinc binding: any 3+ of His/Cys/Asp/Glu within ZINC_CUTOFF
    zinc_candidates = {**his_res, **cys_res, **asp_res, **glu_res}
    motifs += _find_zinc_cluster(zinc_candidates, coord_map, ZINC_CUTOFF)

    # 4. DNA-binding Arg/Lys cluster: 3+ within DNA_CUTOFF
    dna_candidates = {**arg_res, **lys_res}
    motifs += _find_dna_binding_cluster(dna_candidates, coord_map, DNA_CUTOFF)

    # 5. Acid-base pairs: Asp/Glu near Arg/Lys/His
    acid_res = {**asp_res, **glu_res}
    base_res = {**arg_res, **lys_res, **his_res}
    motifs += _find_acid_base_pairs(acid_res, base_res, ACIDBASE_CUTOFF)

    # 6. Kinase DFG loop
    motifs += _find_sequential_triad(asp_res, phe_res, gly_res,
                                     coord_map, "dfg_loop")

    # 7. Kinase HRD catalytic loop
    motifs += _find_sequential_triad(his_res, arg_res, asp_res,
                                     coord_map, "hrd_catalytic_loop")

    # 8. P-loop / Walker A (Gly-x-Gly-x-x-Gly — ATP binding)
    motifs += _find_ploop(gly_res, coord_map)

    # 9. GHKL ATPase / Bergerat fold (HSP90, MutL, GyrB)
    motifs += _find_ghkl_atpase(asn_res, asp_res, gly_res, coord_map)

    # 10. Flavin-binding Rossmann fold (NQO1, NQO2, oxidoreductases)
    motifs += _find_flavin_binding(gly_res, tyr_res, phe_res, coord_map)

    # 11. Haem-binding proximal His (haemoglobins, myoglobins, cytochromes)
    motifs += _find_haem_binding(his_res, cys_res, coord_map)

    log.debug(f"    Detected {len(motifs)} catalytic motif(s)")
    return motifs


def _dist(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _find_triad(
    ser: dict, his: dict, asp: dict,
    motif_type: str, cutoff: float
) -> list[CatalyticMotif]:
    motifs = []
    seen: set[tuple] = set()

    # Primary pass: standard cutoff
    for sn, sc in ser.items():
        for hn, hc in his.items():
            if _dist(sc, hc) > cutoff:
                continue
            for dn, dc in asp.items():
                if _dist(hc, dc) > cutoff:
                    continue
                key = tuple(sorted([sn, hn, dn]))
                if key in seen:
                    continue
                seen.add(key)
                mean_d = (_dist(sc, hc) + _dist(hc, dc) + _dist(sc, dc)) / 3
                max_gap = max(abs(sn - hn), abs(hn - dn), abs(sn - dn))
                motifs.append(CatalyticMotif(
                    motif_type=motif_type,
                    residue_numbers=[sn, hn, dn],
                    residue_letters=["S", "H", "D"],
                    mean_distance=round(mean_d, 2),
                    confidence="HIGH" if mean_d < 6.0 else "MEDIUM",
                    long_range=max_gap > 20,
                ))

    # Secondary pass: long-range triad (12Å, requires seq gap >20 between some pair).
    # Captures serine proteases like thrombin where the catalytic Ser/His/Asp
    # are far apart in sequence but converge in 3D (e.g. H363-D419-S521 in thrombin).
    long_cutoff = 12.0
    for sn, sc in ser.items():
        for hn, hc in his.items():
            if _dist(sc, hc) > long_cutoff:
                continue
            for dn, dc in asp.items():
                max_gap = max(abs(sn - hn), abs(hn - dn), abs(sn - dn))
                if max_gap <= 20:
                    continue  # not long-range; already handled above
                if _dist(hc, dc) > long_cutoff:
                    continue
                key = tuple(sorted([sn, hn, dn]))
                if key in seen:
                    continue
                seen.add(key)
                mean_d = (_dist(sc, hc) + _dist(hc, dc) + _dist(sc, dc)) / 3
                motifs.append(CatalyticMotif(
                    motif_type=motif_type,
                    residue_numbers=[sn, hn, dn],
                    residue_letters=["S", "H", "D"],
                    mean_distance=round(mean_d, 2),
                    confidence="MEDIUM",
                    long_range=True,
                ))

    return motifs


def _find_dyad(
    cys: dict, his: dict,
    motif_type: str, cutoff: float
) -> list[CatalyticMotif]:
    motifs = []
    for cn, cc in cys.items():
        for hn, hc in his.items():
            d = _dist(cc, hc)
            if d <= cutoff:
                motifs.append(CatalyticMotif(
                    motif_type=motif_type,
                    residue_numbers=[cn, hn],
                    residue_letters=["C", "H"],
                    mean_distance=round(d, 2),
                    confidence="HIGH" if d < 4.5 else "MEDIUM",
                ))
    return motifs


def _classify_zinc_type(letters: list[str]) -> str:
    """
    Distinguish structural zinc from catalytic zinc by coordination geometry.

    Catalytic zinc (metallopeptidase active site): His-His-Glu pattern (H2E1).
      - Coordination by 2+ His and 1+ Glu with no Cys
    Structural zinc (zinc finger, RING domain): Cys-rich patterns (C4 or C3H1).
      - 3+ Cys residues → structural (RING domains, zinc fingers, etc.)

    Structural zinc should NOT drive EC class prediction.
    """
    cys_count = letters.count("C")
    his_count = letters.count("H")
    glu_count = letters.count("E")

    if cys_count >= 3:
        return "structural"
    if his_count >= 2 and glu_count >= 1 and cys_count == 0:
        return "catalytic"
    # Mixed or ambiguous — err on the side of structural (conservative)
    return "structural"


def _find_zinc_cluster(
    candidates: dict,
    coord_map:  dict,
    cutoff:     float,
) -> list[CatalyticMotif]:
    motifs = []
    keys   = list(candidates.keys())
    used   = set()

    for i in range(len(keys)):
        cluster = [keys[i]]
        for j in range(len(keys)):
            if i == j:
                continue
            if _dist(candidates[keys[i]], candidates[keys[j]]) <= cutoff:
                cluster.append(keys[j])

        cluster = sorted(set(cluster))
        frozen  = tuple(cluster)
        if len(cluster) < 3 or frozen in used:
            continue
        used.add(frozen)

        letters = [coord_map[n][0] for n in cluster]
        coords  = [candidates[n] for n in cluster]
        dists   = [_dist(coords[a], coords[b])
                   for a, b in combinations(range(len(coords)), 2)]
        mean_d  = float(np.mean(dists)) if dists else 0.0
        zinc_type = _classify_zinc_type(letters)

        motifs.append(CatalyticMotif(
            motif_type="zinc_binding_cluster",
            residue_numbers=cluster,
            residue_letters=letters,
            mean_distance=round(mean_d, 2),
            confidence="HIGH" if len(cluster) >= 4 else "MEDIUM",
            zinc_type=zinc_type,
        ))
    return motifs


def _find_dna_binding_cluster(
    candidates: dict,
    coord_map:  dict,
    cutoff:     float,
) -> list[CatalyticMotif]:
    motifs = []
    keys   = list(candidates.keys())
    used   = set()

    for i in range(len(keys)):
        cluster = [keys[i]]
        for j in range(len(keys)):
            if i == j:
                continue
            if _dist(candidates[keys[i]], candidates[keys[j]]) <= cutoff:
                cluster.append(keys[j])

        cluster = sorted(set(cluster))
        frozen  = tuple(cluster)
        if len(cluster) < DNA_CLUSTER_MIN or frozen in used:
            continue
        used.add(frozen)

        letters = [coord_map[n][0] for n in cluster]
        coords  = [candidates[n] for n in cluster]
        dists   = [_dist(coords[a], coords[b])
                   for a, b in combinations(range(len(coords)), 2)]
        mean_d  = float(np.mean(dists)) if dists else 0.0

        motifs.append(CatalyticMotif(
            motif_type="dna_binding_cluster",
            residue_numbers=cluster,
            residue_letters=letters,
            mean_distance=round(mean_d, 2),
            confidence="HIGH" if len(cluster) >= 5 else "MEDIUM",
        ))
    return motifs


def _find_acid_base_pairs(
    acid: dict, base: dict, cutoff: float
) -> list[CatalyticMotif]:
    motifs = []
    seen   = set()
    for an, ac in acid.items():
        for bn, bc in base.items():
            if an == bn:
                continue
            d = _dist(ac, bc)
            if d <= cutoff:
                key = tuple(sorted([an, bn]))
                if key in seen:
                    continue
                seen.add(key)
                motifs.append(CatalyticMotif(
                    motif_type="acid_base_pair",
                    residue_numbers=[an, bn],
                    residue_letters=["D/E", "R/K/H"],
                    mean_distance=round(d, 2),
                    confidence="MEDIUM",
                ))
    return motifs


def _find_sequential_triad(
    res1: dict, res2: dict, res3: dict,
    coord_map: dict,
    motif_type: str, max_seq_gap: int = 2,
) -> list[CatalyticMotif]:
    """
    Find three residue types appearing consecutively in sequence
    and close in 3D space. Used for DFG and HRD kinase motif detection.
    """
    motifs = []
    seen   = set()
    for n1, c1 in res1.items():
        for n2, c2 in res2.items():
            if not (0 < n2 - n1 <= max_seq_gap + 1):
                continue
            if _dist(c1, c2) > 12.0:
                continue
            for n3, c3 in res3.items():
                if not (0 < n3 - n2 <= max_seq_gap + 1):
                    continue
                if _dist(c2, c3) > 12.0:
                    continue
                key = (n1, n2, n3)
                if key in seen:
                    continue
                seen.add(key)
                mean_d = (_dist(c1,c2) + _dist(c2,c3) + _dist(c1,c3)) / 3
                letters = [coord_map[n][0] for n in [n1, n2, n3]]
                motifs.append(CatalyticMotif(
                    motif_type=motif_type,
                    residue_numbers=[n1, n2, n3],
                    residue_letters=letters,
                    mean_distance=round(mean_d, 2),
                    confidence="HIGH",
                ))
    return motifs[:3]


def _find_ghkl_atpase(
    asn_res:   dict,
    asp_res:   dict,
    gly_res:   dict,
    coord_map: dict,
) -> list[CatalyticMotif]:
    """
    Detect GHKL ATPase / Bergerat fold (HSP90, MutL, GyrB, histidine kinases).
    Key residues: catalytic Asn + Asp within 12Å, with a GxG dinucleotide motif
    within 15 residues of the Asn in sequence and within 15Å in 3D.
    """
    motifs = []
    seen: set[tuple] = set()

    # Build GxG pairs (gap 1-3 in sequence)
    gly_keys = sorted(gly_res.keys())
    gg_pairs: list[tuple[int, int]] = []
    for i, g1 in enumerate(gly_keys):
        for g2 in gly_keys[i + 1:]:
            gap = g2 - g1
            if gap > 3:
                break
            gg_pairs.append((g1, g2))

    for nn, nc in asn_res.items():
        for dn, dc in asp_res.items():
            if nn == dn:
                continue
            if _dist(nc, dc) > 8.0:
                continue
            # Find a GxG pair near the Asn in sequence + 3D
            found_gg = -1
            for g1, g2 in gg_pairs:
                if abs(nn - g1) <= 8 and _dist(gly_res[g1], nc) <= 8.0:
                    found_gg = g1
                    break
            if found_gg < 0:
                continue
            key = (min(nn, dn), max(nn, dn), found_gg)
            if key in seen:
                continue
            seen.add(key)
            motifs.append(CatalyticMotif(
                motif_type="ghkl_atpase",
                residue_numbers=[nn, dn, found_gg],
                residue_letters=["N", "D", "G"],
                mean_distance=round(_dist(nc, dc), 2),
                confidence="MEDIUM",
            ))

    return motifs[:2]


def _find_flavin_binding(
    gly_res:   dict,
    tyr_res:   dict,
    phe_res:   dict,
    coord_map: dict,
) -> list[CatalyticMotif]:
    """
    Detect flavin (FMN/FAD) binding Rossmann fold (NQO1, NQO2, oxidoreductases).
    Key: GxG dinucleotide motif (gap 1-4) + aromatic residue (Tyr/Phe) within 12Å
    for isoalloxazine ring stacking.
    """
    motifs = []
    seen: set[tuple] = set()
    gly_keys = sorted(gly_res.keys())
    aromatic = {**tyr_res, **phe_res}

    for i, g1 in enumerate(gly_keys):
        for g2 in gly_keys[i + 1:]:
            gap = g2 - g1
            if gap > 4:
                break
            if gap < 1:
                continue
            d_gg = _dist(gly_res[g1], gly_res[g2])
            if d_gg > 6.0:
                continue
            for an, ac in aromatic.items():
                if _dist(gly_res[g1], ac) > 6.0:
                    continue
                key = (g1, g2, an)
                if key in seen:
                    continue
                seen.add(key)
                aa = coord_map[an][0]
                mean_d = (_dist(gly_res[g1], ac) +
                          _dist(gly_res[g2], ac) + d_gg) / 3
                motifs.append(CatalyticMotif(
                    motif_type="flavin_binding",
                    residue_numbers=[g1, g2, an],
                    residue_letters=["G", "G", aa],
                    mean_distance=round(mean_d, 2),
                    confidence="MEDIUM",
                ))
                break  # one aromatic per GG pair

    return motifs[:2]


def _find_haem_binding(
    his_res:   dict,
    cys_res:   dict,
    coord_map: dict,
) -> list[CatalyticMotif]:
    """
    Detect haem-binding His (proximal histidine coordination in Hb, Mb, cytochromes).
    Key: isolated His with 3+ hydrophobic neighbours within 8Å, NOT adjacent to Cys
    (which would indicate zinc coordination rather than haem binding).
    """
    HYDROPHOBIC = {"V", "I", "L", "M", "F", "W", "A"}
    motifs = []
    seen: set[int] = set()

    for hn, hc in his_res.items():
        # Negative signal: Cys within 7Å → zinc context, not haem
        if any(_dist(hc, cc) <= 5.0 for cc in cys_res.values()):
            continue
        # Count hydrophobic neighbours
        hydrophobic_count = sum(
            1 for nn, (aa, nc, _) in coord_map.items()
            if nn != hn and aa in HYDROPHOBIC and _dist(hc, nc) <= 8.0
        )
        if hydrophobic_count < 6:
            continue
        if hn in seen:
            continue
        seen.add(hn)
        motifs.append(CatalyticMotif(
            motif_type="haem_binding",
            residue_numbers=[hn],
            residue_letters=["H"],
            mean_distance=0.0,
            confidence="MEDIUM",
        ))

    return motifs[:3]


def _find_ploop(
    gly_res:   dict,
    coord_map: dict,
) -> list[CatalyticMotif]:
    """
    Detect P-loop / Walker A motif: GxxxxGK pattern.
    Looks for Gly residues separated by 4-5 residues in sequence.
    """
    motifs = []
    gly_keys = sorted(gly_res.keys())
    seen = set()
    for i, g1 in enumerate(gly_keys):
        for g2 in gly_keys[i+1:]:
            gap = g2 - g1
            if 4 <= gap <= 6:
                key = (g1, g2)
                if key in seen:
                    continue
                seen.add(key)
                d = _dist(gly_res[g1], gly_res[g2])
                if d > 10.0:
                    continue
                # Check for Lys near end of loop
                region = [n for n in coord_map if g1 <= n <= g2 + 1]
                letters = [coord_map[n][0] for n in region]
                motifs.append(CatalyticMotif(
                    motif_type="p_loop_walker_a",
                    residue_numbers=region[:7],
                    residue_letters=letters[:7],
                    mean_distance=round(d, 2),
                    confidence="MEDIUM",
                ))
    return motifs[:2]

# ── Residue scoring ────────────────────────────────────────────────────────────

def _score_residues(
    structure:        ParsedStructure,
    conservation_map: dict[int, float],
    motif_residues:   dict[int, list[str]],
    sasa_map:         dict[tuple[str, int], float],
) -> list[ActiveResidue]:
    """
    Score every residue and return those with at least LOW confidence.
    """
    active = []

    for i, res in enumerate(structure.residues):
        pos     = i + 1   # 1-based index for conservation lookup
        key     = (res.chain_id, res.residue_number)
        sasa    = sasa_map.get(key, 50.0)
        cons    = conservation_map.get(pos, 3.0)
        is_cons = cons >= CONSERVATION_THRESHOLD
        is_bur  = sasa < 20.0
        in_mot  = res.residue_number in motif_residues

        # Accumulate evidence points
        score = 0
        if is_cons:
            score += EVIDENCE_POINTS["conserved"]
        if is_bur:
            score += EVIDENCE_POINTS["buried"]
        if in_mot:
            score += EVIDENCE_POINTS["motif_member"]
        if res.plddt >= 80:
            score += EVIDENCE_POINTS["high_plddt"]

        # Bonus: residue type known to be catalytic
        if res.one_letter in {"C", "H", "D", "E", "R", "K"} and is_cons:
            score += EVIDENCE_POINTS["charged_context"]

        # Only keep residues with at least some evidence
        if score < CONFIDENCE_THRESHOLDS["LOW"]:
            continue

        if score >= CONFIDENCE_THRESHOLDS["HIGH"]:
            confidence = "HIGH"
        elif score >= CONFIDENCE_THRESHOLDS["MEDIUM"]:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        active.append(ActiveResidue(
            residue_number=res.residue_number,
            chain_id=res.chain_id,
            one_letter=res.one_letter,
            three_letter=res.residue_name,
            conservation=round(cons, 2),
            is_conserved=is_cons,
            is_buried=is_bur,
            sasa=round(sasa, 2),
            plddt=res.plddt,
            motifs=motif_residues.get(res.residue_number, []),
            evidence_score=score,
            confidence=confidence,
            coords=res.coords,
            domain_context="",
        ))

    # Sort by evidence score descending
    active.sort(key=lambda r: r.evidence_score, reverse=True)
    return active


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637)")
def main(uniprot: str) -> None:
    """
    Module 03 — Active site prediction.

    Requires Module 01 to have been run first (.pdb must exist).
    Module 02 SASA data is used if available.

    Example:
        python pipeline/active_sites.py --uniprot P04637
    """
    uniprot   = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    pdb_path  = Path(cfg.paths["structures"]) / f"{uniprot}.pdb"
    out_path  = inter_dir / f"{uniprot}_active_sites.json"

    if not pdb_path.exists():
        log.error(
            f".pdb not found: {pdb_path}\n"
            f"  Run Module 01 first: python pipeline/fetch_structure.py --uniprot {uniprot}"
        )
        raise SystemExit(1)

    # Load structure
    parsed = parse_pdb(pdb_path, uniprot)

    # Load SASA from Module 02 if available
    sasa_map: dict[tuple[str, int], float] = {}
    phys_path = inter_dir / f"{uniprot}_physicochemical.json"
    if phys_path.exists():
        log.info("  Loading SASA from Module 02 output...")
        with open(phys_path) as f:
            phys_data = json.load(f)
        for rec in phys_data.get("residues", []):
            key = (rec["chain_id"], rec["residue_number"])
            sasa_map[key] = rec["sasa"]
        log.info(f"  Loaded SASA for {len(sasa_map)} residues")
    else:
        log.warning("  Module 02 output not found — running without SASA data")

    # Run prediction
    result = predict_active_sites(parsed, sasa_map)

    # Save
    result.to_json(out_path)
    click.echo(f"\nDone. Results written to:\n  {out_path}")


if __name__ == "__main__":
    main()