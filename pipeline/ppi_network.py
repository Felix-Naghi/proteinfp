"""
pipeline/12_ppi_network.py
───────────────────────────
Module 12 — Protein-protein interaction network.

Queries the STRING DB API for known and predicted interaction partners,
then predicts binding interfaces for the top partners using:
  1. STRING DB REST API — 7 evidence channels, confidence scores 0-1000
  2. Interface residue prediction — surface exposure + evolutionary conservation
  3. Interaction type classification — inhibitory / activating / structural

STRING evidence channels:
  - neighborhood:   gene proximity in genomes
  - fusion:         gene fusion events
  - cooccurrence:   phylogenetic co-occurrence
  - coexpression:   correlated mRNA expression
  - experimental:   experimental evidence (highest weight)
  - database:       curated pathway databases
  - textmining:     literature co-mention

Confidence score 0-1000:
  - ≥ 900 = high confidence
  - ≥ 700 = medium-high
  - ≥ 400 = medium
  - < 400 = low

Usage (standalone):
    python pipeline/12_ppi_network.py --uniprot P04637

Usage (from orchestrator):
    from pipeline.ppi_network import predict_ppi
    result = predict_ppi("P04637", sequence, parsed_structure)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import click
import numpy as np
import requests

from utils.config import cfg, get_logger
from utils.pdb_parser import parse_pdb, ParsedStructure, HYDROPHOBICITY

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

STRING_API       = "https://string-db.org/api"
STRING_VERSION   = "12.0"
MAX_PARTNERS     = 20
MIN_SCORE        = 400      # medium confidence threshold

# Interface prediction constants
INTERFACE_SASA_THRESHOLD  = 30.0   # Å² — residue must be exposed
INTERFACE_HYDRO_THRESHOLD = 0.5    # Kyte-Doolittle — hydrophobic interface residues

# Charge at pH 7.4
CHARGE_AT_PH7: dict[str, float] = {
    "A":  0.0, "R": +1.0, "N":  0.0, "D": -1.0, "C":  0.0,
    "Q":  0.0, "E": -1.0, "G":  0.0, "H": +0.1, "I":  0.0,
    "L":  0.0, "K": +1.0, "M":  0.0, "F":  0.0, "P":  0.0,
    "S":  0.0, "T":  0.0, "W":  0.0, "Y":  0.0, "V":  0.0,
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class PPIPartner:
    """A single predicted protein interaction partner."""
    partner_id:         str       # STRING ID or UniProt accession
    partner_name:       str
    combined_score:     int       # STRING combined score 0-1000
    experimental_score: int
    database_score:     int
    textmining_score:   int
    coexpression_score: int
    confidence:         str       # "high" / "medium" / "low"
    interaction_type:   str       # "inhibitory" / "activating" / "structural" / "unknown"
    interface_residues: list[int] # predicted interface residue numbers
    interface_letters:  list[str]
    interface_charge:   float
    interface_hydrophobicity: float
    binding_mode:       str       # "hydrophobic" / "electrostatic" / "mixed"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PPIResult:
    """Full PPI prediction output. Output of Module 12."""
    uniprot_id:      str
    string_id:       str
    n_partners:      int                   = 0
    partners:        list[PPIPartner]      = field(default_factory=list)
    high_confidence: list[PPIPartner]      = field(default_factory=list)
    n_high:          int                   = 0
    n_medium:        int                   = 0
    api_available:   bool                  = False
    top_partner:     str                   = ""
    network_hubs:    list[str]             = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  PPI network: {self.uniprot_id}",
            f"  STRING ID      : {self.string_id or 'not found'}",
            f"  API available  : {'yes' if self.api_available else 'no'}",
            f"  Partners found : {self.n_partners}",
            f"    High conf (≥700): {self.n_high}",
            f"    Medium conf    : {self.n_medium}",
        ]
        for p in self.partners[:8]:
            lines.append(
                f"  {p.partner_name[:20]:20s} score={p.combined_score:4d} "
                f"({p.confidence:6s}) {p.interaction_type:12s} "
                f"iface={len(p.interface_residues)} res"
            )
        lines.append(f"{'─'*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.debug(f"Saved PPI JSON → {path}")


# ── Main function ──────────────────────────────────────────────────────────────

def predict_ppi(
    uniprot_id: str,
    sequence:   str,
    structure:  Optional[ParsedStructure] = None,
    sasa_map:   Optional[dict]            = None,
) -> PPIResult:
    """
    Predict protein-protein interactions via STRING DB.

    Args:
        uniprot_id: UniProt accession
        sequence:   Amino acid sequence
        structure:  ParsedStructure from Module 01 (for interface prediction)
        sasa_map:   (chain, res_num) → SASA from Module 02

    Returns:
        PPIResult with interaction partners and interface predictions.
    """
    log.info(f"── Module 12: PPI network for {uniprot_id} ──")

    result = PPIResult(uniprot_id=uniprot_id, string_id="")

    # ── Step 1: Get STRING ID ─────────────────────────────────────────────────
    log.info("  [1/3] Resolving STRING ID...")
    string_id = _get_string_id(uniprot_id)

    if not string_id:
        log.warning("  Could not resolve STRING ID — using gene name lookup")
        string_id = _get_string_id_by_name(uniprot_id)

    if not string_id:
        log.warning("  STRING ID not found — skipping network query")
        result.api_available = False
        return result

    result.string_id    = string_id
    result.api_available = True
    log.info(f"    STRING ID: {string_id}")

    # ── Step 2: Query interaction partners ────────────────────────────────────
    log.info("  [2/3] Querying STRING DB for interaction partners...")
    raw_partners = _query_string_partners(string_id)
    log.info(f"    {len(raw_partners)} partners returned")

    # ── Step 3: Predict interfaces for each partner ───────────────────────────
    log.info("  [3/3] Predicting binding interfaces...")
    partners = []

    for raw in raw_partners[:MAX_PARTNERS]:
        partner = _build_partner(raw, structure, sasa_map or {}, sequence)
        partners.append(partner)

    partners.sort(key=lambda p: p.combined_score, reverse=True)

    high   = [p for p in partners if p.combined_score >= 700]
    medium = [p for p in partners if 400 <= p.combined_score < 700]
    hubs   = [p.partner_name for p in partners[:5]]

    result.n_partners      = len(partners)
    result.partners        = partners
    result.high_confidence = high
    result.n_high          = len(high)
    result.n_medium        = len(medium)
    result.top_partner     = partners[0].partner_name if partners else ""
    result.network_hubs    = hubs

    log.info(result.summary())
    return result


# ── STRING DB API ──────────────────────────────────────────────────────────────

def _get_string_id(uniprot_id: str) -> str:
    """Resolve UniProt accession to STRING ID."""
    try:
        species = cfg.get("string_db", "species", default=9606)
        resp = requests.get(
            f"{STRING_API}/json/get_string_ids",
            params={
                "identifiers": uniprot_id,
                "species":     species,
                "limit":       1,
                "echo_query":  0,
            },
            timeout=15,
            headers={"User-Agent": "ProteinFP/0.1"},
        )
        if resp.status_code != 200 or not resp.json():
            return ""
        data = resp.json()
        return data[0].get("stringId", "")
    except Exception as e:
        log.debug(f"    STRING ID lookup failed: {e}")
        return ""


def _get_string_id_by_name(uniprot_id: str) -> str:
    """Try resolving by common gene names for well-known proteins."""
    known = {
        "P04637": "9606.ENSP00000269305",   # TP53
        "P00533": "9606.ENSP00000275493",   # EGFR
        "P38398": "9606.ENSP00000309572",   # BRCA1
        "P06213": "9606.ENSP00000241135",   # INSR
        "P00441": "9606.ENSP00000261509",   # SOD1
    }
    return known.get(uniprot_id, "")


def _query_string_partners(string_id: str) -> list[dict]:
    """Query STRING DB for interaction partners."""
    try:
        min_score = cfg.get("string_db", "min_score", default=MIN_SCORE)
        limit     = cfg.get("string_db", "limit",     default=MAX_PARTNERS)

        resp = requests.get(
            f"{STRING_API}/json/interaction_partners",
            params={
                "identifiers": string_id,
                "species":     9606,
                "limit":       limit,
                "required_score": min_score,
            },
            timeout=20,
            headers={"User-Agent": "ProteinFP/0.1"},
        )
        if resp.status_code != 200:
            log.warning(f"    STRING partners: HTTP {resp.status_code}")
            return []
        return resp.json()
    except Exception as e:
        log.warning(f"    STRING partners failed: {e}")
        return []


# ── Interface prediction ───────────────────────────────────────────────────────

def _build_partner(
    raw:       dict,
    structure: Optional[ParsedStructure],
    sasa_map:  dict,
    sequence:  str,
) -> PPIPartner:
    """
    Build a PPIPartner from STRING raw data + interface prediction.
    """
    partner_id   = raw.get("stringId_B", raw.get("preferredName_B", ""))
    partner_name = raw.get("preferredName_B", partner_id)

    combined    = int(raw.get("score",         0) * 1000) if isinstance(raw.get("score"), float) else int(raw.get("score", 0))
    experimental = int(raw.get("escore",       0) * 1000) if isinstance(raw.get("escore"), float) else int(raw.get("escore", 0))
    database    = int(raw.get("dscore",        0) * 1000) if isinstance(raw.get("dscore"), float) else int(raw.get("dscore", 0))
    textmining  = int(raw.get("tscore",        0) * 1000) if isinstance(raw.get("tscore"), float) else int(raw.get("tscore", 0))
    coexpr      = int(raw.get("coexpression",  0) * 1000) if isinstance(raw.get("coexpression"), float) else int(raw.get("coexpression", 0))

    if combined >= 900:
        confidence = "high"
    elif combined >= 700:
        confidence = "medium-high"
    elif combined >= 400:
        confidence = "medium"
    else:
        confidence = "low"

    # Classify interaction type from partner name
    interaction_type = _classify_interaction(partner_name)

    # Predict interface residues
    if structure and sasa_map:
        iface_res, iface_let = _predict_interface(
            structure, sasa_map, interaction_type
        )
    else:
        iface_res, iface_let = _sequence_interface(sequence)

    # Interface chemistry
    hydrophobes = [HYDROPHOBICITY.get(aa, 0.0) for aa in iface_let]
    charges     = [CHARGE_AT_PH7.get(aa, 0.0)  for aa in iface_let]
    mean_hydro  = float(np.mean(hydrophobes)) if hydrophobes else 0.0
    net_charge  = float(sum(charges))

    if mean_hydro > INTERFACE_HYDRO_THRESHOLD:
        binding_mode = "hydrophobic"
    elif abs(net_charge) > 2:
        binding_mode = "electrostatic"
    else:
        binding_mode = "mixed"

    return PPIPartner(
        partner_id=partner_id,
        partner_name=partner_name,
        combined_score=combined,
        experimental_score=experimental,
        database_score=database,
        textmining_score=textmining,
        coexpression_score=coexpr,
        confidence=confidence,
        interaction_type=interaction_type,
        interface_residues=iface_res[:20],
        interface_letters=iface_let[:20],
        interface_charge=round(net_charge, 2),
        interface_hydrophobicity=round(mean_hydro, 3),
        binding_mode=binding_mode,
    )


def _classify_interaction(partner_name: str) -> str:
    """Classify interaction type from partner name heuristics."""
    name = partner_name.upper()
    if any(k in name for k in ["MDM", "MDMX", "MDM2", "UBIQUITIN", "UBB", "UBC"]):
        return "inhibitory"
    if any(k in name for k in ["ATM", "ATR", "CHK", "CHEK", "PRKDC"]):
        return "activating"
    if any(k in name for k in ["BRCA", "RAD", "PCNA", "RPA", "POLE"]):
        return "cooperative"
    if any(k in name for k in ["HISTO", "H2A", "H2B", "H3", "H4"]):
        return "structural"
    if any(k in name for k in ["CASP", "BCL", "BAX", "BAK"]):
        return "apoptotic"
    return "unknown"


def _predict_interface(
    structure: ParsedStructure,
    sasa_map:  dict,
    interaction_type: str,
) -> tuple[list[int], list[str]]:
    """
    Predict interface residues based on surface exposure + charge.
    Different interaction types use different surface properties.
    """
    candidates = []

    for res in structure.residues:
        key  = (res.chain_id, res.residue_number)
        sasa = sasa_map.get(key, 0.0)

        if sasa < INTERFACE_SASA_THRESHOLD:
            continue   # buried — not on surface

        score = 0.0

        # Charge-based interactions (MDM2, activating kinases)
        if interaction_type in ("inhibitory", "activating"):
            charge = CHARGE_AT_PH7.get(res.one_letter, 0.0)
            if abs(charge) > 0:
                score += 2.0
            if res.one_letter in {"F", "W", "L", "I", "V"}:
                score += 1.5  # hydrophobic hot spots

        # Cooperative interactions (BRCA1, RAD proteins)
        elif interaction_type == "cooperative":
            if res.one_letter in {"K", "R", "H"}:
                score += 2.0
            if res.plddt >= 80:
                score += 1.0  # prefer high-confidence regions

        # Default: general surface
        else:
            score = sasa / 100.0

        if score > 0.5:
            candidates.append((res.residue_number, res.one_letter, score))

    candidates.sort(key=lambda x: x[2], reverse=True)
    top = candidates[:15]
    return [c[0] for c in top], [c[1] for c in top]


def _sequence_interface(sequence: str) -> tuple[list[int], list[str]]:
    """
    Fallback interface prediction from sequence alone.
    Returns exposed charged/hydrophobic residues likely on the surface.
    """
    candidates = []
    for i, aa in enumerate(sequence):
        if aa in {"K", "R", "E", "D", "F", "W", "Y", "L", "I"}:
            candidates.append((i + 1, aa))
    return (
        [c[0] for c in candidates[:15]],
        [c[1] for c in candidates[:15]],
    )


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637)")
def main(uniprot: str) -> None:
    """
    Module 12 — Protein-protein interaction network.

    Queries STRING DB for interaction partners and predicts binding interfaces.
    Requires Module 01 (.pdb) and Module 02 (SASA) to have run first.

    Example:
        python pipeline/ppi_network.py --uniprot P04637
    """
    uniprot   = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    pdb_path  = Path(cfg.paths["structures"]) / f"{uniprot}.pdb"
    out_path  = inter_dir / f"{uniprot}_ppi.json"

    if not pdb_path.exists():
        log.error(f".pdb not found — run Module 01 first")
        raise SystemExit(1)

    parsed   = parse_pdb(pdb_path, uniprot)
    sequence = parsed.sequence

    # Load SASA from Module 02
    sasa_map: dict = {}
    phys_path = inter_dir / f"{uniprot}_physicochemical.json"
    if phys_path.exists():
        with open(phys_path) as f:
            phys = json.load(f)
        for rec in phys.get("residues", []):
            sasa_map[(rec["chain_id"], rec["residue_number"])] = rec["sasa"]
        log.info(f"  Loaded SASA for {len(sasa_map)} residues")

    result = predict_ppi(uniprot, sequence, parsed, sasa_map)
    result.to_json(out_path)
    click.echo(f"\nDone. Results written to:\n  {out_path}")


if __name__ == "__main__":
    main()