"""
utils/pdb_parser.py
───────────────────
Shared PDB / structure parsing helpers built on BioPython.
Used by Module 01 (structure fetch), Module 03 (active sites),
Module 04 (binding pockets), and Module 05 (allostery).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import numpy as np
from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue

from utils.config import get_logger

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Standard 3-letter to 1-letter amino acid map
AA3TO1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Non-standard / modified
    "SEC": "U", "PYL": "O", "MSE": "M",
}

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY: dict[str, float] = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ResidueInfo:
    chain_id:       str
    residue_number: int
    insertion_code: str
    residue_name:   str          # 3-letter
    one_letter:     str          # 1-letter (X if unknown)
    plddt:          float        # 0–100; stored in B-factor by AFDB
    is_disordered:  bool
    hydrophobicity: float
    coords:         list[float]  # CA coordinates [x, y, z]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParsedStructure:
    uniprot_id:         str
    pdb_path:           str
    sequence:           str
    length:             int
    residues:           list[ResidueInfo] = field(default_factory=list)
    mean_plddt:         float = 0.0
    disordered_regions: list[tuple[int, int]] = field(default_factory=list)
    # Summary stats
    n_disordered:       int = 0
    high_conf_fraction: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.debug(f"Saved parsed structure JSON → {path}")


# ── Core parser ────────────────────────────────────────────────────────────────

def parse_pdb(
    pdb_path:        str | Path,
    uniprot_id:      str,
    plddt_threshold: float = 70.0,
) -> ParsedStructure:
    """
    Parse an AlphaFold .pdb file and return a rich ParsedStructure object.

    AFDB convention:
      - B-factor column stores pLDDT (0–100) per atom.
      - We take the CA atom's B-factor as the residue-level pLDDT.
      - Chain A is always the protein chain.

    Args:
        pdb_path:        Path to the .pdb file.
        uniprot_id:      UniProt accession (for labelling).
        plddt_threshold: Residues below this are flagged as disordered.

    Returns:
        ParsedStructure with per-residue annotations.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    log.info(f"Parsing structure: {pdb_path.name}")

    parser = PDBParser(QUIET=True)
    structure: Structure = parser.get_structure(uniprot_id, str(pdb_path))

    residue_infos: list[ResidueInfo] = []
    sequence_chars: list[str] = []

    model = structure[0]         # AFDB always has a single model

    for residue in _iter_std_residues(model):
        res_name = residue.get_resname().strip()
        one_letter = AA3TO1.get(res_name, "X")
        chain_id = residue.get_parent().get_id()
        res_id = residue.get_id()

        # pLDDT is stored in the B-factor of the CA atom
        plddt = _get_ca_bfactor(residue)

        # CA coordinates
        ca_coords = _get_ca_coords(residue)

        ri = ResidueInfo(
            chain_id=chain_id,
            residue_number=res_id[1],
            insertion_code=res_id[2].strip(),
            residue_name=res_name,
            one_letter=one_letter,
            plddt=plddt,
            is_disordered=(plddt < plddt_threshold),
            hydrophobicity=HYDROPHOBICITY.get(one_letter, 0.0),
            coords=ca_coords,
        )
        residue_infos.append(ri)
        sequence_chars.append(one_letter)

    if not residue_infos:
        raise ValueError(f"No standard residues found in {pdb_path}")

    sequence  = "".join(sequence_chars)
    plddt_arr = np.array([r.plddt for r in residue_infos])
    mean_plddt = float(np.mean(plddt_arr))

    disordered = _find_disordered_regions(residue_infos)
    n_disordered = sum(1 for r in residue_infos if r.is_disordered)
    high_conf_fraction = float(np.mean(plddt_arr >= plddt_threshold))

    result = ParsedStructure(
        uniprot_id=uniprot_id,
        pdb_path=str(pdb_path),
        sequence=sequence,
        length=len(residue_infos),
        residues=residue_infos,
        mean_plddt=round(mean_plddt, 2),
        disordered_regions=disordered,
        n_disordered=n_disordered,
        high_conf_fraction=round(high_conf_fraction, 3),
    )

    log.info(
        f"  → {result.length} residues | mean pLDDT: {result.mean_plddt:.1f} | "
        f"disordered: {n_disordered} residues in {len(disordered)} region(s)"
    )
    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _iter_std_residues(model) -> Iterator[Residue]:
    """Yield standard amino acid residues only (skip HETATM / water)."""
    for chain in model:
        for residue in chain:
            if residue.get_id()[0] != " ":   # hetero flag
                continue
            if residue.get_resname().strip() not in AA3TO1:
                continue
            yield residue


def _get_ca_bfactor(residue: Residue) -> float:
    """Return CA atom B-factor (= pLDDT in AFDB files). Falls back to 0."""
    try:
        return float(residue["CA"].get_bfactor())
    except KeyError:
        return 0.0


def _get_ca_coords(residue: Residue) -> list[float]:
    """Return CA [x, y, z] coordinates. Returns [0,0,0] if no CA atom."""
    try:
        vec = residue["CA"].get_vector()
        return [round(float(vec[0]), 3),
                round(float(vec[1]), 3),
                round(float(vec[2]), 3)]
    except KeyError:
        return [0.0, 0.0, 0.0]


def _find_disordered_regions(
    residues: list[ResidueInfo],
    min_length: int = 3,
) -> list[tuple[int, int]]:
    """
    Find contiguous runs of disordered residues.
    Only reports runs of >= min_length residues to avoid noise.

    Returns list of (start_residue_number, end_residue_number) tuples.
    """
    regions: list[tuple[int, int]] = []
    in_region = False
    start = 0

    for i, res in enumerate(residues):
        if res.is_disordered and not in_region:
            in_region = True
            start = res.residue_number
        elif not res.is_disordered and in_region:
            in_region = False
            end = residues[i - 1].residue_number
            if (end - start + 1) >= min_length:
                regions.append((start, end))

    # Handle case where protein ends in a disordered region
    if in_region:
        end = residues[-1].residue_number
        if (end - start + 1) >= min_length:
            regions.append((start, end))

    return regions
