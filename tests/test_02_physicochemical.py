"""
tests/test_02_physicochemical.py
─────────────────────────────────
Tests for Module 02 — Physicochemical surface analysis.

Run with:
    python -m pytest tests/test_02_physicochemical.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── Minimal PDB fixture (same style as Module 01 tests) ───────────────────────

MINIMAL_PDB = """\
ATOM      1  N   MET A   1       1.000   2.000   3.000  1.00 92.15           N
ATOM      2  CA  MET A   1       1.500   2.500   3.500  1.00 92.15           C
ATOM      3  C   MET A   1       2.000   3.000   4.000  1.00 92.15           C
ATOM      4  O   MET A   1       2.500   3.500   4.500  1.00 92.15           O
ATOM      5  N   GLU A   2       3.000   4.000   5.000  1.00 87.30           N
ATOM      6  CA  GLU A   2       3.500   4.500   5.500  1.00 87.30           C
ATOM      7  N   SER A   3       4.000   5.000   6.000  1.00 45.20           N
ATOM      8  CA  SER A   3       4.500   5.500   6.500  1.00 45.20           C
ATOM      9  N   ALA A   4       5.000   6.000   7.000  1.00 72.80           N
ATOM     10  CA  ALA A   4       5.500   6.500   7.500  1.00 72.80           C
ATOM     11  N   LYS A   5       6.000   7.000   8.000  1.00 85.00           N
ATOM     12  CA  LYS A   5       6.500   7.500   8.500  1.00 85.00           C
ATOM     13  N   ARG A   6       7.000   8.000   9.000  1.00 88.00           N
ATOM     14  CA  ARG A   6       7.500   8.500   9.500  1.00 88.00           C
ATOM     15  N   LEU A   7       8.000   9.000  10.000  1.00 90.00           N
ATOM     16  CA  LEU A   7       8.500   9.500  10.500  1.00 90.00           C
END
"""


@pytest.fixture
def pdb_file(tmp_path: Path) -> Path:
    p = tmp_path / "P00000.pdb"
    p.write_text(MINIMAL_PDB)
    return p


@pytest.fixture
def parsed_structure(pdb_file):
    from utils.pdb_parser import parse_pdb
    return parse_pdb(pdb_file, "P00000", plddt_threshold=70.0)


# ── Unit tests ─────────────────────────────────────────────────────────────────

class TestPhysicochemResult:

    def test_returns_correct_length(self, parsed_structure):
        from pipeline.physicochemical import compute_physicochemical
        result = compute_physicochemical(parsed_structure)
        assert result.length == 7
        assert len(result.residues) == 7

    def test_charge_values_assigned(self, parsed_structure):
        from pipeline.physicochemical import compute_physicochemical
        result = compute_physicochemical(parsed_structure)
        res_map = {r.residue_number: r for r in result.residues}
        # GLU (residue 2) should be negative
        assert res_map[2].charge == -1.0
        # LYS (residue 5) should be positive
        assert res_map[5].charge == +1.0
        # ARG (residue 6) should be positive
        assert res_map[6].charge == +1.0
        # ALA (residue 4) should be neutral
        assert res_map[4].charge == 0.0

    def test_hydrophobicity_assigned(self, parsed_structure):
        from pipeline.physicochemical import compute_physicochemical, HYDROPHOBICITY
        result = compute_physicochemical(parsed_structure)
        for rec in result.residues:
            expected = HYDROPHOBICITY.get(rec.one_letter, 0.0)
            assert abs(rec.hydrophobicity - expected) < 0.001

    def test_total_charge_is_sum(self, parsed_structure):
        from pipeline.physicochemical import compute_physicochemical, CHARGE_AT_PH7
        result = compute_physicochemical(parsed_structure)
        expected_total = sum(
            CHARGE_AT_PH7.get(r.one_letter, 0.0)
            for r in parsed_structure.residues
        )
        assert abs(result.total_charge - expected_total) < 0.01

    def test_sasa_non_negative(self, parsed_structure):
        from pipeline.physicochemical import compute_physicochemical
        result = compute_physicochemical(parsed_structure)
        for rec in result.residues:
            assert rec.sasa >= 0.0

    def test_sasa_fraction_bounded(self, parsed_structure):
        from pipeline.physicochemical import compute_physicochemical
        result = compute_physicochemical(parsed_structure)
        for rec in result.residues:
            assert 0.0 <= rec.sasa_fraction <= 1.0

    def test_secondary_structure_valid_labels(self, parsed_structure):
        from pipeline.physicochemical import compute_physicochemical, DSSP_LABELS
        result = compute_physicochemical(parsed_structure)
        valid = set(DSSP_LABELS.values())
        for rec in result.residues:
            assert rec.secondary_structure in valid, \
                f"Invalid SS label: {rec.secondary_structure}"

    def test_fractions_sum_to_one(self, parsed_structure):
        from pipeline.physicochemical import compute_physicochemical
        result = compute_physicochemical(parsed_structure)
        total = result.helix_fraction + result.strand_fraction + result.coil_fraction
        assert abs(total - 1.0) < 0.01

    def test_charge_asymmetry_label(self, parsed_structure):
        from pipeline.physicochemical import compute_physicochemical
        result = compute_physicochemical(parsed_structure)
        assert result.charge_asymmetry in ("positive", "negative", "neutral")

    def test_to_dict_json_serialisable(self, parsed_structure):
        from pipeline.physicochemical import compute_physicochemical
        result = compute_physicochemical(parsed_structure)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_to_json_writes_file(self, parsed_structure, tmp_path):
        from pipeline.physicochemical import compute_physicochemical
        result = compute_physicochemical(parsed_structure)
        out = tmp_path / "physchem.json"
        result.to_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["uniprot_id"] == "P00000"
        assert data["length"] == 7

    def test_summary_string(self, parsed_structure):
        from pipeline.physicochemical import compute_physicochemical
        result = compute_physicochemical(parsed_structure)
        s = result.summary()
        assert "P00000" in s
        assert "SASA" in s


class TestSurfacePatches:

    def test_patches_are_lists(self, parsed_structure):
        from pipeline.physicochemical import compute_physicochemical
        result = compute_physicochemical(parsed_structure)
        assert isinstance(result.hydrophobic_patches, list)
        assert isinstance(result.positive_patches, list)
        assert isinstance(result.negative_patches, list)

    def test_patch_residues_are_ints(self, parsed_structure):
        from pipeline.physicochemical import compute_physicochemical
        result = compute_physicochemical(parsed_structure)
        for patch in (result.hydrophobic_patches
                      + result.positive_patches
                      + result.negative_patches):
            for rn in patch.residue_numbers:
                assert isinstance(rn, int)

    def test_patch_size_matches_residue_count(self, parsed_structure):
        from pipeline.physicochemical import compute_physicochemical
        result = compute_physicochemical(parsed_structure)
        for patch in (result.hydrophobic_patches
                      + result.positive_patches
                      + result.negative_patches):
            assert patch.size == len(patch.residue_numbers)