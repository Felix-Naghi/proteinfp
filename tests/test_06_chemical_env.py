"""
tests/test_06_chemical_env.py
──────────────────────────────
Tests for Module 06 — Chemical environment mapping.

Run with:
    python -m pytest tests/test_06_chemical_env.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np


# ── Fixtures ───────────────────────────────────────────────────────────────────

MINIMAL_PDB = """\
ATOM      1  CA  HIS A   1       0.000   0.000   0.000  1.00 85.00           C
ATOM      2  CA  CYS A   2       4.000   0.000   0.000  1.00 85.00           C
ATOM      3  CA  ASP A   3       8.000   0.000   0.000  1.00 85.00           C
ATOM      4  CA  GLU A   4       0.000   4.000   0.000  1.00 85.00           C
ATOM      5  CA  ARG A   5       4.000   4.000   0.000  1.00 85.00           C
ATOM      6  CA  LYS A   6       8.000   4.000   0.000  1.00 85.00           C
ATOM      7  CA  PHE A   7       0.000   8.000   0.000  1.00 85.00           C
ATOM      8  CA  TRP A   8       4.000   8.000   0.000  1.00 85.00           C
ATOM      9  CA  ILE A   9       8.000   8.000   0.000  1.00 85.00           C
ATOM     10  CA  LEU A  10       4.000   4.000   4.000  1.00 85.00           C
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
    return parse_pdb(pdb_file, "P00000")


@pytest.fixture
def coords_lookup(parsed_structure):
    return {r.residue_number: np.array(r.coords)
            for r in parsed_structure.residues}


@pytest.fixture
def sample_active_data():
    return {
        "active_residues": [
            {"residue_number": 1, "confidence": "HIGH"},
            {"residue_number": 2, "confidence": "HIGH"},
            {"residue_number": 3, "confidence": "HIGH"},
        ],
        "catalytic_motifs": [
            {
                "motif_type": "zinc_binding",
                "residue_numbers": [1, 2, 3],
                "residue_letters": ["H", "C", "D"],
            }
        ]
    }


@pytest.fixture
def sample_pocket_data():
    return {
        "pockets": [
            {
                "pocket_id": "P1",
                "lining_residues": [5, 6, 7, 8],
                "lining_letters": ["R", "K", "F", "W"],
            }
        ]
    }


@pytest.fixture
def sample_allo_data():
    return {
        "allosteric_sites": [
            {
                "site_id": "A1",
                "residue_numbers": [9, 10],
                "residue_letters": ["I", "L"],
            }
        ]
    }


# ── Unit tests: _compute_site_env ─────────────────────────────────────────────

class TestComputeSiteEnv:

    def test_returns_site_chem_env(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env, SiteChemEnv
        env = _compute_site_env(
            site_id="test",
            site_type="active",
            residue_numbers=[1, 2, 3],
            residue_letters=["H", "C", "D"],
            coords_lookup=coords_lookup,
        )
        assert isinstance(env, SiteChemEnv)

    def test_site_id_preserved(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env
        env = _compute_site_env("my_site", "active", [1], ["A"], coords_lookup)
        assert env.site_id == "my_site"

    def test_site_type_preserved(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env
        env = _compute_site_env("s1", "binding", [1], ["A"], coords_lookup)
        assert env.site_type == "binding"

    def test_charge_character_positive(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env
        # ARG(+1) + LYS(+1) + ARG(+1) = net +3 → positive
        env = _compute_site_env("s", "active", [1, 2, 3],
                                ["R", "K", "R"], coords_lookup)
        assert env.charge_character == "positive"

    def test_charge_character_negative(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env
        env = _compute_site_env("s", "active", [1, 2, 3],
                                ["D", "E", "D"], coords_lookup)
        assert env.charge_character == "negative"

    def test_charge_character_neutral(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env
        env = _compute_site_env("s", "active", [1, 2],
                                ["A", "G"], coords_lookup)
        assert env.charge_character == "neutral"

    def test_hbond_donors_counted(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env, HBOND_DONORS
        letters = ["R", "S", "N"]
        env = _compute_site_env("s", "active", [1, 2, 3], letters, coords_lookup)
        expected = sum(HBOND_DONORS[aa] for aa in letters)
        assert env.n_hbond_donors == expected

    def test_hbond_acceptors_counted(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env, HBOND_ACCEPTORS
        letters = ["D", "E", "N"]
        env = _compute_site_env("s", "active", [1, 2, 3], letters, coords_lookup)
        expected = sum(HBOND_ACCEPTORS[aa] for aa in letters)
        assert env.n_hbond_acceptors == expected

    def test_aromatic_count(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env
        env = _compute_site_env("s", "active", [1, 2, 3],
                                ["F", "W", "A"], coords_lookup)
        assert env.n_aromatic == 2

    def test_metal_coord_count(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env
        env = _compute_site_env("s", "active", [1, 2, 3, 4],
                                ["H", "C", "D", "E"], coords_lookup)
        assert env.n_metal_coord == 4

    def test_amphipathic_detected(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env
        # ILE (hydrophobic) + ASP (hydrophilic) = amphipathic
        env = _compute_site_env("s", "active", [1, 2],
                                ["I", "D"], coords_lookup)
        assert env.is_amphipathic is True

    def test_hydrophobic_mode(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env
        env = _compute_site_env("s", "active", [1, 2, 3, 4],
                                ["I", "L", "V", "F"], coords_lookup)
        assert env.predicted_binding_mode in ("hydrophobic", "hydrophobic+pi", "mixed")

    def test_hbond_mode(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env
        env = _compute_site_env("s", "active", [1, 2, 3, 4, 5],
                                ["D", "E", "N", "Q", "S"], coords_lookup)
        assert env.predicted_binding_mode in ("h-bond heavy", "mixed")

    def test_metal_mode(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env
        env = _compute_site_env("s", "active", [1, 2, 3, 4],
                                ["H", "C", "D", "E"], coords_lookup)
        assert env.predicted_binding_mode == "metal"

    def test_ligand_efficiency_bounded(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env
        env = _compute_site_env("s", "active", [1, 2, 3],
                                ["H", "C", "D"], coords_lookup)
        assert 0.0 <= env.ligand_efficiency_est <= 1.0

    def test_to_dict_serialisable(self, coords_lookup):
        from pipeline.chemical_env import _compute_site_env
        env = _compute_site_env("s", "active", [1, 2, 3],
                                ["H", "R", "F"], coords_lookup)
        d = env.to_dict()
        json.dumps(d)  # should not raise


# ── Unit tests: metal binding score ───────────────────────────────────────────

class TestMetalBindingScore:

    def test_no_metal_residues_scores_zero(self, coords_lookup):
        from pipeline.chemical_env import _metal_binding_score
        score = _metal_binding_score(["A", "G", "I"], [1, 2, 3], coords_lookup)
        assert score == 0.0

    def test_four_metal_residues_scores_high(self, coords_lookup):
        from pipeline.chemical_env import _metal_binding_score
        score = _metal_binding_score(
            ["H", "C", "D", "E"], [1, 2, 3, 4], coords_lookup
        )
        assert score > 0.4

    def test_score_bounded(self, coords_lookup):
        from pipeline.chemical_env import _metal_binding_score
        score = _metal_binding_score(
            ["H", "C", "D", "E", "H"], [1, 2, 3, 4, 5], coords_lookup
        )
        assert 0.0 <= score <= 1.0


# ── Unit tests: full pipeline ─────────────────────────────────────────────────

class TestMapChemicalEnvironment:

    def test_returns_result_object(self, parsed_structure):
        from pipeline.chemical_env import map_chemical_environment, ChemEnvResult
        result = map_chemical_environment(parsed_structure)
        assert isinstance(result, ChemEnvResult)

    def test_uniprot_id_preserved(self, parsed_structure):
        from pipeline.chemical_env import map_chemical_environment
        result = map_chemical_environment(parsed_structure)
        assert result.uniprot_id == "P00000"

    def test_maps_active_sites(
        self, parsed_structure, sample_active_data
    ):
        from pipeline.chemical_env import map_chemical_environment
        result = map_chemical_environment(
            parsed_structure, active_data=sample_active_data
        )
        assert len(result.active_envs) > 0

    def test_maps_binding_pockets(
        self, parsed_structure, sample_pocket_data
    ):
        from pipeline.chemical_env import map_chemical_environment
        result = map_chemical_environment(
            parsed_structure, pocket_data=sample_pocket_data
        )
        assert len(result.binding_envs) > 0

    def test_maps_allosteric_sites(
        self, parsed_structure, sample_allo_data
    ):
        from pipeline.chemical_env import map_chemical_environment
        result = map_chemical_environment(
            parsed_structure, allo_data=sample_allo_data
        )
        assert len(result.allo_envs) > 0

    def test_n_sites_mapped_correct(
        self, parsed_structure,
        sample_active_data, sample_pocket_data, sample_allo_data
    ):
        from pipeline.chemical_env import map_chemical_environment
        result = map_chemical_environment(
            parsed_structure,
            active_data=sample_active_data,
            pocket_data=sample_pocket_data,
            allo_data=sample_allo_data,
        )
        expected = (len(result.active_envs) +
                    len(result.binding_envs) +
                    len(result.allo_envs))
        assert result.n_sites_mapped == expected

    def test_runs_with_no_data(self, parsed_structure):
        from pipeline.chemical_env import map_chemical_environment
        result = map_chemical_environment(parsed_structure)
        assert result.n_sites_mapped == 0

    def test_to_dict_serialisable(
        self, parsed_structure, sample_active_data
    ):
        from pipeline.chemical_env import map_chemical_environment
        result = map_chemical_environment(
            parsed_structure, active_data=sample_active_data
        )
        json.dumps(result.to_dict())

    def test_to_json_writes_file(
        self, parsed_structure, sample_active_data, tmp_path
    ):
        from pipeline.chemical_env import map_chemical_environment
        result = map_chemical_environment(
            parsed_structure, active_data=sample_active_data
        )
        out = tmp_path / "chem_env.json"
        result.to_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["uniprot_id"] == "P00000"

    def test_summary_string(
        self, parsed_structure, sample_active_data
    ):
        from pipeline.chemical_env import map_chemical_environment
        result = map_chemical_environment(
            parsed_structure, active_data=sample_active_data
        )
        s = result.summary()
        assert "P00000" in s
        assert "Chemical environment" in s