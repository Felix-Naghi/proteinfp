"""
tests/test_05_allosteric.py
────────────────────────────
Tests for Module 05 — Allosteric site prediction.

Run with:
    python -m pytest tests/test_05_allosteric.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest


# ── Fixtures ───────────────────────────────────────────────────────────────────

# 20-residue PDB with enough density to build a contact network
DENSE_PDB = """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 85.00           C
ATOM      2  CA  ARG A   2       4.000   0.000   0.000  1.00 85.00           C
ATOM      3  CA  ASP A   3       8.000   0.000   0.000  1.00 85.00           C
ATOM      4  CA  CYS A   4       0.000   4.000   0.000  1.00 85.00           C
ATOM      5  CA  GLU A   5       4.000   4.000   0.000  1.00 85.00           C
ATOM      6  CA  HIS A   6       8.000   4.000   0.000  1.00 85.00           C
ATOM      7  CA  ILE A   7       0.000   8.000   0.000  1.00 85.00           C
ATOM      8  CA  LYS A   8       4.000   8.000   0.000  1.00 85.00           C
ATOM      9  CA  LEU A   9       8.000   8.000   0.000  1.00 85.00           C
ATOM     10  CA  MET A  10       0.000   0.000   4.000  1.00 85.00           C
ATOM     11  CA  ASN A  11       4.000   0.000   4.000  1.00 85.00           C
ATOM     12  CA  PRO A  12       8.000   0.000   4.000  1.00 85.00           C
ATOM     13  CA  GLN A  13       0.000   4.000   4.000  1.00 85.00           C
ATOM     14  CA  SER A  14       8.000   4.000   4.000  1.00 85.00           C
ATOM     15  CA  THR A  15       0.000   8.000   4.000  1.00 85.00           C
ATOM     16  CA  VAL A  16       4.000   8.000   4.000  1.00 85.00           C
ATOM     17  CA  TRP A  17       8.000   8.000   4.000  1.00 85.00           C
ATOM     18  CA  TYR A  18       0.000   0.000   8.000  1.00 85.00           C
ATOM     19  CA  PHE A  19       4.000   0.000   8.000  1.00 85.00           C
ATOM     20  CA  GLY A  20       8.000   0.000   8.000  1.00 85.00           C
END
"""


@pytest.fixture
def dense_pdb(tmp_path: Path) -> Path:
    p = tmp_path / "P00000.pdb"
    p.write_text(DENSE_PDB)
    return p


@pytest.fixture
def dense_structure(dense_pdb):
    from utils.pdb_parser import parse_pdb
    return parse_pdb(dense_pdb, "P00000")


@pytest.fixture
def sasa_map(dense_structure):
    return {
        (r.chain_id, r.residue_number): 40.0
        for r in dense_structure.residues
    }


@pytest.fixture
def active_set():
    # Residues 1-3 are the "active site" in tests
    return {1, 2, 3}


# ── Unit tests: Kirchhoff matrix ──────────────────────────────────────────────

class TestKirchhoff:

    def test_kirchhoff_symmetric(self, dense_structure):
        from pipeline.allosteric import _build_kirchhoff
        coords = np.array([r.coords for r in dense_structure.residues])
        K, _ = _build_kirchhoff(coords, cutoff=8.0)
        assert np.allclose(K, K.T)

    def test_kirchhoff_diagonal_positive(self, dense_structure):
        from pipeline.allosteric import _build_kirchhoff
        coords = np.array([r.coords for r in dense_structure.residues])
        K, _ = _build_kirchhoff(coords, cutoff=8.0)
        assert np.all(np.diag(K) >= 0)

    def test_kirchhoff_offdiag_nonpositive(self, dense_structure):
        from pipeline.allosteric import _build_kirchhoff
        coords = np.array([r.coords for r in dense_structure.residues])
        K, _ = _build_kirchhoff(coords, cutoff=8.0)
        off_diag = K - np.diag(np.diag(K))
        assert np.all(off_diag <= 0)

    def test_n_contacts_positive(self, dense_structure):
        from pipeline.allosteric import _build_kirchhoff
        coords = np.array([r.coords for r in dense_structure.residues])
        _, n_contacts = _build_kirchhoff(coords, cutoff=8.0)
        assert n_contacts > 0

    def test_larger_cutoff_more_contacts(self, dense_structure):
        from pipeline.allosteric import _build_kirchhoff
        coords = np.array([r.coords for r in dense_structure.residues])
        _, n1 = _build_kirchhoff(coords, cutoff=5.0)
        _, n2 = _build_kirchhoff(coords, cutoff=10.0)
        assert n2 >= n1


# ── Unit tests: GNM correlations ──────────────────────────────────────────────

class TestGNMCorrelations:

    def test_correlation_matrix_shape(self, dense_structure):
        from pipeline.allosteric import _build_kirchhoff, _compute_gnm_correlations
        coords = np.array([r.coords for r in dense_structure.residues])
        K, _ = _build_kirchhoff(coords, cutoff=8.0)
        C = _compute_gnm_correlations(K)
        n = len(dense_structure.residues)
        assert C.shape == (n, n)

    def test_correlation_diagonal_is_one(self, dense_structure):
        from pipeline.allosteric import _build_kirchhoff, _compute_gnm_correlations
        coords = np.array([r.coords for r in dense_structure.residues])
        K, _ = _build_kirchhoff(coords, cutoff=8.0)
        C = _compute_gnm_correlations(K)
        assert np.allclose(np.diag(C), 1.0, atol=0.1)

    def test_correlation_bounded(self, dense_structure):
        from pipeline.allosteric import _build_kirchhoff, _compute_gnm_correlations
        coords = np.array([r.coords for r in dense_structure.residues])
        K, _ = _build_kirchhoff(coords, cutoff=8.0)
        C = _compute_gnm_correlations(K)
        assert np.all(C >= -1.01) and np.all(C <= 1.01)

    def test_correlation_symmetric(self, dense_structure):
        from pipeline.allosteric import _build_kirchhoff, _compute_gnm_correlations
        coords = np.array([r.coords for r in dense_structure.residues])
        K, _ = _build_kirchhoff(coords, cutoff=8.0)
        C = _compute_gnm_correlations(K)
        assert np.allclose(C, C.T, atol=1e-6)


# ── Unit tests: coupling scores ────────────────────────────────────────────────

class TestCouplingScores:

    def test_scores_bounded_0_to_1(self, dense_structure):
        from pipeline.allosteric import (
            _build_kirchhoff, _compute_gnm_correlations, _compute_coupling_scores
        )
        coords = np.array([r.coords for r in dense_structure.residues])
        K, _  = _build_kirchhoff(coords, 8.0)
        C     = _compute_gnm_correlations(K)
        scores = _compute_coupling_scores(C, [0, 1, 2], len(coords))
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0 + 1e-6)

    def test_active_residues_score_high(self, dense_structure):
        from pipeline.allosteric import (
            _build_kirchhoff, _compute_gnm_correlations, _compute_coupling_scores
        )
        coords = np.array([r.coords for r in dense_structure.residues])
        K, _   = _build_kirchhoff(coords, 8.0)
        C      = _compute_gnm_correlations(K)
        active = [0, 1, 2]
        scores = _compute_coupling_scores(C, active, len(coords))
        # Active residues should score at or near maximum
        assert scores[active[0]] >= scores.mean()

    def test_empty_active_returns_zeros(self, dense_structure):
        from pipeline.allosteric import _compute_coupling_scores
        n      = len(dense_structure.residues)
        dummy  = np.ones((n, n))
        scores = _compute_coupling_scores(dummy, [], n)
        assert np.all(scores == 0.0)


# ── Unit tests: full pipeline ─────────────────────────────────────────────────

class TestAllostericPrediction:

    def test_returns_result_object(self, dense_structure, active_set, sasa_map):
        from pipeline.allosteric import predict_allosteric_sites, AllostericResult
        result = predict_allosteric_sites(dense_structure, active_set, sasa_map)
        assert isinstance(result, AllostericResult)

    def test_uniprot_id_preserved(self, dense_structure, active_set, sasa_map):
        from pipeline.allosteric import predict_allosteric_sites
        result = predict_allosteric_sites(dense_structure, active_set, sasa_map)
        assert result.uniprot_id == "P00000"

    def test_enm_computed(self, dense_structure, active_set, sasa_map):
        from pipeline.allosteric import predict_allosteric_sites
        result = predict_allosteric_sites(dense_structure, active_set, sasa_map)
        assert result.enm_computed is True

    def test_n_contacts_positive(self, dense_structure, active_set, sasa_map):
        from pipeline.allosteric import predict_allosteric_sites
        result = predict_allosteric_sites(dense_structure, active_set, sasa_map)
        assert result.n_contacts > 0

    def test_sites_are_list(self, dense_structure, active_set, sasa_map):
        from pipeline.allosteric import predict_allosteric_sites
        result = predict_allosteric_sites(dense_structure, active_set, sasa_map)
        assert isinstance(result.allosteric_sites, list)

    def test_site_confidence_valid(self, dense_structure, active_set, sasa_map):
        from pipeline.allosteric import predict_allosteric_sites
        result = predict_allosteric_sites(dense_structure, active_set, sasa_map)
        valid = {"HIGH", "MEDIUM", "LOW"}
        for site in result.allosteric_sites:
            assert site.confidence in valid

    def test_site_ids_sequential(self, dense_structure, active_set, sasa_map):
        from pipeline.allosteric import predict_allosteric_sites
        result = predict_allosteric_sites(dense_structure, active_set, sasa_map)
        for i, site in enumerate(result.allosteric_sites):
            assert site.site_id == f"A{i+1}"

    def test_allosteric_residues_far_from_active(
        self, dense_structure, active_set, sasa_map
    ):
        from pipeline.allosteric import predict_allosteric_sites, MIN_DISTANCE_FROM_ACTIVE
        result = predict_allosteric_sites(dense_structure, active_set, sasa_map)
        for res in result.allosteric_residues:
            assert res.min_dist_active >= MIN_DISTANCE_FROM_ACTIVE

    def test_allosteric_residues_not_in_active_set(
        self, dense_structure, active_set, sasa_map
    ):
        from pipeline.allosteric import predict_allosteric_sites
        result = predict_allosteric_sites(dense_structure, active_set, sasa_map)
        for res in result.allosteric_residues:
            assert res.residue_number not in active_set

    def test_correlation_scores_bounded(self, dense_structure, active_set, sasa_map):
        from pipeline.allosteric import predict_allosteric_sites
        result = predict_allosteric_sites(dense_structure, active_set, sasa_map)
        for res in result.allosteric_residues:
            assert 0.0 <= res.correlation_score <= 1.0

    def test_to_dict_serialisable(self, dense_structure, active_set, sasa_map):
        from pipeline.allosteric import predict_allosteric_sites
        result = predict_allosteric_sites(dense_structure, active_set, sasa_map)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_to_json_writes_file(self, dense_structure, active_set, sasa_map, tmp_path):
        from pipeline.allosteric import predict_allosteric_sites
        result = predict_allosteric_sites(dense_structure, active_set, sasa_map)
        out = tmp_path / "allosteric.json"
        result.to_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["uniprot_id"] == "P00000"

    def test_n_sites_matches_list(self, dense_structure, active_set, sasa_map):
        from pipeline.allosteric import predict_allosteric_sites
        result = predict_allosteric_sites(dense_structure, active_set, sasa_map)
        assert result.n_sites == len(result.allosteric_sites)

    def test_summary_string(self, dense_structure, active_set, sasa_map):
        from pipeline.allosteric import predict_allosteric_sites
        result = predict_allosteric_sites(dense_structure, active_set, sasa_map)
        s = result.summary()
        assert "P00000" in s
        assert "Allosteric" in s

    def test_no_active_set_still_runs(self, dense_structure, sasa_map):
        from pipeline.allosteric import predict_allosteric_sites, AllostericResult
        result = predict_allosteric_sites(dense_structure, set(), sasa_map)
        assert isinstance(result, AllostericResult)