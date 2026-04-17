"""
tests/test_04_binding_pockets.py
─────────────────────────────────
Tests for Module 04 — Binding pocket detection (alpha-sphere method).

Run with:
    python -m pytest tests/test_04_binding_pockets.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest


# ── Fixtures ───────────────────────────────────────────────────────────────────

MINIMAL_PDB = """\
ATOM      1  N   MET A   1       1.000   2.000   3.000  1.00 92.15           N
ATOM      2  CA  MET A   1       1.500   2.500   3.500  1.00 92.15           C
ATOM      3  N   GLU A   2       3.000   4.000   5.000  1.00 87.30           N
ATOM      4  CA  GLU A   2       3.500   4.500   5.500  1.00 87.30           C
ATOM      5  N   SER A   3       5.000   6.000   7.000  1.00 45.20           N
ATOM      6  CA  SER A   3       5.500   6.500   7.500  1.00 45.20           C
ATOM      7  N   ALA A   4       7.000   8.000   9.000  1.00 72.80           N
ATOM      8  CA  ALA A   4       7.500   8.500   9.500  1.00 72.80           C
END
"""

# A denser PDB with residues clustered together to guarantee pocket detection
DENSE_PDB = """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 85.00           C
ATOM      2  CA  ALA A   2       4.000   0.000   0.000  1.00 85.00           C
ATOM      3  CA  ALA A   3       8.000   0.000   0.000  1.00 85.00           C
ATOM      4  CA  ALA A   4       0.000   4.000   0.000  1.00 85.00           C
ATOM      5  CA  ALA A   5       4.000   4.000   0.000  1.00 85.00           C
ATOM      6  CA  ALA A   6       8.000   4.000   0.000  1.00 85.00           C
ATOM      7  CA  ALA A   7       0.000   8.000   0.000  1.00 85.00           C
ATOM      8  CA  ALA A   8       4.000   8.000   0.000  1.00 85.00           C
ATOM      9  CA  ALA A   9       8.000   8.000   0.000  1.00 85.00           C
ATOM     10  CA  ALA A  10       0.000   0.000   4.000  1.00 85.00           C
ATOM     11  CA  ALA A  11       4.000   0.000   4.000  1.00 85.00           C
ATOM     12  CA  ALA A  12       8.000   0.000   4.000  1.00 85.00           C
ATOM     13  CA  ARG A  13       0.000   4.000   4.000  1.00 85.00           C
ATOM     14  CA  ALA A  14       8.000   4.000   4.000  1.00 85.00           C
ATOM     15  CA  ALA A  15       0.000   8.000   4.000  1.00 85.00           C
ATOM     16  CA  ALA A  16       4.000   8.000   4.000  1.00 85.00           C
ATOM     17  CA  ALA A  17       8.000   8.000   4.000  1.00 85.00           C
ATOM     18  CA  ALA A  18       0.000   0.000   8.000  1.00 85.00           C
ATOM     19  CA  ALA A  19       4.000   0.000   8.000  1.00 85.00           C
ATOM     20  CA  ALA A  20       8.000   0.000   8.000  1.00 85.00           C
END
"""


@pytest.fixture
def minimal_pdb(tmp_path: Path) -> Path:
    p = tmp_path / "P00000.pdb"
    p.write_text(MINIMAL_PDB)
    return p


@pytest.fixture
def dense_pdb(tmp_path: Path) -> Path:
    p = tmp_path / "P00001.pdb"
    p.write_text(DENSE_PDB)
    return p


@pytest.fixture
def minimal_structure(minimal_pdb):
    from utils.pdb_parser import parse_pdb
    return parse_pdb(minimal_pdb, "P00000")


@pytest.fixture
def dense_structure(dense_pdb):
    from utils.pdb_parser import parse_pdb
    return parse_pdb(dense_pdb, "P00001")


# ── Unit tests: druggability scoring ──────────────────────────────────────────

class TestDruggabilityScore:

    def test_score_bounded_0_to_1(self):
        from pipeline.binding_pockets import _druggability_score
        score = _druggability_score(
            burial=0.5, enclosure=0.5, hydro=0.5,
            charge=0.0, n_lining=15, near_active=True, volume=500
        )
        assert 0.0 <= score <= 1.0

    def test_ideal_pocket_scores_high(self):
        from pipeline.binding_pockets import _druggability_score
        score = _druggability_score(
            burial=0.8, enclosure=0.6, hydro=0.5,
            charge=0.0, n_lining=15, near_active=True, volume=600
        )
        assert score >= 0.5

    def test_tiny_pocket_scores_low(self):
        from pipeline.binding_pockets import _druggability_score
        score = _druggability_score(
            burial=0.1, enclosure=0.05, hydro=-2.0,
            charge=5.0, n_lining=2, near_active=False, volume=10
        )
        assert score < 0.5

    def test_active_site_bonus_applied(self):
        from pipeline.binding_pockets import _druggability_score
        base  = _druggability_score(0.4, 0.4, 0.0, 0.0, 10, False, 400)
        bonus = _druggability_score(0.4, 0.4, 0.0, 0.0, 10, True,  400)
        assert bonus > base

    def test_score_never_exceeds_1(self):
        from pipeline.binding_pockets import _druggability_score
        score = _druggability_score(
            burial=10.0, enclosure=10.0, hydro=5.0,
            charge=0.0, n_lining=50, near_active=True, volume=5000
        )
        assert score <= 1.0


# ── Unit tests: candidate clustering ──────────────────────────────────────────

class TestCandidateClustering:

    def test_empty_input_returns_empty(self):
        from pipeline.binding_pockets import _cluster_candidates
        result = _cluster_candidates([])
        assert result == []

    def test_single_candidate(self):
        from pipeline.binding_pockets import _cluster_candidates
        candidates = [([0.0, 0.0, 0.0], 0.5, [0, 1, 2])]
        result = _cluster_candidates(candidates)
        assert len(result) == 1

    def test_nearby_candidates_merged(self):
        from pipeline.binding_pockets import _cluster_candidates, CLUSTER_MERGE_DIST
        # Two candidates very close together should merge into one
        candidates = [
            (np.array([0.0, 0.0, 0.0]), 0.8, [0, 1]),
            (np.array([1.0, 0.0, 0.0]), 0.7, [1, 2]),
        ]
        result = _cluster_candidates(candidates)
        assert len(result) == 1

    def test_distant_candidates_separate(self):
        from pipeline.binding_pockets import _cluster_candidates
        # Two candidates far apart should stay separate
        candidates = [
            (np.array([0.0,  0.0, 0.0]), 0.8, [0, 1]),
            (np.array([50.0, 0.0, 0.0]), 0.7, [5, 6]),
        ]
        result = _cluster_candidates(candidates)
        assert len(result) == 2


# ── Unit tests: full pipeline ─────────────────────────────────────────────────

class TestBindingPocketDetection:

    def test_returns_result_object(self, minimal_structure):
        from pipeline.binding_pockets import detect_binding_pockets, BindingPocketResult
        result = detect_binding_pockets(minimal_structure)
        assert isinstance(result, BindingPocketResult)

    def test_uniprot_id_preserved(self, minimal_structure):
        from pipeline.binding_pockets import detect_binding_pockets
        result = detect_binding_pockets(minimal_structure)
        assert result.uniprot_id == "P00000"

    def test_pockets_are_list(self, minimal_structure):
        from pipeline.binding_pockets import detect_binding_pockets
        result = detect_binding_pockets(minimal_structure)
        assert isinstance(result.pockets, list)

    def test_pocket_volumes_positive(self, dense_structure):
        from pipeline.binding_pockets import detect_binding_pockets
        result = detect_binding_pockets(dense_structure)
        for p in result.pockets:
            assert p.volume_A3 > 0

    def test_druggability_scores_bounded(self, dense_structure):
        from pipeline.binding_pockets import detect_binding_pockets
        result = detect_binding_pockets(dense_structure)
        for p in result.pockets:
            assert 0.0 <= p.druggability_score <= 1.0

    def test_druggability_class_valid(self, dense_structure):
        from pipeline.binding_pockets import detect_binding_pockets
        result = detect_binding_pockets(dense_structure)
        valid = {"high", "medium", "low"}
        for p in result.pockets:
            assert p.druggability_class in valid

    def test_pocket_ids_sequential(self, dense_structure):
        from pipeline.binding_pockets import detect_binding_pockets
        result = detect_binding_pockets(dense_structure)
        for i, p in enumerate(result.pockets):
            assert p.pocket_id == f"P{i+1}"

    def test_n_pockets_matches_list(self, dense_structure):
        from pipeline.binding_pockets import detect_binding_pockets
        result = detect_binding_pockets(dense_structure)
        assert result.n_pockets == len(result.pockets)

    def test_to_dict_serialisable(self, minimal_structure):
        from pipeline.binding_pockets import detect_binding_pockets
        result = detect_binding_pockets(minimal_structure)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_to_json_writes_file(self, minimal_structure, tmp_path):
        from pipeline.binding_pockets import detect_binding_pockets
        result = detect_binding_pockets(minimal_structure)
        out = tmp_path / "pockets.json"
        result.to_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["uniprot_id"] == "P00000"

    def test_active_site_flagging(self, dense_structure):
        from pipeline.binding_pockets import detect_binding_pockets
        result = detect_binding_pockets(
            dense_structure,
            active_site_residues={1, 2, 3, 5}
        )
        for p in result.pockets:
            if any(r in p.lining_residues for r in [1, 2, 3, 5]):
                assert p.near_active_site is True

    def test_summary_string(self, minimal_structure):
        from pipeline.binding_pockets import detect_binding_pockets
        result = detect_binding_pockets(minimal_structure)
        s = result.summary()
        assert "P00000" in s
        assert "Binding pocket" in s

    def test_dense_structure_finds_pockets(self, dense_structure):
        from pipeline.binding_pockets import detect_binding_pockets
        result = detect_binding_pockets(dense_structure)
        assert result.n_pockets > 0