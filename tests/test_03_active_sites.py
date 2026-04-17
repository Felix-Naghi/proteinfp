"""
tests/test_03_active_sites.py
──────────────────────────────
Tests for Module 03 — Active site prediction.

Run with:
    python -m pytest tests/test_03_active_sites.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

# ── Fixtures ───────────────────────────────────────────────────────────────────

# PDB with a serine protease-like triad geometry:
# SER-5, HIS-10, ASP-15 placed within 6A of each other
# Plus ARG/LYS cluster for DNA-binding motif detection
TRIAD_PDB = """\
ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00 85.00           N
ATOM      2  CA  ALA A   1       1.500   1.500   1.500  1.00 85.00           C
ATOM      3  N   ALA A   2       2.000   2.000   2.000  1.00 85.00           N
ATOM      4  CA  ALA A   2       2.500   2.500   2.500  1.00 85.00           C
ATOM      5  N   ALA A   3       3.000   3.000   3.000  1.00 85.00           N
ATOM      6  CA  ALA A   3       3.500   3.500   3.500  1.00 85.00           C
ATOM      7  N   ALA A   4       4.000   4.000   4.000  1.00 85.00           N
ATOM      8  CA  ALA A   4       4.500   4.500   4.500  1.00 85.00           C
ATOM      9  N   SER A   5       5.000   5.000   5.000  1.00 90.00           N
ATOM     10  CA  SER A   5       5.500   5.500   5.500  1.00 90.00           C
ATOM     11  N   ALA A   6       6.000   6.000   6.000  1.00 85.00           N
ATOM     12  CA  ALA A   6       6.500   6.500   6.500  1.00 85.00           C
ATOM     13  N   ALA A   7       5.000   5.000   7.000  1.00 85.00           N
ATOM     14  CA  ALA A   7       5.500   5.500   7.500  1.00 85.00           C
ATOM     15  N   ALA A   8       4.000   6.000   6.000  1.00 85.00           N
ATOM     16  CA  ALA A   8       4.500   6.500   6.500  1.00 85.00           C
ATOM     17  N   HIS A   9       6.000   4.000   6.000  1.00 88.00           N
ATOM     18  CA  HIS A   9       6.500   4.500   6.500  1.00 88.00           C
ATOM     19  N   ASP A  10       7.000   6.000   5.000  1.00 87.00           N
ATOM     20  CA  ASP A  10       7.500   6.500   5.500  1.00 87.00           C
ATOM     21  N   ARG A  11      10.000  10.000  10.000  1.00 88.00           N
ATOM     22  CA  ARG A  11      10.500  10.500  10.500  1.00 88.00           C
ATOM     23  N   ARG A  12      11.000  11.000  11.000  1.00 88.00           N
ATOM     24  CA  ARG A  12      11.500  11.500  11.500  1.00 88.00           C
ATOM     25  N   LYS A  13      12.000  12.000  12.000  1.00 88.00           N
ATOM     26  CA  LYS A  13      12.500  12.500  12.500  1.00 88.00           C
ATOM     27  N   CYS A  14       8.000   8.000   8.000  1.00 86.00           N
ATOM     28  CA  CYS A  14       8.500   8.500   8.500  1.00 86.00           C
ATOM     29  N   HIS A  15       9.000   8.000   9.000  1.00 86.00           N
ATOM     30  CA  HIS A  15       9.500   8.500   9.500  1.00 86.00           C
END
"""


@pytest.fixture
def pdb_file(tmp_path: Path) -> Path:
    p = tmp_path / "P00000.pdb"
    p.write_text(TRIAD_PDB)
    return p


@pytest.fixture
def parsed_structure(pdb_file):
    from utils.pdb_parser import parse_pdb
    return parse_pdb(pdb_file, "P00000", plddt_threshold=70.0)


@pytest.fixture
def sasa_map(parsed_structure):
    # Simulate: most residues exposed, a few buried
    result = {}
    for res in parsed_structure.residues:
        # Bury SER-5, HIS-9, ASP-10 to make them active site candidates
        if res.residue_number in {5, 9, 10}:
            result[(res.chain_id, res.residue_number)] = 5.0
        else:
            result[(res.chain_id, res.residue_number)] = 80.0
    return result


# ── Unit tests ─────────────────────────────────────────────────────────────────

class TestActiveSitePrediction:

    def test_returns_result_object(self, parsed_structure, sasa_map):
        from pipeline.active_sites import predict_active_sites, ActiveSiteResult
        result = predict_active_sites(parsed_structure, sasa_map)
        assert isinstance(result, ActiveSiteResult)

    def test_uniprot_id_preserved(self, parsed_structure, sasa_map):
        from pipeline.active_sites import predict_active_sites
        result = predict_active_sites(parsed_structure, sasa_map)
        assert result.uniprot_id == "P00000"

    def test_active_residues_are_list(self, parsed_structure, sasa_map):
        from pipeline.active_sites import predict_active_sites
        result = predict_active_sites(parsed_structure, sasa_map)
        assert isinstance(result.active_residues, list)

    def test_confidence_values_valid(self, parsed_structure, sasa_map):
        from pipeline.active_sites import predict_active_sites
        result = predict_active_sites(parsed_structure, sasa_map)
        valid = {"HIGH", "MEDIUM", "LOW"}
        for res in result.active_residues:
            assert res.confidence in valid

    def test_evidence_score_positive(self, parsed_structure, sasa_map):
        from pipeline.active_sites import predict_active_sites
        result = predict_active_sites(parsed_structure, sasa_map)
        for res in result.active_residues:
            assert res.evidence_score >= 1

    def test_counts_match_lists(self, parsed_structure, sasa_map):
        from pipeline.active_sites import predict_active_sites
        result = predict_active_sites(parsed_structure, sasa_map)
        assert result.n_high_confidence == sum(
            1 for r in result.active_residues if r.confidence == "HIGH"
        )
        assert result.n_medium_confidence == sum(
            1 for r in result.active_residues if r.confidence == "MEDIUM"
        )

    def test_sasa_values_carried(self, parsed_structure, sasa_map):
        from pipeline.active_sites import predict_active_sites
        result = predict_active_sites(parsed_structure, sasa_map)
        res_map = {r.residue_number: r for r in result.active_residues}
        # Buried residues should show low SASA
        for rn in [5, 9, 10]:
            if rn in res_map:
                assert res_map[rn].sasa < 20.0

    def test_to_dict_serialisable(self, parsed_structure, sasa_map):
        from pipeline.active_sites import predict_active_sites
        result = predict_active_sites(parsed_structure, sasa_map)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_to_json_writes_file(self, parsed_structure, sasa_map, tmp_path):
        from pipeline.active_sites import predict_active_sites
        result = predict_active_sites(parsed_structure, sasa_map)
        out = tmp_path / "active_sites.json"
        result.to_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["uniprot_id"] == "P00000"

    def test_summary_string(self, parsed_structure, sasa_map):
        from pipeline.active_sites import predict_active_sites
        result = predict_active_sites(parsed_structure, sasa_map)
        s = result.summary()
        assert "P00000" in s
        assert "Active site" in s


class TestMotifDetection:

    def test_detects_motifs(self, parsed_structure):
        from pipeline.active_sites import _detect_motifs
        coord_map = {
            r.residue_number: (r.one_letter, r.coords, r.chain_id)
            for r in parsed_structure.residues
        }
        motifs = _detect_motifs(coord_map)
        assert isinstance(motifs, list)

    def test_motif_residue_numbers_are_ints(self, parsed_structure):
        from pipeline.active_sites import _detect_motifs
        coord_map = {
            r.residue_number: (r.one_letter, r.coords, r.chain_id)
            for r in parsed_structure.residues
        }
        for motif in _detect_motifs(coord_map):
            for rn in motif.residue_numbers:
                assert isinstance(rn, int)

    def test_motif_confidence_valid(self, parsed_structure):
        from pipeline.active_sites import _detect_motifs
        coord_map = {
            r.residue_number: (r.one_letter, r.coords, r.chain_id)
            for r in parsed_structure.residues
        }
        valid = {"HIGH", "MEDIUM", "LOW"}
        for motif in _detect_motifs(coord_map):
            assert motif.confidence in valid

    def test_dna_binding_cluster_detected(self, parsed_structure):
        from pipeline.active_sites import _detect_motifs
        coord_map = {
            r.residue_number: (r.one_letter, r.coords, r.chain_id)
            for r in parsed_structure.residues
        }
        motifs = _detect_motifs(coord_map)
        types = [m.motif_type for m in motifs]
        # ARG-11, ARG-12, LYS-13 are close together — should detect DNA cluster
        assert "dna_binding_cluster" in types


class TestConservationProxy:

    def test_proxy_covers_all_residues(self, parsed_structure):
        from pipeline.active_sites import _conservation_proxy
        proxy = _conservation_proxy(parsed_structure.sequence)
        assert len(proxy) == parsed_structure.length

    def test_proxy_scores_in_range(self, parsed_structure):
        from pipeline.active_sites import _conservation_proxy
        proxy = _conservation_proxy(parsed_structure.sequence)
        for score in proxy.values():
            assert 1.0 <= score <= 9.0

    def test_cys_his_get_high_scores(self):
        from pipeline.active_sites import _conservation_proxy
        proxy = _conservation_proxy("CHACD")
        # C=1, H=2 should have higher scores than A=3
        assert proxy[1] > proxy[3]   # C > A
        assert proxy[2] > proxy[3]   # H > A