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


# ── PDB fixtures for new motif tests ──────────────────────────────────────────

# DFG loop: ASP-1, PHE-2, GLY-3 placed close together in 3D and consecutively
DFG_PDB = """\
ATOM      1  N   ASP A   1       1.000   1.000   1.000  1.00 90.00           N
ATOM      2  CA  ASP A   1       1.500   1.500   1.500  1.00 90.00           C
ATOM      3  N   PHE A   2       2.500   1.500   1.500  1.00 90.00           N
ATOM      4  CA  PHE A   2       3.000   2.000   2.000  1.00 90.00           C
ATOM      5  N   GLY A   3       4.000   2.000   2.000  1.00 90.00           N
ATOM      6  CA  GLY A   3       4.500   2.500   2.500  1.00 90.00           C
ATOM      7  N   ALA A   4       6.000   3.000   3.000  1.00 85.00           N
ATOM      8  CA  ALA A   4       6.500   3.500   3.500  1.00 85.00           C
END
"""

# P-loop / Walker A: GLY-1, ALA-2, GLY-3, ALA-4, ALA-5, GLY-6, LYS-7
# Two glycines separated by 4-6 residues → GxxxxGK pattern
PLOOP_PDB = """\
ATOM      1  N   GLY A   1       1.000   1.000   1.000  1.00 88.00           N
ATOM      2  CA  GLY A   1       1.500   1.500   1.500  1.00 88.00           C
ATOM      3  N   ALA A   2       2.000   2.000   2.000  1.00 85.00           N
ATOM      4  CA  ALA A   2       2.500   2.500   2.500  1.00 85.00           C
ATOM      5  N   ALA A   3       3.000   3.000   3.000  1.00 85.00           N
ATOM      6  CA  ALA A   3       3.500   3.500   3.500  1.00 85.00           C
ATOM      7  N   ALA A   4       4.000   4.000   4.000  1.00 85.00           N
ATOM      8  CA  ALA A   4       4.500   4.500   4.500  1.00 85.00           C
ATOM      9  N   ALA A   5       5.000   5.000   5.000  1.00 85.00           N
ATOM     10  CA  ALA A   5       5.500   5.500   5.500  1.00 85.00           C
ATOM     11  N   GLY A   6       6.000   5.500   5.500  1.00 88.00           N
ATOM     12  CA  GLY A   6       6.500   6.000   6.000  1.00 88.00           C
ATOM     13  N   LYS A   7       7.000   6.500   6.500  1.00 88.00           N
ATOM     14  CA  LYS A   7       7.500   7.000   7.000  1.00 88.00           C
END
"""

# Zinc finger: CYS-CYS-HIS-CYS (C3H1 / RING-like structural zinc)
# All four residues within ~7Å of each other
ZINC_FINGER_PDB = """\
ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00 85.00           N
ATOM      2  CA  ALA A   1       1.500   1.500   1.500  1.00 85.00           C
ATOM      3  N   CYS A   2       5.000   5.000   5.000  1.00 88.00           N
ATOM      4  CA  CYS A   2       5.500   5.500   5.500  1.00 88.00           C
ATOM      5  N   CYS A   3       7.000   5.000   5.000  1.00 88.00           N
ATOM      6  CA  CYS A   3       7.500   5.500   5.500  1.00 88.00           C
ATOM      7  N   HIS A   4       6.000   7.000   5.000  1.00 88.00           N
ATOM      8  CA  HIS A   4       6.500   7.500   5.500  1.00 88.00           C
ATOM      9  N   CYS A   5       5.000   6.000   7.000  1.00 88.00           N
ATOM     10  CA  CYS A   5       5.500   6.500   7.500  1.00 88.00           C
ATOM     11  N   ALA A   6      10.000  10.000  10.000  1.00 85.00           N
ATOM     12  CA  ALA A   6      10.500  10.500  10.500  1.00 85.00           C
END
"""


# ── Tests for new motifs ───────────────────────────────────────────────────────

class TestDFGLoopDetection:

    @pytest.fixture
    def dfg_structure(self, tmp_path):
        from utils.pdb_parser import parse_pdb
        p = tmp_path / "PDFG.pdb"
        p.write_text(DFG_PDB)
        return parse_pdb(p, "PDFG", plddt_threshold=70.0)

    def test_dfg_loop_detected(self, dfg_structure):
        from pipeline.active_sites import _detect_motifs
        coord_map = {
            r.residue_number: (r.one_letter, r.coords, r.chain_id)
            for r in dfg_structure.residues
        }
        motifs = _detect_motifs(coord_map)
        types  = [m.motif_type for m in motifs]
        assert "dfg_loop" in types, (
            "DFG loop should be detected for sequential D-F-G within 2 residues"
        )

    def test_dfg_residues_are_d_f_g(self, dfg_structure):
        from pipeline.active_sites import _detect_motifs
        coord_map = {
            r.residue_number: (r.one_letter, r.coords, r.chain_id)
            for r in dfg_structure.residues
        }
        motifs  = _detect_motifs(coord_map)
        dfg     = [m for m in motifs if m.motif_type == "dfg_loop"]
        assert dfg, "Expected at least one dfg_loop motif"
        letters = dfg[0].residue_letters
        # The DFG motif letters should include D, F, G
        assert "D" in letters
        assert "F" in letters
        assert "G" in letters


class TestPLoopDetection:

    @pytest.fixture
    def ploop_structure(self, tmp_path):
        from utils.pdb_parser import parse_pdb
        p = tmp_path / "PPLOOP.pdb"
        p.write_text(PLOOP_PDB)
        return parse_pdb(p, "PPLOOP", plddt_threshold=70.0)

    def test_ploop_detected(self, ploop_structure):
        from pipeline.active_sites import _detect_motifs
        coord_map = {
            r.residue_number: (r.one_letter, r.coords, r.chain_id)
            for r in ploop_structure.residues
        }
        motifs = _detect_motifs(coord_map)
        types  = [m.motif_type for m in motifs]
        assert "p_loop_walker_a" in types, (
            "P-loop should be detected for GxxxxGK pattern"
        )

    def test_ploop_confidence_is_medium(self, ploop_structure):
        from pipeline.active_sites import _detect_motifs
        coord_map = {
            r.residue_number: (r.one_letter, r.coords, r.chain_id)
            for r in ploop_structure.residues
        }
        motifs = _detect_motifs(coord_map)
        ploop  = [m for m in motifs if m.motif_type == "p_loop_walker_a"]
        assert ploop, "Expected p_loop_walker_a motif"
        assert ploop[0].confidence == "MEDIUM"


class TestZincTypeClassification:

    def test_classify_zinc_structural_c4(self):
        from pipeline.active_sites import _classify_zinc_type
        # Cys4 pattern (classic zinc finger) → structural
        assert _classify_zinc_type(["C", "C", "C", "C"]) == "structural"

    def test_classify_zinc_structural_c3h1(self):
        from pipeline.active_sites import _classify_zinc_type
        # Cys3His1 (RING domain) → structural
        assert _classify_zinc_type(["C", "C", "C", "H"]) == "structural"

    def test_classify_zinc_catalytic_h2e1(self):
        from pipeline.active_sites import _classify_zinc_type
        # His2Glu (metallopeptidase active site) → catalytic
        assert _classify_zinc_type(["H", "H", "E"]) == "catalytic"

    def test_zinc_finger_not_metallopeptidase_in_ec(self, tmp_path):
        """Structural zinc (C3H1) must NOT contribute EC 3.4.24 prediction."""
        from pipeline.clean_ec import predict_ec_number, ENZYMATIC_MOTIFS
        # Synthesise an active_result with a C3H1 zinc cluster (structural)
        active_result = {
            "catalytic_motifs": [
                {
                    "motif_type":      "zinc_binding_cluster",
                    "residue_numbers": [2, 3, 4, 5],
                    "residue_letters": ["C", "C", "H", "C"],
                    "mean_distance":   5.0,
                    "confidence":      "HIGH",
                    "zinc_type":       "structural",
                }
            ],
            "n_high_confidence": 4,
        }
        result = predict_ec_number(
            uniprot_id="TEST",
            sequence="ACCHCA",
            active_result=active_result,
            go_result=None,
            homology_result=None,
        )
        # Should NOT predict metallopeptidase (EC 3.4.24) from structural zinc
        assert result.specific_ec != "3.4.24", (
            "Structural zinc (C3H1) should not produce EC 3.4.24 prediction"
        )

    def test_ring_domain_ubiquitin_ligase_from_go(self):
        """RING domain protein with GO:0061630 should get EC 2.3.2 (transferase)."""
        from pipeline.clean_ec import predict_ec_number
        active_result = {
            "catalytic_motifs": [
                {
                    "motif_type":      "zinc_binding_cluster",
                    "residue_numbers": [305, 308, 311, 319],
                    "residue_letters": ["C", "C", "C", "H"],
                    "mean_distance":   5.5,
                    "confidence":      "HIGH",
                    "zinc_type":       "structural",
                }
            ],
            "n_high_confidence": 4,
        }
        go_result = {
            "mf_predictions": [
                {
                    "go_id":    "GO:0061630",
                    "go_name":  "ubiquitin protein ligase activity",
                    "score":    0.88,
                    "evidence": ["domain_annotation"],
                }
            ],
            "bp_predictions": [],
        }
        result = predict_ec_number(
            uniprot_id="TEST",
            sequence="ACCHCAAACCHC",
            active_result=active_result,
            go_result=go_result,
            homology_result=None,
        )
        assert result.is_enzyme, "Ubiquitin ligase should be classified as enzyme"
        # EC class should be 2 (transferase) not 3 (hydrolase/metallopeptidase)
        if result.top_prediction:
            assert result.top_prediction.ec_class == "2", (
                f"Ubiquitin ligase should be EC class 2, got {result.top_prediction.ec_class}"
            )
        # Specific EC should point to ubiquitin ligase sub-class
        assert result.specific_ec.startswith("2.3.2"), (
            f"Expected specific EC 2.3.2.x, got {result.specific_ec}"
        )