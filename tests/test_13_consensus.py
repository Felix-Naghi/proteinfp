"""
tests/test_13_consensus.py
───────────────────────────
Tests for Module 13 — Consensus scoring + final report.

All tests are offline using fixture data.

Run with:
    python -m pytest tests/test_13_consensus.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def full_modules_data():
    """Complete set of mock module outputs for consensus testing."""
    return {
        "structure": {
            "uniprot_id": "P00000",
            "gene_name": "TP53",
            "protein_name": "Cellular tumor antigen p53",
            "organism": "Homo sapiens",
            "length": 393,
            "mean_plddt": 75.0,
        },
        "homology": {
            "n_experimental_hits": 5,
            "all_go_terms": ["GO:0003677", "GO:0006915", "GO:0005634"],
            "all_go_names": ["DNA binding", "apoptotic process", "nucleus"],
            "interpro_domains": [
                {
                    "go_terms": ["GO:0003677"],
                    "go_names": ["DNA binding"],
                }
            ],
            "blast_hits": [
                {
                    "reviewed": True,
                    "function_text": "Acts as a tumor suppressor.",
                    "go_terms": ["GO:0003677"],
                    "go_names": ["DNA binding"],
                    "evidence_weight": 3.0,
                }
            ],
            "top_function": "Acts as a tumor suppressor.",
            "protein_families": ["P53 DNA-binding domain"],
        },
        "go": {
            "mf_predictions": [
                {"go_id": "GO:0003677", "go_name": "DNA binding",
                 "namespace": "MF", "score": 0.85, "evidence": ["experimental_homolog"]},
                {"go_id": "GO:0008270", "go_name": "zinc ion binding",
                 "namespace": "MF", "score": 0.80, "evidence": ["structural_motif"]},
            ],
            "bp_predictions": [
                {"go_id": "GO:0006915", "go_name": "apoptotic process",
                 "namespace": "BP", "score": 0.70, "evidence": ["experimental_homolog"]},
            ],
            "cc_predictions": [
                {"go_id": "GO:0005634", "go_name": "nucleus",
                 "namespace": "CC", "score": 0.80, "evidence": ["structural_motif"]},
            ],
        },
        "active": {
            "active_residues": [
                {"residue_number": 176, "one_letter": "C", "confidence": "HIGH",
                 "motifs": ["zinc_binding_cluster"]},
                {"residue_number": 248, "one_letter": "R", "confidence": "HIGH",
                 "motifs": ["dna_binding_cluster"]},
                {"residue_number": 273, "one_letter": "R", "confidence": "MEDIUM",
                 "motifs": []},
            ],
            "catalytic_motifs": [
                {"motif_type": "zinc_binding_cluster", "confidence": "HIGH",
                 "residue_numbers": [176, 178, 179, 182]},
            ],
        },
        "pockets": {
            "pockets": [
                {"pocket_id": "P1", "volume_A3": 560, "druggability_score": 0.85,
                 "druggability_class": "high", "lining_residues": [176, 248, 273],
                 "near_active_site": True},
            ],
        },
        "allosteric": {
            "allosteric_sites": [
                {"site_id": "A1", "mean_correlation": 0.96,
                 "residue_numbers": [28, 29, 30], "confidence": "HIGH",
                 "min_dist_active": 25.0},
            ],
        },
        "ec": {
            "is_enzyme": False,
            "enzyme_confidence": 0.2,
            "specific_ec": "",
            "top_prediction": None,
        },
        "ppi": {
            "partners": [
                {"partner_name": "MDM2", "combined_score": 999,
                 "interaction_type": "inhibitory",
                 "interface_residues": [19, 23, 26], "confidence": "high"},
                {"partner_name": "ATM", "combined_score": 870,
                 "interaction_type": "activating",
                 "interface_residues": [15, 18], "confidence": "medium-high"},
            ],
        },
        "foldseek": {
            "same_fold_hits": [
                {"tmscore": 0.99, "seq_identity": 0.99,
                 "description": "Cellular tumor antigen p53",
                 "function_inferred": "tumor suppressor DNA binding"},
            ],
        },
    }


@pytest.fixture
def mock_inter_dir(tmp_path, full_modules_data):
    """Write all module JSONs to a temp intermediate directory."""
    inter = tmp_path / "intermediate"
    inter.mkdir()

    file_map = {
        "P00000_structure.json":       "structure",
        "P00000_homology.json":        "homology",
        "P00000_go_predictions.json":  "go",
        "P00000_active_sites.json":    "active",
        "P00000_binding_pockets.json": "pockets",
        "P00000_allosteric.json":      "allosteric",
        "P00000_ec_prediction.json":   "ec",
        "P00000_ppi.json":             "ppi",
        "P00000_foldseek.json":        "foldseek",
    }

    for filename, key in file_map.items():
        p = inter / filename
        p.write_text(json.dumps(full_modules_data[key]))

    return inter


# ── Unit tests: GO aggregation ────────────────────────────────────────────────

class TestGoAggregation:

    def test_aggregates_from_homology(self, full_modules_data):
        from pipeline.consensus import _aggregate_go_terms
        mf, bp, cc = _aggregate_go_terms(full_modules_data)
        all_ids = {t.go_id for t in mf + bp + cc}
        assert "GO:0003677" in all_ids

    def test_aggregates_from_go_module(self, full_modules_data):
        from pipeline.consensus import _aggregate_go_terms
        mf, bp, cc = _aggregate_go_terms(full_modules_data)
        all_ids = {t.go_id for t in mf + bp + cc}
        assert "GO:0008270" in all_ids

    def test_go_terms_sorted_by_score(self, full_modules_data):
        from pipeline.consensus import _aggregate_go_terms
        mf, bp, cc = _aggregate_go_terms(full_modules_data)
        for lst in (mf, bp, cc):
            scores = [t.weighted_score for t in lst]
            assert scores == sorted(scores, reverse=True)

    def test_high_confidence_terms_exist(self, full_modules_data):
        from pipeline.consensus import _aggregate_go_terms
        mf, bp, cc = _aggregate_go_terms(full_modules_data)
        all_terms = mf + bp + cc
        confs = {t.confidence for t in all_terms}
        assert "HIGH" in confs or "MEDIUM" in confs

    def test_no_duplicate_go_ids(self, full_modules_data):
        from pipeline.consensus import _aggregate_go_terms
        mf, bp, cc = _aggregate_go_terms(full_modules_data)
        all_ids = [t.go_id for t in mf + bp + cc]
        assert len(all_ids) == len(set(all_ids))

    def test_namespace_correct(self, full_modules_data):
        from pipeline.consensus import _aggregate_go_terms
        mf, bp, cc = _aggregate_go_terms(full_modules_data)
        for t in mf: assert t.namespace == "MF"
        for t in bp: assert t.namespace == "BP"
        for t in cc: assert t.namespace == "CC"


# ── Unit tests: helper functions ──────────────────────────────────────────────

class TestHelpers:

    def test_extract_top_function_from_swissprot(self, full_modules_data):
        from pipeline.consensus import _extract_top_function
        fn = _extract_top_function(full_modules_data)
        assert "tumor suppressor" in fn.lower()

    def test_extract_location_from_cc(self, full_modules_data):
        from pipeline.consensus import _extract_location
        loc = _extract_location(full_modules_data)
        assert loc != ""

    def test_overall_confidence_high(self, full_modules_data):
        from pipeline.consensus import _overall_confidence
        mf_terms = [type('T', (), {"confidence": "HIGH"})()]
        conf = _overall_confidence(
            n_sources=10, n_exp_hits=5,
            mean_plddt=80.0, go_mf=mf_terms
        )
        assert conf in ("VERY HIGH", "HIGH")

    def test_overall_confidence_low(self):
        from pipeline.consensus import _overall_confidence
        conf = _overall_confidence(
            n_sources=1, n_exp_hits=0,
            mean_plddt=50.0, go_mf=[]
        )
        assert conf in ("VERY LOW", "LOW")

    def test_infer_ns_binding(self):
        from pipeline.consensus import _infer_ns
        assert _infer_ns("DNA binding") == "MF"

    def test_infer_ns_process(self):
        from pipeline.consensus import _infer_ns
        assert _infer_ns("apoptotic process") == "BP"

    def test_infer_ns_nucleus(self):
        from pipeline.consensus import _infer_ns
        assert _infer_ns("nucleus") == "CC"


# ── Unit tests: validation suggestions ───────────────────────────────────────

class TestValidationSuggestions:

    def test_generates_suggestions(self, full_modules_data):
        from pipeline.consensus import _generate_validation_suggestions
        active  = full_modules_data["active"]["active_residues"]
        pockets = full_modules_data["pockets"]["pockets"]
        partners = full_modules_data["ppi"]["partners"]
        go_mf   = []
        sugs = _generate_validation_suggestions(
            active, pockets, partners, go_mf, False, ""
        )
        assert len(sugs) > 0

    def test_alanine_scan_suggested(self, full_modules_data):
        from pipeline.consensus import _generate_validation_suggestions
        active  = full_modules_data["active"]["active_residues"]
        pockets = full_modules_data["pockets"]["pockets"]
        partners = full_modules_data["ppi"]["partners"]
        sugs = _generate_validation_suggestions(
            active, pockets, partners, [], False, ""
        )
        assert any("alanine" in s.lower() or "mutagenesis" in s.lower()
                   for s in sugs)

    def test_ppi_coip_suggested(self, full_modules_data):
        from pipeline.consensus import _generate_validation_suggestions
        active  = full_modules_data["active"]["active_residues"]
        pockets = full_modules_data["pockets"]["pockets"]
        partners = full_modules_data["ppi"]["partners"]
        sugs = _generate_validation_suggestions(
            active, pockets, partners, [], False, ""
        )
        assert any("immunoprecipitation" in s.lower() or "co-ip" in s.lower()
                   for s in sugs)

    def test_max_8_suggestions(self, full_modules_data):
        from pipeline.consensus import _generate_validation_suggestions
        sugs = _generate_validation_suggestions(
            full_modules_data["active"]["active_residues"],
            full_modules_data["pockets"]["pockets"],
            full_modules_data["ppi"]["partners"],
            [], False, ""
        )
        assert len(sugs) <= 8


# ── Unit tests: ConsensusReport ───────────────────────────────────────────────

class TestConsensusReport:

    def test_report_construction(self, full_modules_data):
        from pipeline.consensus import (
            ConsensusReport, _aggregate_go_terms,
            _extract_top_function, _extract_location,
            _generate_validation_suggestions, _overall_confidence
        )
        mf, bp, cc = _aggregate_go_terms(full_modules_data)
        report = ConsensusReport(
            uniprot_id="P00000",
            gene_name="TP53",
            protein_name="Cellular tumor antigen p53",
            organism="Homo sapiens",
            sequence_length=393,
            generated_at="2025-01-01 00:00:00",
            top_function="Acts as a tumor suppressor.",
            is_enzyme=False,
            ec_number="",
            subcellular_location="nucleus",
            go_terms_mf=mf,
            go_terms_bp=bp,
            go_terms_cc=cc,
        )
        assert report.uniprot_id == "P00000"
        assert report.gene_name == "TP53"

    def test_to_dict_serialisable(self, full_modules_data):
        from pipeline.consensus import (
            ConsensusReport, _aggregate_go_terms
        )
        mf, bp, cc = _aggregate_go_terms(full_modules_data)
        report = ConsensusReport(
            uniprot_id="P00000", gene_name="TP53",
            protein_name="p53", organism="Homo sapiens",
            sequence_length=393, generated_at="2025-01-01",
            top_function="tumor suppressor", is_enzyme=False,
            ec_number="", subcellular_location="nucleus",
            go_terms_mf=mf, go_terms_bp=bp, go_terms_cc=cc,
        )
        json.dumps(report.to_dict())

    def test_text_report_contains_key_fields(self, full_modules_data):
        from pipeline.consensus import (
            ConsensusReport, _aggregate_go_terms
        )
        mf, bp, cc = _aggregate_go_terms(full_modules_data)
        report = ConsensusReport(
            uniprot_id="P00000", gene_name="TP53",
            protein_name="Cellular tumor antigen p53",
            organism="Homo sapiens", sequence_length=393,
            generated_at="2025-01-01", top_function="tumor suppressor",
            is_enzyme=False, ec_number="", subcellular_location="nucleus",
            go_terms_mf=mf, go_terms_bp=bp, go_terms_cc=cc,
        )
        text = report.to_text_report()
        assert "P00000" in text
        assert "TP53" in text
        assert "ProteinFP" in text

    def test_to_json_writes_file(self, full_modules_data, tmp_path):
        from pipeline.consensus import (
            ConsensusReport, _aggregate_go_terms
        )
        mf, bp, cc = _aggregate_go_terms(full_modules_data)
        report = ConsensusReport(
            uniprot_id="P00000", gene_name="TP53",
            protein_name="p53", organism="Homo sapiens",
            sequence_length=393, generated_at="2025-01-01",
            top_function="tumor suppressor", is_enzyme=False,
            ec_number="", subcellular_location="nucleus",
            go_terms_mf=mf, go_terms_bp=bp, go_terms_cc=cc,
        )
        out = tmp_path / "report.json"
        report.to_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["uniprot_id"] == "P00000"
        assert data["gene_name"] == "TP53"