"""
tests/test_08_09_10_ai_prediction.py
──────────────────────────────────────
Tests for Modules 08, 09, 10 — AI prediction layer.

Module 08 (ESM-2) GPU tests are skipped by default — ESM-2 requires
downloading a 1.4GB model on first run.

Run with:
    python -m pytest tests/test_08_09_10_ai_prediction.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest


# ── Fixtures ───────────────────────────────────────────────────────────────────

TP53_SEQUENCE = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDP"

@pytest.fixture
def sample_esm2_result():
    """Mock ESM-2 result for offline testing."""
    L = len(TP53_SEQUENCE)
    rng = np.random.default_rng(42)
    return {
        "uniprot_id": "P00000",
        "sequence_length": L,
        "model_name": "esm2_t33_650M_UR50D",
        "embedding_dim": 1280,
        "gpu_used": False,
        "compute_time_sec": 2.5,
        "protein_embedding": rng.standard_normal(1280).tolist(),
        "residue_embeddings": rng.standard_normal((L, 1280)).tolist(),
        "contact_map": (rng.random((L, L)) > 0.7).astype(float).tolist(),
        "predicted_functional_residues": list(range(1, 20)),
        "embedding_norm": 24.5,
    }


@pytest.fixture
def sample_homology_result():
    return {
        "uniprot_id": "P00000",
        "all_go_terms": ["GO:0003677", "GO:0006915", "GO:0005634"],
        "all_go_names": ["DNA binding", "apoptotic process", "nucleus"],
        "n_experimental_hits": 5,
        "interpro_domains": [
            {
                "accession": "IPR011615",
                "name": "P53 DNA-binding domain",
                "database": "PFAM",
                "go_terms": ["GO:0003677"],
                "go_names": ["DNA binding"],
            }
        ],
        "blast_hits": [
            {
                "accession": "P04637",
                "reviewed": True,
                "function_text": "Acts as a tumor suppressor.",
                "go_terms": ["GO:0003677", "GO:0006915"],
                "go_names": ["DNA binding", "apoptosis"],
                "evidence_weight": 3.0,
            }
        ],
    }


@pytest.fixture
def sample_active_result():
    return {
        "n_high_confidence": 8,
        "catalytic_motifs": [
            {
                "motif_type": "zinc_binding_cluster",
                "residue_numbers": [176, 178, 179, 182],
                "confidence": "HIGH",
            },
            {
                "motif_type": "dna_binding_cluster",
                "residue_numbers": [248, 249, 273],
                "confidence": "MEDIUM",
            },
        ],
        "active_residues": [
            {"residue_number": 176, "confidence": "HIGH"},
            {"residue_number": 248, "confidence": "HIGH"},
        ],
    }


# ── Module 08: ESM-2 result structure ─────────────────────────────────────────

class TestESM2ResultStructure:

    def test_result_dataclass_fields(self, sample_esm2_result):
        from pipeline.esm2_embeddings import ESM2Result
        r = ESM2Result(**{
            k: v for k, v in sample_esm2_result.items()
        })
        assert r.uniprot_id == "P00000"
        assert r.embedding_dim == 1280

    def test_protein_embedding_np(self, sample_esm2_result):
        from pipeline.esm2_embeddings import ESM2Result
        r = ESM2Result(**sample_esm2_result)
        arr = r.protein_embedding_np()
        assert arr.shape == (1280,)
        assert arr.dtype == np.float32

    def test_residue_embeddings_np(self, sample_esm2_result):
        from pipeline.esm2_embeddings import ESM2Result
        r = ESM2Result(**sample_esm2_result)
        arr = r.residue_embeddings_np()
        assert arr.shape == (len(TP53_SEQUENCE), 1280)

    def test_summary_string(self, sample_esm2_result):
        from pipeline.esm2_embeddings import ESM2Result
        r = ESM2Result(**sample_esm2_result)
        s = r.summary()
        assert "P00000" in s
        assert "ESM-2" in s

    def test_to_dict_serialisable(self, sample_esm2_result):
        from pipeline.esm2_embeddings import ESM2Result
        r = ESM2Result(**sample_esm2_result)
        json.dumps(r.to_dict())

    def test_to_json_writes_file(self, sample_esm2_result, tmp_path):
        from pipeline.esm2_embeddings import ESM2Result
        r = ESM2Result(**sample_esm2_result)
        out = tmp_path / "esm2.json"
        r.to_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["uniprot_id"] == "P00000"


# ── Module 09: GO term prediction ─────────────────────────────────────────────

class TestGOPrediction:

    def test_returns_deepfri_result(self, sample_homology_result,
                                    sample_active_result, sample_esm2_result):
        from pipeline.deepfri_go import predict_go_terms, DeepFRIResult
        result = predict_go_terms(
            "P00000", TP53_SEQUENCE,
            sample_esm2_result, sample_homology_result, sample_active_result
        )
        assert isinstance(result, DeepFRIResult)

    def test_uniprot_id_preserved(self, sample_homology_result):
        from pipeline.deepfri_go import predict_go_terms
        result = predict_go_terms("P00000", TP53_SEQUENCE,
                                   None, sample_homology_result, None)
        assert result.uniprot_id == "P00000"

    def test_mf_predictions_list(self, sample_homology_result):
        from pipeline.deepfri_go import predict_go_terms
        result = predict_go_terms("P00000", TP53_SEQUENCE,
                                   None, sample_homology_result, None)
        assert isinstance(result.mf_predictions, list)

    def test_bp_predictions_list(self, sample_homology_result):
        from pipeline.deepfri_go import predict_go_terms
        result = predict_go_terms("P00000", TP53_SEQUENCE,
                                   None, sample_homology_result, None)
        assert isinstance(result.bp_predictions, list)

    def test_scores_bounded(self, sample_homology_result):
        from pipeline.deepfri_go import predict_go_terms
        result = predict_go_terms("P00000", TP53_SEQUENCE,
                                   None, sample_homology_result, None)
        for pred in (result.mf_predictions + result.bp_predictions +
                     result.cc_predictions):
            assert 0.0 <= pred.score <= 1.0

    def test_homology_go_terms_appear(self, sample_homology_result):
        from pipeline.deepfri_go import predict_go_terms
        result = predict_go_terms("P00000", TP53_SEQUENCE,
                                   None, sample_homology_result, None)
        all_ids = {p.go_id for p in result.mf_predictions +
                   result.bp_predictions + result.cc_predictions}
        assert "GO:0003677" in all_ids or "GO:0006915" in all_ids

    def test_zinc_motif_adds_metal_binding(self, sample_active_result):
        from pipeline.deepfri_go import predict_go_terms
        result = predict_go_terms("P00000", TP53_SEQUENCE,
                                   None, None, sample_active_result)
        all_ids = {p.go_id for p in result.mf_predictions}
        assert "GO:0008270" in all_ids or "GO:0046872" in all_ids

    def test_dna_motif_adds_dna_binding(self, sample_active_result):
        from pipeline.deepfri_go import predict_go_terms
        result = predict_go_terms("P00000", TP53_SEQUENCE,
                                   None, None, sample_active_result)
        all_ids = {p.go_id for p in result.mf_predictions}
        assert "GO:0003677" in all_ids

    def test_n_predictions_matches(self, sample_homology_result,
                                    sample_active_result):
        from pipeline.deepfri_go import predict_go_terms
        result = predict_go_terms("P00000", TP53_SEQUENCE,
                                   None, sample_homology_result,
                                   sample_active_result)
        expected = (len(result.mf_predictions) + len(result.bp_predictions) +
                    len(result.cc_predictions))
        assert result.n_predictions == expected

    def test_to_dict_serialisable(self, sample_homology_result):
        from pipeline.deepfri_go import predict_go_terms
        result = predict_go_terms("P00000", TP53_SEQUENCE,
                                   None, sample_homology_result, None)
        json.dumps(result.to_dict())

    def test_to_json_writes_file(self, sample_homology_result, tmp_path):
        from pipeline.deepfri_go import predict_go_terms
        result = predict_go_terms("P00000", TP53_SEQUENCE,
                                   None, sample_homology_result, None)
        out = tmp_path / "go.json"
        result.to_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["uniprot_id"] == "P00000"

    def test_runs_with_no_inputs(self):
        from pipeline.deepfri_go import predict_go_terms
        result = predict_go_terms("P00000", TP53_SEQUENCE)
        assert result.uniprot_id == "P00000"

    def test_summary_string(self, sample_homology_result):
        from pipeline.deepfri_go import predict_go_terms
        result = predict_go_terms("P00000", TP53_SEQUENCE,
                                   None, sample_homology_result, None)
        s = result.summary()
        assert "P00000" in s


# ── Module 10: EC number prediction ───────────────────────────────────────────

class TestECPrediction:

    def test_returns_ec_result(self, sample_active_result):
        from pipeline.clean_ec import predict_ec_number, ECResult
        result = predict_ec_number("P00000", TP53_SEQUENCE,
                                    sample_active_result, None, None)
        assert isinstance(result, ECResult)

    def test_uniprot_id_preserved(self):
        from pipeline.clean_ec import predict_ec_number
        result = predict_ec_number("P00000", TP53_SEQUENCE)
        assert result.uniprot_id == "P00000"

    def test_is_enzyme_bool(self):
        from pipeline.clean_ec import predict_ec_number
        result = predict_ec_number("P00000", TP53_SEQUENCE)
        assert isinstance(result.is_enzyme, bool)

    def test_tp53_not_enzyme(self, sample_active_result, sample_homology_result):
        from pipeline.clean_ec import predict_ec_number
        from pipeline.deepfri_go import predict_go_terms
        go_result = predict_go_terms(
            "P00000", TP53_SEQUENCE, None, sample_homology_result,
            sample_active_result
        ).to_dict()
        result = predict_ec_number(
            "P00000", TP53_SEQUENCE,
            sample_active_result, go_result, sample_homology_result
        )
        # TP53 is a transcription factor, not an enzyme
        assert result.is_enzyme is False

    def test_zinc_motif_gives_ec3(self, sample_active_result):
        from pipeline.clean_ec import predict_ec_number
        # Zinc binding cluster → metallopeptidase → EC 3
        active = {
            "n_high_confidence": 4,
            "catalytic_motifs": [
                {
                    "motif_type": "zinc_binding_cluster",
                    "residue_numbers": [1, 2, 3, 4],
                    "confidence": "HIGH",
                }
            ],
            "active_residues": [],
        }
        result = predict_ec_number("P00000", TP53_SEQUENCE, active, None, None)
        ec_classes = [p.ec_class for p in result.all_predictions]
        assert "3" in ec_classes

    def test_serine_triad_gives_ec3(self):
        from pipeline.clean_ec import predict_ec_number
        active = {
            "n_high_confidence": 3,
            "catalytic_motifs": [
                {
                    "motif_type": "serine_protease_triad",
                    "residue_numbers": [57, 102, 195],
                    "confidence": "HIGH",
                }
            ],
            "active_residues": [],
        }
        result = predict_ec_number("P00000", TP53_SEQUENCE, active, None, None)
        assert result.specific_ec == "3.4.21"

    def test_scores_bounded(self, sample_active_result):
        from pipeline.clean_ec import predict_ec_number
        result = predict_ec_number("P00000", TP53_SEQUENCE,
                                    sample_active_result, None, None)
        for pred in result.all_predictions:
            assert 0.0 <= pred.score <= 1.0

    def test_enzyme_confidence_bounded(self):
        from pipeline.clean_ec import predict_ec_number
        result = predict_ec_number("P00000", TP53_SEQUENCE)
        assert 0.0 <= result.enzyme_confidence <= 1.0

    def test_to_dict_serialisable(self, sample_active_result):
        from pipeline.clean_ec import predict_ec_number
        result = predict_ec_number("P00000", TP53_SEQUENCE,
                                    sample_active_result)
        json.dumps(result.to_dict())

    def test_to_json_writes_file(self, sample_active_result, tmp_path):
        from pipeline.clean_ec import predict_ec_number
        result = predict_ec_number("P00000", TP53_SEQUENCE,
                                    sample_active_result)
        out = tmp_path / "ec.json"
        result.to_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["uniprot_id"] == "P00000"

    def test_summary_string(self, sample_active_result):
        from pipeline.clean_ec import predict_ec_number
        result = predict_ec_number("P00000", TP53_SEQUENCE,
                                    sample_active_result)
        s = result.summary()
        assert "P00000" in s
        assert "EC prediction" in s

    def test_runs_with_no_inputs(self):
        from pipeline.clean_ec import predict_ec_number
        result = predict_ec_number("P00000", TP53_SEQUENCE)
        assert isinstance(result.is_enzyme, bool)


# ── GO namespace inference ─────────────────────────────────────────────────────

class TestGoNamespaceInference:

    def test_binding_is_mf(self):
        from pipeline.deepfri_go import _go_namespace
        assert _go_namespace("GO:0005488", "binding") == "MF"

    def test_activity_is_mf(self):
        from pipeline.deepfri_go import _go_namespace
        assert _go_namespace("GO:0003824", "catalytic activity") == "MF"

    def test_process_is_bp(self):
        from pipeline.deepfri_go import _go_namespace
        assert _go_namespace("GO:0006915", "apoptotic process") == "BP"

    def test_nucleus_is_cc(self):
        from pipeline.deepfri_go import _go_namespace
        assert _go_namespace("GO:0005634", "nucleus") == "CC"

    def test_membrane_is_cc(self):
        from pipeline.deepfri_go import _go_namespace
        assert _go_namespace("GO:0016020", "membrane") == "CC"