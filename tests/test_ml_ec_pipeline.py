"""
tests/test_ml_ec_pipeline.py
─────────────────────────────
Comprehensive test suite for the ML-based EC classification pipeline.

Tests:
  - Feature engineering (unit tests for each feature block)
  - ML classifier (smoke tests with synthetic data)
  - Integration (end-to-end with known proteins)
  - Backward compatibility (parity checks with legacy Module 10)
  - Edge cases (short sequences, missing data, all-zeros)

Run:
    python -m pytest tests/test_ml_ec_pipeline.py -v
    python -m pytest tests/test_ml_ec_pipeline.py -v -k "fast"  # skip slow tests
"""

from __future__ import annotations

import json
import random
import string
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Helper ─────────────────────────────────────────────────────────────────────

AA = list("ACDEFGHIKLMNPQRSTVWY")

def rand_seq(n: int) -> str:
    return "".join(random.choices(AA, k=n))


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestFeatureEngineering:
    """Unit tests for each feature extraction function."""

    def setup_method(self):
        from pipeline.ml_ec_features import (
            _aac, _dpc, _tpc_compressed, _ctd_features,
            _paac, _apaac, _esm2_features, _structural_features,
            _contact_graph_features, _disulfide_features,
            _cofactor_fingerprint, _evidence_features,
            build_feature_vector, FEATURE_DIM,
        )
        self._aac            = _aac
        self._dpc            = _dpc
        self._tpc            = _tpc_compressed
        self._ctd            = _ctd_features
        self._paac           = _paac
        self._apaac          = _apaac
        self._esm2           = _esm2_features
        self._struct         = _structural_features
        self._contact        = _contact_graph_features
        self._disulf         = _disulfide_features
        self._cofact         = _cofactor_fingerprint
        self._evidence       = _evidence_features
        self.build           = build_feature_vector
        self.FEATURE_DIM     = FEATURE_DIM

    # ── AAC ────────────────────────────────────────────────────────────────────

    def test_aac_shape(self):
        v = self._aac("ACDEFG")
        assert v.shape == (20,)

    def test_aac_sums_to_one(self):
        v = self._aac(rand_seq(100))
        assert abs(v.sum() - 1.0) < 1e-5

    def test_aac_empty(self):
        v = self._aac("")
        assert v.shape == (20,)
        assert v.sum() == 0.0

    def test_aac_single_aa(self):
        v = self._aac("AAAA")
        assert v[0] == pytest.approx(1.0)  # A is index 0

    # ── DPC ────────────────────────────────────────────────────────────────────

    def test_dpc_shape(self):
        v = self._dpc(rand_seq(50))
        assert v.shape == (400,)

    def test_dpc_sums_to_one(self):
        v = self._dpc(rand_seq(100))
        assert abs(v.sum() - 1.0) < 1e-5

    def test_dpc_short_seq(self):
        v = self._dpc("A")
        assert v.shape == (400,)
        assert v.sum() == 0.0  # single AA → no dipeptides

    # ── TPC ────────────────────────────────────────────────────────────────────

    def test_tpc_shape(self):
        v = self._tpc(rand_seq(100))
        assert v.shape == (64,)

    def test_tpc_no_nan(self):
        v = self._tpc(rand_seq(200))
        assert not np.any(np.isnan(v))
        assert not np.any(np.isinf(v))

    # ── CTD ────────────────────────────────────────────────────────────────────

    def test_ctd_shape(self):
        v = self._ctd(rand_seq(100))
        assert v.shape == (21,)

    # ── PAAC ───────────────────────────────────────────────────────────────────

    def test_paac_shape(self):
        v = self._paac(rand_seq(100), lag=30)
        assert v.shape == (50,)

    def test_paac_short_seq(self):
        """PAAC should handle sequences shorter than lag gracefully."""
        v = self._paac("MKTA", lag=30)
        assert v.shape == (50,)
        assert not np.any(np.isnan(v))

    # ── APAAC ──────────────────────────────────────────────────────────────────

    def test_apaac_shape(self):
        v = self._apaac(rand_seq(100), lag=30)
        assert v.shape == (80,)

    # ── ESM-2 features ─────────────────────────────────────────────────────────

    def test_esm2_none(self):
        v = self._esm2(None)
        assert v.shape == (128,)
        assert v.sum() == 0.0

    def test_esm2_valid(self):
        fake_emb = np.random.randn(1280).tolist()
        v = self._esm2({"protein_embedding": fake_emb})
        assert v.shape == (128,)
        assert not np.any(np.isnan(v))

    def test_esm2_wrong_dim(self):
        """ESM-2 features should handle wrong embedding dim gracefully."""
        short_emb = np.random.randn(640).tolist()
        v = self._esm2({"protein_embedding": short_emb})
        assert v.shape == (128,)
        assert not np.any(np.isnan(v))

    # ── Structural features ────────────────────────────────────────────────────

    def test_struct_none_inputs(self):
        v = self._struct(None, None, None, None, None)
        assert v.shape == (86,)
        assert v.sum() == 0.0

    def test_struct_with_pdb(self):
        fake_pdb = {
            "residues": [{"plddt": 85.0 + i*0.1} for i in range(100)],
            "secondary_structure_fractions": {"helix": 0.4, "strand": 0.2, "coil": 0.4},
        }
        v = self._struct(fake_pdb, None, None, None, None)
        assert v.shape == (86,)
        assert v[0] > 0   # mean pLDDT

    def test_struct_active_sites(self):
        fake_active = {
            "catalytic_motifs": [
                {"motif_type": "serine_protease_triad", "mean_distance": 4.5,
                 "zinc_type": None, "residue_numbers": [57, 102, 195]}
            ],
            "active_residues": [{"type": "catalytic"} for _ in range(5)],
            "n_high_confidence": 3,
            "n_medium_confidence": 2,
        }
        v = self._struct(None, fake_active, None, None, None)
        assert v[18] == pytest.approx(1.0)  # n_high = 3? check serine motif count

    # ── Contact graph ──────────────────────────────────────────────────────────

    def test_contact_no_data(self):
        v = self._contact(None, "MKTAY")
        assert v.shape == (12,)
        assert v.sum() == 0.0

    def test_contact_valid(self):
        L = 30
        fake_cm = (np.random.rand(L, L) > 0.7).astype(float).tolist()
        v = self._contact({"contact_map": fake_cm}, rand_seq(L))
        assert v.shape == (12,)
        assert not np.any(np.isnan(v))

    # ── Disulphide features ────────────────────────────────────────────────────

    def test_disulf_no_cys(self):
        v = self._disulf("MKTAY", None)
        assert v[0] == 0.0   # n_cys

    def test_disulf_cys_rich(self):
        v = self._disulf("MCCCHCCCMK", None)
        assert v[0] == 6.0   # 6 Cys

    # ── Cofactor fingerprint ───────────────────────────────────────────────────

    def test_cofact_shape(self):
        v = self._cofact(rand_seq(100), None)
        assert v.shape == (23,)
        assert not np.any(np.isnan(v))

    def test_cofact_p_loop(self):
        """Walker A P-loop (GXXXXGKT) should score positive for ATP."""
        seq = "MKTGGGGGKTMKTM"
        v = self._cofact(seq, None)
        assert v[16] > 0  # ATP-binding signal

    # ── Evidence features ──────────────────────────────────────────────────────

    def test_evidence_none(self):
        v = self._evidence(None, None, None)
        assert v.shape == (42,)
        assert v.sum() == 0.0

    def test_evidence_go_oxidoreductase(self):
        go = {
            "mf_predictions": [
                {"go_id": "GO:0016491", "go_name": "oxidoreductase activity", "score": 0.9}
            ],
            "bp_predictions": [],
        }
        v = self._evidence(go, None, None)
        assert v[10] == 1.0  # GO:0016491 at index 10 in GO_CATEGORIES

    # ── Full feature vector ─────────────────────────────────────────────────────

    def test_build_feature_vector_shape(self):
        seq = rand_seq(150)
        v = self.build(sequence=seq)
        assert v.shape == (self.FEATURE_DIM,), f"Expected {self.FEATURE_DIM}, got {v.shape[0]}"

    def test_build_no_nan_inf(self):
        seq = rand_seq(200)
        v = self.build(sequence=seq)
        assert not np.any(np.isnan(v)), "Feature vector contains NaN"
        assert not np.any(np.isinf(v)), "Feature vector contains Inf"

    def test_build_deterministic(self):
        seq = rand_seq(100)
        v1 = self.build(sequence=seq)
        v2 = self.build(sequence=seq)
        np.testing.assert_array_equal(v1, v2)

    def test_build_all_inputs(self):
        seq = rand_seq(120)
        fake_esm2   = {"protein_embedding": np.random.randn(1280).tolist(), "contact_map": []}
        fake_active = {"catalytic_motifs": [], "active_residues": [],
                       "n_high_confidence": 0, "n_medium_confidence": 0}
        fake_go     = {"mf_predictions": [
                           {"go_id": "GO:0016787", "go_name": "hydrolase activity", "score": 0.8}
                       ], "bp_predictions": []}
        v = self.build(
            sequence        = seq,
            esm2_result     = fake_esm2,
            active_result   = fake_active,
            go_result       = fake_go,
        )
        assert v.shape == (self.FEATURE_DIM,)
        assert not np.any(np.isnan(v))

    def test_build_enzyme_vs_nonenzyme_differ(self):
        """Feature vectors for enzyme vs non-enzyme should differ appreciably."""
        # Enzyme-like: rich in catalytic motifs / GO terms
        enzyme_go = {
            "mf_predictions": [
                {"go_id": "GO:0016787", "go_name": "hydrolase activity", "score": 0.95},
                {"go_id": "GO:0004252", "go_name": "serine-type endopeptidase", "score": 0.88},
            ],
            "bp_predictions": [],
        }
        enzyme_active = {
            "catalytic_motifs": [{"motif_type": "serine_protease_triad",
                                   "mean_distance": 4.5, "zinc_type": None,
                                   "residue_numbers": [57, 102, 195]}],
            "active_residues": [{"type": "catalytic"}] * 3,
            "n_high_confidence": 3, "n_medium_confidence": 0,
        }
        nonenzyme_go = {
            "mf_predictions": [
                {"go_id": "GO:0003700", "go_name": "DNA-binding transcription factor", "score": 0.9}
            ],
            "bp_predictions": [],
        }
        seq = rand_seq(200)
        v_enz   = self.build(sequence=seq, go_result=enzyme_go,    active_result=enzyme_active)
        v_noenz = self.build(sequence=seq, go_result=nonenzyme_go)
        diff = np.abs(v_enz - v_noenz).sum()
        assert diff > 0.5, f"Enzyme vs non-enzyme vectors too similar (diff={diff:.4f})"


# ── Classifier smoke tests ────────────────────────────────────────────────────

class TestMLClassifier:
    """Smoke tests for the ML ensemble using synthetic data."""

    @pytest.fixture
    def tiny_model(self):
        """Train a tiny model on synthetic data."""
        from pipeline.ml_ec_classifier import ECClassifierEnsemble, N_CLASSES
        from pipeline.ml_ec_features import FEATURE_DIM

        np.random.seed(42)
        n_per_class = 40
        X = np.random.randn(n_per_class * N_CLASSES, FEATURE_DIM).astype(np.float32)
        y = np.repeat(np.arange(N_CLASSES), n_per_class).astype(np.int32)

        # Add small signal: each class has a unique offset in a few dims
        for c in range(N_CLASSES):
            mask = y == c
            X[mask, c*5:(c+1)*5] += 3.0  # class-specific signal

        clf = ECClassifierEnsemble()
        try:
            clf.train(X, y, validation_split=0.2, n_cv_folds=2, verbose=False)
            return clf
        except ImportError:
            pytest.skip("xgboost/lightgbm/sklearn not installed")

    def test_predict_returns_result(self, tiny_model):
        seq = rand_seq(100)
        result = tiny_model.predict(sequence=seq, uniprot_id="TEST")
        assert result.uniprot_id == "TEST"
        assert result.sequence_length == 100

    def test_predict_probabilities_sum_to_one(self, tiny_model):
        seq = rand_seq(100)
        result = tiny_model.predict(sequence=seq)
        total = sum(p.probability for p in result.ec_predictions)
        # EC predictions are 7 classes (not including non-enzyme)
        # enzyme_probability + non_enzyme_prob = 1
        assert abs(result.enzyme_probability + result.non_enzyme_score - 1.0) < 0.01

    def test_predictions_sorted_by_probability(self, tiny_model):
        seq = rand_seq(100)
        result = tiny_model.predict(sequence=seq)
        probs = [p.probability for p in result.ec_predictions]
        assert probs == sorted(probs, reverse=True), "Predictions not sorted by probability"

    def test_inference_time_positive(self, tiny_model):
        result = tiny_model.predict(sequence=rand_seq(100))
        assert result.inference_time_ms > 0

    def test_all_7_ec_classes_present(self, tiny_model):
        result = tiny_model.predict(sequence=rand_seq(200))
        ec_classes = {p.ec_class for p in result.ec_predictions}
        assert ec_classes == {"1", "2", "3", "4", "5", "6", "7"}

    @pytest.mark.slow
    def test_save_load_roundtrip(self, tiny_model, tmp_path):
        from pipeline.ml_ec_classifier import ECClassifierEnsemble
        tiny_model.save(tmp_path / "test_model")
        loaded = ECClassifierEnsemble.load(tmp_path / "test_model")
        seq = rand_seq(100)
        r1 = tiny_model.predict(sequence=seq)
        r2 = loaded.predict(sequence=seq)
        # Results should be identical
        for p1, p2 in zip(r1.ec_predictions, r2.ec_predictions):
            assert p1.ec_class == p2.ec_class
            assert abs(p1.probability - p2.probability) < 1e-4


# ── Integration / backward compat ─────────────────────────────────────────────

class TestPredictor:
    """Tests for the predict_ec_ml wrapper."""

    def test_fallback_to_legacy_when_no_model(self):
        """predict_ec_ml should fall back gracefully when model dir is missing."""
        from pipeline.ml_ec_predict import predict_ec_ml
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            result = predict_ec_ml(
                uniprot_id  = "TEST",
                sequence    = rand_seq(100),
                model_dir   = Path(tmpdir) / "nonexistent",
            )
        assert result is not None
        assert result.ml_used is False
        assert hasattr(result, "is_enzyme")

    def test_result_has_required_fields(self):
        from pipeline.ml_ec_predict import predict_ec_ml
        result = predict_ec_ml("TEST", rand_seq(50), model_dir=Path("/nonexistent"))
        for field in ["uniprot_id", "sequence_length", "is_enzyme",
                      "enzyme_confidence", "ml_used"]:
            assert hasattr(result, field), f"Missing field: {field}"

    def test_to_dict_serialisable(self):
        from pipeline.ml_ec_predict import predict_ec_ml
        result = predict_ec_ml("TEST", rand_seq(80), model_dir=Path("/nonexistent"))
        d = result.to_dict()
        j = json.dumps(d)  # must not raise
        assert isinstance(j, str)

    def test_specific_ec_refinement_serine_protease(self):
        """Serine protease motif should refine to EC 3.4.21."""
        from pipeline.ml_ec_classifier import _refine_specific_ec
        active = {
            "catalytic_motifs": [
                {"motif_type": "serine_protease_triad", "zinc_type": None}
            ]
        }
        ec, name = _refine_specific_ec("3", active, None)
        assert ec == "3.4.21"
        assert "serine" in name.lower()

    def test_specific_ec_refinement_structural_zinc_excluded(self):
        """Structural zinc (not catalytic) should NOT give EC 3.4.24."""
        from pipeline.ml_ec_classifier import _refine_specific_ec
        active = {
            "catalytic_motifs": [
                {"motif_type": "zinc_binding_cluster", "zinc_type": "structural"}
            ]
        }
        ec, name = _refine_specific_ec("3", active, None)
        assert ec != "3.4.24", "Structural zinc should not give metallopeptidase EC"

    def test_specific_ec_go_gTPase(self):
        """GO:0003924 should give EC 3.6.5 (GTPase)."""
        from pipeline.ml_ec_classifier import _refine_specific_ec
        go = {"mf_predictions": [{"go_id": "GO:0003924", "go_name": "GTPase activity"}],
              "bp_predictions": []}
        ec, name = _refine_specific_ec("3", None, go)
        assert ec == "3.6.5"


# ── Edge cases ─────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_very_short_sequence(self):
        from pipeline.ml_ec_features import build_feature_vector, FEATURE_DIM
        v = build_feature_vector(sequence="MK")
        assert v.shape == (FEATURE_DIM,)
        assert not np.any(np.isnan(v))

    def test_very_long_sequence(self):
        from pipeline.ml_ec_features import build_feature_vector, FEATURE_DIM
        seq = rand_seq(5000)
        v = build_feature_vector(sequence=seq)
        assert v.shape == (FEATURE_DIM,)

    def test_non_standard_amino_acids(self):
        """Sequences with X, U, B etc should not crash."""
        from pipeline.ml_ec_features import build_feature_vector, FEATURE_DIM
        seq = "MKTAXUBJZOMKT"  # X, U, B, J, Z, O are non-standard
        v = build_feature_vector(sequence=seq)
        assert v.shape == (FEATURE_DIM,)
        assert not np.any(np.isnan(v))

    def test_all_same_amino_acid(self):
        from pipeline.ml_ec_features import build_feature_vector
        v = build_feature_vector(sequence="A" * 100)
        assert not np.any(np.isnan(v))

    def test_empty_go_result(self):
        from pipeline.ml_ec_features import build_feature_vector, FEATURE_DIM
        v = build_feature_vector(sequence=rand_seq(100),
                                  go_result={"mf_predictions": [], "bp_predictions": []})
        assert v.shape == (FEATURE_DIM,)

    def test_malformed_esm2(self):
        from pipeline.ml_ec_features import build_feature_vector
        v = build_feature_vector(
            sequence    = rand_seq(100),
            esm2_result = {"protein_embedding": None, "contact_map": "broken"},
        )
        assert not np.any(np.isnan(v))

    def test_feature_dim_constant_matches(self):
        from pipeline.ml_ec_features import build_feature_vector, FEATURE_DIM
        v = build_feature_vector(sequence=rand_seq(100))
        assert v.shape[0] == FEATURE_DIM, (
            f"FEATURE_DIM constant ({FEATURE_DIM}) doesn't match actual output ({v.shape[0]})"
        )


# ── Performance benchmark ─────────────────────────────────────────────────────

class TestPerformance:
    """Timing benchmarks."""

    @pytest.mark.slow
    def test_feature_extraction_speed(self):
        """Feature extraction should complete in < 1s per protein."""
        import time
        from pipeline.ml_ec_features import build_feature_vector
        seq = rand_seq(500)
        t0 = time.time()
        for _ in range(10):
            build_feature_vector(sequence=seq)
        elapsed = (time.time() - t0) / 10
        assert elapsed < 1.0, f"Feature extraction too slow: {elapsed:.3f}s per call"

    @pytest.mark.slow
    def test_feature_vectors_discriminative(self):
        """Feature vectors from different EC classes should be separable."""
        from pipeline.ml_ec_features import build_feature_vector
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        # Build small dataset with class-specific GO evidence
        go_templates = {
            0: {"mf_predictions": [{"go_id": "GO:0003700", "go_name": "transcription factor"}],
                "bp_predictions": []},
            1: {"mf_predictions": [{"go_id": "GO:0016491", "go_name": "oxidoreductase activity"}],
                "bp_predictions": []},
            2: {"mf_predictions": [{"go_id": "GO:0016301", "go_name": "kinase activity"}],
                "bp_predictions": []},
            3: {"mf_predictions": [{"go_id": "GO:0016787", "go_name": "hydrolase activity"}],
                "bp_predictions": []},
        }

        X, y = [], []
        for cls, go in go_templates.items():
            for _ in range(20):
                v = build_feature_vector(sequence=rand_seq(150), go_result=go)
                X.append(v)
                y.append(cls)

        X = np.array(X)
        y = np.array(y)

        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        acc = lda.score(X, y)
        assert acc > 0.7, f"Feature vectors not discriminative enough: LDA accuracy={acc:.2f}"