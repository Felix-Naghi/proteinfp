"""
tests/test_12_ppi_network.py
─────────────────────────────
Tests for Module 12 — Protein-protein interaction network.

All tests are offline — no network calls.

Run with:
    python -m pytest tests/test_12_ppi_network.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np


# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_RAW_PARTNERS = [
    {
        "stringId_B":       "9606.ENSP00000258400",
        "preferredName_B":  "MDM2",
        "score":            0.999,
        "escore":           0.950,
        "dscore":           0.900,
        "tscore":           0.850,
        "coexpression":     0.200,
    },
    {
        "stringId_B":       "9606.ENSP00000309572",
        "preferredName_B":  "BRCA1",
        "score":            0.920,
        "escore":           0.800,
        "dscore":           0.700,
        "tscore":           0.600,
        "coexpression":     0.300,
    },
    {
        "stringId_B":       "9606.ENSP00000278616",
        "preferredName_B":  "ATM",
        "score":            0.870,
        "escore":           0.750,
        "dscore":           0.650,
        "tscore":           0.550,
        "coexpression":     0.100,
    },
    {
        "stringId_B":       "9606.ENSP00000354356",
        "preferredName_B":  "UNKNOWN_PROTEIN",
        "score":            0.450,
        "escore":           0.100,
        "dscore":           0.200,
        "tscore":           0.300,
        "coexpression":     0.050,
    },
]

MINIMAL_PDB = """\
ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 85.00           C
ATOM      2  CA  ARG A   2      14.000  10.000  10.000  1.00 85.00           C
ATOM      3  CA  ASP A   3      18.000  10.000  10.000  1.00 85.00           C
ATOM      4  CA  LEU A   4      22.000  10.000  10.000  1.00 85.00           C
ATOM      5  CA  LYS A   5      26.000  10.000  10.000  1.00 85.00           C
END
"""


@pytest.fixture
def pdb_file(tmp_path):
    p = tmp_path / "P00000.pdb"
    p.write_text(MINIMAL_PDB)
    return p


@pytest.fixture
def parsed_structure(pdb_file):
    from utils.pdb_parser import parse_pdb
    return parse_pdb(pdb_file, "P00000")


@pytest.fixture
def sasa_map(parsed_structure):
    return {
        (r.chain_id, r.residue_number): 60.0
        for r in parsed_structure.residues
    }


# ── Unit tests: interaction classification ────────────────────────────────────

class TestInteractionClassification:

    def test_mdm2_is_inhibitory(self):
        from pipeline.ppi_network import _classify_interaction
        assert _classify_interaction("MDM2") == "inhibitory"

    def test_atm_is_activating(self):
        from pipeline.ppi_network import _classify_interaction
        assert _classify_interaction("ATM") == "activating"

    def test_brca1_is_cooperative(self):
        from pipeline.ppi_network import _classify_interaction
        assert _classify_interaction("BRCA1") == "cooperative"

    def test_unknown_is_unknown(self):
        from pipeline.ppi_network import _classify_interaction
        assert _classify_interaction("ZZZNVL") == "unknown"

    def test_case_insensitive(self):
        from pipeline.ppi_network import _classify_interaction
        assert _classify_interaction("mdm2") == "inhibitory"
        assert _classify_interaction("Atm")  == "activating"


# ── Unit tests: partner building ──────────────────────────────────────────────

class TestPartnerBuilding:

    def test_builds_ppi_partner(self, parsed_structure, sasa_map):
        from pipeline.ppi_network import _build_partner, PPIPartner
        raw = SAMPLE_RAW_PARTNERS[0]
        partner = _build_partner(raw, parsed_structure, sasa_map, "MEESA")
        assert isinstance(partner, PPIPartner)

    def test_partner_name_preserved(self, parsed_structure, sasa_map):
        from pipeline.ppi_network import _build_partner
        raw = SAMPLE_RAW_PARTNERS[0]
        partner = _build_partner(raw, parsed_structure, sasa_map, "MEESA")
        assert partner.partner_name == "MDM2"

    def test_score_converted_to_int(self, parsed_structure, sasa_map):
        from pipeline.ppi_network import _build_partner
        raw = SAMPLE_RAW_PARTNERS[0]
        partner = _build_partner(raw, parsed_structure, sasa_map, "MEESA")
        assert isinstance(partner.combined_score, int)
        assert partner.combined_score == 999

    def test_high_confidence_flagged(self, parsed_structure, sasa_map):
        from pipeline.ppi_network import _build_partner
        raw = SAMPLE_RAW_PARTNERS[0]   # score 0.999
        partner = _build_partner(raw, parsed_structure, sasa_map, "MEESA")
        assert partner.confidence == "high"

    def test_medium_confidence_flagged(self, parsed_structure, sasa_map):
        from pipeline.ppi_network import _build_partner
        raw = SAMPLE_RAW_PARTNERS[3]   # score 0.450
        partner = _build_partner(raw, parsed_structure, sasa_map, "MEESA")
        assert partner.confidence == "medium"

    def test_interface_residues_are_ints(self, parsed_structure, sasa_map):
        from pipeline.ppi_network import _build_partner
        partner = _build_partner(
            SAMPLE_RAW_PARTNERS[0], parsed_structure, sasa_map, "MEESA"
        )
        for rn in partner.interface_residues:
            assert isinstance(rn, int)

    def test_binding_mode_valid(self, parsed_structure, sasa_map):
        from pipeline.ppi_network import _build_partner
        for raw in SAMPLE_RAW_PARTNERS:
            partner = _build_partner(raw, parsed_structure, sasa_map, "MEESA")
            assert partner.binding_mode in ("hydrophobic", "electrostatic", "mixed")

    def test_to_dict_serialisable(self, parsed_structure, sasa_map):
        from pipeline.ppi_network import _build_partner
        partner = _build_partner(
            SAMPLE_RAW_PARTNERS[0], parsed_structure, sasa_map, "MEESA"
        )
        json.dumps(partner.to_dict())

    def test_mdm2_is_inhibitory_type(self, parsed_structure, sasa_map):
        from pipeline.ppi_network import _build_partner
        partner = _build_partner(
            SAMPLE_RAW_PARTNERS[0], parsed_structure, sasa_map, "MEESA"
        )
        assert partner.interaction_type == "inhibitory"


# ── Unit tests: interface prediction ─────────────────────────────────────────

class TestInterfacePrediction:

    def test_returns_residue_numbers(self, parsed_structure, sasa_map):
        from pipeline.ppi_network import _predict_interface
        rns, letters = _predict_interface(parsed_structure, sasa_map, "inhibitory")
        assert isinstance(rns, list)
        assert all(isinstance(r, int) for r in rns)

    def test_returns_letters(self, parsed_structure, sasa_map):
        from pipeline.ppi_network import _predict_interface
        rns, letters = _predict_interface(parsed_structure, sasa_map, "inhibitory")
        assert len(rns) == len(letters)

    def test_sequence_fallback_works(self):
        from pipeline.ppi_network import _sequence_interface
        rns, letters = _sequence_interface("MRKDEFILW")
        assert len(rns) > 0
        assert len(rns) == len(letters)


# ── Unit tests: PPIResult ─────────────────────────────────────────────────────

class TestPPIResult:

    def _make_result(self, parsed_structure, sasa_map):
        from pipeline.ppi_network import _build_partner, PPIResult
        partners = [
            _build_partner(raw, parsed_structure, sasa_map, "MEESA")
            for raw in SAMPLE_RAW_PARTNERS
        ]
        partners.sort(key=lambda p: p.combined_score, reverse=True)
        high = [p for p in partners if p.combined_score >= 700]
        return PPIResult(
            uniprot_id="P00000",
            string_id="9606.ENSP00000269305",
            n_partners=len(partners),
            partners=partners,
            high_confidence=high,
            n_high=len(high),
            n_medium=sum(1 for p in partners if 400 <= p.combined_score < 700),
            api_available=True,
            top_partner=partners[0].partner_name if partners else "",
        )

    def test_result_construction(self, parsed_structure, sasa_map):
        from pipeline.ppi_network import PPIResult
        result = self._make_result(parsed_structure, sasa_map)
        assert result.uniprot_id == "P00000"
        assert result.n_partners == 4

    def test_partners_sorted_by_score(self, parsed_structure, sasa_map):
        result = self._make_result(parsed_structure, sasa_map)
        scores = [p.combined_score for p in result.partners]
        assert scores == sorted(scores, reverse=True)

    def test_high_confidence_subset(self, parsed_structure, sasa_map):
        result = self._make_result(parsed_structure, sasa_map)
        for p in result.high_confidence:
            assert p.combined_score >= 700

    def test_top_partner_is_mdm2(self, parsed_structure, sasa_map):
        result = self._make_result(parsed_structure, sasa_map)
        assert result.top_partner == "MDM2"

    def test_summary_string(self, parsed_structure, sasa_map):
        result = self._make_result(parsed_structure, sasa_map)
        s = result.summary()
        assert "P00000" in s
        assert "PPI" in s

    def test_to_dict_serialisable(self, parsed_structure, sasa_map):
        result = self._make_result(parsed_structure, sasa_map)
        json.dumps(result.to_dict())

    def test_to_json_writes_file(self, parsed_structure, sasa_map, tmp_path):
        result = self._make_result(parsed_structure, sasa_map)
        out = tmp_path / "ppi.json"
        result.to_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["uniprot_id"] == "P00000"
        assert data["n_partners"] == 4