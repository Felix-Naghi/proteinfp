"""
tests/test_01_fetch_structure.py
─────────────────────────────────
Tests for Module 01.

Run with:
    pytest tests/test_01_fetch_structure.py -v

Tests are layered:
  1. Unit tests — test the parser and helpers with local fixture data (no network)
  2. Integration test — hits the real AFDB (requires internet; skipped in CI)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# ── Fixtures ───────────────────────────────────────────────────────────────────

# Minimal valid AFDB-style PDB content (real TP53 residues, fake coordinates)
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
END
"""

KNOWN_LOW_PLDDT_RESIDUE = 3    # SER A 3 has pLDDT 45.20 — should be flagged disordered


@pytest.fixture
def pdb_file(tmp_path: Path) -> Path:
    """Write the minimal PDB to a temp file and return its path."""
    p = tmp_path / "P00000.pdb"
    p.write_text(MINIMAL_PDB)
    return p


# ── Unit tests: PDB parser ─────────────────────────────────────────────────────

class TestPdbParser:

    def test_parse_returns_correct_length(self, pdb_file):
        from utils.pdb_parser import parse_pdb
        result = parse_pdb(pdb_file, "P00000", plddt_threshold=70.0)
        assert result.length == 4, "Expected 4 residues in minimal PDB"

    def test_sequence_correct(self, pdb_file):
        from utils.pdb_parser import parse_pdb
        result = parse_pdb(pdb_file, "P00000")
        assert result.sequence == "MESA", f"Expected MESA, got {result.sequence}"

    def test_plddt_extraction(self, pdb_file):
        from utils.pdb_parser import parse_pdb
        result = parse_pdb(pdb_file, "P00000", plddt_threshold=70.0)
        # SER (index 2) should have pLDDT 45.20
        ser_res = result.residues[2]
        assert ser_res.one_letter == "S"
        assert abs(ser_res.plddt - 45.20) < 0.01

    def test_disordered_flagging(self, pdb_file):
        from utils.pdb_parser import parse_pdb
        result = parse_pdb(pdb_file, "P00000", plddt_threshold=70.0)
        # SER at position 3 should be disordered
        ser_res = result.residues[2]
        assert ser_res.is_disordered is True

    def test_high_confidence_not_disordered(self, pdb_file):
        from utils.pdb_parser import parse_pdb
        result = parse_pdb(pdb_file, "P00000", plddt_threshold=70.0)
        # MET at position 1 (pLDDT 92.15) should NOT be disordered
        met_res = result.residues[0]
        assert met_res.is_disordered is False

    def test_missing_file_raises(self, tmp_path):
        from utils.pdb_parser import parse_pdb
        with pytest.raises(FileNotFoundError):
            parse_pdb(tmp_path / "nonexistent.pdb", "P00000")

    def test_hydrophobicity_assigned(self, pdb_file):
        from utils.pdb_parser import parse_pdb, HYDROPHOBICITY
        result = parse_pdb(pdb_file, "P00000")
        for res in result.residues:
            expected = HYDROPHOBICITY.get(res.one_letter, 0.0)
            assert abs(res.hydrophobicity - expected) < 0.001

    def test_to_dict_serialisable(self, pdb_file):
        from utils.pdb_parser import parse_pdb
        result = parse_pdb(pdb_file, "P00000")
        d = result.to_dict()
        # Must be JSON-serialisable
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_to_json_writes_file(self, pdb_file, tmp_path):
        from utils.pdb_parser import parse_pdb
        result = parse_pdb(pdb_file, "P00000")
        out = tmp_path / "out.json"
        result.to_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["uniprot_id"] == "P00000"
        assert data["length"] == 4


# ── Unit tests: StructureResult ────────────────────────────────────────────────

class TestStructureResult:

    def test_summary_contains_key_fields(self, pdb_file):
        from utils.pdb_parser import parse_pdb
        from pipeline.fetch_structure import StructureResult
        parsed = parse_pdb(pdb_file, "P00000")
        result = StructureResult(
            uniprot_id="P00000",
            gene_name="TEST",
            protein_name="Test protein",
            organism="Homo sapiens",
            sequence=parsed.sequence,
            length=parsed.length,
            pdb_path=str(pdb_file),
            json_path="/tmp/out.json",
            mean_plddt=parsed.mean_plddt,
            high_conf_frac=parsed.high_conf_fraction,
            n_disordered=parsed.n_disordered,
            disordered_regions=parsed.disordered_regions,
            afdb_version="4",
            uniprot_reviewed=True,
            parsed=parsed,
        )
        summary = result.summary()
        assert "P00000" in summary
        assert "TEST" in summary
        assert "Swiss-Prot" in summary


# ── Integration tests (require network; skip in offline environments) ──────────

@pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("OFFLINE") == "true",
    reason="Integration test skipped: set OFFLINE=true to skip, or running in CI"
)
class TestFetchStructureIntegration:

    def test_fetch_tp53(self, tmp_path, monkeypatch):
        """
        Full integration test against real AFDB + UniProt APIs.
        Uses TP53 (P04637) — one of the most well-characterised human proteins.
        This test confirms the network calls, download, and parsing all work end-to-end.
        """
        import yaml
        from utils import config as cfg_module

        # Redirect data directories to temp path so we don't pollute the project
        fake_cfg = {
            "paths": {
                "structures":   str(tmp_path / "structures"),
                "intermediate": str(tmp_path / "intermediate"),
                "reports":      str(tmp_path / "reports"),
                "input":        str(tmp_path / "input"),
                "logs":         str(tmp_path / "logs"),
            },
            "afdb": {
                "base_url":      "https://alphafold.ebi.ac.uk/api",
                "structure_url": "https://alphafold.ebi.ac.uk/files",
                "model_version": 4,
                "plddt_threshold": 70.0,
                "max_retries":   3,
                "retry_delay_sec": 1.0,
                "timeout_sec":   30,
            },
            "uniprot": {
                "base_url":    "https://rest.uniprot.org/uniprotkb",
                "timeout_sec": 20,
            },
            "logging": {"level": "WARNING", "log_to_file": False},
        }

        # Patch the global config to point to temp dirs
        monkeypatch.setattr(cfg_module, "cfg", cfg_module._Config(fake_cfg))

        from pipeline.fetch_structure import fetch_structure
        result = fetch_structure("P04637")

        assert result.uniprot_id == "P04637"
        assert result.gene_name.upper() in ("TP53", "P53")
        assert result.length > 0
        assert result.mean_plddt > 0
        assert Path(result.pdb_path).exists()
        assert Path(result.json_path).exists()
        assert result.sequence.startswith("M")    # TP53 starts with Met

    def test_invalid_uniprot_raises(self, tmp_path):
        """Requesting a non-existent UniProt ID should raise ValueError."""
        from pipeline.fetch_structure import fetch_structure
        with pytest.raises(ValueError, match="not found in AlphaFold DB"):
            fetch_structure("XXXXXXXX")
