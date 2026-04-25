"""
tests/test_14_molecular_dynamics.py
────────────────────────────────────
Tests for Module 14 — Molecular dynamics.

Offline tests use mocked OpenMM and synthetic trajectories so the test suite
stays fast and runnable without a GPU or internet. The Integration test runs
a real 100 ps implicit-solvent simulation and is gated behind the
`-k Integration` selector, matching the convention used by the other modules.

Run with:
    python -m pytest tests/test_14_molecular_dynamics.py -v
    python -m pytest tests/test_14_molecular_dynamics.py -v -k "Integration"
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


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
def fake_traj():
    """Synthetic Cα trajectory: (n_frames=20, n_ca=10, 3) Å."""
    rng = np.random.default_rng(42)
    base = np.array([
        [ 0, 0, 0], [ 4, 0, 0], [ 8, 0, 0], [ 0, 4, 0], [ 4, 4, 0],
        [ 8, 4, 0], [ 0, 8, 0], [ 4, 8, 0], [ 8, 8, 0], [ 4, 4, 4],
    ], dtype=np.float32)
    # Add per-residue noise — residues 0–4 stiff, residues 5–9 floppy
    noise = rng.normal(0, 0.3, size=(20, 10, 3)).astype(np.float32)
    noise[:, 5:, :] *= 5.0
    return base[None] + noise


@pytest.fixture
def sample_active_data():
    return {
        "active_residues": [
            {"residue_number": 1, "confidence": "HIGH"},
            {"residue_number": 2, "confidence": "HIGH"},
        ],
        "catalytic_motifs": [
            {"motif_type": "zinc_binding", "residue_numbers": [1, 2, 3]}
        ],
    }


@pytest.fixture
def sample_pocket_data():
    return {
        "pockets": [
            {"pocket_id": "P1", "lining_residues": [5, 6, 7, 8]},
        ]
    }


@pytest.fixture
def sample_allo_data():
    return {
        "allosteric_sites": [
            {"site_id": "A1", "residue_numbers": [9, 10]},
        ]
    }


# ── Pure-function tests (no OpenMM needed) ────────────────────────────────────

class TestTrajectoryDescriptors:

    def test_radius_of_gyration_shape_and_sign(self, fake_traj):
        from pipeline.molecular_dynamics import _radius_of_gyration
        rg = _radius_of_gyration(fake_traj)
        assert rg.shape == (20,)
        assert np.all(rg > 0)

    def test_rmsd_first_frame_is_zero(self, fake_traj):
        from pipeline.molecular_dynamics import _rmsd_to_reference
        rmsd = _rmsd_to_reference(fake_traj)
        assert rmsd.shape == (20,)
        assert rmsd[0] == pytest.approx(0.0, abs=1e-6)
        assert np.all(rmsd[1:] >= 0)

    def test_rmsf_per_residue_length_matches(self, fake_traj):
        from pipeline.molecular_dynamics import _rmsf_per_residue
        rmsf = _rmsf_per_residue(fake_traj)
        assert rmsf.shape == (10,)
        assert np.all(rmsf >= 0)

    def test_rmsf_detects_floppy_residues(self, fake_traj):
        """Residues 5–9 were given 5× more noise — RMSF must reflect that."""
        from pipeline.molecular_dynamics import _rmsf_per_residue
        rmsf = _rmsf_per_residue(fake_traj)
        assert rmsf[5:].mean() > 3.0 * rmsf[:5].mean()


class TestSiteAnalysis:

    def test_active_site_dynamics_extracted(
        self, pdb_file, fake_traj, sample_active_data
    ):
        from pipeline.molecular_dynamics import _analyze_sites
        from utils.pdb_parser import parse_pdb
        struct = parse_pdb(pdb_file, "P00000")
        rmsf = np.array([0.5, 0.6, 0.7, 0.5, 0.5, 3.0, 3.5, 3.2, 3.8, 4.0])

        sites = _analyze_sites(
            structure    = struct,
            rmsf_per_res = rmsf,
            coords_array = fake_traj,
            active_data  = sample_active_data,
            pocket_data  = None,
            allo_data    = None,
        )
        assert any(s.site_type == "active" for s in sites)
        active = [s for s in sites if s.site_type == "active"]
        assert active[0].mean_rmsf < 1.0           # rigid active site
        assert 0.0 <= active[0].stability_score <= 1.0

    def test_pocket_dynamics_separate_from_active(
        self, pdb_file, fake_traj, sample_pocket_data
    ):
        from pipeline.molecular_dynamics import _analyze_sites
        from utils.pdb_parser import parse_pdb
        struct = parse_pdb(pdb_file, "P00000")
        rmsf = np.array([0.5, 0.6, 0.7, 0.5, 0.5, 3.0, 3.5, 3.2, 3.8, 4.0])

        sites = _analyze_sites(
            structure    = struct,
            rmsf_per_res = rmsf,
            coords_array = fake_traj,
            active_data  = None,
            pocket_data  = sample_pocket_data,
            allo_data    = None,
        )
        pockets = [s for s in sites if s.site_type == "pocket"]
        assert len(pockets) == 1
        assert pockets[0].mean_rmsf > 2.0          # floppy pocket
        assert pockets[0].pocket_vol_drift > 0

    def test_allosteric_sites_handled(
        self, pdb_file, fake_traj, sample_allo_data
    ):
        from pipeline.molecular_dynamics import _analyze_sites
        from utils.pdb_parser import parse_pdb
        struct = parse_pdb(pdb_file, "P00000")
        rmsf = np.ones(10)

        sites = _analyze_sites(
            structure    = struct,
            rmsf_per_res = rmsf,
            coords_array = fake_traj,
            active_data  = None,
            pocket_data  = None,
            allo_data    = sample_allo_data,
        )
        assert any(s.site_type == "allo" for s in sites)


class TestResultSerialisation:

    def test_to_json_roundtrip(self, tmp_path):
        from pipeline.molecular_dynamics import MDResult, SiteDynamics
        import json

        r = MDResult(
            uniprot_id="P00000",
            n_atoms=100,
            n_residues=10,
            forcefield="amber14",
            production_ns=1.0,
            n_frames=20,
            site_dynamics=[SiteDynamics(
                site_id="P1", site_type="pocket",
                residue_numbers=[5, 6, 7],
                mean_rmsf=2.5, max_rmsf=4.0,
            )],
        )
        out = tmp_path / "r.json"
        r.to_json(out)
        loaded = json.loads(out.read_text())
        assert loaded["uniprot_id"] == "P00000"
        assert loaded["site_dynamics"][0]["site_id"] == "P1"

    def test_summary_contains_uniprot(self):
        from pipeline.molecular_dynamics import MDResult
        r = MDResult(uniprot_id="P12345", n_residues=10, production_ns=1.0)
        assert "P12345" in r.summary()


class TestPDBNotFound:

    def test_missing_pdb_raises(self, tmp_path):
        from pipeline.molecular_dynamics import run_md
        with pytest.raises(FileNotFoundError):
            run_md("PNOEXIST", pdb_path=tmp_path / "nope.pdb")


# ── Integration test (gated, requires OpenMM + a few minutes) ─────────────────

@pytest.mark.integration
class TestIntegration:
    """
    Real 100 ps implicit-solvent run on a tiny structure.
    Only runs with: pytest -k "Integration"
    """

    def test_short_implicit_run(self, pdb_file):
        pytest.importorskip("openmm")
        from pipeline.molecular_dynamics import run_md
        result = run_md(
            uniprot_id    = "P00000",
            pdb_path      = pdb_file,
            production_ns = 0.1,           # 100 ps
            implicit      = True,
            use_gpu       = False,
        )
        assert result.n_frames > 0
        assert result.mean_rg > 0
        assert len(result.rmsf_per_residue) == result.n_residues