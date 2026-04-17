"""
tests/test_11_foldseek.py
──────────────────────────
Tests for Module 11 — Foldseek structural similarity search.

All API tests are offline — no network calls in the test suite.

Run with:
    python -m pytest tests/test_11_foldseek.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest


# ── Fixtures ───────────────────────────────────────────────────────────────────

MINIMAL_PDB = """\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 85.00           C
ATOM      2  CA  ARG A   2       4.000   5.000   6.000  1.00 85.00           C
ATOM      3  CA  ASP A   3       7.000   8.000   9.000  1.00 85.00           C
END
"""

SAMPLE_RAW_HITS = [
    {
        "target": "1TUP",
        "taxName": "Cellular tumor antigen p53",
        "tmscore": 0.95,
        "rmsd": 0.8,
        "seqId": 0.99,
        "qCov": 0.98,
        "eval": 0.0,
        "_db": "pdb100",
    },
    {
        "target": "2OCJ",
        "taxName": "p53 DNA-binding domain",
        "tmscore": 0.85,
        "rmsd": 1.2,
        "seqId": 0.45,
        "qCov": 0.90,
        "eval": 1e-50,
        "_db": "pdb100",
    },
    {
        "target": "AF-Q8WUF5-F1",
        "taxName": "Tumour protein p53",
        "tmscore": 0.72,
        "rmsd": 2.1,
        "seqId": 0.15,
        "qCov": 0.80,
        "eval": 1e-20,
        "_db": "afdb50",
    },
    {
        "target": "3ABC",
        "taxName": "Unrelated protein",
        "tmscore": 0.35,
        "rmsd": 5.0,
        "seqId": 0.10,
        "qCov": 0.50,
        "eval": 0.1,
        "_db": "pdb100",
    },
]

SAMPLE_HOMOLOGY = {
    "protein_families": ["P53 DNA-binding domain", "P53 tetramerisation motif"],
    "blast_hits": [
        {
            "accession": "P04637",
            "description": "Cellular tumor antigen p53 [Homo sapiens]",
            "identity_pct": 100.0,
            "coverage_pct": 100.0,
            "e_value": 0.0,
            "function_text": "Acts as a tumor suppressor.",
            "reviewed": True,
            "evidence_weight": 3.0,
        },
        {
            "accession": "P02340",
            "description": "Cellular tumor antigen p53 [Mus musculus]",
            "identity_pct": 77.0,
            "coverage_pct": 95.0,
            "e_value": 1e-150,
            "function_text": "Mouse p53.",
            "reviewed": True,
            "evidence_weight": 3.0,
        },
    ],
}


@pytest.fixture
def pdb_file(tmp_path: Path) -> Path:
    p = tmp_path / "P00000.pdb"
    p.write_text(MINIMAL_PDB)
    return p


@pytest.fixture
def homology_json(tmp_path: Path) -> Path:
    p = tmp_path / "intermediate" / "P00000_homology.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(SAMPLE_HOMOLOGY))
    return p


# ── Unit tests: hit parsing ────────────────────────────────────────────────────

class TestHitParsing:

    def test_parses_tmscore(self):
        from pipeline.foldseek import _parse_hits
        hits = _parse_hits(SAMPLE_RAW_HITS)
        assert hits[0].tmscore == 0.95

    def test_parses_rmsd(self):
        from pipeline.foldseek import _parse_hits
        hits = _parse_hits(SAMPLE_RAW_HITS)
        assert hits[0].rmsd == 0.8

    def test_parses_seq_identity(self):
        from pipeline.foldseek import _parse_hits
        hits = _parse_hits(SAMPLE_RAW_HITS)
        assert hits[0].seq_identity == 0.99

    def test_same_fold_flagged(self):
        from pipeline.foldseek import _parse_hits
        hits = _parse_hits(SAMPLE_RAW_HITS)
        same_fold = [h for h in hits if h.is_same_fold]
        assert len(same_fold) == 3   # tmscore >= 0.5: 0.95, 0.85, 0.72

    def test_low_tmscore_not_same_fold(self):
        from pipeline.foldseek import _parse_hits
        hits = _parse_hits(SAMPLE_RAW_HITS)
        low = [h for h in hits if h.tmscore < 0.5]
        assert all(not h.is_same_fold for h in low)

    def test_sorted_by_tmscore(self):
        from pipeline.foldseek import _parse_hits
        hits = _parse_hits(SAMPLE_RAW_HITS)
        scores = [h.tmscore for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_database_preserved(self):
        from pipeline.foldseek import _parse_hits
        hits = _parse_hits(SAMPLE_RAW_HITS)
        dbs = {h.database for h in hits}
        assert "pdb100" in dbs
        assert "afdb50" in dbs

    def test_empty_hits_returns_empty(self):
        from pipeline.foldseek import _parse_hits
        hits = _parse_hits([])
        assert hits == []

    def test_hit_to_dict_serialisable(self):
        from pipeline.foldseek import _parse_hits
        hits = _parse_hits(SAMPLE_RAW_HITS)
        for hit in hits:
            json.dumps(hit.to_dict())


# ── Unit tests: function inference ────────────────────────────────────────────

class TestFunctionInference:

    def test_infers_from_descriptions(self):
        from pipeline.foldseek import _parse_hits, _infer_functions
        hits = _parse_hits(SAMPLE_RAW_HITS)
        same_fold = [h for h in hits if h.is_same_fold]
        fns = _infer_functions(same_fold)
        assert len(fns) > 0

    def test_deduplicates_functions(self):
        from pipeline.foldseek import _parse_hits, _infer_functions
        hits = _parse_hits(SAMPLE_RAW_HITS)
        same_fold = [h for h in hits if h.is_same_fold]
        fns = _infer_functions(same_fold)
        assert len(fns) == len(set(fns))

    def test_max_five_functions(self):
        from pipeline.foldseek import _parse_hits, _infer_functions
        hits = _parse_hits(SAMPLE_RAW_HITS * 10)
        same_fold = [h for h in hits if h.is_same_fold]
        fns = _infer_functions(same_fold)
        assert len(fns) <= 5


# ── Unit tests: FoldseekResult ────────────────────────────────────────────────

class TestFoldseekResult:

    def test_result_construction(self):
        from pipeline.foldseek import FoldseekResult, _parse_hits
        hits = _parse_hits(SAMPLE_RAW_HITS)
        same_fold = [h for h in hits if h.is_same_fold]
        result = FoldseekResult(
            uniprot_id="P00000",
            pdb_path="/tmp/test.pdb",
            n_hits=len(hits),
            hits=hits,
            same_fold_hits=same_fold,
            n_same_fold=len(same_fold),
            top_tmscore=hits[0].tmscore if hits else 0.0,
            api_available=True,
        )
        assert result.uniprot_id == "P00000"
        assert result.n_hits == len(hits)
        assert result.n_same_fold == len(same_fold)

    def test_summary_string(self):
        from pipeline.foldseek import FoldseekResult, _parse_hits
        hits = _parse_hits(SAMPLE_RAW_HITS)
        result = FoldseekResult(
            uniprot_id="P00000",
            pdb_path="/tmp/test.pdb",
            n_hits=len(hits),
            hits=hits,
            same_fold_hits=[h for h in hits if h.is_same_fold],
            n_same_fold=sum(1 for h in hits if h.is_same_fold),
            api_available=True,
        )
        s = result.summary()
        assert "P00000" in s
        assert "Foldseek" in s

    def test_to_dict_serialisable(self):
        from pipeline.foldseek import FoldseekResult, _parse_hits
        hits = _parse_hits(SAMPLE_RAW_HITS)
        result = FoldseekResult(
            uniprot_id="P00000",
            pdb_path="/tmp/test.pdb",
            hits=hits,
            n_hits=len(hits),
        )
        json.dumps(result.to_dict())

    def test_to_json_writes_file(self, tmp_path):
        from pipeline.foldseek import FoldseekResult, _parse_hits
        hits = _parse_hits(SAMPLE_RAW_HITS)
        result = FoldseekResult(
            uniprot_id="P00000",
            pdb_path="/tmp/test.pdb",
            hits=hits,
            n_hits=len(hits),
        )
        out = tmp_path / "foldseek.json"
        result.to_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["uniprot_id"] == "P00000"

    def test_novel_hits_low_seq_identity(self):
        from pipeline.foldseek import FoldseekResult, _parse_hits
        hits = _parse_hits(SAMPLE_RAW_HITS)
        same_fold = [h for h in hits if h.is_same_fold]
        novel     = [h for h in same_fold if h.seq_identity < 0.3]
        result = FoldseekResult(
            uniprot_id="P00000",
            pdb_path="/tmp/test.pdb",
            same_fold_hits=same_fold,
            novel_hits=novel,
        )
        for h in result.novel_hits:
            assert h.seq_identity < 0.3


# ── Unit tests: fallback ──────────────────────────────────────────────────────

class TestFallback:

    def test_fallback_uses_blast_hits(self, tmp_path, monkeypatch):
        from pipeline.foldseek import FoldseekResult, _apply_fallback
        import pipeline.foldseek as fs_module

        # Write homology JSON to intermediate dir
        inter_dir = tmp_path / "intermediate"
        inter_dir.mkdir()
        hom_path = inter_dir / "P00000_homology.json"
        hom_path.write_text(json.dumps(SAMPLE_HOMOLOGY))

        # Patch cfg paths
        monkeypatch.setattr(
            fs_module.cfg, "_data",
            {**fs_module.cfg._data, "paths": {
                **fs_module.cfg._data.get("paths", {}),
                "intermediate": str(inter_dir),
            }}
        )

        result = FoldseekResult(uniprot_id="P00000", pdb_path="/tmp/t.pdb")
        _apply_fallback(result, "P00000")
        assert result.n_hits > 0

    def test_fallback_result_serialisable(self, tmp_path, monkeypatch):
        from pipeline.foldseek import FoldseekResult, _apply_fallback
        import pipeline.foldseek as fs_module

        inter_dir = tmp_path / "intermediate"
        inter_dir.mkdir()
        hom_path = inter_dir / "P00000_homology.json"
        hom_path.write_text(json.dumps(SAMPLE_HOMOLOGY))

        monkeypatch.setattr(
            fs_module.cfg, "_data",
            {**fs_module.cfg._data, "paths": {
                **fs_module.cfg._data.get("paths", {}),
                "intermediate": str(inter_dir),
            }}
        )

        result = FoldseekResult(uniprot_id="P00000", pdb_path="/tmp/t.pdb")
        _apply_fallback(result, "P00000")
        json.dumps(result.to_dict())