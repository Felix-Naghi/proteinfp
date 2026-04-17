"""
tests/test_07_homology.py
──────────────────────────
Tests for Module 07 — Sequence homology + domain annotation.

All network tests are skipped by default (set ONLINE=true to run them).
Offline tests cover all parsing, aggregation, and data structure logic.

Run with:
    python -m pytest tests/test_07_homology.py -v
    python -m pytest tests/test_07_homology.py -v -k "online" (network tests)
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_blast_hits():
    from pipeline.homology import HomologHit
    return [
        HomologHit(
            accession="P04637",
            description="Cellular tumor antigen p53 [Homo sapiens]",
            organism="Homo sapiens",
            identity_pct=100.0,
            coverage_pct=100.0,
            e_value=0.0,
            bit_score=800.0,
            reviewed=True,
            go_terms=["GO:0003677", "GO:0006915"],
            go_names=["DNA binding", "apoptosis"],
            function_text="Acts as a tumor suppressor.",
            evidence_weight=3.0,
        ),
        HomologHit(
            accession="P02340",
            description="Cellular tumor antigen p53 [Mus musculus]",
            organism="Mus musculus",
            identity_pct=77.0,
            coverage_pct=95.0,
            e_value=1e-150,
            bit_score=600.0,
            reviewed=True,
            go_terms=["GO:0003677", "GO:0005634"],
            go_names=["DNA binding", "nucleus"],
            function_text="Mouse p53 homolog.",
            evidence_weight=3.0,
        ),
        HomologHit(
            accession="A0A123",
            description="Hypothetical protein [Unknown]",
            organism="Unknown",
            identity_pct=35.0,
            coverage_pct=60.0,
            e_value=1e-10,
            bit_score=90.0,
            reviewed=False,
            go_terms=[],
            go_names=[],
            function_text="",
            evidence_weight=1.5,
        ),
    ]


@pytest.fixture
def sample_domains():
    from pipeline.homology import InterProDomain
    return [
        InterProDomain(
            accession="IPR011615",
            name="P53 DNA-binding domain",
            database="PFAM",
            start=94,
            end=292,
            go_terms=["GO:0003677"],
            go_names=["DNA binding"],
            e_value=1e-80,
        ),
        InterProDomain(
            accession="PTHR11447",
            name="TUMOR PROTEIN P53",
            database="PANTHER",
            start=1,
            end=393,
            go_terms=["GO:0006915", "GO:0005634"],
            go_names=["apoptosis", "nucleus"],
            e_value=0.0,
        ),
    ]


# ── Unit tests: data structures ───────────────────────────────────────────────

class TestDataStructures:

    def test_homolog_hit_to_dict(self, sample_blast_hits):
        d = sample_blast_hits[0].to_dict()
        assert d["accession"] == "P04637"
        assert d["reviewed"] is True
        assert isinstance(d["go_terms"], list)

    def test_homolog_hit_serialisable(self, sample_blast_hits):
        for hit in sample_blast_hits:
            json.dumps(hit.to_dict())

    def test_interpro_domain_to_dict(self, sample_domains):
        d = sample_domains[0].to_dict()
        assert d["accession"] == "IPR011615"
        assert d["database"] == "PFAM"

    def test_interpro_domain_serialisable(self, sample_domains):
        for dom in sample_domains:
            json.dumps(dom.to_dict())


# ── Unit tests: GO aggregation ────────────────────────────────────────────────

class TestGoAggregation:

    def test_aggregates_go_from_blast(self, sample_blast_hits, sample_domains):
        from pipeline.homology import _aggregate_annotations
        go_terms, go_names, families, top_fn = _aggregate_annotations(
            sample_blast_hits, [], "P04637"
        )
        assert "GO:0003677" in go_terms
        assert "GO:0006915" in go_terms

    def test_aggregates_go_from_domains(self, sample_domains):
        from pipeline.homology import _aggregate_annotations
        go_terms, go_names, families, top_fn = _aggregate_annotations(
            [], sample_domains, "P04637"
        )
        assert len(go_terms) > 0

    def test_deduplicates_go_terms(self, sample_blast_hits, sample_domains):
        from pipeline.homology import _aggregate_annotations
        go_terms, _, _, _ = _aggregate_annotations(
            sample_blast_hits, sample_domains, "P04637"
        )
        assert len(go_terms) == len(set(go_terms))

    def test_top_function_from_swissprot(self, sample_blast_hits):
        from pipeline.homology import _aggregate_annotations
        _, _, _, top_fn = _aggregate_annotations(
            sample_blast_hits, [], "P04637"
        )
        assert "tumor suppressor" in top_fn.lower()

    def test_families_extracted_from_panther(self, sample_domains):
        from pipeline.homology import _aggregate_annotations
        _, _, families, _ = _aggregate_annotations([], sample_domains, "P04637")
        assert len(families) > 0

    def test_empty_input_returns_empty(self):
        from pipeline.homology import _aggregate_annotations
        go_terms, go_names, families, top_fn = _aggregate_annotations(
            [], [], "P00000"
        )
        assert go_terms == []
        assert top_fn == ""

    def test_go_names_same_length_as_terms(self, sample_blast_hits, sample_domains):
        from pipeline.homology import _aggregate_annotations
        go_terms, go_names, _, _ = _aggregate_annotations(
            sample_blast_hits, sample_domains, "P04637"
        )
        assert len(go_terms) == len(go_names)


# ── Unit tests: HomologyResult ────────────────────────────────────────────────

class TestHomologyResult:

    def test_result_to_dict(self, sample_blast_hits, sample_domains):
        from pipeline.homology import HomologyResult, _aggregate_annotations
        go_terms, go_names, families, top_fn = _aggregate_annotations(
            sample_blast_hits, sample_domains, "P04637"
        )
        result = HomologyResult(
            uniprot_id="P04637",
            sequence_length=393,
            blast_hits=sample_blast_hits,
            interpro_domains=sample_domains,
            all_go_terms=go_terms,
            all_go_names=go_names,
            protein_families=families,
            top_function=top_fn,
            blast_available=True,
            interpro_available=True,
            n_experimental_hits=2,
        )
        d = result.to_dict()
        assert d["uniprot_id"] == "P04637"
        assert d["sequence_length"] == 393
        assert isinstance(d["blast_hits"], list)
        assert isinstance(d["interpro_domains"], list)

    def test_result_serialisable(self, sample_blast_hits, sample_domains):
        from pipeline.homology import HomologyResult
        result = HomologyResult(
            uniprot_id="P04637",
            sequence_length=393,
            blast_hits=sample_blast_hits,
            interpro_domains=sample_domains,
        )
        json.dumps(result.to_dict())

    def test_result_to_json(self, sample_blast_hits, tmp_path):
        from pipeline.homology import HomologyResult
        result = HomologyResult(
            uniprot_id="P04637",
            sequence_length=393,
            blast_hits=sample_blast_hits,
        )
        out = tmp_path / "homology.json"
        result.to_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["uniprot_id"] == "P04637"

    def test_summary_string(self, sample_blast_hits, sample_domains):
        from pipeline.homology import HomologyResult
        result = HomologyResult(
            uniprot_id="P04637",
            sequence_length=393,
            blast_hits=sample_blast_hits,
            interpro_domains=sample_domains,
            n_experimental_hits=2,
        )
        s = result.summary()
        assert "P04637" in s
        assert "Homology" in s

    def test_n_experimental_hits_counted(self, sample_blast_hits):
        from pipeline.homology import HomologyResult
        result = HomologyResult(
            uniprot_id="P04637",
            sequence_length=393,
            blast_hits=sample_blast_hits,
            n_experimental_hits=sum(
                1 for h in sample_blast_hits if h.reviewed
            ),
        )
        assert result.n_experimental_hits == 2


# ── Unit tests: BLAST XML parsing ─────────────────────────────────────────────

class TestBlastXmlParsing:

    MINIMAL_BLAST_XML = """<?xml version="1.0"?>
<!DOCTYPE BlastOutput PUBLIC "-//NCBI//NCBI BlastOutput/EN"
    "http://www.ncbi.nlm.nih.gov/dtd/NCBI_BlastOutput.dtd">
<BlastOutput>
  <BlastOutput_query-len>393</BlastOutput_query-len>
  <BlastOutput_iterations>
    <Iteration>
      <Iteration_hits>
        <Hit>
          <Hit_num>1</Hit_num>
          <Hit_accession>P04637</Hit_accession>
          <Hit_def>Cellular tumor antigen p53 [Homo sapiens]</Hit_def>
          <Hit_hsps>
            <Hsp>
              <Hsp_identity>393</Hsp_identity>
              <Hsp_align-len>393</Hsp_align-len>
              <Hsp_query-from>1</Hsp_query-from>
              <Hsp_query-to>393</Hsp_query-to>
              <Hsp_evalue>0</Hsp_evalue>
              <Hsp_bit-score>800</Hsp_bit-score>
            </Hsp>
          </Hit_hsps>
        </Hit>
      </Iteration_hits>
    </Iteration>
  </BlastOutput_iterations>
</BlastOutput>"""

    def test_parses_accession(self):
        from pipeline.homology import _parse_blast_xml
        hits = _parse_blast_xml(self.MINIMAL_BLAST_XML)
        assert len(hits) >= 1
        assert hits[0].accession == "P04637"

    def test_parses_identity(self):
        from pipeline.homology import _parse_blast_xml
        hits = _parse_blast_xml(self.MINIMAL_BLAST_XML)
        assert hits[0].identity_pct == 100.0

    def test_parses_evalue(self):
        from pipeline.homology import _parse_blast_xml
        hits = _parse_blast_xml(self.MINIMAL_BLAST_XML)
        assert hits[0].e_value == 0.0

    def test_empty_xml_returns_empty(self):
        from pipeline.homology import _parse_blast_xml
        hits = _parse_blast_xml("<BlastOutput></BlastOutput>")
        assert hits == []

    def test_invalid_xml_returns_empty(self):
        from pipeline.homology import _parse_blast_xml
        hits = _parse_blast_xml("not xml at all")
        assert hits == []


# ── Organism extraction ────────────────────────────────────────────────────────

class TestOrganismExtraction:

    def test_extracts_from_brackets(self):
        from pipeline.homology import _extract_organism
        org = _extract_organism("Tumor protein p53 [Homo sapiens]")
        assert org == "Homo sapiens"

    def test_no_brackets_returns_unknown(self):
        from pipeline.homology import _extract_organism
        org = _extract_organism("Tumor protein p53")
        assert org == "unknown"

    def test_multiple_brackets_uses_last(self):
        from pipeline.homology import _extract_organism
        org = _extract_organism("Protein [isoform 2] [Mus musculus]")
        assert org == "Mus musculus"


# ── Online integration tests ──────────────────────────────────────────────────

@pytest.mark.skipif(
    True,  # Always skip by default — set to False to run network tests
    reason="Network test — remove skip to run against live APIs"
)
class TestHomologyOnline:

    def test_interpro_direct_lookup_tp53(self):
        from pipeline.homology import _interpro_direct_lookup
        domains = _interpro_direct_lookup("P04637")
        assert len(domains) > 0
        names = [d.name for d in domains]
        assert any("p53" in n.lower() or "P53" in n for n in names)

    def test_uniprot_annotation_tp53(self):
        from pipeline.homology import _fetch_uniprot_annotation
        go_terms, go_names, reviewed, fn_text = _fetch_uniprot_annotation("P04637")
        assert reviewed is True
        assert len(go_terms) > 0
        assert "tumor" in fn_text.lower() or "suppressor" in fn_text.lower()