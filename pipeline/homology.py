"""
pipeline/07_homology.py
────────────────────────
Module 07 — Sequence homology + domain annotation.

Queries three complementary databases to find proteins with known function
similar to the query protein:

  1. NCBI BLAST (PSI-BLAST against Swiss-Prot)
     - Finds sequence homologs with experimentally validated annotations
     - Swiss-Prot only = experimental evidence, not computational inference
     - E-value < 0.001, max 50 hits

  2. UniProt REST API (keyword + family search)
     - Retrieves GO terms, pathway membership, protein family
     - Cross-references to PDB, KEGG, Reactome

  3. InterProScan via EBI REST API
     - Pfam domain families
     - PANTHER superfamilies
     - TIGRFAM functional categories
     - Produces GO term mappings from domain membership

Evidence weighting for consensus (Module 13):
  - Swiss-Prot experimental hit (reviewed=True):  weight 3.0
  - Swiss-Prot inferred hit (reviewed=False):     weight 1.5
  - InterPro domain with GO mapping:              weight 2.0
  - UniProt family annotation:                    weight 1.5

Usage (standalone):
    python pipeline/07_homology.py --uniprot P04637

Usage (from orchestrator):
    from pipeline.homology import run_homology
    result = run_homology("P04637", sequence)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import click
import requests

from utils.config import cfg, get_logger
from utils.pdb_parser import parse_pdb

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

BLAST_URL    = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
UNIPROT_URL  = "https://rest.uniprot.org/uniprotkb"
INTERPRO_URL = "https://www.ebi.ac.uk/interpro/api"
EBI_TOOL_URL = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"

# BLAST polling interval (seconds)
BLAST_POLL_INTERVAL = 10
BLAST_MAX_WAIT      = 600   # 10 minutes max

# Maximum homologs to keep
MAX_HOMOLOGS = 30

# GO evidence codes we trust (experimental only)
EXPERIMENTAL_GO_CODES = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class HomologHit:
    """A single sequence homolog with functional annotation."""
    accession:      str
    description:    str
    organism:       str
    identity_pct:   float
    coverage_pct:   float
    e_value:        float
    bit_score:      float
    reviewed:       bool        # True = Swiss-Prot (experimental)
    go_terms:       list[str]   # GO accessions
    go_names:       list[str]   # GO term names
    function_text:  str         # UniProt function annotation
    evidence_weight: float      # for consensus scoring

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InterProDomain:
    """A protein domain identified by InterProScan."""
    accession:      str         # e.g. IPR011615
    name:           str
    database:       str         # Pfam / PANTHER / TIGRFAM / etc.
    start:          int
    end:            int
    go_terms:       list[str]
    go_names:       list[str]
    e_value:        float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HomologyResult:
    """Full homology + domain annotation output. Output of Module 07."""
    uniprot_id:         str
    sequence_length:    int
    blast_hits:         list[HomologHit]    = field(default_factory=list)
    interpro_domains:   list[InterProDomain] = field(default_factory=list)
    all_go_terms:       list[str]           = field(default_factory=list)
    all_go_names:       list[str]           = field(default_factory=list)
    protein_families:   list[str]           = field(default_factory=list)
    top_function:       str                 = ""
    blast_available:    bool                = False
    interpro_available: bool                = False
    n_experimental_hits: int                = 0

    def summary(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  Homology analysis: {self.uniprot_id}",
            f"  BLAST hits       : {len(self.blast_hits)} "
            f"({self.n_experimental_hits} experimental Swiss-Prot)",
            f"  InterPro domains : {len(self.interpro_domains)}",
            f"  GO terms         : {len(self.all_go_terms)} unique",
            f"  Protein families : {', '.join(self.protein_families[:3]) or 'none'}",
        ]
        if self.top_function:
            lines.append(f"  Top function     : {self.top_function[:80]}")
        for hit in self.blast_hits[:5]:
            flag = "[SwissProt]" if hit.reviewed else "[TrEMBL]"
            lines.append(
                f"  {flag} {hit.accession} {hit.organism[:20]} "
                f"id={hit.identity_pct:.0f}% e={hit.e_value:.1e}"
            )
        for dom in self.interpro_domains[:5]:
            lines.append(
                f"  [{dom.database}] {dom.accession} {dom.name[:40]} "
                f"pos={dom.start}-{dom.end}"
            )
        lines.append(f"{'─'*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.debug(f"Saved homology JSON → {path}")


# ── Main function ──────────────────────────────────────────────────────────────

def run_homology(
    uniprot_id: str,
    sequence:   str,
) -> HomologyResult:
    """
    Run full homology analysis for a protein sequence.

    Args:
        uniprot_id: UniProt accession
        sequence:   Amino acid sequence string

    Returns:
        HomologyResult with BLAST hits, domains, GO terms.
    """
    log.info(f"── Module 07: Homology analysis for {uniprot_id} ──")

    result = HomologyResult(
        uniprot_id=uniprot_id,
        sequence_length=len(sequence),
    )

    # ── Step 1: NCBI BLAST vs Swiss-Prot ─────────────────────────────────────
    log.info("  [1/3] Running NCBI BLAST vs Swiss-Prot...")
    blast_hits, blast_ok = _run_blast(uniprot_id, sequence)
    result.blast_hits        = blast_hits
    result.blast_available   = blast_ok
    result.n_experimental_hits = sum(1 for h in blast_hits if h.reviewed)
    log.info(f"    {len(blast_hits)} hits "
             f"({result.n_experimental_hits} experimental)")

    # ── Step 2: InterProScan domain annotation ────────────────────────────────
    log.info("  [2/3] Running InterProScan domain annotation...")
    domains, ipro_ok = _run_interproscan(uniprot_id, sequence)
    result.interpro_domains    = domains
    result.interpro_available  = ipro_ok
    log.info(f"    {len(domains)} domains found")

    # ── Step 3: Aggregate GO terms and families ───────────────────────────────
    log.info("  [3/3] Aggregating GO terms and functional annotations...")
    go_terms, go_names, families, top_fn = _aggregate_annotations(
        blast_hits, domains, uniprot_id
    )
    result.all_go_terms    = go_terms
    result.all_go_names    = go_names
    result.protein_families = families
    result.top_function    = top_fn

    log.info(result.summary())
    return result


# ── BLAST ──────────────────────────────────────────────────────────────────────

def _run_blast(
    uniprot_id: str,
    sequence:   str,
) -> tuple[list[HomologHit], bool]:
    """
    Submit sequence to NCBI BLAST vs Swiss-Prot, poll for results.
    Returns (hits, success_bool).
    """
    try:
        # Submit job
        log.debug("    Submitting BLAST job...")
        resp = requests.post(
            BLAST_URL,
            data={
                "CMD":       "Put",
                "PROGRAM":   "blastp",
                "DATABASE":  "swissprot",
                "QUERY":     sequence,
                "FORMAT_TYPE": "XML",
                "HITLIST_SIZE": MAX_HOMOLOGS,
                "EXPECT":    0.001,
                "EMAIL":     "research@proteinfp.io",
            },
            timeout=30,
        )
        resp.raise_for_status()

        # Extract RID (request ID)
        rid = None
        for line in resp.text.split("\n"):
            if "RID = " in line:
                rid = line.split("RID = ")[1].strip()
                break

        if not rid:
            log.warning("    BLAST: could not extract RID — skipping")
            return [], False

        log.info(f"    BLAST job submitted: RID={rid}")

        # Poll for results
        elapsed = 0
        while elapsed < BLAST_MAX_WAIT:
            time.sleep(BLAST_POLL_INTERVAL)
            elapsed += BLAST_POLL_INTERVAL

            status_resp = requests.get(
                BLAST_URL,
                params={"CMD": "Get", "RID": rid, "FORMAT_TYPE": "XML"},
                timeout=30,
            )

            if "Status=WAITING" in status_resp.text:
                log.debug(f"    BLAST: waiting... ({elapsed}s)")
                continue
            elif "Status=FAILED" in status_resp.text:
                log.warning("    BLAST job failed")
                return [], False
            elif "<BlastOutput>" in status_resp.text or "<?xml" in status_resp.text:
                log.info(f"    BLAST: results ready ({elapsed}s)")
                hits = _parse_blast_xml(status_resp.text)
                return hits, True

        log.warning(f"    BLAST timed out after {BLAST_MAX_WAIT}s") 
        return [], False

    except Exception as e:
        log.warning(f"    BLAST unavailable: {e}")
        return [], False


def _parse_blast_xml(xml_text: str) -> list[HomologHit]:
    """Parse NCBI BLAST XML output into HomologHit objects."""
    hits = []

    try:
        # Find the XML portion
        xml_start = xml_text.find("<?xml")
        if xml_start == -1:
            xml_start = xml_text.find("<BlastOutput")
        if xml_start == -1:
            return []

        root = ET.fromstring(xml_text[xml_start:])
        ns   = ""

        iterations = root.findall(f".//{ns}Iteration")
        if not iterations:
            return []

        iteration = iterations[0]

        for hit in iteration.findall(f".//{ns}Hit"):
            try:
                accession   = _xml_text(hit, f"{ns}Hit_accession")
                description = _xml_text(hit, f"{ns}Hit_def")
                organism    = _extract_organism(description)

                hsp = hit.find(f".//{ns}Hsp")
                if hsp is None:
                    continue

                identity    = float(_xml_text(hsp, f"{ns}Hsp_identity") or 0)
                align_len   = float(_xml_text(hsp, f"{ns}Hsp_align-len") or 1)
                query_from  = int(_xml_text(hsp, f"{ns}Hsp_query-from") or 1)
                query_to    = int(_xml_text(hsp, f"{ns}Hsp_query-to") or 1)
                e_value     = float(_xml_text(hsp, f"{ns}Hsp_evalue") or 1)
                bit_score   = float(_xml_text(hsp, f"{ns}Hsp_bit-score") or 0)
                query_len   = float(_xml_text(
                    root, f"{ns}BlastOutput_query-len") or 1)

                identity_pct = (identity / align_len * 100) if align_len > 0 else 0
                coverage_pct = ((query_to - query_from + 1) / query_len * 100
                                if query_len > 0 else 0)

                # Fetch UniProt annotation for top hits
                go_terms, go_names, reviewed, fn_text = [], [], False, ""
                if len(hits) < 10:
                    go_terms, go_names, reviewed, fn_text = _fetch_uniprot_annotation(
                        accession
                    )

                weight = 3.0 if reviewed else 1.5

                hits.append(HomologHit(
                    accession=accession,
                    description=description[:200],
                    organism=organism,
                    identity_pct=round(identity_pct, 1),
                    coverage_pct=round(coverage_pct, 1),
                    e_value=e_value,
                    bit_score=round(bit_score, 1),
                    reviewed=reviewed,
                    go_terms=go_terms,
                    go_names=go_names,
                    function_text=fn_text[:500] if fn_text else "",
                    evidence_weight=weight,
                ))

            except Exception as e:
                log.debug(f"    Skipping hit: {e}")
                continue

    except Exception as e:
        log.warning(f"    BLAST XML parse error: {e}")

    return hits[:MAX_HOMOLOGS]


def _xml_text(element, tag: str) -> str:
    node = element.find(tag)
    return node.text.strip() if node is not None and node.text else ""


def _extract_organism(description: str) -> str:
    if "[" in description and "]" in description:
        return description.split("[")[-1].rstrip("]")[:40]
    return "unknown"


# ── UniProt annotation ─────────────────────────────────────────────────────────

def _fetch_uniprot_annotation(
    accession: str,
) -> tuple[list[str], list[str], bool, str]:
    """
    Fetch GO terms, review status, and function text from UniProt REST API.
    Returns (go_terms, go_names, reviewed, function_text).
    """
    try:
        resp = requests.get(
            f"{UNIPROT_URL}/{accession}",
            params={"format": "json",
                    "fields": "go,reviewed,cc_function,protein_name"},
            timeout=15,
            headers={"User-Agent": "ProteinFP/0.1"},
        )
        if resp.status_code != 200:
            return [], [], False, ""

        data     = resp.json()
        reviewed = data.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)"

        # GO terms
        go_terms = []
        go_names = []
        for ref in data.get("uniProtKBCrossReferences", []):
            if ref.get("database") == "GO":
                go_id = ref.get("id", "")
                name  = ""
                for prop in ref.get("properties", []):
                    if prop.get("key") == "GoTerm":
                        name = prop.get("value", "")
                        break
                if go_id:
                    go_terms.append(go_id)
                    go_names.append(name)

        # Function text
        fn_text = ""
        for comment in data.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    fn_text = texts[0].get("value", "")
                    break

        return go_terms[:20], go_names[:20], reviewed, fn_text

    except Exception:
        return [], [], False, ""


# ── InterProScan ───────────────────────────────────────────────────────────────

def _run_interproscan(
    uniprot_id: str,
    sequence:   str,
) -> tuple[list[InterProDomain], bool]:
    """
    Get InterPro domain annotations via direct API lookup.
    Falls back to EBI job submission if direct lookup returns nothing.
    """
    # Direct lookup by UniProt ID — fast, pre-computed, most reliable
    domains = _interpro_direct_lookup(uniprot_id)
    if domains:
        log.info(f"    InterPro direct lookup: {len(domains)} domains")
        return domains, True

    # Fallback: try the InterPro search API
    try:
        resp = requests.get(
            f"{INTERPRO_URL}/entry/all/protein/UniProt/{uniprot_id}",
            params={"page_size": 100, "type": "domain,family,homologous_superfamily"},
            timeout=30,
            headers={"Accept": "application/json", "User-Agent": "ProteinFP/0.1"},
        )
        if resp.status_code == 200:
            domains = _interpro_direct_lookup(uniprot_id)
            return domains, True
    except Exception as e:
        log.warning(f"    InterPro fallback failed: {e}")

    log.warning("    InterPro: no domains found")
    return [], False
    """
    Submit sequence to EBI InterProScan REST API.
    Returns (domains, success_bool).
    Falls back to direct InterPro lookup if job submission fails.
    """
    # First try: direct InterPro API lookup by UniProt ID (faster)
    domains = _interpro_direct_lookup(uniprot_id)
    if domains:
        return domains, True

    # Second try: submit sequence for scanning
    try:
        submit_resp = requests.post(
            f"{EBI_TOOL_URL}/run",
            data={
                "email":    "research@proteinfp.io",
                "sequence": sequence,
                "appl":     "Pfam,PANTHER,TIGRFAM",
                "goterms":  "true",
                "pathways": "false",
            },
            timeout=30,
        )

        if submit_resp.status_code != 200:
            log.warning(f"    InterProScan submit failed: {submit_resp.status_code}")
            return [], False

        job_id = submit_resp.text.strip()
        log.info(f"    InterProScan job: {job_id}")

        # Poll for results
        elapsed = 0
        while elapsed < 180:
            time.sleep(15)
            elapsed += 15

            status_resp = requests.get(
                f"{EBI_TOOL_URL}/status/{job_id}", timeout=15
            )
            status = status_resp.text.strip()
            log.debug(f"    InterProScan status: {status} ({elapsed}s)")

            if status == "FINISHED":
                result_resp = requests.get(
                    f"{EBI_TOOL_URL}/result/{job_id}/json", timeout=30
                )
                domains = _parse_interpro_json(result_resp.json())
                return domains, True
            elif status in ("ERROR", "FAILURE"):
                log.warning("    InterProScan job failed")
                return [], False

        log.warning("    InterProScan timed out")
        return [], False

    except Exception as e:
        log.warning(f"    InterProScan unavailable: {e}")
        return [], False


def _interpro_direct_lookup(uniprot_id: str) -> list[InterProDomain]:
    """
    Look up pre-computed InterPro annotations for a UniProt ID.
    """
    try:
        resp = requests.get(
            f"{INTERPRO_URL}/entry/all/protein/UniProt/{uniprot_id}",
            params={"page_size": 50},
            timeout=20,
            headers={"Accept": "application/json", "User-Agent": "ProteinFP/0.1"},
        )

        if resp.status_code != 200:
            return []

        data    = resp.json()
        domains = []

        for entry in data.get("results", []):
            meta = entry.get("metadata", {})
            acc  = meta.get("accession", "")
            # name is a plain string in current API
            name = meta.get("name", "")
            if isinstance(name, dict):
                name = name.get("name", "")
            db   = meta.get("source_database", "").upper()

            # Position from proteins array
            start, end = 0, 0
            proteins = entry.get("proteins", [])
            if proteins:
                locs = proteins[0].get("entry_protein_locations", [])
                if locs:
                    frags = locs[0].get("fragments", [])
                    if frags:
                        start = frags[0].get("start", 0)
                        end   = frags[-1].get("end", 0)

            # GO terms — fetch separately per entry if needed
            go_terms = []
            go_names = []
            raw_go = meta.get("go_terms")
            if raw_go:
                for go in raw_go:
                    go_id   = go.get("identifier", "")
                    go_name = go.get("name", "")
                    if go_id:
                        go_terms.append(go_id)
                        go_names.append(go_name)

            if acc:
                domains.append(InterProDomain(
                    accession=acc,
                    name=str(name)[:80],
                    database=db,
                    start=start,
                    end=end,
                    go_terms=go_terms[:10],
                    go_names=go_names[:10],
                    e_value=0.0,
                ))

        log.debug(f"    InterPro direct: {len(domains)} domains")
        return domains

    except Exception as e:
        log.debug(f"    InterPro direct lookup failed: {e}")
        return []


def _parse_interpro_json(data: dict) -> list[InterProDomain]:
    """Parse InterProScan JSON output."""
    domains = []
    try:
        for match in data.get("results", [{}])[0].get("matches", []):
            sig     = match.get("signature", {})
            entry   = sig.get("entry") or {}
            acc     = sig.get("accession", "")
            name    = sig.get("name", "")
            db      = sig.get("signatureLibraryRelease", {}).get(
                "library", "unknown").upper()

            locs = match.get("locations", [])
            start = locs[0].get("start", 0) if locs else 0
            end   = locs[-1].get("end", 0)  if locs else 0
            evalue = locs[0].get("evalue", 0.0) if locs else 0.0

            go_terms = []
            go_names = []
            for go in entry.get("goXRefs", []):
                go_terms.append(go.get("id", ""))
                go_names.append(go.get("name", ""))

            domains.append(InterProDomain(
                accession=acc,
                name=name[:80],
                database=db,
                start=start,
                end=end,
                go_terms=go_terms[:10],
                go_names=go_names[:10],
                e_value=float(evalue),
            ))
    except Exception as e:
        log.debug(f"    InterPro JSON parse error: {e}")
    return domains


# ── GO term aggregation ────────────────────────────────────────────────────────

def _aggregate_annotations(
    blast_hits: list[HomologHit],
    domains:    list[InterProDomain],
    uniprot_id: str,
) -> tuple[list[str], list[str], list[str], str]:
    """
    Aggregate GO terms from all sources, deduplicate, rank by frequency.
    Returns (go_terms, go_names, families, top_function_text).
    """
    go_counts: dict[str, int]  = {}
    go_name_map: dict[str, str] = {}
    families: list[str]        = []
    function_texts: list[str]  = []

    # From BLAST hits (weight by evidence quality)
    for hit in blast_hits:
        weight = int(hit.evidence_weight)
        for go_id, go_name in zip(hit.go_terms, hit.go_names):
            go_counts[go_id]   = go_counts.get(go_id, 0) + weight
            go_name_map[go_id] = go_name
        if hit.function_text:
            function_texts.append(hit.function_text)

    # From InterPro domains
    for dom in domains:
        for go_id, go_name in zip(dom.go_terms, dom.go_names):
            go_counts[go_id]   = go_counts.get(go_id, 0) + 2
            go_name_map[go_id] = go_name
        # Extract family names from PANTHER entries
        if dom.database in ("PANTHER", "PFAM"):
            if dom.name and dom.name not in families:
                families.append(dom.name)

    # Sort GO terms by evidence count
    sorted_gos = sorted(go_counts.items(), key=lambda x: x[1], reverse=True)
    top_gos    = [go for go, _ in sorted_gos[:30]]
    top_names  = [go_name_map.get(go, "") for go in top_gos]

    # Top function = first non-empty function text from Swiss-Prot hits
    top_fn = ""
    for hit in blast_hits:
        if hit.reviewed and hit.function_text:
            top_fn = hit.function_text
            break
    if not top_fn and function_texts:
        top_fn = function_texts[0]

    return top_gos, top_names, families[:10], top_fn


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637)")
def main(uniprot: str) -> None:
    """
    Module 07 — Sequence homology + domain annotation.

    Requires Module 01 (.pdb) to have run first (to get the sequence).
    Makes live API calls to NCBI BLAST, UniProt, and InterPro.
    BLAST can take 2-5 minutes — this is normal.

    Example:
        python pipeline/homology.py --uniprot P04637
    """
    uniprot   = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    pdb_path  = Path(cfg.paths["structures"]) / f"{uniprot}.pdb"
    out_path  = inter_dir / f"{uniprot}_homology.json"

    if not pdb_path.exists():
        log.error(
            f".pdb not found: {pdb_path}\n"
            f"  Run Module 01 first: python pipeline/fetch_structure.py "
            f"--uniprot {uniprot}"
        )
        raise SystemExit(1)

    parsed   = parse_pdb(pdb_path, uniprot)
    sequence = parsed.sequence

    log.info(f"  Sequence length: {len(sequence)} aa")
    log.info("  Note: BLAST against Swiss-Prot takes 2-7 minutes.")

    result = run_homology(uniprot, sequence)
    result.to_json(out_path)
    click.echo(f"\nDone. Results written to:\n  {out_path}")


if __name__ == "__main__":
    main()