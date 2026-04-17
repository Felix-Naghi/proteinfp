"""
pipeline/01_fetch_structure.py
───────────────────────────────
Module 01 — Fetch protein structure + metadata from the AlphaFold DB.

Given a UniProt accession ID (e.g. "P04637"), this module:
  1. Queries the AFDB summary API to confirm the protein exists and get metadata.
  2. Downloads the AlphaFold .pdb file.
  3. Fetches the amino acid sequence + functional metadata from UniProt.
  4. Parses the structure into a rich ResidueInfo object (via utils/pdb_parser).
  5. Saves the .pdb locally and writes a structured JSON summary.

This is the entry point for EVERY protein in the pipeline.
All downstream modules receive the path to the .pdb + the ParsedStructure object.

Usage (standalone test):
    python pipeline/01_fetch_structure.py --uniprot P04637
    python pipeline/01_fetch_structure.py --uniprot Q9Y6I9 --force

Usage (from orchestrator):
    from pipeline.fetch_structure import fetch_structure
    result = fetch_structure("P04637")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import click
import requests
from tqdm import tqdm

from utils.config import cfg, get_logger
from utils.pdb_parser import parse_pdb, ParsedStructure

log = get_logger(__name__)


# ── Return type ────────────────────────────────────────────────────────────────

@dataclass
class StructureResult:
    """
    Everything Module 01 produces. Passed to every downstream module.
    """
    uniprot_id:       str
    gene_name:        str
    protein_name:     str
    organism:         str
    sequence:         str
    length:           int
    pdb_path:         str                  # absolute path to .pdb file
    json_path:        str                  # absolute path to parsed JSON
    mean_plddt:       float
    high_conf_frac:   float                # fraction of residues with pLDDT >= threshold
    n_disordered:     int
    disordered_regions: list[tuple[int, int]]
    afdb_version:     str
    uniprot_reviewed: bool                 # True = Swiss-Prot (experimental); False = TrEMBL
    parsed:           Optional[ParsedStructure] = None   # full residue-level data

    def summary(self) -> str:
        reviewed = "Swiss-Prot (reviewed)" if self.uniprot_reviewed else "TrEMBL (unreviewed)"
        return (
            f"\n{'─'*60}\n"
            f"  Protein    : {self.protein_name}\n"
            f"  Gene       : {self.gene_name}\n"
            f"  UniProt    : {self.uniprot_id}  ({reviewed})\n"
            f"  Organism   : {self.organism}\n"
            f"  Length     : {self.length} aa\n"
            f"  Mean pLDDT : {self.mean_plddt:.1f}\n"
            f"  High-conf  : {self.high_conf_frac*100:.1f}% of residues\n"
            f"  Disordered : {self.n_disordered} residues in "
            f"{len(self.disordered_regions)} region(s)\n"
            f"  .pdb saved : {self.pdb_path}\n"
            f"{'─'*60}"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("parsed", None)    # don't serialise the full residue list here
        return d

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ── Main function ──────────────────────────────────────────────────────────────

def fetch_structure(
    uniprot_id: str,
    force:      bool = False,
) -> StructureResult:
    """
    Fetch and parse the AlphaFold structure for a UniProt accession.

    Args:
        uniprot_id : UniProt accession, e.g. "P04637"
        force      : Re-download even if .pdb already exists locally

    Returns:
        StructureResult with paths, metadata, and ParsedStructure.

    Raises:
        ValueError  : protein not found in AFDB or UniProt
        RuntimeError: download failed after retries
    """
    uniprot_id = uniprot_id.strip().upper()
    log.info(f"── Module 01: Fetching structure for {uniprot_id} ──")

    struct_dir = Path(cfg.paths["structures"])
    inter_dir  = Path(cfg.paths["intermediate"])
    pdb_path   = struct_dir / f"{uniprot_id}.pdb"
    json_path  = inter_dir  / f"{uniprot_id}_structure.json"

    # ── Step 1: Check AFDB existence ──────────────────────────────────────────
    log.info("  [1/4] Querying AlphaFold DB...")
    afdb_meta = _query_afdb(uniprot_id)

    # ── Step 2: Download .pdb ─────────────────────────────────────────────────
    if pdb_path.exists() and not force:
        log.info(f"  [2/4] .pdb already cached: {pdb_path.name}  (use --force to re-download)")
    else:
        log.info(f"  [2/4] Downloading .pdb...")
        _download_pdb(uniprot_id, afdb_meta, pdb_path)

    # ── Step 3: Fetch UniProt metadata ────────────────────────────────────────
    log.info("  [3/4] Fetching UniProt metadata...")
    uniprot_meta = _query_uniprot(uniprot_id)

    # ── Step 4: Parse the structure ───────────────────────────────────────────
    log.info("  [4/4] Parsing structure...")
    plddt_threshold = float(cfg.get("afdb", "plddt_threshold", default=70.0))
    parsed = parse_pdb(pdb_path, uniprot_id, plddt_threshold=plddt_threshold)

    # ── Assemble result ───────────────────────────────────────────────────────
    result = StructureResult(
        uniprot_id=uniprot_id,
        gene_name=uniprot_meta.get("gene_name", "unknown"),
        protein_name=uniprot_meta.get("protein_name", "unknown"),
        organism=uniprot_meta.get("organism", "unknown"),
        sequence=parsed.sequence,
        length=parsed.length,
        pdb_path=str(pdb_path.resolve()),
        json_path=str(json_path.resolve()),
        mean_plddt=parsed.mean_plddt,
        high_conf_frac=parsed.high_conf_fraction,
        n_disordered=parsed.n_disordered,
        disordered_regions=parsed.disordered_regions,
        afdb_version=afdb_meta.get("latestVersion", "unknown"),
        uniprot_reviewed=uniprot_meta.get("reviewed", False),
        parsed=parsed,
    )

    result.to_json(json_path)
    log.info(result.summary())
    return result


# ── AFDB API helpers ───────────────────────────────────────────────────────────

def _query_afdb(uniprot_id: str) -> dict:
    """
    Call the AFDB summary endpoint to confirm the protein exists
    and get the model file URL.

    Returns the first entry from the AFDB response list.
    Raises ValueError if not found.
    """
    url = f"{cfg.afdb['base_url']}/prediction/{uniprot_id}"
    response = _get_with_retry(url, label=f"AFDB lookup {uniprot_id}")

    if response.status_code in (400, 404):
        raise ValueError(
            f"'{uniprot_id}' not found in AlphaFold DB.\n"
            f"  → Check: https://alphafold.ebi.ac.uk/entry/{uniprot_id}\n"
            f"  → Verify the UniProt ID is correct and the protein has an AF model."
        )

    response.raise_for_status()
    data = response.json()

    if not data:
        raise ValueError(f"AFDB returned empty response for {uniprot_id}")

    entry = data[0]
    log.debug(f"    AFDB entry found: model v{entry.get('latestVersion', '?')}, "
              f"length {entry.get('uniprotEnd', '?')} aa")
    return entry


def _download_pdb(
    uniprot_id: str,
    afdb_meta:  dict,
    out_path:   Path,
) -> None:
    """
    Download the AlphaFold .pdb file.

    AFDB file naming convention:
      AF-{UNIPROT_ID}-F1-model_v{VERSION}.pdb
    """
    version  = afdb_meta.get("latestVersion", cfg.afdb["model_version"])
    filename = f"AF-{uniprot_id}-F1-model_v{version}.pdb"
    url      = f"{cfg.afdb['structure_url']}/{filename}"

    log.debug(f"    Downloading: {url}")
    response = _get_with_retry(url, label=f"PDB download {uniprot_id}", stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=f"    {filename}", leave=False
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    log.info(f"    Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


# ── UniProt API helpers ────────────────────────────────────────────────────────

def _query_uniprot(uniprot_id: str) -> dict:
    """
    Fetch protein name, gene name, organism, and review status from UniProt REST API.
    Returns a flat dict of the fields we need.
    """
    url = f"{cfg.uniprot['base_url']}/{uniprot_id}"
    params = {
        "fields": "gene_names,protein_name,organism_name,reviewed",
        "format": "json",
    }

    response = _get_with_retry(url, label=f"UniProt {uniprot_id}", params=params)

    if response.status_code == 400 or response.status_code == 404:
        log.warning(f"    UniProt returned {response.status_code} for {uniprot_id}. "
                    f"Using placeholder metadata.")
        return {"gene_name": "unknown", "protein_name": "unknown",
                "organism": "unknown", "reviewed": False}

    response.raise_for_status()
    data = response.json()

    # Parse gene name
    gene_names = data.get("genes", [])
    gene_name = (gene_names[0].get("geneName", {}).get("value", "unknown")
                 if gene_names else "unknown")

    # Parse protein name (recommended name preferred)
    prot_desc  = data.get("proteinDescription", {})
    rec_name   = prot_desc.get("recommendedName", {})
    protein_name = (rec_name.get("fullName", {}).get("value")
                    or prot_desc.get("submissionNames", [{}])[0]
                       .get("fullName", {}).get("value", "unknown"))

    # Parse organism
    organism = data.get("organism", {}).get("scientificName", "unknown")

    # Reviewed = Swiss-Prot (manually curated)
    reviewed = data.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)"

    result = {
        "gene_name":    gene_name,
        "protein_name": protein_name,
        "organism":     organism,
        "reviewed":     reviewed,
    }
    log.debug(f"    UniProt: {gene_name} | {protein_name} | {organism} | "
              f"{'reviewed' if reviewed else 'unreviewed'}")
    return result


# ── HTTP retry helper ──────────────────────────────────────────────────────────

def _get_with_retry(
    url:     str,
    label:   str   = "",
    params:  dict  = None,
    stream:  bool  = False,
    timeout: float = None,
) -> requests.Response:
    """
    GET with exponential backoff retry.
    Raises RuntimeError if all retries are exhausted.
    """
    max_retries  = int(cfg.get("afdb", "max_retries", default=3))
    retry_delay  = float(cfg.get("afdb", "retry_delay_sec", default=2.0))
    timeout      = timeout or float(cfg.get("afdb", "timeout_sec", default=30))

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                url,
                params=params,
                stream=stream,
                timeout=timeout,
                headers={"User-Agent": "ProteinFP/0.1 (research pipeline)"},
            )
            # Don't retry on 404 — protein genuinely doesn't exist
            if resp.status_code == 404:
                return resp
            # Retry on server errors
            if resp.status_code >= 500:
                raise requests.HTTPError(f"Server error {resp.status_code}")
            return resp

        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to reach {label} after {max_retries} attempts.\n"
                    f"  Last error: {e}\n"
                    f"  URL: {url}"
                ) from e
            wait = retry_delay * (2 ** (attempt - 1))
            log.warning(f"    {label}: attempt {attempt} failed ({e}). "
                        f"Retrying in {wait:.0f}s...")
            time.sleep(wait)


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--uniprot", "-u",
    required=True,
    help="UniProt accession ID (e.g. P04637 for human TP53)"
)
@click.option(
    "--force", "-f",
    is_flag=True,
    default=False,
    help="Re-download the .pdb even if it already exists locally"
)
def main(uniprot: str, force: bool) -> None:
    """
    Module 01 — Fetch protein structure from AlphaFold DB.

    Example:
        python pipeline/01_fetch_structure.py --uniprot P04637
        python pipeline/01_fetch_structure.py --uniprot Q9Y6I9 --force
    """
    try:
        result = fetch_structure(uniprot, force=force)
        click.echo(f"\nDone. Results written to:\n  {result.json_path}")
    except (ValueError, RuntimeError) as e:
        log.error(str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
