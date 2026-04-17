"""
pipeline/11_foldseek.py
────────────────────────
Module 11 — Structural similarity search via Foldseek API.

Searches the entire PDB and AlphaFold DB for proteins with the same 3D fold
as the query, regardless of sequence similarity. This catches "structural
homologs" — proteins that share function but have diverged so far in
sequence that BLAST misses them.

Uses the Foldseek web API (search.foldseek.com) — no local installation needed.
Foldseek encodes protein structures as "3Di sequences" (structural alphabet)
and searches them at near-BLAST speed.

Why this matters:
  - Two proteins can share the same fold (and function) with <20% sequence identity
  - BLAST misses these; Foldseek finds them
  - Structural homologs are the strongest evidence for function transfer
    when sequence homology is absent

Databases searched:
  - PDB100     — all experimentally determined structures
  - AlphaFold DB — all predicted structures (200M+ proteins)

Usage (standalone):
    python pipeline/11_foldseek.py --uniprot P04637

Usage (from orchestrator):
    from pipeline.foldseek import run_foldseek
    result = run_foldseek("P04637", pdb_path)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import click
import requests

from utils.config import cfg, get_logger
from utils.pdb_parser import parse_pdb

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

FOLDSEEK_API     = "https://search.foldseek.com/api"
MAX_RESULTS      = 50
POLL_INTERVAL    = 5    # seconds
MAX_WAIT         = 120  # seconds

# TM-score threshold for "same fold" (0.5 = generally same fold)
TMSCORE_THRESHOLD = 0.5

# Databases to search
DATABASES = ["pdb100", "afdb50"]


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class FoldseekHit:
    """A single structural similarity hit."""
    target:          str       # PDB ID or UniProt accession
    description:     str
    tmscore:         float     # 0-1, higher = more similar fold
    rmsd:            float     # Å, lower = more similar
    seq_identity:    float     # fraction 0-1
    query_coverage:  float     # fraction of query covered
    e_value:         float
    database:        str       # "pdb100" or "afdb50"
    is_same_fold:    bool      # tmscore >= 0.5
    function_inferred: str     # description if same fold and known function

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FoldseekResult:
    """Full Foldseek structural search output. Output of Module 11."""
    uniprot_id:         str
    pdb_path:           str
    n_hits:             int              = 0
    hits:               list[FoldseekHit] = field(default_factory=list)
    same_fold_hits:     list[FoldseekHit] = field(default_factory=list)
    n_same_fold:        int              = 0
    top_tmscore:        float            = 0.0
    novel_hits:         list[FoldseekHit] = field(default_factory=list)
    api_available:      bool             = False
    inferred_functions: list[str]        = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  Foldseek structural search: {self.uniprot_id}",
            f"  API available  : {'yes' if self.api_available else 'no'}",
            f"  Total hits     : {self.n_hits}",
            f"  Same fold (≥0.5): {self.n_same_fold}",
            f"  Novel hits     : {len(self.novel_hits)} "
            f"(same fold, low seq identity)",
        ]
        for hit in self.same_fold_hits[:5]:
            lines.append(
                f"  [{hit.database}] {hit.target} "
                f"TM={hit.tmscore:.2f} RMSD={hit.rmsd:.1f}Å "
                f"id={hit.seq_identity*100:.0f}% "
                f"{hit.description[:40]}"
            )
        if self.inferred_functions:
            lines.append(f"  Inferred functions:")
            for fn in self.inferred_functions[:3]:
                lines.append(f"    - {fn}")
        lines.append(f"{'─'*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.debug(f"Saved Foldseek JSON → {path}")


# ── Main function ──────────────────────────────────────────────────────────────

def run_foldseek(
    uniprot_id: str,
    pdb_path:   str | Path,
) -> FoldseekResult:
    """
    Run Foldseek structural similarity search via web API.

    Args:
        uniprot_id: UniProt accession for labelling
        pdb_path:   Path to .pdb file

    Returns:
        FoldseekResult with structural hits and inferred functions.
    """
    log.info(f"── Module 11: Foldseek structural search for {uniprot_id} ──")

    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    result = FoldseekResult(
        uniprot_id=uniprot_id,
        pdb_path=str(pdb_path),
    )

    # ── Submit search ─────────────────────────────────────────────────────────
    log.info("  [1/3] Submitting structure to Foldseek API...")
    ticket_id, ok = _submit_foldseek(pdb_path)

    if not ok or not ticket_id:
        log.warning("  Foldseek API unavailable — using structure-based fallback")
        result.api_available = False
        _apply_fallback(result, uniprot_id)
        return result

    result.api_available = True
    log.info(f"    Ticket: {ticket_id}")

    # ── Poll for results ──────────────────────────────────────────────────────
    log.info("  [2/3] Waiting for results...")
    raw_hits, poll_ok = _poll_foldseek(ticket_id)

    if not poll_ok:
        log.warning("  Foldseek timed out — using fallback")
        _apply_fallback(result, uniprot_id)
        return result

    # ── Parse and score hits ──────────────────────────────────────────────────
    log.info(f"  [3/3] Parsing {len(raw_hits)} hits...")
    hits = _parse_hits(raw_hits)

    same_fold = [h for h in hits if h.is_same_fold]
    novel     = [h for h in same_fold if h.seq_identity < 0.3]
    functions = _infer_functions(same_fold)

    result.n_hits            = len(hits)
    result.hits              = hits
    result.same_fold_hits    = same_fold
    result.n_same_fold       = len(same_fold)
    result.top_tmscore       = max((h.tmscore for h in hits), default=0.0)
    result.novel_hits        = novel
    result.inferred_functions = functions

    log.info(result.summary())
    return result


# ── Foldseek API ───────────────────────────────────────────────────────────────

def _submit_foldseek(pdb_path: Path) -> tuple[str, bool]:
    """
    Submit a .pdb file to the Foldseek web API.
    Returns (ticket_id, success).
    """
    try:
        with open(pdb_path, "rb") as f:
            pdb_content = f.read()

        resp = requests.post(
            f"{FOLDSEEK_API}/ticket",
            files={"q": (pdb_path.name, pdb_content, "application/octet-stream")},
            data={
                "mode":      "3diaa",
                "database[]": DATABASES,
            },
            timeout=30,
            headers={"User-Agent": "ProteinFP/0.1"},
        )

        if resp.status_code != 200:
            log.warning(f"    Foldseek submit failed: HTTP {resp.status_code}")
            return "", False

        data      = resp.json()
        ticket_id = data.get("id", "")

        if not ticket_id:
            log.warning("    Foldseek: no ticket ID returned")
            return "", False

        return ticket_id, True

    except Exception as e:
        log.warning(f"    Foldseek submit error: {e}")
        return "", False


def _poll_foldseek(ticket_id: str) -> tuple[list[dict], bool]:
    """
    Poll Foldseek until results are ready.
    Returns (raw_hits_list, success).
    """
    elapsed = 0

    while elapsed < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        try:
            status_resp = requests.get(
                f"{FOLDSEEK_API}/ticket/{ticket_id}",
                timeout=15,
            )

            if status_resp.status_code != 200:
                continue

            status_data = status_resp.json()
            status      = status_data.get("status", "")

            log.debug(f"    Foldseek status: {status} ({elapsed}s)")

            if status == "COMPLETE":
                # Fetch results
                result_resp = requests.get(
                    f"{FOLDSEEK_API}/result/{ticket_id}/0",
                    params={"format": "json", "limit": MAX_RESULTS},
                    timeout=30,
                )
                if result_resp.status_code == 200:
                    data = result_resp.json()
                    hits = data.get("results", [])
                    # Flatten across databases
                    all_hits = []
                    for db_result in hits:
                        db_name = db_result.get("db", "unknown")
                        for hit in db_result.get("alignments", [[]])[0]:
                            hit["_db"] = db_name
                            all_hits.append(hit)
                    log.info(f"    Got {len(all_hits)} hits")
                    return all_hits, True
                else:
                    log.warning(f"    Result fetch failed: {result_resp.status_code}")
                    return [], False

            elif status in ("ERROR", "FAILED"):
                log.warning("    Foldseek job failed")
                return [], False

        except Exception as e:
            log.debug(f"    Poll error: {e}")
            continue

    log.warning(f"    Foldseek timed out after {MAX_WAIT}s")
    return [], False


def _parse_hits(raw_hits: list[dict]) -> list[FoldseekHit]:
    """Parse raw Foldseek JSON hits into FoldseekHit objects."""
    hits = []

    for raw in raw_hits:
        try:
            target      = raw.get("target", raw.get("name", "unknown"))
            description = raw.get("taxName", raw.get("description", ""))
            tmscore     = float(raw.get("prob", raw.get("tmscore", raw.get("tmScore", 0.0))))
            rmsd        = float(raw.get("rmsd", 0.0))
            raw_seqid = float(raw.get("seqId", raw.get("seq_id", 0.0)))
            seqid = raw_seqid / 100.0 if raw_seqid > 1.0 else raw_seqid
            qcov        = float(raw.get("qCov", raw.get("q_cov", 0.0)))
            evalue      = float(raw.get("eval", raw.get("evalue", 1.0)))
            db          = raw.get("_db", "unknown")

            # Infer description from target if empty
            if not description and len(target) == 4:
                description = f"PDB entry {target}"

            is_same_fold = tmscore >= TMSCORE_THRESHOLD

            hits.append(FoldseekHit(
                target=target,
                description=description[:200],
                tmscore=round(tmscore, 3),
                rmsd=round(rmsd, 2),
                seq_identity=round(seqid, 3),
                query_coverage=round(qcov, 3),
                e_value=evalue,
                database=db,
                is_same_fold=is_same_fold,
                function_inferred="" if seqid > 0.9 else description[:100],
            ))

        except Exception as e:
            log.debug(f"    Skipping hit: {e}")
            continue

    hits.sort(key=lambda h: h.tmscore, reverse=True)
    return hits[:MAX_RESULTS]


def _infer_functions(same_fold_hits: list[FoldseekHit]) -> list[str]:
    """Extract unique function descriptions from same-fold hits."""
    seen      = set()
    functions = []
    for hit in same_fold_hits:
        desc = hit.function_inferred if hit.function_inferred and len(hit.function_inferred) > 15 else hit.target
        if desc and desc not in seen and len(desc) > 10:
            seen.add(desc)
            functions.append(desc[:120])
        if len(functions) >= 5:
            break
    return functions


# ── Fallback when API unavailable ─────────────────────────────────────────────

def _apply_fallback(result: FoldseekResult, uniprot_id: str) -> None:
    """
    When Foldseek API is unavailable, use pre-computed structural family
    annotations from the InterPro data already in Module 07 output.
    This provides structural context without the API.
    """
    inter_dir   = Path(cfg.paths["intermediate"])
    homology_path = inter_dir / f"{uniprot_id}_homology.json"

    if not homology_path.exists():
        log.warning("    No homology data available for fallback")
        return

    with open(homology_path) as f:
        homology = json.load(f)

    # Use InterPro domain descriptions as structural family proxies
    families = homology.get("protein_families", [])
    if families:
        result.inferred_functions = families[:5]
        log.info(f"    Fallback: using {len(families)} InterPro family annotations")

    # Create synthetic hits from BLAST homologs (sequence = proxy for structure)
    for blast_hit in homology.get("blast_hits", [])[:10]:
        result.hits.append(FoldseekHit(
            target=blast_hit.get("accession", ""),
            description=blast_hit.get("description", "")[:200],
            tmscore=min(0.5 + blast_hit.get("identity_pct", 0) / 200, 1.0),
            rmsd=max(0.5, 3.0 - blast_hit.get("identity_pct", 0) / 50),
            seq_identity=blast_hit.get("identity_pct", 0) / 100,
            query_coverage=blast_hit.get("coverage_pct", 0) / 100,
            e_value=blast_hit.get("e_value", 1.0),
            database="blast_proxy",
            is_same_fold=blast_hit.get("identity_pct", 0) > 30,
            function_inferred=blast_hit.get("function_text", "")[:100],
        ))

    result.n_hits        = len(result.hits)
    result.same_fold_hits = [h for h in result.hits if h.is_same_fold]
    result.n_same_fold   = len(result.same_fold_hits)
    result.top_tmscore   = max(
        (h.tmscore for h in result.hits), default=0.0
    )
    result.novel_hits    = [
        h for h in result.same_fold_hits if h.seq_identity < 0.3
    ]

    log.info(f"    Fallback: {result.n_hits} proxy hits from BLAST homologs")
    log.info(result.summary())


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637)")
def main(uniprot: str) -> None:
    """
    Module 11 — Foldseek structural similarity search.

    Searches the PDB and AlphaFold DB for proteins with the same 3D fold.
    Requires Module 01 (.pdb file) to have run first.

    Example:
        python pipeline/foldseek.py --uniprot P04637
    """
    uniprot   = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    pdb_path  = Path(cfg.paths["structures"]) / f"{uniprot}.pdb"
    out_path  = inter_dir / f"{uniprot}_foldseek.json"

    if not pdb_path.exists():
        log.error(f".pdb not found — run Module 01 first")
        raise SystemExit(1)

    result = run_foldseek(uniprot, pdb_path)
    result.to_json(out_path)
    click.echo(f"\nDone. Results written to:\n  {out_path}")


if __name__ == "__main__":
    main()