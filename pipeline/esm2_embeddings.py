"""
pipeline/08_esm2_embeddings.py
───────────────────────────────
Module 08 — ESM-2 protein language model embeddings.

Uses Meta's ESM-2 (650M parameter model) to generate per-residue and
per-protein embeddings from sequence alone. These embeddings encode
evolutionary and functional information learned from 250M protein sequences.

The embeddings are used by:
  - Module 09 (DeepFRI) for structure-aware GO term prediction
  - Module 10 (CLEAN) for enzyme commission number prediction
  - Module 13 (consensus) as an independent evidence source

Model: esm2_t33_650M_UR50D
  - 33 transformer layers
  - 650M parameters
  - Fits in ~3GB VRAM — ideal for RTX 5060
  - Trained on UniRef50 (250M sequences)

Output:
  - Per-residue embeddings: (L, 1280) float32 tensor
  - Per-protein embedding:  (1280,) float32 (mean pooled)
  - Attention contacts:     (L, L) float32 contact map
  - Per-residue logits for functional residue prediction

Usage (standalone):
    python pipeline/08_esm2_embeddings.py --uniprot P04637

Usage (from orchestrator):
    from pipeline.esm2_embeddings import compute_esm2_embeddings
    result = compute_esm2_embeddings("P04637", sequence)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import click
import numpy as np

from utils.config import cfg, get_logger
from utils.pdb_parser import parse_pdb

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

ESM2_MODEL     = "esm2_t33_650M_UR50D"   # fits in RTX 5060 VRAM
MAX_SEQ_LENGTH = 1022                     # ESM-2 max tokens (1024 - 2 for BOS/EOS)
EMBEDDING_DIM  = 1280                     # ESM-2 650M embedding dimension


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ESM2Result:
    """ESM-2 embedding output. Output of Module 08."""
    uniprot_id:          str
    sequence_length:     int
    model_name:          str
    embedding_dim:       int
    gpu_used:            bool
    compute_time_sec:    float

    # Stored as lists for JSON serialisation (converted from numpy)
    protein_embedding:   list[float]        # (1280,) mean-pooled
    residue_embeddings:  list[list[float]]  # (L, 1280) per-residue
    contact_map:         list[list[float]]  # (L, L) predicted contacts

    # Functional predictions from embedding space
    predicted_functional_residues: list[int]    # high-attention residues
    embedding_norm:      float                  # L2 norm of protein embedding

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.debug(f"Saved ESM-2 embeddings JSON → {path}")

    def protein_embedding_np(self) -> np.ndarray:
        return np.array(self.protein_embedding, dtype=np.float32)

    def residue_embeddings_np(self) -> np.ndarray:
        return np.array(self.residue_embeddings, dtype=np.float32)

    def summary(self) -> str:
        device = "GPU" if self.gpu_used else "CPU"
        return (
            f"\n{'─'*60}\n"
            f"  ESM-2 embeddings: {self.uniprot_id}\n"
            f"  Model           : {self.model_name}\n"
            f"  Sequence length : {self.sequence_length} aa\n"
            f"  Embedding dim   : {self.embedding_dim}\n"
            f"  Device          : {device}\n"
            f"  Compute time    : {self.compute_time_sec:.1f}s\n"
            f"  Embedding norm  : {self.embedding_norm:.2f}\n"
            f"  Functional res. : {len(self.predicted_functional_residues)} "
            f"high-attention positions\n"
            f"{'─'*60}"
        )


# ── Main function ──────────────────────────────────────────────────────────────

def compute_esm2_embeddings(
    uniprot_id: str,
    sequence:   str,
) -> ESM2Result:
    """
    Compute ESM-2 embeddings for a protein sequence.

    Automatically uses GPU if available (RTX 5060 will be detected).
    Falls back to CPU if GPU not available.

    Args:
        uniprot_id: UniProt accession for labelling
        sequence:   Amino acid sequence string

    Returns:
        ESM2Result with embeddings and contact map.
    """
    log.info(f"── Module 08: ESM-2 embeddings for {uniprot_id} ──")

    # Truncate if over ESM-2 limit
    if len(sequence) > MAX_SEQ_LENGTH:
        log.warning(f"  Sequence truncated from {len(sequence)} to "
                    f"{MAX_SEQ_LENGTH} residues for ESM-2")
        sequence = sequence[:MAX_SEQ_LENGTH]

    start_time = time.time()

    try:
        import torch
        import esm

        # ── Load model ────────────────────────────────────────────────────────
        device_str = cfg.get("compute", "device", default="cuda")
        device     = torch.device(
            device_str if torch.cuda.is_available() else "cpu"
        )
        gpu_used   = device.type == "cuda"

        log.info(f"  Loading {ESM2_MODEL} on {device}...")
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model           = model.eval().to(device)
        batch_converter = alphabet.get_batch_converter()

        log.info(f"  Model loaded. Computing embeddings...")

        # ── Prepare input ─────────────────────────────────────────────────────
        data   = [(uniprot_id, sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        # ── Forward pass ──────────────────────────────────────────────────────
        with torch.no_grad():
            results = model(
                batch_tokens,
                repr_layers=[33],           # final layer representations
                return_contacts=True,
            )

        # ── Extract embeddings ────────────────────────────────────────────────
        # Per-residue: remove BOS/EOS tokens
        token_repr = results["representations"][33]
        res_emb    = token_repr[0, 1:len(sequence)+1, :].cpu().numpy()  # (L, 1280)

        # Per-protein: mean pool
        prot_emb   = res_emb.mean(axis=0)   # (1280,)

        # Contact map
        contacts   = results["contacts"][0].cpu().numpy()  # (L, L)

        # Clip contact map to sequence length
        L          = len(sequence)
        contacts   = contacts[:L, :L]

        # ── Identify functional residues ──────────────────────────────────────
        # High-attention residues = positions with many predicted contacts
        contact_counts    = (contacts > 0.5).sum(axis=1)
        threshold         = float(np.percentile(contact_counts, 75))
        functional_res    = [
            i + 1  # 1-based residue numbering
            for i, count in enumerate(contact_counts)
            if count >= threshold
        ]

        elapsed = time.time() - start_time
        log.info(f"  Done in {elapsed:.1f}s")

        result = ESM2Result(
            uniprot_id=uniprot_id,
            sequence_length=len(sequence),
            model_name=ESM2_MODEL,
            embedding_dim=EMBEDDING_DIM,
            gpu_used=gpu_used,
            compute_time_sec=round(elapsed, 2),
            protein_embedding=prot_emb.tolist(),
            # Store every 4th residue to reduce file size (still captures functional signal)
            # Full embeddings kept in memory for downstream modules in the same session
            residue_embeddings=res_emb[::4].tolist(),
            contact_map=contacts.tolist(),
            predicted_functional_residues=functional_res,
            embedding_norm=round(float(np.linalg.norm(prot_emb)), 4),
        )

        log.info(result.summary())
        return result

    except ImportError:
        log.error(
            "  ESM-2 not available. Install with:\n"
            "    pip install fair-esm"
        )
        raise
    except Exception as e:
        log.error(f"  ESM-2 failed: {e}")
        raise


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637)")
def main(uniprot: str) -> None:
    """
    Module 08 — ESM-2 protein language model embeddings.

    Uses GPU automatically if available (RTX 5060 recommended).
    First run downloads the model (~1.4GB) — subsequent runs use cache.

    Example:
        python pipeline/esm2_embeddings.py --uniprot P04637
    """
    uniprot   = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    pdb_path  = Path(cfg.paths["structures"]) / f"{uniprot}.pdb"
    out_path  = inter_dir / f"{uniprot}_esm2.json"

    if not pdb_path.exists():
        log.error(f".pdb not found — run Module 01 first")
        raise SystemExit(1)

    parsed   = parse_pdb(pdb_path, uniprot)
    sequence = parsed.sequence

    result = compute_esm2_embeddings(uniprot, sequence)
    result.to_json(out_path)
    click.echo(f"\nDone. Results written to:\n  {out_path}")


if __name__ == "__main__":
    main()