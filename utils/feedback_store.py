"""
utils/feedback_store.py
────────────────────────
The persistent feedback memory that turns ProteinFP from a one-shot pipeline
into a learning system.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THIS EXISTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every time the pipeline scores a candidate (docked SMILES, designed antibody,
PROTAC linker, etc.), that score is an experimental measurement of how well
the current pocket / epitope model predicts reality.

In the old pipeline, those scores died at process exit. Each new run started
from scratch — the pocket model from AlphaFold geometry, the fragment library
from a static seed list. There was no memory.

The FeedbackStore is the memory. Every evaluation gets appended to disk with
its full context (pocket descriptor at the time, generation, weights, scaffold,
fingerprint). Downstream modules then:

  • refit_pocket_model()  — recompute pocket descriptor from what actually bound
  • seed_next_run()       — biased seeding toward proven scaffolds
  • detect_dead_ends()    — avoid moves that historically wasted generations
  • cross_protein_xfer()  — learn from structurally related proteins

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN DECISIONS (intentional, not accidental)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. JSONL append-only log    — O(1) write, crash-safe, never rewrites bulk data
  2. Schema versioning        — old records keep parsing as schema evolves
  3. Pocket context captured  — every score carries the pocket descriptor it
                                  was measured against, so we can later separate
                                  "bad molecule" from "stale pocket model"
  4. Fingerprint index        — Morgan FP hash → fast dedup + similarity lookup
  5. Aggregate cache          — summary stats refreshed lazily, not per-write
  6. Three storage tiers      —
      events/   :  data/feedback/{uid}/events.jsonl     (append-only, raw)
      index/    :  data/feedback/{uid}/index.json       (fingerprint hash table)
      stats/    :  data/feedback/{uid}/stats.json       (aggregates, derived)
  7. Modality-agnostic        — small molecules, antibodies, PROTACs, ADCs all
                                  go to the same store with a `modality` tag
  8. Run-tagged               — every event knows which CLI invocation made it,
                                  so you can replay or audit any particular run

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCHEMA (v1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A single event (one row in events.jsonl):

  {
    "schema_version":   1,
    "event_id":         "evt_a3f9c2...",      # uuid4 hex, primary key
    "run_id":           "run_20260512T085847",# groups events from one CLI run
    "timestamp":        "2026-05-12T08:58:47Z",
    "uniprot_id":       "P04637",
    "modality":         "small_molecule",     # | antibody | protac | adc | cart | allosteric
    "target_site":      "P1",                 # pocket / epitope identifier
    "target_kind":      "orthosteric",        # | allosteric | ppi_interface | epitope

    # Candidate identity (modality-dependent fields go in 'payload')
    "identity": {
      "smiles":          "...",               # for small molecules
      "fp_hash":         "5e1c...",           # Morgan FP -> sha1 hex (16 chars)
      "scaffold":        "...",               # Murcko scaffold SMILES
      "heavy_atoms":     22,
    },

    # The actual measurement
    "score": {
      "primary":         -8.4,                # main metric (Vina dG, affinity...)
      "primary_kind":    "vina_dG_kcal_mol",
      "secondary":       {                    # other metrics, free-form
         "le":           0.41,
         "qed":          0.72,
         "fitness":      0.83,
         "novelty":      0.15
      },
      "valid":           true,                # false = pose failed, score is garbage
    },

    # Provenance — what produced this candidate
    "origin": {
      "generation":      3,
      "parent_smiles":   "...",               # for evolved candidates
      "mutation_op":     "fragment_grow",     # | crossover | seed | random
      "weights": {                            # fitness weights at this gen
         "w_bind":       0.55, "w_le": 0.22, "w_nov": 0.18, "w_cyp": 0.05
      }
    },

    # The pocket descriptor AT THE TIME of evaluation — critical for refit
    "pocket_snapshot": {
      "volume_A3":       1260.0,
      "druggability":    0.90,
      "charge":          1.5,
      "hydrophobicity":  0.42,
      "hbd_capacity":    4,
      "hba_capacity":    6,
      "center":          [-15.1, 10.1, 19.6],
      "model_version":   "raw_v0"             # tracks which pocket model was used
    },

    # Modality-specific extras
    "payload": {}
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  from utils.feedback_store import FeedbackStore

  # Open (or create) the store for one protein
  fb = FeedbackStore("P04637")

  # Start a new run — returns a run_id you tag all events with
  run_id = fb.start_run(modality="small_molecule",
                        target_site="P1",
                        pocket_snapshot={...})

  # Record one evaluation
  fb.record(
      smiles="c1ccccc1Cc2ncccc2",
      primary_score=-8.4,
      primary_kind="vina_dG_kcal_mol",
      secondary={"le": 0.41, "qed": 0.72, "fitness": 0.83},
      origin={"generation": 3, "mutation_op": "fragment_grow"},
      valid=True,
  )

  # Query whether you've seen a molecule before
  if fb.has_seen(smiles="c1ccccc1Cc2ncccc2"):
      prior = fb.lookup(smiles="c1ccccc1Cc2ncccc2")

  # Get top N ever observed (across all runs)
  top10 = fb.top_n(n=10, valid_only=True)

  # Refresh aggregate stats (cheap, reads index only unless forced)
  stats = fb.stats(refresh=False)

  # End the run — flushes any buffered writes, updates stats
  fb.end_run()

The store is process-safe via fcntl advisory locks on Linux (best-effort
on other platforms). Multiple ProteinFP runs against the SAME protein
should serialize their writes; runs against DIFFERENT proteins are
independent and parallel-safe (each protein has its own directory).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

# Optional RDKit — used for fingerprint hashing. The store still works without
# it (falls back to SMILES-string hashing), it just loses near-dup detection.
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    _RDKIT = True
except ImportError:
    _RDKIT = False

# Optional fcntl — Linux/Mac advisory locking
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

SCHEMA_VERSION = 1

# Morgan fingerprint settings — must stay consistent across versions or
# fp_hash values become incompatible with the existing index.
FP_RADIUS = 2
FP_NBITS  = 2048
FP_HASH_LEN = 16    # truncated sha1 → 16 hex chars (64 bits, collision-safe enough)

# Filenames
EVENTS_FILE = "events.jsonl"
INDEX_FILE  = "index.json"
STATS_FILE  = "stats.json"
LOCK_FILE   = ".lock"

# Aggregate refresh policy: rebuild stats every N new events
STATS_REFRESH_EVERY = 25


# ══════════════════════════════════════════════════════════════════════════════
# DATA TYPES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeedbackEvent:
    """A single recorded evaluation. Serialized as one JSONL row."""
    schema_version:   int
    event_id:         str
    run_id:           str
    timestamp:        str
    uniprot_id:       str
    modality:         str
    target_site:      str
    target_kind:      str
    identity:         dict
    score:            dict
    origin:           dict
    pocket_snapshot:  dict
    payload:          dict = field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Single-line JSON — newline is appended by writer."""
        return json.dumps(asdict(self), separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "FeedbackEvent":
        # Forward-compatible: ignore unknown keys, fill missing with defaults
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known}
        filtered.setdefault("payload", {})
        return cls(**filtered)


# ══════════════════════════════════════════════════════════════════════════════
# THE STORE
# ══════════════════════════════════════════════════════════════════════════════

class FeedbackStore:
    """
    Persistent, append-only feedback log for one protein.

    One instance == one (uniprot_id, root_dir) pair. Multiple instances pointing
    at the same directory will coordinate via file locking.

    Lifecycle:
        store = FeedbackStore("P04637")           # open / create
        run_id = store.start_run(...)             # begin a run
        store.record(...)                         # record an evaluation
        store.record(...)
        store.end_run()                           # flush + refresh stats
    """

    def __init__(
        self,
        uniprot_id: str,
        root_dir:   Optional[Path] = None,
    ):
        self.uniprot_id = uniprot_id.strip().upper()
        if not self.uniprot_id:
            raise ValueError("uniprot_id must be non-empty")

        # Resolve root: explicit arg → config → ./data/feedback fallback
        self.root_dir = self._resolve_root(root_dir)
        self.dir      = self.root_dir / self.uniprot_id
        self.dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.dir / EVENTS_FILE
        self.index_path  = self.dir / INDEX_FILE
        self.stats_path  = self.dir / STATS_FILE
        self.lock_path   = self.dir / LOCK_FILE

        # Touch files so first reads don't have to special-case missing
        self.events_path.touch(exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"fp_index": {}, "n_events": 0}))
        if not self.stats_path.exists():
            self.stats_path.write_text(json.dumps({"schema_version": SCHEMA_VERSION,
                                                    "uniprot_id":     self.uniprot_id,
                                                    "n_events":       0}))

        # In-memory state for the current run
        self._current_run_id:        Optional[str]  = None
        self._current_modality:      Optional[str]  = None
        self._current_target_site:   Optional[str]  = None
        self._current_target_kind:   Optional[str]  = None
        self._current_pocket_snap:   Optional[dict] = None
        self._events_since_refresh:  int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # RUN LIFECYCLE
    # ──────────────────────────────────────────────────────────────────────────

    def start_run(
        self,
        modality:        str,
        target_site:     str,
        pocket_snapshot: Optional[dict] = None,
        target_kind:     str = "orthosteric",
        run_id:          Optional[str]  = None,
    ) -> str:
        """
        Begin a run. All subsequent record() calls inherit this context until
        end_run() or the next start_run().

        Returns the run_id (auto-generated if not supplied).
        """
        if run_id is None:
            run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") \
                     + "_" + uuid.uuid4().hex[:6]

        self._current_run_id       = run_id
        self._current_modality     = modality
        self._current_target_site  = target_site
        self._current_target_kind  = target_kind
        self._current_pocket_snap  = dict(pocket_snapshot) if pocket_snapshot else {}
        self._events_since_refresh = 0
        return run_id

    def end_run(self) -> dict:
        """Flush, refresh aggregate stats, return the refreshed stats dict."""
        stats = self._refresh_stats()
        self._current_run_id      = None
        self._current_modality    = None
        self._current_target_site = None
        self._current_target_kind = None
        self._current_pocket_snap = None
        self._events_since_refresh = 0
        return stats

    # ──────────────────────────────────────────────────────────────────────────
    # WRITE PATH
    # ──────────────────────────────────────────────────────────────────────────

    def record(
        self,
        primary_score:   float,
        primary_kind:    str  = "vina_dG_kcal_mol",
        smiles:          Optional[str]  = None,
        sequence:        Optional[str]  = None,    # for antibodies / peptides
        identity_extra:  Optional[dict] = None,
        secondary:       Optional[dict] = None,
        origin:          Optional[dict] = None,
        valid:           bool = True,
        payload:         Optional[dict] = None,
        # Override run context (rarely needed)
        run_id:          Optional[str]  = None,
        modality:        Optional[str]  = None,
        target_site:     Optional[str]  = None,
        target_kind:     Optional[str]  = None,
        pocket_snapshot: Optional[dict] = None,
    ) -> str:
        """
        Append one evaluation event. Returns the event_id.

        Typical usage during evolution:
            store.record(
                smiles="...",
                primary_score=-8.4,
                primary_kind="vina_dG_kcal_mol",
                secondary={"le": 0.41, "qed": 0.72, "fitness": 0.83},
                origin={"generation": 3, "mutation_op": "fragment_grow"},
                valid=True,
            )

        Notes
        -----
        - At least one of smiles or sequence (or identity_extra) must be provided
          so the candidate has an identity.
        - If valid=False, the event is still written (we want to learn from
          failures too) but score.valid=False signals downstream code to ignore
          the primary score for fitting purposes.
        """
        # Need an active run, OR caller supplies the context explicitly
        rid   = run_id          or self._current_run_id
        mod   = modality        or self._current_modality
        tsite = target_site     or self._current_target_site
        tkind = target_kind     or self._current_target_kind
        psnap = pocket_snapshot if pocket_snapshot is not None \
                                else (self._current_pocket_snap or {})

        if rid is None or mod is None or tsite is None:
            raise RuntimeError(
                "FeedbackStore.record() called outside an active run. "
                "Call start_run(modality=..., target_site=...) first, or pass "
                "run_id/modality/target_site explicitly."
            )

        # Build identity block
        identity = self._build_identity(smiles=smiles, sequence=sequence,
                                         extra=identity_extra)
        if not identity:
            raise ValueError(
                "record() needs at least one identity field "
                "(smiles, sequence, or identity_extra)."
            )

        event = FeedbackEvent(
            schema_version  = SCHEMA_VERSION,
            event_id        = "evt_" + uuid.uuid4().hex,
            run_id          = rid,
            timestamp       = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            uniprot_id      = self.uniprot_id,
            modality        = mod,
            target_site     = tsite,
            target_kind     = tkind or "orthosteric",
            identity        = identity,
            score = {
                "primary":      float(primary_score),
                "primary_kind": primary_kind,
                "secondary":    dict(secondary) if secondary else {},
                "valid":        bool(valid),
            },
            origin          = dict(origin) if origin else {},
            pocket_snapshot = dict(psnap),
            payload         = dict(payload) if payload else {},
        )

        # Persist
        with self._lock():
            self._append_event(event)
            self._update_index(event)
            self._events_since_refresh += 1
            if self._events_since_refresh >= STATS_REFRESH_EVERY:
                self._refresh_stats()
                self._events_since_refresh = 0

        return event.event_id

    # ──────────────────────────────────────────────────────────────────────────
    # READ PATH
    # ──────────────────────────────────────────────────────────────────────────

    def has_seen(
        self,
        smiles:   Optional[str] = None,
        fp_hash:  Optional[str] = None,
    ) -> bool:
        """Constant-time lookup via fingerprint index."""
        if fp_hash is None:
            if smiles is None:
                return False
            fp_hash = self._fp_hash(smiles)
            if fp_hash is None:
                return False
        idx = self._load_index()
        return fp_hash in idx.get("fp_index", {})

    def lookup(
        self,
        smiles:   Optional[str] = None,
        fp_hash:  Optional[str] = None,
    ) -> Optional[dict]:
        """
        Return the index entry for a molecule (best score, count, last seen).
        For the full event history, use iter_events().
        """
        if fp_hash is None and smiles is not None:
            fp_hash = self._fp_hash(smiles)
        if fp_hash is None:
            return None
        idx = self._load_index()
        return idx.get("fp_index", {}).get(fp_hash)

    def iter_events(
        self,
        modality:    Optional[str] = None,
        target_site: Optional[str] = None,
        valid_only:  bool          = False,
        run_id:      Optional[str] = None,
    ) -> Iterator[FeedbackEvent]:
        """Stream events from disk. Filters applied row-by-row (cheap)."""
        if not self.events_path.exists():
            return
        with self.events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if modality    and d.get("modality")    != modality:    continue
                if target_site and d.get("target_site") != target_site: continue
                if run_id      and d.get("run_id")      != run_id:      continue
                if valid_only  and not d.get("score", {}).get("valid", True): continue
                yield FeedbackEvent.from_dict(d)

    def top_n(
        self,
        n:            int = 10,
        modality:     Optional[str] = None,
        target_site:  Optional[str] = None,
        valid_only:   bool = True,
        prefer_lower: bool = True,    # docking scores: lower is better
    ) -> list[dict]:
        """
        Return the top-N events by primary score.

        For docking ΔG, lower is better (prefer_lower=True, the default).
        For affinity or fitness metrics, set prefer_lower=False.
        """
        events: list[dict] = []
        for ev in self.iter_events(modality=modality,
                                    target_site=target_site,
                                    valid_only=valid_only):
            events.append({
                "identity":     ev.identity,
                "score":        ev.score,
                "origin":       ev.origin,
                "run_id":       ev.run_id,
                "target_site":  ev.target_site,
                "modality":     ev.modality,
                "event_id":     ev.event_id,
            })
        events.sort(key=lambda e: e["score"]["primary"], reverse=not prefer_lower)
        return events[:n]

    def stats(self, refresh: bool = False) -> dict:
        """Return cached aggregate stats. refresh=True forces recomputation."""
        if refresh:
            return self._refresh_stats()
        try:
            return json.loads(self.stats_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return self._refresh_stats()

    # ──────────────────────────────────────────────────────────────────────────
    # INTERNALS — IDENTITY / FINGERPRINTING
    # ──────────────────────────────────────────────────────────────────────────

    def _build_identity(
        self,
        smiles:    Optional[str],
        sequence:  Optional[str],
        extra:     Optional[dict],
    ) -> dict:
        identity: dict = {}
        if smiles:
            identity["smiles"] = smiles
            fph = self._fp_hash(smiles)
            if fph:
                identity["fp_hash"] = fph
            scaff = self._murcko(smiles)
            if scaff:
                identity["scaffold"] = scaff
            ha = self._heavy_atoms(smiles)
            if ha is not None:
                identity["heavy_atoms"] = ha
        if sequence:
            identity["sequence"] = sequence
            # Sequence hash — sha1 truncated for stability across runs
            identity["seq_hash"] = hashlib.sha1(
                sequence.encode("utf-8")).hexdigest()[:FP_HASH_LEN]
        if extra:
            for k, v in extra.items():
                if k not in identity:
                    identity[k] = v
        return identity

    @staticmethod
    def _fp_hash(smiles: str) -> Optional[str]:
        """Canonical SMILES → Morgan FP → truncated sha1 hex."""
        if not smiles:
            return None
        if not _RDKIT:
            # Fallback: hash the raw SMILES string. Loses canonicalization
            # benefits but still gives stable dedup for identical strings.
            return hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:FP_HASH_LEN]
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp  = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
            bits = fp.ToBitString()
            return hashlib.sha1(bits.encode("ascii")).hexdigest()[:FP_HASH_LEN]
        except Exception:
            return None

    @staticmethod
    def _murcko(smiles: str) -> Optional[str]:
        if not _RDKIT or not smiles:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        except Exception:
            return None

    @staticmethod
    def _heavy_atoms(smiles: str) -> Optional[int]:
        if not _RDKIT or not smiles:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol.GetNumHeavyAtoms() if mol else None
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # INTERNALS — STORAGE
    # ──────────────────────────────────────────────────────────────────────────

    def _append_event(self, event: FeedbackEvent) -> None:
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(event.to_jsonl())
            f.write("\n")

    def _load_index(self) -> dict:
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"fp_index": {}, "n_events": 0}

    def _save_index(self, idx: dict) -> None:
        # Atomic write: tmp file + rename
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(idx, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, self.index_path)

    def _update_index(self, event: FeedbackEvent) -> None:
        """
        Update fp_index with this event. Each entry tracks:
            best_score, n_observations, first_seen, last_seen, last_event_id
        """
        idx = self._load_index()
        fp_index: dict = idx.get("fp_index", {})

        fph = event.identity.get("fp_hash") or event.identity.get("seq_hash")
        if fph:
            entry = fp_index.get(fph, {
                "best_score":      None,
                "best_kind":       None,
                "n_observations":  0,
                "first_seen":      event.timestamp,
                "last_seen":       event.timestamp,
                "last_event_id":   event.event_id,
                "identity_repr":   event.identity.get("smiles")
                                    or event.identity.get("sequence", "")[:40],
                "modality":        event.modality,
            })
            entry["n_observations"] += 1
            entry["last_seen"]       = event.timestamp
            entry["last_event_id"]   = event.event_id

            # Track best score (lower=better for docking; upstream can rescore)
            new_primary = event.score["primary"]
            if event.score.get("valid", True):
                if entry["best_score"] is None or new_primary < entry["best_score"]:
                    entry["best_score"] = new_primary
                    entry["best_kind"]  = event.score["primary_kind"]

            fp_index[fph] = entry

        idx["fp_index"] = fp_index
        idx["n_events"] = idx.get("n_events", 0) + 1
        self._save_index(idx)

    def _refresh_stats(self) -> dict:
        """
        Recompute aggregate stats by streaming the events file.
        Cheap enough for stores up to ~1M events (linear scan).
        """
        stats = {
            "schema_version":   SCHEMA_VERSION,
            "uniprot_id":       self.uniprot_id,
            "refreshed_at":     datetime.now(timezone.utc)
                                  .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_events":         0,
            "n_valid":          0,
            "n_invalid":        0,
            "n_unique":         0,
            "n_runs":           0,
            "by_modality":      {},
            "by_target_site":   {},
            "best_per_site":    {},     # site_id -> {score, smiles, event_id}
        }

        seen_fp:  set[str] = set()
        seen_run: set[str] = set()

        for ev in self.iter_events():
            stats["n_events"] += 1
            valid = ev.score.get("valid", True)
            stats["n_valid"]    += int(valid)
            stats["n_invalid"]  += int(not valid)
            seen_run.add(ev.run_id)

            fph = ev.identity.get("fp_hash") or ev.identity.get("seq_hash")
            if fph:
                seen_fp.add(fph)

            stats["by_modality"][ev.modality] = \
                stats["by_modality"].get(ev.modality, 0) + 1
            stats["by_target_site"][ev.target_site] = \
                stats["by_target_site"].get(ev.target_site, 0) + 1

            if valid:
                site = ev.target_site
                primary = ev.score["primary"]
                cur = stats["best_per_site"].get(site)
                if cur is None or primary < cur["primary_score"]:
                    stats["best_per_site"][site] = {
                        "primary_score":  primary,
                        "primary_kind":   ev.score["primary_kind"],
                        "event_id":       ev.event_id,
                        "run_id":         ev.run_id,
                        "modality":       ev.modality,
                        "identity_repr":  ev.identity.get("smiles")
                                          or ev.identity.get("sequence", "")[:40],
                    }

        stats["n_unique"] = len(seen_fp)
        stats["n_runs"]   = len(seen_run)

        tmp = self.stats_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        os.replace(tmp, self.stats_path)
        return stats

    # ──────────────────────────────────────────────────────────────────────────
    # INTERNALS — LOCKING
    # ──────────────────────────────────────────────────────────────────────────

    @contextmanager
    def _lock(self):
        """
        Best-effort advisory lock around write operations.

        Linux/Mac: fcntl LOCK_EX, blocking. Robust against concurrent runs.
        Windows / no-fcntl: noop. Concurrent writes to events.jsonl are still
        line-atomic on most filesystems, but index/stats can race — recommend
        single-writer per protein on those platforms.
        """
        if not _HAS_FCNTL:
            yield
            return

        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    # ──────────────────────────────────────────────────────────────────────────
    # INTERNALS — PATH RESOLUTION
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_root(root_dir: Optional[Path]) -> Path:
        if root_dir is not None:
            return Path(root_dir).resolve()
        # Try config
        try:
            from utils.config import cfg  # type: ignore
            base = cfg.paths.get("feedback") if hasattr(cfg, "paths") \
                   else cfg.get("paths", "feedback")
            if base:
                return Path(base).resolve()
            # Fallback: sibling of intermediate
            inter = cfg.paths.get("intermediate") if hasattr(cfg, "paths") \
                    else cfg.get("paths", "intermediate")
            if inter:
                return (Path(inter).parent / "feedback").resolve()
        except Exception:
            pass
        # Last resort: ./data/feedback under cwd
        return (Path.cwd() / "data" / "feedback").resolve()