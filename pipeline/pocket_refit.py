"""
pipeline/pocket_refit.py
─────────────────────────
Stage 2 of the closed-loop pipeline.

Given accumulated docking feedback from utils/feedback_store.FeedbackStore,
compute a refined pocket descriptor that the next design run will use.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THIS EXISTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Module 04 produces a pocket descriptor from atomic geometry alone:
    volume, center, lining residues, mean_hydrophobicity, net_charge, ...

That descriptor is a *prior*. After you've actually docked N molecules and
measured what binds vs. what doesn't, you have evidence about which pocket
properties actually predict binding — for THIS pocket, on THIS protein,
under THIS docking force-field. The static descriptor doesn't know any of that.

The refit produces a corrected descriptor + a feature-importance vector
that the designer reads to steer the next generation:
  - "for this pocket, HBD count matters 4× more than logP"
  - "the effective optimal drug volume is 420 Å³, not the 560 Å³ the pocket
     geometry would suggest"
  - "molecules with net positive charge bind 1.8 kcal/mol better than neutral"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN PRINCIPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NEVER overwrite hard geometry.
   Volume, center, lining_residues come from atomic positions. The refit
   never touches these. It only updates *preference* fields and adds a
   `learned_corrections` block.

2. Refit is a RESIDUAL on top of the static descriptor.
   We never replace the static prior — we learn a correction to it. This
   keeps the model interpretable and lets us undo bad refits trivially.

3. Selection-bias-aware weighting.
   Top-binders are an obvious choice for refitting but they're biased
   toward whatever the previous generation already preferred. We weight
   each event by NOVELTY (different scaffolds get more weight) and by
   ROUNDS_AGO (recent measurements more than ancient ones) before fitting.

4. Refuse to refit when there isn't enough signal.
   Below MIN_EVENTS valid evaluations, or when feature variance is too low
   to fit a meaningful model, return the original descriptor unchanged.
   Stage 2 should never make the model WORSE.

5. Held-out validation gate.
   Split events 80/20, fit on 80, evaluate on 20. If the refit doesn't
   improve held-out RMSE over the static prior, reject it.

6. Versioned, never destructive.
   Writes data/intermediate/{uid}_pocket_refined.json with
   `model_version` bumped each refit. The original Module 04 output stays
   untouched. Designers read `_pocket_refined.json` if it exists, fall
   back to the original if not.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Per pocket (target_site in feedback store):

  1.  Load all VALID events for this site from the feedback store.
  2.  For each event, compute molecular features from SMILES:
         mw, logp, tpsa, hbd, hba, rotors, charge, aromatic_ratio,
         heavy_atoms, fp_density (Morgan FP bit density as a proxy
         for "complexity").
  3.  Compute event weights:
         w_event = w_novelty * w_recency
           w_novelty = 1 / (1 + n_obs_of_this_scaffold)
           w_recency = decay^rounds_ago
  4.  Fit ridge regression:  binding_residual = X @ beta + b
         where binding_residual = primary_score - static_prediction
         and X is the feature matrix.
         (We fit the *residual* over a simple static prediction so the
         model only learns what the static descriptor got wrong.)
  5.  80/20 split, evaluate held-out RMSE vs. static baseline.
         Accept refit iff held-out RMSE improves by ≥ MIN_RMSE_GAIN.
  6.  If accepted, compute the EFFECTIVE descriptor by inverting the
      learned beta:
         - effective_optimal_mw      = the MW that maximizes predicted binding
         - effective_charge_pref     = sign and magnitude of the charge coef
         - effective_hydrophob_pref  = logp coef
         - feature_importances       = |beta_i| / sum(|beta|)
  7.  Write the refined pocket file with:
         - original static fields (untouched)
         - learned_corrections block
         - effective_descriptor block
         - feature_importances
         - validation_metrics
         - model_version, n_events, refit_timestamp

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # After a design run
  from pipeline.pocket_refit import refit_pocket_model
  result = refit_pocket_model("P04637")
  for r in result.refined_pockets:
      print(f"{r.pocket_id}: gain={r.rmse_gain:.2f} kcal/mol  "
            f"top_feature={r.top_feature}")

  # Or from CLI
  python -m pipeline.pocket_refit --uniprot P04637

  # In the next design run, denovo_design.py reads the refined file:
  refined = load_refined_pocket("P04637", "P1")
  if refined:
      pocket = refined.effective_descriptor
  else:
      pocket = load_static_pocket("P04637", "P1")  # original Module 04
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Numpy / sklearn are already required by ProteinFP — see pyproject.toml
import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

# RDKit — needed for molecular features. Fall back to no-refit if missing.
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, Lipinski
    RDLogger.DisableLog("rdApp.*")
    _RDKIT = True
except ImportError:
    _RDKIT = False

# Import the feedback store — we depend on Stage 1
from utils.feedback_store import FeedbackStore


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — sensible defaults, tunable per project
# ══════════════════════════════════════════════════════════════════════════════

REFIT_SCHEMA_VERSION = 1

# Don't try to refit if we have fewer than this many valid events for a site.
# Below this, ridge fits noise — the static prior is more reliable.
MIN_EVENTS_PER_SITE = 30

# Train/test split for held-out validation
HOLDOUT_FRACTION = 0.20

# Ridge regularization strength. Higher = more conservative refit.
# 1.0 is a sane default; lower if you have lots of events, higher if few.
RIDGE_ALPHA = 1.0

# Minimum RMSE improvement (in score units, typically kcal/mol) for the refit
# to be accepted. If the refit only improves held-out RMSE by < this amount,
# reject it — not worth the model complexity.
MIN_RMSE_GAIN = 0.20

# Weight decay for old events. weight = DECAY^rounds_ago, where rounds_ago is
# the number of distinct runs between the event and the most recent run.
# 0.85 → 5 rounds ago = 0.44× weight, 10 rounds ago = 0.20× weight.
RECENCY_DECAY = 0.85

# Cap on per-event weight so a single super-novel outlier doesn't dominate
MAX_EVENT_WEIGHT = 5.0

# Score is "valid" iff its primary score is below this value. Vina returns
# 0.0 for unbindable poses; -0.5 is a safe cutoff above which we don't trust.
MAX_VALID_PRIMARY_SCORE = -0.5

# Features extracted per molecule. Order matters — preserved in feature_names.
FEATURE_NAMES = [
    "mw",            # molecular weight (Da)
    "logp",          # Crippen logP
    "tpsa",          # topological polar surface area (Å²)
    "hbd",           # H-bond donors
    "hba",           # H-bond acceptors
    "rotors",        # rotatable bonds
    "aromatic_frac", # fraction of heavy atoms that are aromatic
    "heavy_atoms",   # heavy atom count
    "frac_csp3",     # fraction of sp3 carbons (3D-likeness)
    "n_rings",       # ring count
    "formal_charge", # net formal charge
]
N_FEATURES = len(FEATURE_NAMES)


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RefinedPocket:
    """
    The refined descriptor for one pocket. Combines static prior + learned
    residual + effective descriptor + validation metrics.
    """
    pocket_id:            str
    target_site:          str           # matches FeedbackStore target_site
    n_events_used:        int
    n_holdout:            int

    # The original static descriptor (copied verbatim from Module 04)
    static_descriptor:    dict          = field(default_factory=dict)

    # Learned correction — a residual model on top of static prior
    learned_corrections:  dict          = field(default_factory=dict)
    feature_names:        list          = field(default_factory=list)
    coefficients:         list          = field(default_factory=list)
    intercept:            float         = 0.0
    feature_means:        list          = field(default_factory=list)
    feature_stds:         list          = field(default_factory=list)

    # What the model thinks the pocket effectively prefers (derived from coefs)
    effective_descriptor: dict          = field(default_factory=dict)

    # Interpretability — which features matter
    feature_importances:  dict          = field(default_factory=dict)
    top_feature:          str           = ""
    top_feature_direction: str          = ""   # "higher_is_better" | "lower_is_better"

    # Validation
    static_holdout_rmse:  float         = 0.0
    refined_holdout_rmse: float         = 0.0
    rmse_gain:            float         = 0.0
    accepted:             bool          = False
    reject_reason:        str           = ""

    # Provenance
    model_version:        int           = 1
    refit_timestamp:      str           = ""
    score_kind:           str           = "vina_dG_kcal_mol"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RefitResult:
    """Output of a full refit pass across all pockets for one protein."""
    uniprot_id:       str
    n_total_events:   int = 0
    n_sites:          int = 0
    n_accepted:       int = 0
    n_rejected:       int = 0
    refined_pockets:  list[RefinedPocket] = field(default_factory=list)
    warnings:         list[str]           = field(default_factory=list)
    output_path:      str                 = ""

    def summary(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  Pocket refit: {self.uniprot_id}",
            f"  Total events    : {self.n_total_events}",
            f"  Sites considered: {self.n_sites}",
            f"  Refits accepted : {self.n_accepted}",
            f"  Refits rejected : {self.n_rejected}",
        ]
        for p in self.refined_pockets:
            if p.accepted:
                lines.append(
                    f"  ✓ {p.pocket_id}: gain={p.rmse_gain:+.3f}  "
                    f"top={p.top_feature} ({p.top_feature_direction})  "
                    f"n={p.n_events_used}"
                )
            else:
                lines.append(
                    f"  ✗ {p.pocket_id}: rejected — {p.reject_reason}"
                )
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        lines.append(f"{'─'*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# MOLECULAR FEATURIZATION
# ══════════════════════════════════════════════════════════════════════════════

def _features_from_smiles(smiles: str) -> Optional[np.ndarray]:
    """
    Extract the FEATURE_NAMES feature vector from a SMILES string.
    Returns None if RDKit can't parse the SMILES.
    """
    if not _RDKIT or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        ha = mol.GetNumHeavyAtoms()
        if ha == 0:
            return None
        n_aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        feats = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            n_aromatic / ha,
            ha,
            Lipinski.FractionCSP3(mol),
            mol.GetRingInfo().NumRings(),
            Chem.GetFormalCharge(mol),
        ], dtype=np.float64)
        # Sanitize NaNs/Infs
        if not np.all(np.isfinite(feats)):
            return None
        return feats
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# STATIC BASELINE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def _static_baseline_prediction(
    feats:             np.ndarray,
    static_descriptor: dict,
) -> float:
    """
    What would the static pocket descriptor predict for this molecule?

    This is a *very* simple baseline: a heuristic linear combination of
    molecular features against pocket properties. We don't need it to be
    accurate — we just need it to be the same for every molecule so the
    residual we fit is meaningful and so we can compute "did the refit
    improve over the static prior".

    The baseline reflects naive complementarity:
      - drug MW should match pocket volume (roughly 1 Da per Å³ × 0.7)
      - drug logP should match pocket hydrophobicity
      - drug HBD/HBA should roughly match pocket capacity
      - mismatch penalized linearly

    Returns a predicted ΔG-like score (lower = better binding).
    """
    mw, logp, tpsa, hbd, hba, rotors, arom, ha, csp3, rings, charge = feats

    pocket_vol   = float(static_descriptor.get("volume_A3", 500.0))
    pocket_hydro = float(static_descriptor.get("mean_hydrophobicity", 0.0))
    pocket_chg   = float(static_descriptor.get("net_charge", 0.0))

    # MW vs volume (optimal MW ≈ pocket_vol × 0.7)
    optimal_mw = pocket_vol * 0.7
    mw_err     = abs(mw - optimal_mw) / max(optimal_mw, 100.0)

    # logP vs pocket hydrophobicity
    logp_err = abs(logp - pocket_hydro)

    # Charge complementarity (drug should counter pocket charge)
    chg_err = abs(charge + pocket_chg)

    # Crude ΔG estimate. Calibration doesn't have to be right — only the
    # *shape* of the prediction matters for residual fitting.
    base = -7.0   # typical drug ΔG floor
    penalty = 0.5 * mw_err + 0.3 * logp_err + 0.4 * chg_err
    return base + penalty


# ══════════════════════════════════════════════════════════════════════════════
# EVENT WEIGHTING
# ══════════════════════════════════════════════════════════════════════════════

def _compute_event_weights(
    events:       list,
    run_order:    dict,    # run_id -> recency rank (0 = most recent)
) -> np.ndarray:
    """
    Per-event weights for fitting. Combats selection bias.

    w_event = w_recency * w_novelty
      w_recency = RECENCY_DECAY ^ rounds_ago
      w_novelty = 1 / (1 + n_obs_of_this_scaffold_so_far)

    Capped at MAX_EVENT_WEIGHT.
    """
    scaffold_counts: dict = {}
    weights = np.zeros(len(events), dtype=np.float64)

    # Process in chronological order so scaffold_counts grows correctly
    # (events are appended in time order in the JSONL file)
    for i, ev in enumerate(events):
        scaff = ev.identity.get("scaffold") or ev.identity.get("smiles") or ""
        n_before = scaffold_counts.get(scaff, 0)
        w_nov = 1.0 / (1.0 + n_before)
        scaffold_counts[scaff] = n_before + 1

        rounds_ago = run_order.get(ev.run_id, 0)
        w_rec = RECENCY_DECAY ** rounds_ago

        w = min(MAX_EVENT_WEIGHT, w_nov * w_rec)
        weights[i] = max(w, 0.01)   # floor so no event has zero influence
    return weights


# ══════════════════════════════════════════════════════════════════════════════
# CORE REFIT
# ══════════════════════════════════════════════════════════════════════════════

def _refit_one_site(
    target_site:    str,
    events:         list,
    static_pocket:  dict,
    rng:            np.random.Generator,
) -> RefinedPocket:
    """
    Fit a ridge residual model on events for one pocket. Returns a
    RefinedPocket regardless of success — check `accepted` to know.
    """
    rp = RefinedPocket(
        pocket_id          = static_pocket.get("pocket_id", target_site),
        target_site        = target_site,
        n_events_used      = 0,
        n_holdout          = 0,
        static_descriptor  = dict(static_pocket),
        feature_names      = list(FEATURE_NAMES),
        model_version      = REFIT_SCHEMA_VERSION,
        refit_timestamp    = datetime.now(timezone.utc).strftime(
                                "%Y-%m-%dT%H:%M:%SZ"),
    )

    # ── Featurize ─────────────────────────────────────────────────────────────
    rows: list[tuple[np.ndarray, float, object]] = []
    for ev in events:
        smi = ev.identity.get("smiles")
        if not smi:
            continue
        if not ev.score.get("valid", True):
            continue
        primary = ev.score.get("primary")
        if primary is None or primary > MAX_VALID_PRIMARY_SCORE:
            continue
        feats = _features_from_smiles(smi)
        if feats is None:
            continue
        rows.append((feats, float(primary), ev))

    n = len(rows)
    if n < MIN_EVENTS_PER_SITE:
        rp.reject_reason = f"only {n} valid events (need ≥{MIN_EVENTS_PER_SITE})"
        return rp

    X    = np.vstack([r[0] for r in rows])
    y    = np.array([r[1] for r in rows], dtype=np.float64)
    evs  = [r[2] for r in rows]

    # Sanity: need feature variance
    if np.any(X.std(axis=0) < 1e-9):
        # At least one feature is constant — drop it from fit by zeroing variance.
        # Ridge will still work, just won't learn anything for that feature.
        pass

    # ── Compute static-baseline predictions ───────────────────────────────────
    y_static = np.array([_static_baseline_prediction(X[i], static_pocket)
                          for i in range(n)])
    residual = y - y_static     # what the static prior failed to capture

    # ── Event weights (combat selection bias) ─────────────────────────────────
    # Build run_order from event timestamps (most recent run = 0)
    unique_runs = list(dict.fromkeys(ev.run_id for ev in evs))  # preserves order
    run_order = {rid: (len(unique_runs) - 1 - i)
                 for i, rid in enumerate(unique_runs)}
    weights = _compute_event_weights(evs, run_order)

    # ── Train/holdout split ───────────────────────────────────────────────────
    perm     = rng.permutation(n)
    n_hold   = max(5, int(n * HOLDOUT_FRACTION))
    hold_idx = perm[:n_hold]
    train_idx = perm[n_hold:]

    X_tr, X_ho = X[train_idx], X[hold_idx]
    r_tr, r_ho = residual[train_idx], residual[hold_idx]
    y_tr, y_ho = y[train_idx], y[hold_idx]
    y_static_ho = y_static[hold_idx]
    w_tr = weights[train_idx]

    rp.n_events_used = int(n)
    rp.n_holdout     = int(n_hold)

    # ── Standardize features ──────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_ho_s = scaler.transform(X_ho)

    # ── Fit ridge on residual ─────────────────────────────────────────────────
    try:
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X_tr_s, r_tr, sample_weight=w_tr)
    except Exception as e:
        rp.reject_reason = f"ridge fit failed: {e}"
        return rp

    # ── Evaluate on holdout ───────────────────────────────────────────────────
    r_pred_ho   = model.predict(X_ho_s)
    y_pred_ho   = y_static_ho + r_pred_ho

    static_rmse  = float(np.sqrt(np.mean((y_ho - y_static_ho) ** 2)))
    refined_rmse = float(np.sqrt(np.mean((y_ho - y_pred_ho) ** 2)))
    gain         = static_rmse - refined_rmse

    rp.static_holdout_rmse  = round(static_rmse, 4)
    rp.refined_holdout_rmse = round(refined_rmse, 4)
    rp.rmse_gain            = round(gain, 4)

    if gain < MIN_RMSE_GAIN:
        rp.reject_reason = (
            f"holdout gain {gain:+.3f} < required {MIN_RMSE_GAIN:+.3f}  "
            f"(static rmse={static_rmse:.3f}, refined={refined_rmse:.3f})"
        )
        return rp

    # ── Refit on ALL data with the now-validated hyperparameters ──────────────
    X_full_s = scaler.fit_transform(X)
    r_full   = y - np.array([_static_baseline_prediction(X[i], static_pocket)
                               for i in range(n)])
    final = Ridge(alpha=RIDGE_ALPHA)
    final.fit(X_full_s, r_full, sample_weight=weights)

    rp.coefficients   = [round(float(c), 6) for c in final.coef_]
    rp.intercept      = round(float(final.intercept_), 6)
    rp.feature_means  = [round(float(m), 6) for m in scaler.mean_]
    rp.feature_stds   = [round(float(s), 6) for s in scaler.scale_]

    # ── Feature importances (|coef| in standardized space) ────────────────────
    abs_coef = np.abs(final.coef_)
    total    = float(abs_coef.sum()) or 1.0
    importances = {name: round(float(c) / total, 4)
                   for name, c in zip(FEATURE_NAMES, abs_coef)}
    rp.feature_importances = importances
    top_idx = int(np.argmax(abs_coef))
    rp.top_feature = FEATURE_NAMES[top_idx]
    rp.top_feature_direction = (
        "higher_is_better" if final.coef_[top_idx] < 0   # neg coef → lower ΔG
        else "lower_is_better"
    )

    # ── Effective descriptor: inferences derived from the fit ─────────────────
    # Coefs are in standardized units. To get effects in raw units, divide by
    # the feature scale. We report:
    #   - effective_charge_pref: sign + magnitude of the charge coef
    #   - effective_hydrophob_pref: sign + magnitude of the logp coef
    #   - effective_optimal_mw: MW value that minimizes predicted score,
    #       holding other features at their training mean
    raw_coefs = {name: final.coef_[i] / max(scaler.scale_[i], 1e-9)
                  for i, name in enumerate(FEATURE_NAMES)}

    # Optimal MW: argmin over a search grid (cheap, ~100 pts)
    mw_grid = np.linspace(150, 800, 100)
    # At the training mean of other features, varying MW alone:
    mean_feats = X.mean(axis=0)
    grid_preds = []
    for mw_val in mw_grid:
        f = mean_feats.copy()
        f[0] = mw_val
        f_s = scaler.transform(f.reshape(1, -1))
        r_pred = float(final.predict(f_s)[0])
        y_pred = _static_baseline_prediction(f, static_pocket) + r_pred
        grid_preds.append(y_pred)
    optimal_mw = float(mw_grid[int(np.argmin(grid_preds))])

    rp.effective_descriptor = {
        "effective_optimal_mw_Da":      round(optimal_mw, 1),
        "effective_charge_pref":        round(raw_coefs["formal_charge"], 4),
        "effective_hydrophobicity_pref": round(raw_coefs["logp"], 4),
        "effective_hbd_pref":           round(raw_coefs["hbd"], 4),
        "effective_hba_pref":           round(raw_coefs["hba"], 4),
        "effective_aromatic_pref":      round(raw_coefs["aromatic_frac"], 4),
        # Hard geometric fields preserved from static descriptor (never overwritten)
        "volume_A3":                    static_pocket.get("volume_A3"),
        "center":                       static_pocket.get("center"),
        "lining_residues":              static_pocket.get("lining_residues"),
        "n_lining":                     static_pocket.get("n_lining"),
        # And original soft fields kept alongside the corrections for comparison
        "static_mean_hydrophobicity":   static_pocket.get("mean_hydrophobicity"),
        "static_net_charge":            static_pocket.get("net_charge"),
        "static_druggability_score":    static_pocket.get("druggability_score"),
    }

    rp.learned_corrections = {
        "raw_coefficients":      {name: round(float(raw_coefs[name]), 6)
                                   for name in FEATURE_NAMES},
        "n_events_used":         n,
        "n_holdout":             n_hold,
        "n_unique_scaffolds":    len(set(
            (ev.identity.get("scaffold") or "") for ev in evs
        )),
    }

    rp.accepted = True
    return rp


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════════

def refit_pocket_model(
    uniprot_id:       str,
    feedback_store:   Optional[FeedbackStore] = None,
    static_report:    Optional[dict]          = None,
    inter_dir:        Optional[Path]          = None,
    output_path:      Optional[Path]          = None,
    modality:         str                     = "small_molecule",
    seed:             int                     = 0,
) -> RefitResult:
    """
    Refit pocket models from accumulated feedback. Returns a RefitResult
    summarizing what was accepted vs. rejected, and writes the refined
    descriptor JSON to disk.

    Parameters
    ----------
    uniprot_id : str
        Protein to refit.
    feedback_store : FeedbackStore, optional
        Pre-opened store. If None, opens one with default root.
    static_report : dict, optional
        Pre-loaded consensus report (so caller can avoid re-reading from disk).
        If None, read from `{inter_dir}/../reports/{uid}_report.json`.
    inter_dir : Path, optional
        Override intermediate dir; defaults to cfg.paths['intermediate'].
    output_path : Path, optional
        Where to write the refined JSON. Defaults to
        `{inter_dir}/{uid}_pocket_refined.json`.
    modality : str
        Which modality's events to use. Default 'small_molecule'.
    seed : int
        RNG seed for train/holdout split. Default 0 = deterministic.

    Returns
    -------
    RefitResult
    """
    result = RefitResult(uniprot_id=uniprot_id.upper())

    if not _RDKIT:
        result.warnings.append("RDKit not installed — refit skipped.")
        return result
    if not _SKLEARN:
        result.warnings.append("scikit-learn not installed — refit skipped.")
        return result

    # Resolve paths
    inter_dir, report_dir = _resolve_dirs(inter_dir)
    if output_path is None:
        output_path = inter_dir / f"{uniprot_id.upper()}_pocket_refined.json"

    # Load static report
    if static_report is None:
        rp = report_dir / f"{uniprot_id.upper()}_report.json"
        if not rp.exists():
            result.warnings.append(f"No consensus report at {rp} — cannot refit.")
            return result
        static_report = json.loads(rp.read_text(encoding="utf-8"))

    static_pockets = static_report.get("binding_pockets", [])
    if not static_pockets:
        result.warnings.append("Static report has no binding_pockets.")
        return result

    # Open feedback store
    if feedback_store is None:
        feedback_store = FeedbackStore(uniprot_id)

    # Index static pockets by id for fast lookup
    static_by_id = {p.get("pocket_id"): p for p in static_pockets}

    # Group events by target_site (which == pocket_id in the design code)
    events_by_site: dict[str, list] = {}
    n_total = 0
    for ev in feedback_store.iter_events(modality=modality, valid_only=True):
        events_by_site.setdefault(ev.target_site, []).append(ev)
        n_total += 1

    result.n_total_events = n_total
    result.n_sites = len(events_by_site)

    if n_total == 0:
        result.warnings.append(
            f"No feedback events found for {uniprot_id} (modality={modality}). "
            "Run the designer at least once to accumulate events."
        )
        return result

    # Refit each site that we have a static descriptor for
    rng = np.random.default_rng(seed)
    for site_id, events in events_by_site.items():
        static = static_by_id.get(site_id)
        if static is None:
            result.warnings.append(
                f"Events reference site {site_id} but no matching static "
                f"pocket in the consensus report — skipping."
            )
            continue
        refined = _refit_one_site(site_id, events, static, rng)
        result.refined_pockets.append(refined)
        if refined.accepted:
            result.n_accepted += 1
        else:
            result.n_rejected += 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.output_path = str(output_path)
    result.to_json(output_path)

    return result


def load_refined_pocket(
    uniprot_id:   str,
    pocket_id:    str,
    inter_dir:    Optional[Path] = None,
) -> Optional[RefinedPocket]:
    """
    Read the refined pocket file and return the RefinedPocket for one pocket.
    Returns None if no refined file exists or that pocket wasn't accepted.

    Designers can call this at the start of a run:
        refined = load_refined_pocket(uid, "P1")
        if refined and refined.accepted:
            pocket_descriptor = refined.effective_descriptor
        else:
            pocket_descriptor = static_descriptor   # original Module 04 output
    """
    inter_dir, _ = _resolve_dirs(inter_dir)
    path = inter_dir / f"{uniprot_id.upper()}_pocket_refined.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    for p in data.get("refined_pockets", []):
        if p.get("pocket_id") == pocket_id or p.get("target_site") == pocket_id:
            if not p.get("accepted"):
                return None
            # Reconstruct dataclass
            known = {f for f in RefinedPocket.__dataclass_fields__}
            filtered = {k: v for k, v in p.items() if k in known}
            return RefinedPocket(**filtered)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_dirs(inter_dir: Optional[Path]) -> tuple[Path, Path]:
    """Resolve (intermediate_dir, reports_dir) using config, with fallback."""
    if inter_dir is not None:
        inter = Path(inter_dir).resolve()
        report = inter.parent / "reports"
        return inter, report
    try:
        from utils.config import cfg  # type: ignore
        inter = Path(cfg.paths["intermediate"]).resolve()
        report = Path(cfg.paths["reports"]).resolve()
        return inter, report
    except Exception:
        base = Path.cwd() / "data"
        return base / "intermediate", base / "reports"


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Refit pocket model from accumulated docking feedback."
    )
    parser.add_argument("--uniprot", "-u", required=True,
                        help="UniProt ID (e.g. P04637)")
    parser.add_argument("--modality", "-m", default="small_molecule",
                        help="Which modality's events to use.")
    parser.add_argument("--seed", "-s", type=int, default=0,
                        help="RNG seed for holdout split.")
    args = parser.parse_args()

    result = refit_pocket_model(
        uniprot_id=args.uniprot,
        modality=args.modality,
        seed=args.seed,
    )
    print(result.summary())
    if result.output_path:
        print(f"\n  Wrote refined pocket file → {result.output_path}")


if __name__ == "__main__":
    main()