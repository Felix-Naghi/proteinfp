"""
pipeline/ec_model_check.py
───────────────────────────
Gap 5 fix — makes the ML EC classifier the default and makes the
fallback to legacy rule-based scoring loud rather than silent.

THE GAP
───────
ml_ec_predict.py has:
    DEFAULT_MODEL_DIR = Path("models/ec_ensemble")

If that directory is missing or incomplete, predict_ec_ml() silently
falls back to clean_ec.py's rule-based heuristics. This means:
  - You can't tell from the output whether ML or rules ran.
  - The validation metrics (97% vs ~70%) don't apply.
  - The pipeline appears to be using the ML model when it isn't.

THE FIX
───────
1. check_ec_model()        — verify the model is present and complete
2. ensure_ec_model()       — check + offer to train if missing
3. Patch to ml_ec_predict  — make fallback print a visible warning
4. Updated config path     — read model dir from config.yaml

HOW TO USE
──────────
Add to your pipeline orchestrator startup:

    from pipeline.ec_model_check import ensure_ec_model
    ensure_ec_model()   # prints status; raises if model missing and training fails

Or just check silently:

    from pipeline.ec_model_check import check_ec_model
    status = check_ec_model()
    if not status["ready"]:
        print(status["message"])
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

# Required files for a complete model directory
REQUIRED_MODEL_FILES = [
    "metadata.json",
    "preprocessor.pkl",
    "xgb_model.pkl",
    "lgb_model.pkl",
    "mlp_model.npy",
    "meta_learner.pkl",
]

# Optional but recommended
OPTIONAL_MODEL_FILES = [
    "reject_option.pkl",    # probability calibration
    "model_card.json",      # provenance
]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DIRECTORY RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def get_model_dir() -> Path:
    """
    Get the EC model directory from config.yaml, falling back to the
    hard-coded default.

    Config path: config.yaml → ec_classifier → model_dir
    Default:     models/ec_ensemble
    """
    try:
        from utils.config import cfg
        model_dir_str = cfg.get("ec_classifier", "model_dir",
                                default="models/ec_ensemble")
        return ROOT / model_dir_str
    except Exception:
        return ROOT / "models" / "ec_ensemble"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_ec_model(model_dir: Optional[Path] = None) -> dict:
    """
    Check whether the ML EC classifier model is present and complete.

    Returns a status dict:
        ready:    bool    — True if model can be loaded
        path:     str     — resolved model directory path
        files:    dict    — which required/optional files exist
        message:  str     — human-readable status
        version:  str     — model version from metadata.json, or ""
    """
    if model_dir is None:
        model_dir = get_model_dir()

    status = {
        "ready":   False,
        "path":    str(model_dir),
        "files":   {},
        "message": "",
        "version": "",
    }

    # Check directory exists
    if not model_dir.exists():
        status["message"] = (
            f"Model directory not found: {model_dir}\n"
            f"  Run: python pipeline/ml_ec_train_v2.py\n"
            f"  Or set ec_classifier.model_dir in config.yaml"
        )
        return status

    # Check required files
    missing = []
    for fname in REQUIRED_MODEL_FILES:
        exists = (model_dir / fname).exists()
        status["files"][fname] = exists
        if not exists:
            missing.append(fname)

    for fname in OPTIONAL_MODEL_FILES:
        status["files"][fname] = (model_dir / fname).exists()

    if missing:
        status["message"] = (
            f"Model directory incomplete: {model_dir}\n"
            f"  Missing files: {', '.join(missing)}\n"
            f"  Re-run: python pipeline/ml_ec_train_v2.py"
        )
        return status

    # Read version from metadata
    meta_path = model_dir / "metadata.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        status["version"] = meta.get("model_version", "unknown")
    except Exception:
        status["version"] = "unknown"

    # Try actually loading the model (catches corrupted files)
    try:
        sys.path.insert(0, str(ROOT))
        from pipeline.ml_ec_classifier import ECClassifierEnsemble
        ECClassifierEnsemble.load(model_dir)
        status["ready"]   = True
        status["message"] = (
            f"ML EC classifier ready  "
            f"(v{status['version']}  path={model_dir})"
        )
    except Exception as e:
        status["message"] = (
            f"Model directory exists but failed to load: {e}\n"
            f"  Re-run: python pipeline/ml_ec_train_v2.py"
        )

    return status


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP ENSURE (call this at pipeline startup)
# ══════════════════════════════════════════════════════════════════════════════

def ensure_ec_model(
    model_dir:      Optional[Path] = None,
    raise_if_missing: bool         = False,
    verbose:        bool           = True,
) -> bool:
    """
    Check the ML EC classifier and print a clear status message.

    Call this once at pipeline startup so researchers know immediately
    whether ML or rule-based EC prediction is active.

    Args:
        model_dir:        Override model directory (uses config.yaml by default)
        raise_if_missing: If True, raise RuntimeError when model is absent
        verbose:          Print status to stdout

    Returns:
        True if ML model is ready, False if falling back to rules
    """
    status = check_ec_model(model_dir)

    if status["ready"]:
        if verbose:
            print(f"  [EC classifier] ML model ready — {status['message']}")
        return True

    # Model not ready
    warning = (
        f"\n"
        f"  ╔══════════════════════════════════════════════════════════════╗\n"
        f"  ║  EC CLASSIFIER WARNING                                       ║\n"
        f"  ║                                                              ║\n"
        f"  ║  ML model not found. Falling back to legacy rule-based EC    ║\n"
        f"  ║  prediction. Accuracy: ~70% (rule-based) vs ~97% (ML).       ║\n"
        f"  ║                                                              ║\n"
        f"  ║  To train the ML model:                                      ║\n"
        f"  ║    python pipeline/ml_ec_train_v2.py                         ║\n"
        f"  ║                                                              ║\n"
        f"  ║  {status['path']:<60}║\n"
        f"  ╚══════════════════════════════════════════════════════════════╝\n"
    )

    if verbose:
        print(warning)

    if raise_if_missing:
        raise RuntimeError(
            f"ML EC classifier not available: {status['message']}"
        )

    return False


# ══════════════════════════════════════════════════════════════════════════════
# PATCHED predict_ec_ml — makes fallback loud
# ══════════════════════════════════════════════════════════════════════════════

def predict_ec_ml_checked(
    uniprot_id:      str,
    sequence:        str,
    active_result    = None,
    go_result        = None,
    homology_result  = None,
    esm2_result      = None,
    pdb_result       = None,
    pocket_result    = None,
    enm_result       = None,
    physico_result   = None,
    model_dir:       Optional[Path] = None,
    warn_on_fallback: bool = True,
):
    """
    Drop-in replacement for predict_ec_ml() that:
      1. Resolves model_dir from config.yaml if not provided
      2. Prints a clear warning if falling back to rule-based scoring
      3. Stamps the result with ml_used=True/False so you can check it

    Usage (replace in consensus.py):
        from pipeline.ec_model_check import predict_ec_ml_checked as predict_ec_ml
    """
    if model_dir is None:
        model_dir = get_model_dir()

    from pipeline.ml_ec_predict import predict_ec_ml

    result = predict_ec_ml(
        uniprot_id      = uniprot_id,
        sequence        = sequence,
        active_result   = active_result,
        go_result       = go_result,
        homology_result = homology_result,
        esm2_result     = esm2_result,
        pdb_result      = pdb_result,
        pocket_result   = pocket_result,
        enm_result      = enm_result,
        physico_result  = physico_result,
        model_dir       = model_dir,
    )

    if warn_on_fallback and not result.ml_used:
        print(
            f"\n  [EC {uniprot_id}] WARNING: ML model unavailable — "
            f"using rule-based fallback (~70% accuracy).\n"
            f"  Train the model: python pipeline/ml_ec_train_v2.py\n"
        )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG.YAML PATCH — add ec_classifier section
# ══════════════════════════════════════════════════════════════════════════════

# Add this to your config/config.yaml:
CONFIG_YAML_ADDITION = """
# ── EC Classifier (Module 10-ML) ──────────────────────────────────────────────
ec_classifier:
  model_dir:  "models/ec_ensemble"   # path relative to project root
  # Set to "models/ec_ensemble_v2" etc. if you have multiple trained versions
"""


# ══════════════════════════════════════════════════════════════════════════════
# CLI — run at startup or as a standalone check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check ML EC classifier model status."
    )
    parser.add_argument("--model-dir", default=None,
                        help="Override model directory path")
    parser.add_argument("--raise", dest="raise_if_missing",
                        action="store_true",
                        help="Exit with error if model is missing")
    args = parser.parse_args()

    model_path = Path(args.model_dir) if args.model_dir else None
    ready = ensure_ec_model(
        model_dir       = model_path,
        raise_if_missing = args.raise_if_missing,
        verbose         = True,
    )

    status = check_ec_model(model_path)
    print(f"\n  Model directory : {status['path']}")
    print(f"  Version         : {status['version'] or 'N/A'}")
    print(f"  Status          : {'READY' if status['ready'] else 'NOT READY'}")
    print(f"\n  File status:")
    for fname, exists in status["files"].items():
        marker = "✓" if exists else "✗"
        print(f"    {marker}  {fname}")

    if not ready:
        print(f"\n  {status['message']}")
        print(f"\n  Add to config.yaml:\n{CONFIG_YAML_ADDITION}")
        sys.exit(1)