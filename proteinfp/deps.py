"""
proteinfp/deps.py
──────────────────
Dependency availability checker.

Every optional dependency is checked ONCE at import time and the result
cached.  Modules throughout the pipeline call has_X() before importing
anything optional, so missing packages degrade gracefully rather than
crashing with an ImportError.

Usage in pipeline modules:
    from proteinfp.deps import has_openmm, has_rdkit, has_esm2, require

    if has_openmm():
        from openmm import ...    # safe — only runs if OpenMM is present
    else:
        log.info("  Module 14 skipped — OpenMM not installed")
        log.info("  Install: pip install proteinfp[sim]")

    # OR — raise with a helpful install message:
    require("rdkit", extra="chem",
            context="de novo molecular design (Module 15)")
"""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from typing import Optional

# ── Install hints ──────────────────────────────────────────────────────────────
# Maps package name → (pip extra, human description)
_INSTALL_HINTS: dict[str, tuple[str, str]] = {
    "openmm":      ("sim",       "molecular dynamics simulation (Module 14)"),
    "rdkit":       ("chem",      "de novo molecular design (Module 15)"),
    "fair_esm":    ("ml",        "ESM-2 protein language model (Module 08)"),
    "esm":         ("ml",        "ESM-2 protein language model (Module 08)"),
    "torch":       ("ml",        "ML-based EC classification (Module 10)"),
    "xgboost":     ("ml",        "ML-based EC classification (Module 10)"),
    "lightgbm":    ("ml",        "ML-based EC classification (Module 10)"),
    "freesasa":    ("structure", "SASA surface analysis (Module 02)"),
    "scanpy":      ("grn",       "scRNA-seq preprocessing (GRN-01)"),
    "anndata":     ("grn",       "scRNA-seq preprocessing (GRN-01)"),
    "vina":        (None,        "AutoDock Vina docking — install from https://vina.scripps.edu"),
}


@lru_cache(maxsize=None)
def _check(package: str) -> bool:
    """Return True if `package` is importable. Cached after first call."""
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False


# ── Public checkers ────────────────────────────────────────────────────────────

def has_openmm() -> bool:
    """OpenMM — required for Module 14 (molecular dynamics)."""
    return _check("openmm")


def has_rdkit() -> bool:
    """RDKit — required for Module 15 (de novo molecular design)."""
    return _check("rdkit")


def has_torch() -> bool:
    """PyTorch — required for ESM-2 and ML EC classifier."""
    return _check("torch")


def has_esm2() -> bool:
    """ESM-2 (fair-esm) — required for Module 08 (protein embeddings)."""
    return _check("esm") or _check("fair_esm")


def has_xgboost() -> bool:
    """XGBoost — required for ML EC ensemble."""
    return _check("xgboost")


def has_lightgbm() -> bool:
    """LightGBM — required for ML EC ensemble."""
    return _check("lightgbm")


def has_ml_stack() -> bool:
    """Full ML stack: PyTorch + XGBoost + LightGBM."""
    return has_torch() and has_xgboost() and has_lightgbm()


def has_freesasa() -> bool:
    """freesasa — required for SASA computation (Module 02)."""
    return _check("freesasa")


def has_scanpy() -> bool:
    """scanpy — required for scRNA-seq preprocessing (GRN-01)."""
    return _check("scanpy")


def has_grn_stack() -> bool:
    """Full GRN stack: scanpy + anndata."""
    return has_scanpy() and _check("anndata")


def has_vina(vina_path: Optional[str] = None) -> bool:
    """
    Check whether AutoDock Vina is available.
    Checks the PATH or an explicit path if given.
    """
    import shutil
    if vina_path:
        from pathlib import Path
        return Path(vina_path).exists()
    return shutil.which("vina") is not None or shutil.which("vina.exe") is not None


# ── Require helper (raises with install instructions) ─────────────────────────

def require(
    package:  str,
    extra:    Optional[str] = None,
    context:  str           = "",
) -> None:
    """
    Assert a package is available, raising ImportError with install
    instructions if not.

    Args:
        package:  importable package name (e.g. "rdkit", "openmm")
        extra:    pip extra to suggest (e.g. "chem", "sim")
        context:  human description of what this enables

    Example:
        require("rdkit", extra="chem", context="de novo design")
        # → ImportError: rdkit is required for de novo design.
        #   Install: pip install proteinfp[chem]
    """
    if _check(package):
        return

    hint_extra, hint_context = _INSTALL_HINTS.get(package, (extra, context))
    ctx_str  = context or hint_context or f"this feature"
    pkg_name = package.replace("_", "-")

    if hint_extra:
        install_cmd = f"pip install proteinfp[{hint_extra}]"
    else:
        install_cmd = f"pip install {pkg_name}"

    raise ImportError(
        f"\n"
        f"  {pkg_name} is required for {ctx_str}.\n"
        f"  Install: {install_cmd}\n"
    )


# ── Status report ──────────────────────────────────────────────────────────────

def status_report() -> str:
    """
    Return a human-readable table of all optional dependency statuses.
    Called by `proteinfp --check-deps`.
    """
    checks = [
        ("Core pipeline (Modules 01-13)",   True,               "always available"),
        ("SASA / DSSP (Module 02)",         has_freesasa(),     "pip install proteinfp[structure]"),
        ("ESM-2 embeddings (Module 08)",    has_esm2(),         "pip install proteinfp[ml]"),
        ("ML EC classifier (Module 10)",    has_ml_stack(),     "pip install proteinfp[ml]"),
        ("De novo design (Module 15)",      has_rdkit(),        "pip install proteinfp[chem]"),
        ("Molecular dynamics (Module 14)",  has_openmm(),       "pip install proteinfp[sim]"),
        ("GRN / scRNA-seq (GRN-01)",        has_grn_stack(),    "pip install proteinfp[grn]"),
        ("AutoDock Vina",                   has_vina(),         "https://vina.scripps.edu"),
    ]

    lines = [
        "",
        "  ProteinFP — dependency status",
        "  " + "─" * 55,
        f"  {'Component':<38} {'Status':>8}",
        "  " + "─" * 55,
    ]

    for name, available, install_hint in checks:
        status = "✓  OK" if available else "✗  missing"
        lines.append(f"  {name:<38} {status}")
        if not available and install_hint:
            lines.append(f"  {'':38}   → {install_hint}")

    lines += ["  " + "─" * 55, ""]
    return "\n".join(lines)