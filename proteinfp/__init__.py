"""
ProteinFP — end-to-end protein function prediction and drug candidate design.

Quick start:
    from proteinfp import run
    result = run("P04637")          # TP53
    print(result.report_path)

CLI:
    proteinfp --uniprot P04637
"""

from proteinfp.orchestrator import run_pipeline as run
from proteinfp.deps import status_report, has_rdkit, has_openmm, has_esm2

__all__ = ["run", "status_report", "has_rdkit", "has_openmm", "has_esm2"]

try:
    from importlib.metadata import version
    __version__ = version("proteinfp")
except Exception:
    __version__ = "0.1.0-dev"