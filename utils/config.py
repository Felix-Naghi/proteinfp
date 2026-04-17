"""
utils/config.py
───────────────
Loads config.yaml and exposes a typed config object used by every module.
Import like: from utils.config import cfg
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import colorlog
import yaml

# ── Locate the project root ────────────────────────────────────────────────────
# Works regardless of which directory you run from inside the project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class _Config:
    """Thin wrapper around the YAML dict — gives dot-access to top-level keys."""

    def __init__(self, data: dict[str, Any]):
        self._data = data
        # Resolve all path values relative to project root
        for key, val in data.get("paths", {}).items():
            abs_path = PROJECT_ROOT / val
            abs_path.mkdir(parents=True, exist_ok=True)
            data["paths"][key] = str(abs_path)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'")

    def get(self, *keys: str, default: Any = None) -> Any:
        """Safe nested access: cfg.get('afdb', 'timeout_sec', default=30)"""
        node = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
        return node


cfg = _Config(_load_yaml(CONFIG_PATH))


# ── Logger factory ─────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Returns a coloured console logger + optional file logger.
    Usage: log = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    level_str: str = cfg.get("logging", "level", default="INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

    # Coloured console handler
    console = colorlog.StreamHandler()
    console.setLevel(level)
    console.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(name)s] %(levelname)s%(reset)s  %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        }
    ))
    logger.addHandler(console)

    # File handler (optional)
    if cfg.get("logging", "log_to_file", default=False):
        log_dir = Path(cfg.paths["logs"])
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)

    return logger
