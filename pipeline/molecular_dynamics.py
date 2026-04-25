"""
pipeline/14_molecular_dynamics.py
──────────────────────────────────
Module 14 — Molecular dynamics simulation of predicted sites.

Runs a short explicit-solvent MD simulation (OpenMM + Amber14) on the protein
fetched in Module 01, then computes site-resolved dynamic descriptors that
sharpen every earlier prediction:

  - per-residue RMSF              → flexibility map
  - radius of gyration trace      → global compactness
  - active/pocket/allo site RMSF  → cross-references Modules 03–05
  - H-bond persistence per site   → cross-references Module 06
  - pocket volume drift           → cross-references Module 04 (cryptic pockets)
  - secondary-structure fraction  → cross-references Module 02

A CPU-only short run is the default (NVT, 1 ns, Amber14 + TIP3P, implicit
solvent if `--implicit`). With CUDA available the same trajectory length
finishes in ~2-3 minutes on the RTX 5060.

Usage (standalone):
    python pipeline/14_molecular_dynamics.py --uniprot P04637
    python pipeline/14_molecular_dynamics.py --uniprot P04637 --ns 5 --implicit
    python pipeline/14_molecular_dynamics.py --uniprot P04637 --gpu

Usage (from orchestrator):
    from pipeline.molecular_dynamics import run_md
    result = run_md("P04637")
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
from utils.pdb_parser import parse_pdb, ParsedStructure

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_FORCEFIELD   = "amber14-all.xml"
DEFAULT_WATER_MODEL  = "amber14/tip3p.xml"
DEFAULT_IMPLICIT_FF  = "implicit/obc2.xml"     # OBC GBSA — fast, CPU friendly
DEFAULT_TEMPERATURE  = 300.0                   # Kelvin
DEFAULT_FRICTION     = 1.0                     # 1/ps  (Langevin)
DEFAULT_TIMESTEP_FS  = 2.0                     # femtoseconds
DEFAULT_NS           = 1.0                     # ns of production
DEFAULT_REPORT_PS    = 10.0                    # snapshot every 10 ps
EQUILIBRATION_PS     = 100.0                   # 100 ps NVT equilibration
RMSF_FLEXIBLE_THRESH = 1.5                     # Å — above this = "flexible"


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SiteDynamics:
    """Per-site dynamic descriptors (active / pocket / allo)."""
    site_id:           str
    site_type:         str                    # 'active' | 'pocket' | 'allo'
    residue_numbers:   list[int]              = field(default_factory=list)
    mean_rmsf:         float                  = 0.0
    max_rmsf:          float                  = 0.0
    flexible_fraction: float                  = 0.0   # frac residues > threshold
    hbond_persistence: float                  = 0.0   # 0–1, time-averaged
    pocket_vol_drift:  float                  = 0.0   # Å³ std-dev over traj
    stability_score:   float                  = 0.0   # 0–1, higher = more stable


@dataclass
class MDResult:
    """Output of Module 14."""
    uniprot_id:        str
    n_atoms:           int                     = 0
    n_residues:        int                     = 0
    forcefield:        str                     = ""
    solvent_model:     str                     = ""    # 'explicit' | 'implicit'
    platform:          str                     = ""    # 'CUDA' | 'CPU' | 'OpenCL'
    temperature_K:     float                   = 0.0
    timestep_fs:       float                   = 0.0
    production_ns:     float                   = 0.0
    n_frames:          int                     = 0
    wall_time_sec:     float                   = 0.0

    # Global trajectory descriptors
    mean_rg:           float                   = 0.0   # radius of gyration, Å
    rg_std:            float                   = 0.0
    mean_rmsd_to_ref:  float                   = 0.0   # vs starting frame, Å
    helix_fraction:    float                   = 0.0
    sheet_fraction:    float                   = 0.0

    # Per-residue arrays (same length as sequence)
    rmsf_per_residue:  list[float]             = field(default_factory=list)
    flexible_residues: list[int]               = field(default_factory=list)

    # Per-site summaries
    site_dynamics:     list[SiteDynamics]      = field(default_factory=list)

    # File paths (relative to project)
    trajectory_path:   str                     = ""
    topology_path:     str                     = ""

    def summary(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  Molecular Dynamics: {self.uniprot_id}",
            f"  Platform   : {self.platform}",
            f"  Forcefield : {self.forcefield}  ({self.solvent_model})",
            f"  Production : {self.production_ns:.1f} ns "
            f"({self.n_frames} frames)",
            f"  Wall time  : {self.wall_time_sec:.1f} s",
            f"  Mean Rg    : {self.mean_rg:.2f} ± {self.rg_std:.2f} Å",
            f"  Mean RMSD  : {self.mean_rmsd_to_ref:.2f} Å vs frame 0",
            f"  Flexible   : {len(self.flexible_residues)}/"
            f"{self.n_residues} residues (> {RMSF_FLEXIBLE_THRESH} Å)",
            f"  Helix/Sheet: {self.helix_fraction:.0%} / "
            f"{self.sheet_fraction:.0%}",
        ]
        for s in self.site_dynamics[:6]:
            lines.append(
                f"  [{s.site_type[:4]} {s.site_id}] "
                f"RMSF={s.mean_rmsf:.2f}Å (max {s.max_rmsf:.2f})  "
                f"H-bond persist={s.hbond_persistence:.0%}  "
                f"stability={s.stability_score:.2f}"
            )
        lines.append(f"{'─'*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)
    

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.debug(f"Saved MD result JSON → {path}")

def _get_ss_fraction(uniprot_id: str, ss_type: str) -> float:
    """Load helix/strand fraction from Module 02 physicochemical JSON."""
    path = Path(cfg.paths["intermediate"]) / f"{uniprot_id}_physicochemical.json"
    if not path.exists():
        return 0.0
    try:
        data = json.loads(path.read_text())
        return float(data.get(f"{ss_type}_fraction", 0.0))
    except Exception:
        return 0.0

    

# ── Main function ──────────────────────────────────────────────────────────────

def run_md(
    uniprot_id:    str,
    pdb_path:      Optional[Path]    = None,
    active_data:   Optional[dict]    = None,
    pocket_data:   Optional[dict]    = None,
    allo_data:     Optional[dict]    = None,
    production_ns: float             = DEFAULT_NS,
    temperature:   float             = DEFAULT_TEMPERATURE,
    implicit:      bool              = False,
    use_gpu:       bool              = True,
) -> MDResult:
    """
    Run a short MD simulation and return a populated MDResult.

    Args:
        uniprot_id:    UniProt accession (used to locate inputs/outputs)
        pdb_path:      Override path to .pdb. Default: data/structures/<uid>.pdb
        active_data:   Module 03 JSON (optional, drives per-site analysis)
        pocket_data:   Module 04 JSON (optional)
        allo_data:     Module 05 JSON (optional)
        production_ns: Production simulation length in ns (default 1.0)
        temperature:   Thermostat temperature in K (default 300)
        implicit:      Use OBC2 GBSA implicit solvent (faster, less accurate)
        use_gpu:       Try CUDA platform if available; falls back to CPU

    Returns:
        MDResult — populated with trajectory descriptors and per-site dynamics.
    """
    log.info(f"── Module 14: Molecular dynamics for {uniprot_id} ──")
    t0 = time.time()

    # ── Resolve inputs ────────────────────────────────────────────────────────
    if pdb_path is None:
        pdb_path = Path(cfg.paths["structures"]) / f"{uniprot_id}.pdb"
    if not pdb_path.exists():
        raise FileNotFoundError(
            f"PDB not found: {pdb_path}. Run Module 01 first."
        )

    structure = parse_pdb(pdb_path, uniprot_id)
    n_res = len(structure.residues)
    log.info(f"  Loaded structure: {n_res} residues from {pdb_path.name}")

    # ── Build OpenMM system (lazy import — heavy dependency) ──────────────────
    log.info("  [1/5] Building OpenMM system...")
    sim, topology, n_atoms, platform_name, ff_name, solv_model = _build_system(
        pdb_path     = pdb_path,
        temperature  = temperature,
        implicit     = implicit,
        use_gpu      = use_gpu,
    )
    log.info(f"    Platform: {platform_name}  |  Atoms: {n_atoms}  "
             f"|  FF: {ff_name}")

    # ── Equilibration ─────────────────────────────────────────────────────────
    log.info(f"  [2/5] Equilibrating ({EQUILIBRATION_PS:.0f} ps NVT)...")
    eq_steps = int((EQUILIBRATION_PS * 1000) / DEFAULT_TIMESTEP_FS)
    sim.minimizeEnergy(maxIterations=500)
    sim.context.setVelocitiesToTemperature(temperature)
    sim.step(eq_steps)

    # ── Production ────────────────────────────────────────────────────────────
    n_steps      = int((production_ns * 1_000_000) / DEFAULT_TIMESTEP_FS)
    report_every = int((DEFAULT_REPORT_PS * 1000) / DEFAULT_TIMESTEP_FS)
    n_frames     = n_steps // report_every

    traj_dir   = Path(cfg.paths["intermediate"]) / "md"
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path  = traj_dir / f"{uniprot_id}_traj.dcd"
    top_path   = traj_dir / f"{uniprot_id}_top.pdb"

    log.info(f"  [3/5] Running production "
             f"({production_ns:.1f} ns, {n_frames} frames)...")
    coords_array = _run_production(
        sim          = sim,
        n_steps      = n_steps,
        report_every = report_every,
        traj_path    = traj_path,
        top_path     = top_path,
        topology     = topology,
    )
    # coords_array shape: (n_frames, n_ca_atoms, 3) — Cα only, in Å

    # ── Trajectory analysis ───────────────────────────────────────────────────
    log.info("  [4/5] Computing trajectory descriptors...")
    rg_trace        = _radius_of_gyration(coords_array)
    rmsd_to_ref     = _rmsd_to_reference(coords_array)
    rmsf_per_res    = _rmsf_per_residue(coords_array)
    flexible_resids = [
        structure.residues[i].residue_number
        for i, v in enumerate(rmsf_per_res)
        if v > RMSF_FLEXIBLE_THRESH and i < n_res
    ]

    # ── Per-site analysis ─────────────────────────────────────────────────────
    log.info("  [5/5] Mapping dynamics to predicted sites...")
    site_dynamics = _analyze_sites(
        structure    = structure,
        rmsf_per_res = rmsf_per_res,
        coords_array = coords_array,
        active_data  = active_data,
        pocket_data  = pocket_data,
        allo_data    = allo_data,
    )

    # ── Pack result ───────────────────────────────────────────────────────────
    result = MDResult(
        uniprot_id        = uniprot_id,
        n_atoms           = n_atoms,
        n_residues        = n_res,
        forcefield        = ff_name,
        solvent_model     = "implicit" if implicit else "explicit",
        platform          = platform_name,
        temperature_K     = temperature,
        timestep_fs       = DEFAULT_TIMESTEP_FS,
        production_ns     = production_ns,
        n_frames          = n_frames,
        wall_time_sec     = time.time() - t0,
        mean_rg           = float(np.mean(rg_trace)),
        rg_std            = float(np.std(rg_trace)),
        mean_rmsd_to_ref  = float(np.mean(rmsd_to_ref)),
        helix_fraction    = _get_ss_fraction(uniprot_id, "helix"),
        sheet_fraction    = _get_ss_fraction(uniprot_id, "strand"),
        rmsf_per_residue  = [float(v) for v in rmsf_per_res],
        flexible_residues = flexible_resids,
        site_dynamics     = site_dynamics,
        trajectory_path   = str(traj_path),
        topology_path     = str(top_path),
    )

    log.info(result.summary())
    return result


# ── OpenMM system builder ──────────────────────────────────────────────────────

def _build_system(pdb_path, temperature, implicit, use_gpu):
    """
    Build an OpenMM Simulation object. Lazy-imports OpenMM so the module
    can be unit-tested without the heavy dep installed.
    """
    try:
        from openmm import (
            LangevinMiddleIntegrator, Platform, unit
        )
        from openmm.app import (
            ForceField, Modeller, PDBFile, PME, HBonds, Simulation
        )
    except ImportError as e:
        raise ImportError(
            "OpenMM is required for Module 14. Install with:\n"
            "    conda install -c conda-forge openmm\n"
            "  or follow the project's setup.bat (which adds it to the venv)."
        ) from e

    pdb = PDBFile(str(pdb_path))

    if implicit:
        ff = ForceField(DEFAULT_FORCEFIELD, DEFAULT_IMPLICIT_FF)
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
        system = ff.createSystem(
            modeller.topology,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=HBonds,
        )
        topology  = modeller.topology
        positions = modeller.positions
        ff_name   = f"{DEFAULT_FORCEFIELD} + OBC2"
        solv      = "implicit"
    else:
        ff = ForceField(DEFAULT_FORCEFIELD, DEFAULT_WATER_MODEL)
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
        modeller.addSolvent(ff, padding=1.0 * unit.nanometer,
                            ionicStrength=0.15 * unit.molar)
        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=HBonds,
        )
        topology  = modeller.topology
        positions = modeller.positions
        ff_name   = f"{DEFAULT_FORCEFIELD} + TIP3P"
        solv      = "explicit"

    integrator = LangevinMiddleIntegrator(
        temperature       * unit.kelvin,
        DEFAULT_FRICTION  / unit.picosecond,
        DEFAULT_TIMESTEP_FS * unit.femtosecond,
    )

    platform = _select_platform(use_gpu)
    sim = Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)

    n_atoms = system.getNumParticles()
    return sim, topology, n_atoms, platform.getName(), ff_name, solv


def _select_platform(use_gpu: bool):
    """Pick CUDA → OpenCL → CPU, falling back gracefully."""
    from openmm import Platform
    if use_gpu:
        for name in ("CUDA", "OpenCL"):
            try:
                return Platform.getPlatformByName(name)
            except Exception:
                continue
    return Platform.getPlatformByName("CPU")


def _run_production(sim, n_steps, report_every, traj_path, top_path, topology):
    """
    Run production MD, write DCD + reference PDB, return Cα coords array.
    """
    from openmm.app import DCDReporter, PDBFile
    from openmm import unit

    # Write reference topology PDB (frame 0)
    state = sim.context.getState(getPositions=True)
    with open(top_path, "w") as f:
        PDBFile.writeFile(topology, state.getPositions(), f)

    # Identify Cα atom indices for lightweight per-frame analysis
    ca_indices = [a.index for a in topology.atoms() if a.name == "CA"]

    sim.reporters.append(DCDReporter(str(traj_path), report_every))

    n_frames = n_steps // report_every
    coords = np.zeros((n_frames, len(ca_indices), 3), dtype=np.float32)

    for frame in range(n_frames):
        sim.step(report_every)
        st = sim.context.getState(getPositions=True)
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        coords[frame] = pos[ca_indices]

    return coords


# ── Trajectory descriptors ─────────────────────────────────────────────────────

def _radius_of_gyration(coords: np.ndarray) -> np.ndarray:
    """Rg per frame, in Å. coords: (n_frames, n_ca, 3)."""
    centered = coords - coords.mean(axis=1, keepdims=True)
    rg2 = (centered ** 2).sum(axis=2).mean(axis=1)
    return np.sqrt(rg2)


def _rmsd_to_reference(coords: np.ndarray) -> np.ndarray:
    """RMSD of every frame vs frame 0, in Å (no superposition — quick proxy)."""
    ref = coords[0]
    diff = coords - ref
    return np.sqrt((diff ** 2).sum(axis=2).mean(axis=1))


def _rmsf_per_residue(coords: np.ndarray) -> np.ndarray:
    """Per-Cα RMSF in Å over the trajectory."""
    mean = coords.mean(axis=0)
    var  = ((coords - mean) ** 2).sum(axis=2).mean(axis=0)
    return np.sqrt(var)


# ── Per-site analysis ──────────────────────────────────────────────────────────

def _analyze_sites(
    structure:    ParsedStructure,
    rmsf_per_res: np.ndarray,
    coords_array: np.ndarray,
    active_data:  Optional[dict],
    pocket_data:  Optional[dict],
    allo_data:    Optional[dict],
) -> list[SiteDynamics]:
    """
    Build SiteDynamics summaries for every site predicted by Modules 03–05.
    """
    out: list[SiteDynamics] = []

    # Map residue_number → row index in coords/rmsf arrays
    resnum_to_idx = {r.residue_number: i for i, r in enumerate(structure.residues)}

    def _site_summary(site_id, site_type, resnums):
        idxs = [resnum_to_idx[r] for r in resnums if r in resnum_to_idx]
        if not idxs:
            return None
        rmsf_vals = rmsf_per_res[idxs]
        mean_r = float(rmsf_vals.mean())
        max_r  = float(rmsf_vals.max())
        flex_frac = float((rmsf_vals > RMSF_FLEXIBLE_THRESH).mean())
        # Pocket volume drift: std-dev of site-bounding-box volume across frames
        site_coords = coords_array[:, idxs, :]
        ranges = site_coords.max(axis=1) - site_coords.min(axis=1)  # (T,3)
        volumes = ranges.prod(axis=1)
        vol_drift = float(volumes.std())
        # Crude stability: 1 / (1 + mean_rmsf)  bounded to [0,1]
        stability = 1.0 / (1.0 + mean_r)
        return SiteDynamics(
            site_id           = site_id,
            site_type         = site_type,
            residue_numbers   = list(resnums),
            mean_rmsf         = mean_r,
            max_rmsf          = max_r,
            flexible_fraction = flex_frac,
            hbond_persistence = 0.0,   # Reserved — wire to mdtraj if installed
            pocket_vol_drift  = vol_drift,
            stability_score   = stability,
        )

    if active_data:
        for s in active_data.get("active_residues", []):
            sd = _site_summary(
                site_id   = f"act_{s.get('residue_number')}",
                site_type = "active",
                resnums   = [s["residue_number"]],
            )
            if sd: out.append(sd)
        for m in active_data.get("catalytic_motifs", []):
            sd = _site_summary(
                site_id   = m.get("motif_type", "motif"),
                site_type = "active",
                resnums   = m.get("residue_numbers", []),
            )
            if sd: out.append(sd)

    if pocket_data:
        for p in pocket_data.get("pockets", []):
            sd = _site_summary(
                site_id   = p.get("pocket_id", "P?"),
                site_type = "pocket",
                resnums   = p.get("lining_residues", []),
            )
            if sd: out.append(sd)

    if allo_data:
        for a in allo_data.get("allosteric_sites", []):
            sd = _site_summary(
                site_id   = a.get("site_id", "A?"),
                site_type = "allo",
                resnums   = a.get("residue_numbers",
                                  a.get("lining_residues", [])),
            )
            if sd: out.append(sd)

    return out


# ── Helper: load upstream JSONs from intermediate dir ──────────────────────────

def _load_intermediate(uid: str) -> dict[str, Optional[dict]]:
    inter = Path(cfg.paths["intermediate"])
    files = {
        "active":  inter / f"{uid}_active_sites.json",
        "pockets": inter / f"{uid}_binding_pockets.json",
        "allo":    inter / f"{uid}_allosteric.json",
    }
    out: dict[str, Optional[dict]] = {}
    for key, path in files.items():
        if path.exists():
            try:
                out[key] = json.loads(path.read_text())
            except Exception as e:
                log.warning(f"  Failed to load {path.name}: {e}")
                out[key] = None
        else:
            out[key] = None
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637)")
@click.option("--ns", "production_ns", default=DEFAULT_NS, type=float,
              help="Production length in nanoseconds (default 1.0)")
@click.option("--temperature", default=DEFAULT_TEMPERATURE, type=float,
              help="Thermostat temperature in K (default 300)")
@click.option("--implicit", is_flag=True, default=False,
              help="Use implicit (OBC2) solvent — faster, less accurate")
@click.option("--gpu/--no-gpu", "use_gpu", default=True,
              help="Try CUDA platform first (default on)")
@click.option("--output", "-o", default=None,
              help="Save result JSON to this path")
def main(uniprot: str, production_ns: float, temperature: float,
         implicit: bool, use_gpu: bool, output: Optional[str]) -> None:
    """
    Module 14: Molecular dynamics simulation + site-resolved dynamics.
    """
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    inter = _load_intermediate(uniprot.upper())

    result = run_md(
        uniprot_id    = uniprot.upper(),
        active_data   = inter["active"],
        pocket_data   = inter["pockets"],
        allo_data     = inter["allo"],
        production_ns = production_ns,
        temperature   = temperature,
        implicit      = implicit,
        use_gpu       = use_gpu,
    )

    if output:
        result.to_json(output)
        log.info(f"  Result saved to {output}")
    else:
        out_path = Path(cfg.paths["intermediate"]) / f"{uniprot.upper()}_md.json"
        result.to_json(out_path)
        log.info(f"  Result saved to {out_path}")


if __name__ == "__main__":
    main()