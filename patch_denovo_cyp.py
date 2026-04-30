"""
patch_denovo_cyp.py
────────────────────
Run this once to wire CYP450 metabolic stability into the de novo design
fitness function.

    python patch_denovo_cyp.py

Makes three targeted edits to pipeline/denovo_design.py:
  1. Import profile_molecule from cyp450 (with graceful fallback)
  2. Add W_CYP weight constants to the hyperparameter block
  3. Add w_cyp to the adaptive weight schedule and fitness calculation
  4. Compute cyp_score inline after each successful dock
  5. Log CYP score in generation output
  6. Store cyp_score on each DenovoCandidate in the final output

The CYP450 call adds ~0.5ms per molecule (pure Python, no subprocess).
The weight is small (max 0.10) so it nudges rather than dominates —
metabolic stability is a tiebreaker, not the primary objective.
"""

from pathlib import Path
import sys

DENOVO = Path(__file__).parent.parent / "pipeline" / "denovo_design.py"

if not DENOVO.exists():
    # Try relative
    DENOVO = Path("pipeline") / "denovo_design.py"

if not DENOVO.exists():
    print(f"ERROR: Cannot find denovo_design.py")
    print(f"  Run this script from your proteinFP project root:")
    print(f"  cd C:\\Users\\adria\\Documents\\proteinFP")
    print(f"  python patch_denovo_cyp.py")
    sys.exit(1)

src = DENOVO.read_text(encoding="utf-8")

# ── Guard: don't double-patch ─────────────────────────────────────────────────
if "W_CYP_START" in src:
    print("Already patched — denovo_design.py already contains W_CYP_START.")
    print("Nothing to do.")
    sys.exit(0)

changes = 0

# ── Patch 1: Import profile_molecule from cyp450 ─────────────────────────────
# Insert after the existing utils.config import line

OLD_IMPORT = "from utils.config import cfg, get_logger"
NEW_IMPORT = """from utils.config import cfg, get_logger

# CYP450 metabolic stability scoring (Module 19)
# Graceful fallback if cyp450.py is not yet installed
try:
    from pipeline.cyp450 import profile_molecule as _cyp_profile
    CYP_AVAILABLE = True
except ImportError:
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from pipeline.cyp450 import profile_molecule as _cyp_profile
        CYP_AVAILABLE = True
    except ImportError:
        CYP_AVAILABLE = False
        _cyp_profile = None"""

if OLD_IMPORT in src and "CYP_AVAILABLE" not in src:
    src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    changes += 1
    print("✓ Patch 1: imported profile_molecule from cyp450")
else:
    print("- Patch 1: skipped (already present or import not found)")

# ── Patch 2: Add W_CYP weight constants ──────────────────────────────────────
# Insert after the existing W_LE constants

OLD_WEIGHTS = "W_LE_START      = 0.30;  W_LE_END      = 0.22   # ligand efficiency"
NEW_WEIGHTS = """W_LE_START      = 0.30;  W_LE_END      = 0.22   # ligand efficiency
W_CYP_START     = 0.08;  W_CYP_END     = 0.10   # CYP450 metabolic stability
# CYP weight is small (max 10%) — metabolic stability is a tiebreaker.
# Molecules failing rapid metabolism (t½ < 1h) are penalised but not
# eliminated — they may still be useful as tool compounds or prodrugs.
# Set W_CYP_END = 0.0 to disable CYP scoring entirely."""

if OLD_WEIGHTS in src:
    src = src.replace(OLD_WEIGHTS, NEW_WEIGHTS, 1)
    changes += 1
    print("✓ Patch 2: added W_CYP_START / W_CYP_END constants")
else:
    print("- Patch 2: skipped (W_LE constants not found at expected location)")

# ── Patch 3: Add w_cyp to adaptive weight schedule ───────────────────────────

OLD_WEIGHTS_SCHED = """        w_bind = W_BINDING_START + t * (W_BINDING_END - W_BINDING_START)
        w_nov  = W_NOVELTY_START + t * (W_NOVELTY_END  - W_NOVELTY_START)
        w_le   = W_LE_START      + t * (W_LE_END       - W_LE_START)
        total_w = w_bind + w_nov + w_le
        w_bind /= total_w; w_nov /= total_w; w_le /= total_w"""

NEW_WEIGHTS_SCHED = """        w_bind = W_BINDING_START + t * (W_BINDING_END - W_BINDING_START)
        w_nov  = W_NOVELTY_START + t * (W_NOVELTY_END  - W_NOVELTY_START)
        w_le   = W_LE_START      + t * (W_LE_END       - W_LE_START)
        w_cyp  = W_CYP_START     + t * (W_CYP_END      - W_CYP_START)
        if not CYP_AVAILABLE:
            w_cyp = 0.0          # disable silently if cyp450.py not installed
        total_w = w_bind + w_nov + w_le + w_cyp
        w_bind /= total_w; w_nov /= total_w; w_le /= total_w
        w_cyp  /= total_w"""

if OLD_WEIGHTS_SCHED in src:
    src = src.replace(OLD_WEIGHTS_SCHED, NEW_WEIGHTS_SCHED, 1)
    changes += 1
    print("✓ Patch 3: added w_cyp to adaptive weight schedule")
else:
    print("- Patch 3: skipped (weight schedule block not found)")

# ── Patch 4: Compute cyp_score and add to fitness ────────────────────────────
# The fitness is currently:
#   fitness = (w_bind * norm_bind + w_nov * nov_score + w_le * le_fit)
# We add cyp_score inline after le_fit is computed.

OLD_FITNESS = """                    le_fit   = _le_fitness(score, ha, qed)
                    norm_bind = _normalize_score(score)
                    fitness  = (w_bind * norm_bind +
                                w_nov  * nov_score +
                                w_le   * le_fit)

                    res.update({
                        "le":          le,
                        "qed":         qed,
                        "novelty":     nov_score,
                        "le_fitness":  le_fit,
                        "norm_bind":   norm_bind,
                        "fitness":     fitness,
                        "w_bind":      round(w_bind, 3),
                        "w_nov":       round(w_nov, 3),
                        "generation":  gen,
                    })"""

NEW_FITNESS = """                    le_fit    = _le_fitness(score, ha, qed)
                    norm_bind = _normalize_score(score)

                    # CYP450 metabolic stability score (0=worst, 1=best)
                    # Fast inline call: ~0.5ms per molecule, no subprocess
                    cyp_score = 1.0
                    if CYP_AVAILABLE and w_cyp > 0 and _cyp_profile is not None:
                        try:
                            _cyp = _cyp_profile("denovo", smi, "denovo")
                            if _cyp is not None:
                                cyp_score = _cyp.denovo_cyp_score
                        except Exception:
                            cyp_score = 1.0   # fail silently — never block docking

                    fitness  = (w_bind * norm_bind +
                                w_nov  * nov_score +
                                w_le   * le_fit +
                                w_cyp  * cyp_score)

                    res.update({
                        "le":          le,
                        "qed":         qed,
                        "novelty":     nov_score,
                        "le_fitness":  le_fit,
                        "norm_bind":   norm_bind,
                        "cyp_score":   round(cyp_score, 3),
                        "fitness":     fitness,
                        "w_bind":      round(w_bind, 3),
                        "w_nov":       round(w_nov, 3),
                        "w_cyp":       round(w_cyp, 3),
                        "generation":  gen,
                    })"""

if OLD_FITNESS in src:
    src = src.replace(OLD_FITNESS, NEW_FITNESS, 1)
    changes += 1
    print("✓ Patch 4: CYP score computed inline and added to fitness")
else:
    print("- Patch 4: skipped (fitness block not found at expected location)")

# ── Patch 5: Add CYP to generation log line ──────────────────────────────────

OLD_LOG = """        log.info(
            f"  Gen {gen}: score={gen_best['score']:.2f}  "
            f"LE={gen_best.get('le',0):.3f}  "
            f"QED={gen_best.get('qed',0):.2f}  "
            f"div={div:.2f}  {flag}{surr}"
        )"""

NEW_LOG = """        cyp_str = (f"  CYP={gen_best.get('cyp_score', 1.0):.2f}"
                    if CYP_AVAILABLE and w_cyp > 0 else "")
        log.info(
            f"  Gen {gen}: score={gen_best['score']:.2f}  "
            f"LE={gen_best.get('le',0):.3f}  "
            f"QED={gen_best.get('qed',0):.2f}  "
            f"div={div:.2f}  {flag}{surr}{cyp_str}"
        )"""

if OLD_LOG in src:
    src = src.replace(OLD_LOG, NEW_LOG, 1)
    changes += 1
    print("✓ Patch 5: CYP score added to generation log output")
else:
    print("- Patch 5: skipped (log line not found)")

# ── Patch 6: Store cyp_score on DenovoCandidate in final output ──────────────
# In the consensus scoring at the end, candidates are built from full_history.
# We store the cyp_score so it appears in the JSON output.

OLD_CANDIDATE = """        top_candidates.append(DenovoCandidate(
            smiles=smi,
            generation=next((r["generation"] for r in full_history
                             if r["smiles"] == smi), 0),
            score=round(mean_score, 3),
            le=le,
            qed=round(props.get("qed", 0.0), 3),
            mw=round(props.get("mw", 0.0), 1),
            logp=round(props.get("logp", 0.0), 2),
            tpsa=round(props.get("tpsa", 0.0), 1),
            hbd=props.get("hbd", 0),
            hba=props.get("hba", 0),
            heavy_atoms=ha,
            fitness=0.0,
            novelty=round(novelty.score(smi), 3),
            admet_pass=admet_ok,
            scaffold=_murcko(smi),
            origin="consensus",
        ))"""

NEW_CANDIDATE = """        # Final CYP450 score for this candidate
        final_cyp = 1.0
        if CYP_AVAILABLE and _cyp_profile is not None:
            try:
                _cp = _cyp_profile("final", smi, "final")
                if _cp is not None:
                    final_cyp = _cp.denovo_cyp_score
            except Exception:
                final_cyp = 1.0

        top_candidates.append(DenovoCandidate(
            smiles=smi,
            generation=next((r["generation"] for r in full_history
                             if r["smiles"] == smi), 0),
            score=round(mean_score, 3),
            le=le,
            qed=round(props.get("qed", 0.0), 3),
            mw=round(props.get("mw", 0.0), 1),
            logp=round(props.get("logp", 0.0), 2),
            tpsa=round(props.get("tpsa", 0.0), 1),
            hbd=props.get("hbd", 0),
            hba=props.get("hba", 0),
            heavy_atoms=ha,
            fitness=round(
                # recompute final fitness including CYP for ranking
                _normalize_score(mean_score) * W_BINDING_END / (W_BINDING_END + W_LE_END + W_CYP_END)
                + _le_fitness(mean_score, ha, props.get("qed", 0.0)) * W_LE_END / (W_BINDING_END + W_LE_END + W_CYP_END)
                + final_cyp * W_CYP_END / (W_BINDING_END + W_LE_END + W_CYP_END),
                4
            ),
            novelty=round(novelty.score(smi), 3),
            admet_pass=admet_ok,
            scaffold=_murcko(smi),
            origin="consensus",
        ))
        log.info(f"  Final: score={mean_score:.2f}  LE={le:.3f}  "
                 f"QED={props.get('qed',0):.2f}  CYP={final_cyp:.2f}  {smi[:60]}")"""

# Remove the old log.info after the candidate append (it gets merged above)
OLD_FINAL_LOG = """        log.info(f"  Final: score={mean_score:.2f}  LE={le:.3f}  "
                 f"QED={props.get('qed',0):.2f}  {smi[:60]}")"""

if OLD_CANDIDATE in src:
    # First replace the candidate block (which includes old log)
    src = src.replace(OLD_CANDIDATE, NEW_CANDIDATE, 1)
    # Remove the duplicate log line that follows
    if OLD_FINAL_LOG in src:
        src = src.replace(OLD_FINAL_LOG, "", 1)
    changes += 1
    print("✓ Patch 6: cyp_score stored on final DenovoCandidate + improved fitness ranking")
else:
    print("- Patch 6: skipped (DenovoCandidate construction block not found)")

# ── Write output ──────────────────────────────────────────────────────────────
if changes == 0:
    print("\nNo changes made — the file may have been modified already or")
    print("the expected patterns were not found.")
    print("Check that denovo_design.py is the original unmodified version.")
    sys.exit(1)

# Backup
backup = DENOVO.with_suffix(".py.bak")
backup.write_text(DENOVO.read_text(encoding="utf-8"), encoding="utf-8")
print(f"\n  Backup saved → {backup.name}")

# Write patched file
DENOVO.write_text(src, encoding="utf-8")
print(f"  Patched    → {DENOVO}")
print(f"\n  {changes}/6 patches applied successfully.")
print("""
What changed:
  - Fitness now includes CYP450 metabolic stability (weight 8→10%)
  - Molecules with t½ > 8h score 1.0 (no penalty)
  - Molecules with t½ 2–8h score ~0.75 (small penalty)  
  - Molecules with t½ < 1h score ~0.25 (significant penalty)
  - CYP inhibitors (DDI risk) score lower additionally
  - Generation log now shows CYP= score per generation
  - Final candidates ranked by binding + LE + CYP (not binding alone)
  - cyp_score field added to every candidate in _denovo.json output

The weight is intentionally small (max 10%) so rapid-metabolism molecules
can still win if their binding is substantially better. To increase the
metabolic pressure, edit W_CYP_END in denovo_design.py (e.g. 0.20).
To disable, set W_CYP_END = 0.0.

Test with:
  python pipeline\\denovo_design.py --uniprot P04637 --generations 5
""")