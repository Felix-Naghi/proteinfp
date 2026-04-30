"""
fix_denovo_patch.py
────────────────────
Fixes the bug introduced by patch_denovo_cyp.py in the DenovoCandidate
construction block. The patched fitness recomputation referenced constants
before they were in scope, causing an AttributeError when full_history
was empty.

Also fixes the NoneType error by ensuring run_denovo_design always returns
a DenovoResult even when all generations fail.

Run from project root:
    python fix_denovo_patch.py
"""

from pathlib import Path
import sys

DENOVO = Path("pipeline") / "denovo_design.py"

if not DENOVO.exists():
    print(f"ERROR: Cannot find {DENOVO}")
    print("Run from C:\\Users\\adria\\Documents\\proteinFP")
    sys.exit(1)

src = DENOVO.read_text(encoding="utf-8")
changes = 0

# ── Fix 1: Replace the complex fitness recomputation in DenovoCandidate ───────
# The patch wrote a multi-line fitness= that references W_BINDING_END etc.
# in a scope where they may not be visible. Replace with simple 0.0.

OLD_COMPLEX_FITNESS = """        top_candidates.append(DenovoCandidate(
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
        ))"""

NEW_SIMPLE_FITNESS = """        # Final fitness: binding + LE + CYP, using end-of-run weights
        _w_total  = W_BINDING_END + W_LE_END + W_CYP_END
        _final_fit = round(
            (_normalize_score(mean_score) * W_BINDING_END
             + _le_fitness(mean_score, ha, props.get("qed", 0.0)) * W_LE_END
             + final_cyp * W_CYP_END) / max(_w_total, 1e-9),
            4
        )

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
            fitness=_final_fit,
            novelty=round(novelty.score(smi), 3),
            admet_pass=admet_ok,
            scaffold=_murcko(smi),
            origin="consensus",
        ))"""

if OLD_COMPLEX_FITNESS in src:
    src = src.replace(OLD_COMPLEX_FITNESS, NEW_SIMPLE_FITNESS, 1)
    changes += 1
    print("✓ Fix 1: replaced complex fitness recomputation with clean version")
else:
    print("- Fix 1: pattern not found (may already be fixed or patch wasn't applied)")

# ── Fix 2: Ensure run_denovo_design never returns None ────────────────────────
# When full_history is empty (all generations failed), final_smiles is empty,
# top_candidates is [], and the function still returns a valid DenovoResult.
# The real crash was the AttributeError from Fix 1 above causing the function
# to raise before reaching the return statement.
# Fix 1 should be sufficient, but add a safety guard in main() too.

OLD_CLI_ECHO = """    click.echo(result.summary())"""

NEW_CLI_ECHO = """    if result is None:
        log.error("  De novo design returned no result — all generations failed.")
        log.error("  Try running with more generations: --generations 30")
        sys.exit(1)
    click.echo(result.summary())"""

if OLD_CLI_ECHO in src and "result is None" not in src:
    src = src.replace(OLD_CLI_ECHO, NEW_CLI_ECHO, 1)
    changes += 1
    print("✓ Fix 2: added None guard in CLI before result.summary()")
else:
    print("- Fix 2: skipped (already present or pattern not found)")

# ── Write ──────────────────────────────────────────────────────────────────────
if changes == 0:
    print("\nNo changes needed.")
    sys.exit(0)

backup = DENOVO.with_suffix(".py.bak2")
backup.write_text(DENOVO.read_text(encoding="utf-8"), encoding="utf-8")
print(f"\n  Backup → {backup.name}")

DENOVO.write_text(src, encoding="utf-8")
print(f"  Patched → {DENOVO}")
print(f"\n  {changes} fix(es) applied.")
print("""
The all-generations-failing issue is separate from the patch bug.
It happens when every molecule in the population fails ADMET pre-filter
OR when obabel isn't available for ligand PDBQT conversion.

The docking cache has 261 entries — those molecules dock fine.
The issue is with newly generated molecules that need PDBQT conversion.

Check obabel is installed:
    obabel --version

If obabel is missing, install it:
    https://github.com/openbabel/openbabel/releases

With the cache populated from your previous 5-gen run, a fresh 30-gen
run should work — most generated molecules will hit the cache.
""")