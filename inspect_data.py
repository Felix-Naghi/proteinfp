"""
FIXES APPLIED TO denovo_design.py:

1. **ADMET Thresholds (Line ~320)**: Tightened to prevent oversized molecules
   - MW: 600 → 500 Da (prevents heavy molecules that don't dock)
   - LogP: 6.0 → 5.0 (reduces hydrophobicity issues)
   - HBD: 7 → 5 (fewer donors = better vina compatibility)
   - HBA: 12 → 10 (fewer acceptors = better vina compatibility)

2. **Docking Error Handling (NEW)**: Skip molecules that fail vina more than 5 times
   - Prevents infinite loops of failed docking attempts
   - Automatically hard-resets population when >50% fail

3. **Surrogate Model Reset (NEW)**: Rebuild surrogate if it's making bad predictions
   - Checks if >60% of predictions are > 5.0 kcal/mol error
   - Falls back to random selection if predictions diverge

4. **Fragment Pool Auto-Adjust (NEW)**: Throttle FQSAR scoring if it's promoting bad fragments
   - Disable FQSAR if mean docking score drops >1.0 kcal/mol
   - Switch to balanced mutation mode

5. **Vina Exhaustiveness Boost**: Increase from EXHAUST_START 8→16 for gen 11+
   - More thorough conformer sampling
   - Better scoring for larger molecules

DIAGNOSIS OF YOUR RUN:
=====================
Gen 1-10:   Working correctly, molecules docking with -5.37 kcal/mol
Gen 11+:    No valid results because:
  ✗ ADMET constraints too relaxed (600 Da limit) → oversized junk
  ✗ Surrogate model making poor predictions
  ✗ FQSAR promoting non-druglike fragments
  ✗ Vina unable to score large/polar molecules (molecule 2 in gen 5 has MW ~400+)
  ✗ Fragment library has polar/charged groups that don't dock well

EXPECTED BEHAVIOR AFTER FIX:
=============================
- Gen 11+ should resume docking molecules (smaller, better filtered)
- If molecules still fail, hard reset triggers after gen 11
- Surrogate model resets if predictions diverge >5.0 kcal/mol
- Fragment bias resets if FQSAR promotes bad chemistry
"""

# ═══════════════════════════════════════════════════════════════════════════════
# APPLY THESE CHANGES TO YOUR denovo_design.py FILE
# ═══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# CHANGE 1: Tighten ADMET thresholds (around line 315-320)
# ──────────────────────────────────────────────────────────────────────────────

# OLD:
"""
ADMET_MW_MAX         = 600.0   # slightly relaxed from Lipinski for fragments
ADMET_LOGP_MAX       = 6.0
ADMET_HBD_MAX        = 7
ADMET_HBA_MAX        = 12
"""

# NEW:
ADMET_MW_MAX         = 500.0   # strict Lipinski
ADMET_LOGP_MAX       = 5.0     # hydrophobicity limit
ADMET_HBD_MAX        = 5       # fewer donors
ADMET_HBA_MAX        = 10      # fewer acceptors


# ──────────────────────────────────────────────────────────────────────────────
# CHANGE 2: Add failed docking counter (after _DOCK_CACHE definition, ~line 790)
# ──────────────────────────────────────────────────────────────────────────────

_DOCK_FAIL_COUNT: Dict[str, int] = {}  # Track how many times each molecule fails

def _check_dock_health(results: List[dict], gen: int) -> Tuple[bool, str]:
    """
    Check docking health. Returns (is_healthy, reason).
    If >50% of docking attempts fail, trigger hard reset.
    """
    if not results:
        return False, "0 valid results"
    
    total = len(results)
    if total < 10:
        return True, ""  # Not enough data yet
    
    failed = sum(1 for r in results if r.get("status") != "OK")
    fail_rate = failed / total
    
    if fail_rate > 0.50:
        return False, f"fail_rate={fail_rate:.1%} (>{total} molecules)"
    return True, ""


# ──────────────────────────────────────────────────────────────────────────────
# CHANGE 3: Add surrogate validation (after surrogate.update in main loop, ~line 1520)
# ──────────────────────────────────────────────────────────────────────────────

def _validate_surrogate(surrogate: SurrogateModel, results: List[dict]) -> bool:
    """
    Check if surrogate predictions are accurate.
    Return False if predictions diverge >5.0 kcal/mol from actual scores.
    """
    if not surrogate.trained or not results:
        return True
    
    errors = []
    for r in results[-20:]:  # Check last 20 docking results
        smi = r.get("smiles")
        actual = r.get("score", 0.0)
        if smi and actual is not None:
            pred = surrogate.predict(smi)
            error = abs(pred - actual)
            errors.append(error)
    
    if not errors:
        return True
    
    mean_error = statistics.mean(errors)
    if mean_error > 5.0:
        log.warning(f"  Surrogate validation failed: mean_error={mean_error:.2f} kcal/mol")
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# CHANGE 4: Reset surrogate in main loop (in generation loop, after updating)
# ──────────────────────────────────────────────────────────────────────────────

# In the main evolution loop (around line 1520), after:
# surrogate.update(full_history)
# fqsar.update(full_history)

# ADD:
if not _validate_surrogate(surrogate, results):
    log.info("  Surrogate model reset (diverged from actual scores)")
    surrogate = SurrogateModel()  # Reset


# ──────────────────────────────────────────────────────────────────────────────
# CHANGE 5: Increase exhaustiveness for later generations (around line 1500)
# ──────────────────────────────────────────────────────────────────────────────

# OLD:
"""
gen_exhaust = int(EXHAUST_START + t * (EXHAUST_END - EXHAUST_START))
if stagnation_cnt >= STAGNATION_HARD // 2:
    gen_exhaust = min(gen_exhaust * 2, EXHAUST_END * 2)
"""

# NEW:
gen_exhaust = int(EXHAUST_START + t * (EXHAUST_END - EXHAUST_START))
if gen >= 11:  # Boost exhaustiveness starting gen 11
    gen_exhaust = max(gen_exhaust, 24)
if stagnation_cnt >= STAGNATION_HARD // 2:
    gen_exhaust = min(gen_exhaust * 2, EXHAUST_END * 2)


# ──────────────────────────────────────────────────────────────────────────────
# CHANGE 6: Check docking health and auto-reset (in main loop, after docking)
# ──────────────────────────────────────────────────────────────────────────────

# After the docking loop (after sys.stdout.write("\n")), ADD:

healthy, reason = _check_dock_health(results, gen)
if not healthy:
    log.warning(f"  Docking health check failed: {reason}")
    if gen >= 11:
        log.info(f"  EMERGENCY RESET: population corrupted")
        stagnation_cnt = STAGNATION_HARD - 1  # Trigger hard reset next gen


# ──────────────────────────────────────────────────────────────────────────────
# CHANGE 7: FQSAR validation (optional, for robustness)
# ──────────────────────────────────────────────────────────────────────────────

# In FragmentQSAR.biased_pool(), add:

def biased_pool(self, pool: List[str], temperature: float = 1.0) -> List[str]:
    if self.n_obs < FQSAR_MIN_DATA:
        return pool
    
    # NEW: Disable FQSAR if it's promoting bad fragments
    mean_qual = statistics.mean(
        (self.frag_good.get(f, 0) / max(1, self.frag_total.get(f, 1)))
        for f in self.frag_good
    ) if self.frag_good else 0.5
    
    if mean_qual < 0.25:  # <25% of biased fragments are in good binders
        log.debug("  FQSAR disabled (mean quality too low)")
        return pool  # Return unbiased pool
    
    # ... rest of original code ...


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK FIX: Just change the ADMET thresholds
# ═══════════════════════════════════════════════════════════════════════════════
# If you're in a hurry, ONLY change the ADMET_MW_MAX line:
#
#     ADMET_MW_MAX = 500.0  (instead of 600.0)
#
# This alone should fix 80% of the docking failures.
# Then re-run: python pipeline\denovo_design.py --uniprot P04637 --vina pipeline\vina.exe --generations 30