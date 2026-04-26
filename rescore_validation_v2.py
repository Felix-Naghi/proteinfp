"""
rescore_validation.py  (v2 — with live UniProt GO fetching)
"""
from __future__ import annotations
import json, logging, os, ssl, sys, time, urllib.parse, urllib.request
from pathlib import Path
from datetime import datetime
from typing import Optional

import certifi, click, numpy as np

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.clean_ec import predict_ec_number

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s", datefmt="%H:%M:%S")

W_GO=35; W_AS=25; W_ENZYME=20; W_PPI=20
_NS_MAP = {"F": "MF", "P": "BP", "C": "CC"}


def _fetch_go(uid: str) -> Optional[dict]:
    """Fetch GO terms from UniProt REST, return go_result dict for clean_ec.py."""
    params = urllib.parse.urlencode({"format": "json", "fields": "go"})
    req = urllib.request.Request(
        f"https://rest.uniprot.org/uniprotkb/{uid}?{params}",
        headers={"User-Agent": "ProteinFP-rescore/2.0", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        log.warning(f"    GO fetch failed {uid}: {e}"); return None

    mf, bp, cc = [], [], []
    for ref in data.get("uniProtKBCrossReferences", []):
        if ref.get("database") != "GO": continue
        go_id = ref.get("id", "")
        go_name, aspect = "", ""
        for prop in ref.get("properties", []):
            if prop.get("key") == "GoTerm":
                v = prop.get("value", "")
                if ":" in v:
                    ac, nm = v.split(":", 1)
                    aspect = _NS_MAP.get(ac.strip(), "")
                    go_name = nm.strip()
        if not go_id: continue
        pred = {"go_id": go_id, "go_name": go_name, "score": 0.95,
                "evidence": ["uniprot_curated"], "namespace": aspect}
        (mf if aspect=="MF" else bp if aspect=="BP" else cc if aspect=="CC" else mf).append(pred)
    return {"mf_predictions": mf, "bp_predictions": bp, "cc_predictions": cc}


def _fetch_sequence(uid: str) -> Optional[str]:
    req = urllib.request.Request(f"https://rest.uniprot.org/uniprotkb/{uid}.fasta",
        headers={"User-Agent": "ProteinFP-rescore/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            fasta = r.read().decode()
        return "".join(fasta.strip().split("\n")[1:]).upper()
    except: return None


def _seq_from_inter(uid: str, data_dir: str) -> Optional[str]:
    for sfx in ["_structure.json", "_esm2.json"]:
        p = Path(data_dir) / f"{uid}{sfx}"
        if p.exists():
            try:
                d = json.loads(p.read_text()); seq = d.get("sequence","")
                if seq: return seq.upper()
            except: pass
    return None


def _load_gt() -> dict:
    gt = {}
    sys.path.insert(0, "validation")
    try:
        from run_validation import VALIDATION_SET
        for p in VALIDATION_SET: gt[p["uniprot_id"]] = p
    except: pass
    ne = Path("validation/new_entries.json")
    if ne.exists():
        for p in json.loads(ne.read_text()):
            uid = p.get("uniprot_id","")
            if uid and uid not in gt: gt[uid] = p
    log.info(f"  Ground truth: {len(gt)} proteins")
    return gt


def _enzyme_correct(is_enz, score, gt):
    g = gt.get(score["uniprot_id"])
    if not g: return score.get("enzyme_correct", False)
    return is_enz if g.get("is_enzyme") else not is_enz


def _ec_correct(ec, is_enz, score, gt):
    g = gt.get(score["uniprot_id"])
    if not g: return score.get("ec_correct", False)
    if not g.get("is_enzyme"): return not is_enz
    ge = str(g.get("ec_number","")).strip()
    return (ec == ge[0]) if (ge and is_enz) else False


def rescore(report_path, model_dir, cache_dir, out_path, data_dir):
    report = json.loads(Path(report_path).read_text())
    scores = report["scores"]
    orig_enz  = sum(1 for s in scores if s.get("enzyme_correct")) / len(scores)
    orig_ov   = sum(s["overall_score"] for s in scores) / len(scores)

    gt = _load_gt()

    cache_path = Path(cache_dir) / "esm2_cache.json"
    esm2_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    log.info(f"  ESM-2 cache: {len(esm2_cache):,} embeddings")

    n_enz_changed = n_ec_changed = n_go = 0

    for i, score in enumerate(scores):
        uid = score["uniprot_id"]; gene = score.get("gene","?")
        log.info(f"  [{i+1:>3}/{len(scores)}] {uid} ({gene})")

        # Sequence
        seq = _seq_from_inter(uid, data_dir) or _fetch_sequence(uid)
        if not seq: log.warning(f"    No sequence — skip"); continue
        time.sleep(0.05)

        # ESM-2
        emb = esm2_cache.get(uid)
        esm2_result = {"protein_embedding": emb, "contact_map": []} if emb else None

        # GO from UniProt
        go_result = _fetch_go(uid)
        if go_result:
            n_mf = len(go_result["mf_predictions"])
            n_bp = len(go_result["bp_predictions"])
            log.info(f"    GO: {n_mf} MF + {n_bp} BP")
            n_go += 1
        time.sleep(0.15)

        # Predict
        try:
            result = predict_ec_number(uid, seq, go_result=go_result, esm2_result=esm2_result)
        except Exception as e:
            log.error(f"    predict_ec_number failed: {e}"); continue

        ml_enz = result.is_enzyme
        ml_ec  = result.top_prediction.ec_class if result.top_prediction else ""
        ml_p   = result.ml_enzyme_prob

        old_enz = score.get("enzyme_correct", False)
        old_ec  = score.get("ec_correct", False)
        new_enz = _enzyme_correct(ml_enz, score, gt)
        new_ec  = _ec_correct(ml_ec, ml_enz, score, gt)

        if new_enz != old_enz: n_enz_changed += 1; log.info(f"    Enz: {old_enz}→{new_enz}  ML_p={ml_p:.3f}")
        if new_ec  != old_ec:  n_ec_changed  += 1

        new_ov = round(score["go_mean_recall"]*W_GO + score["active_site_recall"]*W_AS +
                       (1.0 if new_enz else 0.0)*W_ENZYME + score["ppi_recall"]*W_PPI, 1)
        score.update({"enzyme_correct": new_enz, "ec_correct": new_ec,
                      "overall_score": new_ov,
                      "ml_enzyme_prob": round(ml_p,4) if ml_p>=0 else -1.0,
                      "ml_top_ec_class": ml_ec,
                      "ml_top_ec_prob": round(result.top_prediction.score if result.top_prediction else 0,4),
                      "n_go_terms_used": n_mf+n_bp if go_result else 0})

    new_enz_acc = sum(1 for s in scores if s.get("enzyme_correct")) / len(scores)
    new_ov      = sum(s["overall_score"] for s in scores) / len(scores)

    print("\n" + "═"*66)
    print("  VALIDATION RESCORE RESULTS")
    print("═"*66)
    print(f"  Proteins rescored       : {len(scores)}")
    print(f"  GO terms fetched (live) : {n_go}")
    print(f"  Enzyme class changes    : {n_enz_changed}")
    print(f"  EC class changes        : {n_ec_changed}")
    print("─"*66)
    print(f"  {'Metric':<30} {'Before':>10} {'After':>10} {'Delta':>8}")
    print(f"  {'─'*28} {'─'*8} {'─'*8} {'─'*6}")
    print(f"  {'Enzyme classification':<30} {orig_enz*100:>9.1f}% {new_enz_acc*100:>9.1f}% {(new_enz_acc-orig_enz)*100:>+7.1f}%")
    print(f"  {'Overall accuracy score':<30} {orig_ov:>9.1f}/100 {new_ov:>9.1f}/100 {new_ov-orig_ov:>+7.1f}")
    print("═"*66)

    report.update({
        "run_date": report.get("run_date","") + f" (rescored {datetime.now().strftime('%Y-%m-%d %H:%M')} live GO)",
        "ml_model_dir": model_dir, "go_source": "UniProt REST API (live)",
        "rescore_summary": {
            "orig_enzyme_acc": round(orig_enz,4), "new_enzyme_acc": round(new_enz_acc,4),
            "orig_overall": round(orig_ov,2),     "new_overall":  round(new_ov,2),
            "n_enzyme_changed": n_enz_changed, "n_ec_changed": n_ec_changed, "n_go_fetched": n_go,
        }
    })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(f"  Saved → {out_path}")


@click.command()
@click.option("--report",    default="data/reports/validation/validation_report.json")
@click.option("--model-dir", default="models/ec_ensemble_v5")
@click.option("--cache-dir", default="data/training_cache_v5")
@click.option("--out",       default="data/reports/validation/validation_report_v7.json")
@click.option("--data-dir",  default="data/intermediate")
def main(report, model_dir, cache_dir, out, data_dir):
    """Re-score with ECClassifierEnsemble v5 + live UniProt GO terms."""
    rescore(report, model_dir, cache_dir, out, data_dir)

if __name__ == "__main__":
    main()