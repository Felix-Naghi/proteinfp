import json
from pathlib import Path

f = Path('sim/step07_validate.py')
text = f.read_text(encoding='utf-8')

old = '''        if pred is not None and pred > 0:
            error  = pred - exp
            status = "OK"
            exp_pKis.append(exp)
            pred_pKis.append(pred)'''

new = '''        if pred is not None and pred > 0:
            # Apply ML correction directly from saved model
            ml_path = ROOT / "data" / "sim" / "ml_correction_model.json"
            if ml_path.exists():
                import numpy as _np
                _m = json.loads(ml_path.read_text())
                _w = _np.array(_m["weights"], dtype=_np.float64)
                _b = float(_m["bias"])
                dG   = result.get("dG_corrected", -30) or -30
                fill = result.get("fill_ratio", 0.5) or 0.5
                _feat = _np.array([
                    max(-100, min(0, dG)) / -100,
                    (pair.get("logP", 2) + 5) / 10,
                    pair.get("mw", 400) / 600,
                    pair.get("psa", 80) / 200,
                    pair.get("hbd", 2) / 10,
                    pair.get("hba", 5) / 15,
                    min(fill, 2.0) / 2.0,
                    float(pred) / 10,
                ], dtype=_np.float64)
                pred = round(pred + float(_np.dot(_w, _feat) + _b), 3)
            error  = pred - exp
            status = "OK"
            exp_pKis.append(exp)
            pred_pKis.append(pred)'''

if old in text:
    text = text.replace(old, new)
    f.write_text(text, encoding='utf-8')
    print('Fixed: ML correction applied in validation script')
else:
    print('Block not found')
