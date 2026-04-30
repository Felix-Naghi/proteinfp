import json
from pathlib import Path

f = Path('sim/step04_binding_probability.py')
text = f.read_text(encoding='utf-8')

# Fix pKi to use corrected dG not physics dG
old = '''    Kd_physics   = dG_to_Kd(dG_physics)
    Kd_corrected = dG_to_Kd(dG_corrected)

    # pKi = -log10(Ki in M)
    pKi = -math.log10(max(Kd_corrected * 1e-6, 1e-15))

    # Binding probability using corrected Kd
    p_competent = sum(
        s.get("probability", 0)
        for s in states
        if s.get("name") in {"active", "apo", "allosteric_open"}
    )
    p_binding = (drug_conc_uM / (drug_conc_uM + Kd_corrected)) * p_competent'''

new = '''    Kd_physics   = dG_to_Kd(dG_physics)
    Kd_corrected = dG_to_Kd(dG_corrected)

    # pKi uses corrected Kd
    pKi = -math.log10(max(Kd_corrected * 1e-6, 1e-15))
    pKi = round(float(pKi), 3)

    # Binding probability using corrected Kd
    p_competent = sum(
        s.get("probability", 0)
        for s in states
        if s.get("name") in {"active", "apo", "allosteric_open"}
    )
    p_binding = (drug_conc_uM / (drug_conc_uM + Kd_corrected)) * p_competent'''

if old in text:
    text = text.replace(old, new)
    f.write_text(text, encoding='utf-8')
    print('Fixed: pKi now uses corrected Kd')
else:
    # Check what the actual pKi line looks like
    for i, line in enumerate(text.splitlines()):
        if 'pKi = -math' in line:
            print(f'Line {i}: {line}')
