from pathlib import Path
f = Path('sim/step07_validate.py')
t = f.read_text(encoding='utf-8')
t = t.replace(
    '"pearson_r":  round(r, 4),',
    '"pearson_r":  round(float(r), 4),'
)
f.write_text(t, encoding='utf-8')
print('Fixed')
