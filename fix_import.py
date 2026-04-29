from pathlib import Path

f = Path("sim/step03_drug_distribution.py")
text = f.read_text(encoding="utf-8")

# Fix the broken import line
old = "from sim.step01_cell_environment01_cell_environment import DRUGS as _DRUGS"
new = "# drug props loaded from saved JSON file"

text = text.replace(old, new)
f.write_text(text, encoding="utf-8")
print("Fixed import in step03")
