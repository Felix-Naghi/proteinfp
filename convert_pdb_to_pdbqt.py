# fix_feedback_tags.py  — run once from project root
import json
from pathlib import Path

feedback_dir = Path("data/feedback/P04637")
events_path  = feedback_dir / "events.jsonl"
backup_path  = feedback_dir / "events.jsonl.bak"

# Back up first
import shutil
shutil.copy(events_path, backup_path)
print(f"Backed up to {backup_path}")

# Rewrite, replacing active_site → P1
fixed = 0
lines_out = []
with open(events_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        if ev.get("target_site") == "active_site":
            ev["target_site"] = "P1"
            fixed += 1
        lines_out.append(json.dumps(ev, separators=(",", ":")))

with open(events_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines_out) + "\n")

print(f"Fixed {fixed} events: active_site → P1")
print(f"Total events: {len(lines_out)}")