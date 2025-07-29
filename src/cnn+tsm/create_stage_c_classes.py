"""
create_stage_c_classes.py
─────────────────────────
Writes `stage_c_classes.txt` in the exact order
Stage C used during training (alphabetical folder names).
"""

from pathlib import Path

# ←------ EDIT to your train-frames root
TRAIN_DIR = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\data\images\train")

cls_names = sorted(d.name for d in TRAIN_DIR.iterdir() if d.is_dir())
print(f"Found {len(cls_names)} classes.")

out_file = r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\src\cnn+tsm\stage_c_classes.txt"   # e.g. saves next to images
with open(out_file, "w") as f:
    f.write("\n".join(cls_names))

print(f"Class list written to {out_file}")
