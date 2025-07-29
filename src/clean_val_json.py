"""
clean_val_json.py
--------------------
Same logic as the training cleaner — but for the validation set.

What it does:
1. Loads ASL100_val.json
2. Keeps only those entries whose trimmed video exists
3. Saves a new ASL100_val_cleaned.json
"""

import json
from pathlib import Path

# --- Setup paths ---
base_path = Path("C:/Users/seman/Desktop/clg/2nd_sem/research_practicum/code")
json_file = base_path / "data" / "lists" / "ASL100_val.json"
video_root = base_path / "data" / "clips" / "val"
save_path = base_path / "data" / "lists" / "ASL100_val_cleaned.json"

# --- Load original validation list ---
with open(json_file, "r") as f:
    video_list = json.load(f)

# --- Filter valid clips ---
valid_clips = []
for clip in video_list:
    class_dir = video_root / clip["clean_text"]
    filename = clip["file"].replace(" ", "_") + ".mp4"
    full_path = class_dir / filename
    if full_path.exists():
        valid_clips.append(clip)

# --- Save cleaned list ---
with open(save_path, "w") as f:
    json.dump(valid_clips, f)

print(f"Cleaned validation clips saved: {len(valid_clips)} valid entries")
