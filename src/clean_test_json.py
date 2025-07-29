"""
clean_test_json.py
--------------------
This script filters the ASL100_test.json to keep only the test clips
that actually exist on disk (i.e., trimmed .mp4 files).

Steps:
1. Loads ASL100_test.json
2. Checks whether each clip exists under clips/test/
3. Saves filtered list to ASL100_test_cleaned.json
"""

import json
from pathlib import Path

# --- Setup paths ---
base_path   = Path("C:/Users/seman/Desktop/clg/2nd_sem/research_practicum/code")
json_file   = base_path / "data" / "lists" / "ASL100_test.json"
video_root  = base_path / "data" / "clips" / "test"
save_path   = base_path / "data" / "lists" / "ASL100_test_cleaned.json"

# --- Load original test list ---
with open(json_file, "r") as f:
    video_list = json.load(f)

# --- Filter only existing clips ---
valid_clips = []
for clip in video_list:
    class_dir = video_root / clip["clean_text"]
    filename  = clip["file"].replace(" ", "_") + ".mp4"
    full_path = class_dir / filename
    if full_path.exists():
        valid_clips.append(clip)

# --- Save cleaned list ---
with open(save_path, "w") as f:
    json.dump(valid_clips, f)

print(f"Cleaned test clips saved: {len(valid_clips)} valid entries")
