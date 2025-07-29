"""
clean_train_json.py
--------------------
This script filters the training JSON file to include only the usable clips
(i.e. clips whose corresponding trimmed videos actually exist on disk).

What it does:
1. Loads ASL100_train.json
2. For each clip entry, checks whether the trimmed .mp4 video file exists
3. Keeps only valid entries
4. Saves the cleaned JSON file as ASL100_train_cleaned.json
"""

import json
from pathlib import Path

# --- Setup paths ---
base_path = Path("C:/Users/seman/Desktop/clg/2nd_sem/research_practicum/code")
json_file = base_path / "data" / "lists" / "ASL100_train.json"
video_root = base_path / "data" / "clips" / "train"
save_path = base_path / "data" / "lists" / "ASL100_train_cleaned.json"

# --- Load original training list ---
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

print(f"Cleaned training clips saved: {len(valid_clips)} valid entries")
