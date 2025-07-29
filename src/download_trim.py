"""
download_trim.py

PURPOSE:
--------
This script downloads and trims ASL100 YouTube videos:
✔️ Downloads full videos into: data/raw/{split}/{sign}/
✔️ Trims segments into: data/clips/{split}/{sign}/

USAGE:
------
    python download_trim.py
"""

import json
import pathlib
import yt_dlp
import ffmpeg
import tqdm
import os

# Add ffmpeg to PATH
os.environ["PATH"] += os.pathsep + r"C:\installed\programs\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin"

# Base path setup
base_path = pathlib.Path("C:/Users/seman/Desktop/clg/2nd_sem/research_practicum/code")
raw_root = base_path / "data" / "raw"
clips_root = base_path / "data" / "clips"

# Download a YouTube video to raw folder
def download_youtube_video(url, raw_path):
    if raw_path.exists():
        return  # Already downloaded

    try:
        yt_dlp.YoutubeDL({
            'outtmpl': str(raw_path),
            'format': 'bestvideo[ext=mp4]',
            'ffmpeg_location': r'C:\installed\programs\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe',
            'quiet': True
        }).download([url])
    except Exception as e:
        print(f"[ERROR] Failed to download: {url}\nReason: {e}")

# Trim the video into target file using ffmpeg
def trim_video(raw_path, clip_path, start_time, end_time):
    if clip_path.exists():
        return  # Already trimmed

    try:
        (
            ffmpeg
            .input(str(raw_path), ss=start_time, to=end_time)
            .output(str(clip_path), codec='copy', loglevel='quiet')
            .run()
        )
    except Exception as e:
        print(f"[ERROR] Failed to trim: {clip_path}\nReason: {e}")

# Process all clips for a split
def process_split(split_name):
    print(f"\n[PROCESSING SPLIT] {split_name.upper()}")

    json_path = base_path / "data" / "lists" / f"ASL100_{split_name}.json"
    raw_split_root = raw_root / split_name
    clip_split_root = clips_root / split_name

    raw_split_root.mkdir(parents=True, exist_ok=True)
    clip_split_root.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        clip_list = json.load(f)

    for clip in tqdm.tqdm(clip_list, desc=f"{split_name}"):
        class_name = clip["clean_text"]

        # Paths
        class_raw_dir = raw_split_root / class_name
        class_clip_dir = clip_split_root / class_name
        class_raw_dir.mkdir(parents=True, exist_ok=True)
        class_clip_dir.mkdir(parents=True, exist_ok=True)

        video_id = clip["url"].split("v=")[-1]
        raw_file = class_raw_dir / f"{video_id}.mp4"
        trimmed_filename = clip["file"].replace(" ", "_") + ".mp4"
        trimmed_file = class_clip_dir / trimmed_filename

        # Step 1: Download full raw video
        download_youtube_video(clip["url"], raw_file)

        # Step 2: Trim required segment only
        trim_video(raw_file, trimmed_file, clip["start_time"], clip["end_time"])

# Run for all splits
if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)
