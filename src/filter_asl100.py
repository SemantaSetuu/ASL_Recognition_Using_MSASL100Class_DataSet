"""
filter_asl100.py

PURPOSE:
--------
This script filters the full MS-ASL dataset to only keep the first 100 sign classes.
MS-ASL originally has 1000+ classes, but for our project, we are only using the ASL100 subset.

WHAT THIS SCRIPT DOES:
-----------------------
1. Loads the full list of all MS-ASL sign classes from MSASL_classes.json.
2. Keeps only the first 100 sign class names.
3. Reads the full MS-ASL split files: train, val, and test.
4. Filters out all videos that do not belong to those 100 classes.
5. Saves new JSON files with only ASL100 samples into:
   → code/data/lists/ASL100_train.json
   → code/data/lists/ASL100_val.json
   → code/data/lists/ASL100_test.json
6. Prints how many videos were saved in each split.

USAGE:
------
Just run this file once:
    python filter_asl100.py
"""

import json
from pathlib import Path

# Define the root directory of the project where "code" folder lives
base_path = Path("C:/Users/seman/Desktop/clg/2nd_sem/research_practicum/code")


def main(msasl_root: Path):
    # Step 1: Load the full list of all sign classes (1000+ classes)
    msasl_class_file = msasl_root / "MS-ASL" / "MSASL_classes.json"
    with open(msasl_class_file, 'r', encoding='utf-8') as f:
        all_classes = json.load(f)

    # Step 2: Select only the first 100 class names
    first_100_signs = all_classes[:100]
    keep_set = set(first_100_signs)  # convert to set for faster lookup

    # Step 3: Loop through all dataset splits
    for split_name in ['train', 'val', 'test']:
        # Step 3.1: Load the full split file (train/val/test)
        split_file = msasl_root / "MS-ASL" / f"MSASL_{split_name}.json"
        with open(split_file, 'r', encoding='utf-8') as infile:
            video_list = json.load(infile)

        # Step 3.2: Filter videos that belong to the selected 100 signs
        filtered_list = []
        for video_data in video_list:
            if video_data['clean_text'] in keep_set:
                filtered_list.append(video_data)

        # Step 3.3: Define path to save the new filtered JSON file
        output_path = msasl_root / "data" / "lists" / f"ASL100_{split_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 3.4: Save the filtered list to the new file
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(filtered_list, outfile, indent=2)

        # Step 3.5: Print how many clips were saved
        print(f"{output_path.name}: {len(filtered_list)} clips")


# If this file is being run directly (not imported), call main()
if __name__ == '__main__':
    main(base_path)
