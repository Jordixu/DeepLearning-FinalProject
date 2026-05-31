import re
import random
from pathlib import Path

DATASET_DIR = Path(r"d:\Git\DeepLearning-FinalProject\data\combine_asl_dataset")

base_pattern = re.compile(r"^color_\d+_(\d+)\.png$")
copy_pattern = re.compile(r"^color_\d+_(\d+)\s+\(\d+\)\.png$")

deleted = 0
kept = 0

for class_dir in sorted(DATASET_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    # Collect base frames sorted by frame number
    base_frames = sorted(
        [f for f in class_dir.iterdir() if base_pattern.match(f.name)],
        key=lambda f: int(base_pattern.match(f.name).group(1))
    )
    if not base_frames:
        continue

    # Keep every 10th base frame
    keep_stems = {f.stem for i, f in enumerate(base_frames) if i % 10 == 0}

    # Process all color_ files
    for f in list(class_dir.iterdir()):
        if not f.name.startswith("color_"):
            continue
        # Derive stem: strip copy suffix
        stem = re.sub(r"\s+\(\d+\)$", "", f.stem)
        if stem in keep_stems:
            if not re.search(r"\(\d+\)", f.name):
                kept += 1
        else:
            f.unlink()
            deleted += 1

print(f"Deleted: {deleted} files")
print(f"Kept (base frames): {kept}")
