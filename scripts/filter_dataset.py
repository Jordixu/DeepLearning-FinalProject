import os
import re
import random
from collections import defaultdict
from pathlib import Path

DATASET_DIR = Path(r"d:\Git\DeepLearning-FinalProject\data\combine_asl_dataset")
DRY_RUN = False  # Set to True to preview without deleting

pattern = re.compile(r"^(hand\d+)_(.+?)_(.+?)_seg_(\d+)_cropped\.(jpeg|png)$")

def main():
    to_delete = []
    to_keep = []

    for class_dir in sorted(DATASET_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        # Group files by (hand, yyyy)
        groups = defaultdict(list)
        for f in class_dir.iterdir():
            if not f.is_file():
                continue
            m = pattern.match(f.name)
            if not m:
                continue
            hand, label, yyyy, seg, ext = m.groups()
            groups[(hand, yyyy)].append((int(seg), ext, f))

        for (hand, yyyy), files in groups.items():
            # Pick a random segment number from those available
            seg_numbers = list({seg for seg, ext, f in files})
            chosen_seg = random.choice(seg_numbers)

            for seg, ext, f in files:
                if seg == chosen_seg and ext == "jpeg":
                    to_keep.append(f)
                else:
                    to_delete.append(f)

    print(f"Files to keep:  {len(to_keep)}")
    print(f"Files to delete: {len(to_delete)}")

    if DRY_RUN:
        print("\n[DRY RUN] Sample files that would be deleted:")
        for f in to_delete[:20]:
            print(f"  DELETE: {f.relative_to(DATASET_DIR)}")
        print("\n[DRY RUN] Sample files that would be kept:")
        for f in to_keep[:10]:
            print(f"  KEEP:   {f.relative_to(DATASET_DIR)}")
        return

    for f in to_delete:
        f.unlink()

    print("Done.")

if __name__ == "__main__":
    random.seed(42)
    main()
