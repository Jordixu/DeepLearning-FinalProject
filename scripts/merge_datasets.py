"""
Merge 5 ASL datasets into a single flat folder structure.

Sources:
  1. data/archive (2)/asl_alphabet_train/asl_alphabet_train/   prefix: arc  (class subfolders)
     data/archive (2)/asl_alphabet_test/asl_alphabet_test/     prefix: arct (flat: {Class}_test.jpg)
  2. data/ASL_Raw_Images/asl_dataset/                           prefix: raw  (class subfolders)
  3. data/combine_asl_dataset/                                  prefix: com  (class subfolders)
  4. data/synthetic_alphabet/{Train,Test}_Alphabet/             prefix: syna (class subfolders)
  5. data/synthetic_numbers/{Train,Test}_Nums/                  prefix: synn (class subfolders)

Output: data/merged_dataset/{class}/
  - Letters uppercase (A-Z)
  - Digits as-is (0-9, 10)
  - "nothing" and "Blank" -> "blank"
  - "del", "space" kept as-is
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent / "data"
OUT = ROOT / "merged_dataset"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# Classes to skip entirely.
_SKIP_CLASSES = {"del", "space", "0"}


def norm_class(name: str) -> str | None:
    """Return canonical class name, or None to skip this class entirely."""
    low = name.lower()
    if low in _SKIP_CLASSES:
        return None
    if low in ("nothing", "blank"):
        return "blank"
    if len(name) == 1 and name.isalpha():
        return name.upper()
    return name  # digits, "10", etc.


def collect_sources() -> list[tuple[str, str, Path, list[Path]]]:
    """Return list of (prefix, cls_name, src_dir_or_None, files)."""
    entries: list[tuple[str, str, list[Path]]] = []

    def _add_dir(prefix: str, cls_dir: Path) -> None:
        cls_name = norm_class(cls_dir.name)
        if cls_name is None:
            return
        files = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        entries.append((prefix, cls_name, files))

    # 1a. archive (2) train: class subfolders
    arc_train = ROOT / "archive (2)" / "asl_alphabet_train" / "asl_alphabet_train"
    for cls_dir in sorted(arc_train.iterdir()):
        if cls_dir.is_dir():
            _add_dir("arc", cls_dir)

    # 1b. archive (2) test: flat folder, {Class}_test.jpg
    arc_test = ROOT / "archive (2)" / "asl_alphabet_test" / "asl_alphabet_test"
    if arc_test.exists():
        flat_files: dict[str, list[Path]] = {}
        for f in sorted(arc_test.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                raw_cls = f.stem.replace("_test", "")
                cls_name = norm_class(raw_cls)
                if cls_name is not None:
                    flat_files.setdefault(cls_name, []).append(f)
        for cls_name, files in sorted(flat_files.items()):
            entries.append(("arct", cls_name, files))

    # 2. ASL_Raw_Images: class subfolders
    raw_root = ROOT / "ASL_Raw_Images" / "asl_dataset"
    for cls_dir in sorted(raw_root.iterdir()):
        if cls_dir.is_dir():
            _add_dir("raw", cls_dir)

    # 3. combine_asl_dataset: class subfolders
    com_root = ROOT / "combine_asl_dataset"
    for cls_dir in sorted(com_root.iterdir()):
        if cls_dir.is_dir():
            _add_dir("com", cls_dir)

    # 4. synthetic_alphabet (Train + Test): class subfolders
    for split in ("Train_Alphabet", "Test_Alphabet"):
        split_root = ROOT / "synthetic_alphabet" / split
        for cls_dir in sorted(split_root.iterdir()):
            if cls_dir.is_dir():
                _add_dir("syna", cls_dir)

    # 5. synthetic_numbers (Train + Test): class subfolders
    for split in ("Train_Nums", "Test_Nums"):
        split_root = ROOT / "synthetic_numbers" / split
        for cls_dir in sorted(split_root.iterdir()):
            if cls_dir.is_dir():
                _add_dir("synn", cls_dir)

    return entries


def merge():
    if OUT.exists():
        print(f"Removing existing {OUT} ...")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    entries = collect_sources()
    counters: dict[str, int] = {}
    total_copied = 0
    class_counts: dict[str, int] = {}

    for prefix, cls_name, files in entries:
        if not files:
            continue
        out_cls = OUT / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)

        idx = counters.get(cls_name, 0)
        for src in files:
            dst_name = f"{prefix}_{idx:06d}{src.suffix.lower()}"
            shutil.copy2(src, out_cls / dst_name)
            idx += 1
            total_copied += 1
            if total_copied % 5000 == 0:
                print(f"  {total_copied} files copied...")

        counters[cls_name] = idx
        class_counts[cls_name] = class_counts.get(cls_name, 0) + len(files)

    print(f"\nDone. {total_copied} files copied to {OUT}")
    print(f"\nClass distribution ({len(class_counts)} classes):")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls:8s}: {count:6d}")


if __name__ == "__main__":
    merge()
