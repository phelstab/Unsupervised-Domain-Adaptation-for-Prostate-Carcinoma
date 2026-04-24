#!/usr/bin/env python3
"""
Move patient directories from picai_public_images_fold* subdirectories
directly into input/images/ so the preprocessing pipeline can find them.

Before: input/images/picai_public_images_fold0/10000/10000_xxx_t2w.mha
After:  input/images/10000/10000_xxx_t2w.mha
"""

import shutil
from pathlib import Path

# Resolve project root the same way download.py does (setup/../)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def unfold_images(images_dir=None):
    images_dir = PROJECT_ROOT / (images_dir if images_dir is not None else "input/images")
    fold_dirs = sorted(d for d in images_dir.glob("picai_public_images_fold*") if d.is_dir())

    if not fold_dirs:
        print(f"No picai_public_images_fold* directories found in {images_dir}")
        print("Images may already be in the correct structure.")
        return

    print(f"Found {len(fold_dirs)} fold directories: {[d.name for d in fold_dirs]}")

    moved = 0
    skipped = 0

    for fold_dir in fold_dirs:
        patient_dirs = [d for d in fold_dir.iterdir() if d.is_dir()]
        print(f"\n{fold_dir.name}: {len(patient_dirs)} patient directories")

        for patient_dir in sorted(patient_dirs):
            dest = images_dir / patient_dir.name

            if dest.exists():
                print(f"  Skipped {patient_dir.name} (already exists)")
                skipped += 1
                continue

            shutil.move(str(patient_dir), str(dest))
            moved += 1

        remaining = list(fold_dir.iterdir())
        if not remaining:
            fold_dir.rmdir()
            print(f"  Removed empty {fold_dir.name}/")

    print(f"\nDone: {moved} patient dirs moved, {skipped} skipped")


if __name__ == "__main__":
    unfold_images()
