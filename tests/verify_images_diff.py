#!/usr/bin/env python3
"""
Compare one or more images and report whether they are identical.

Usage examples:
  uv run python tests/verify_images_diff.py out/grid/color_processed.png out/prob4/color_processed.png
  uv run python tests/verify_images_diff.py --glob "out/**/color_processed.png"
  uv run python tests/verify_images_diff.py  # defaults to glob out/**/color_processed.png

Exits with non-zero code if --assert-different is provided and any pair is identical,
or if --assert-identical is provided and any pair differs.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import hashlib
import numpy as np
from PIL import Image


def find_images(inputs: List[str], glob_pattern: str | None) -> List[Path]:
    paths: list[Path] = []
    if inputs:
        for raw in inputs:
            p = Path(raw)
            if p.is_file():
                paths.append(p)
            elif p.is_dir():
                pattern = glob_pattern or "**/*.png"
                paths.extend(sorted(p.rglob(pattern)))
            else:
                # treat as glob relative to cwd
                paths.extend(sorted(Path.cwd().glob(raw)))
    else:
        pattern = glob_pattern or "out/**/color_processed.png"
        paths = sorted(Path.cwd().glob(pattern))
    # de-duplicate while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def load_image_rgba(path: Path) -> Tuple[np.ndarray, Tuple[int, int], str]:
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    sha = hashlib.sha256(arr.tobytes()).hexdigest()
    return arr, img.size, sha


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare images by hash and pixel-wise diffs")
    parser.add_argument("inputs", nargs="*", help="Image files, directories, or glob patterns")
    parser.add_argument("--glob", dest="glob", default=None, help="Glob used when inputs include directories or when no inputs are provided (default: out/**/color_processed.png)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--assert-different", action="store_true", help="Exit non-zero if any pair of images is identical")
    group.add_argument("--assert-identical", action="store_true", help="Exit non-zero if any pair of images differs")
    args = parser.parse_args(list(argv) if argv is not None else None)

    paths = find_images(args.inputs, args.glob)
    if len(paths) < 2:
        print("Need at least two images to compare.")
        return 2

    print("Images to compare:")
    for p in paths:
        print(f"  - {p}")

    info: dict[str, dict] = {}
    for p in paths:
        arr, size, sha = load_image_rgba(p)
        info[str(p)] = {"array": arr, "size": size, "sha256": sha}

    print("\nImage sizes and hashes:")
    for k, v in info.items():
        print(f"  {k}: size={v['size']}, sha256={v['sha256']}")

    print("\nPairwise comparisons:")
    names = list(info.keys())
    any_equal = False
    any_differ = False
    for a, b in itertools.combinations(names, 2):
        A = info[a]["array"]
        B = info[b]["array"]
        same_shape = A.shape == B.shape
        if not same_shape:
            print(f"  {Path(a).name} vs {Path(b).name}: DIFFER (shape {A.shape} vs {B.shape})")
            any_differ = True
            continue
        eq = np.array_equal(A, B)
        if eq:
            print(f"  {Path(a).name} vs {Path(b).name}: IDENTICAL")
            any_equal = True
        else:
            diff = np.abs(A.astype(np.int32) - B.astype(np.int32))
            changed = int((diff.sum(axis=-1) > 0).sum())
            total = int(diff.sum())
            print(f"  {Path(a).name} vs {Path(b).name}: DIFFER (changed_pixels={changed}, sum_abs={total})")
            any_differ = True

    if args.assert_different and any_equal:
        return 1
    if args.assert_identical and any_differ:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

