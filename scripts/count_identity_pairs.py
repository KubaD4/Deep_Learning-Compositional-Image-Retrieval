#!/usr/bin/env python3
"""Count image pairs available for each CelebA identity.

The input file is opened read-only and is never modified.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


DEFAULT_IDENTITY_FILE = Path("celeba/identity_CelebA.txt")


def load_identity_counts(path: Path) -> Counter[int]:
    """Return the number of distinct image filenames for every person ID."""
    counts: Counter[int] = Counter()
    seen_filenames: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected 'filename person_id', "
                    f"found {len(fields)} fields"
                )

            filename, identity_raw = fields
            if filename in seen_filenames:
                raise ValueError(
                    f"{path}:{line_number}: duplicate image filename {filename!r}"
                )

            try:
                identity_id = int(identity_raw)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid person ID {identity_raw!r}"
                ) from error

            seen_filenames.add(filename)
            counts[identity_id] += 1

    if not counts:
        raise ValueError(f"No identity rows found in {path}")

    return counts


def unordered_pairs(image_count: int) -> int:
    """Number of unique pairs {A, B}: C(n, 2)."""
    return image_count * (image_count - 1) // 2


def directional_pairs(image_count: int) -> int:
    """Number of ordered training directions A->B and B->A: n*(n-1)."""
    return image_count * (image_count - 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "identity_file",
        nargs="?",
        type=Path,
        default=DEFAULT_IDENTITY_FILE,
        help=f"CelebA identity file (default: {DEFAULT_IDENTITY_FILE})",
    )
    parser.add_argument(
        "--per-identity",
        action="store_true",
        help="Print image and pair counts for every person ID.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of identities with most images to show in the summary.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    counts = load_identity_counts(args.identity_file)

    rows = [
        (
            identity_id,
            image_count,
            unordered_pairs(image_count),
            directional_pairs(image_count),
        )
        for identity_id, image_count in counts.items()
    ]

    total_images = sum(row[1] for row in rows)
    total_unordered = sum(row[2] for row in rows)
    total_directional = sum(row[3] for row in rows)
    identities_with_pairs = sum(row[1] >= 2 for row in rows)

    if args.per_identity:
        print("person_id\timages\tunique_pairs\tdirectional_pairs")
        for row in sorted(rows):
            print("\t".join(map(str, row)))
        print()

    print(f"Identity file: {args.identity_file}")
    print(f"Images read: {total_images:,}")
    print(f"Person IDs: {len(rows):,}")
    print(f"Person IDs with at least 2 images: {identities_with_pairs:,}")
    print(f"Total unique unordered pairs C(n,2): {total_unordered:,}")
    print(f"Total directional pairs n*(n-1): {total_directional:,}")

    if args.top > 0:
        print(f"\nTop {args.top} identities by image count:")
        print("person_id\timages\tunique_pairs\tdirectional_pairs")
        for row in sorted(rows, key=lambda item: (-item[1], item[0]))[: args.top]:
            print("\t".join(map(str, row)))


if __name__ == "__main__":
    main()
