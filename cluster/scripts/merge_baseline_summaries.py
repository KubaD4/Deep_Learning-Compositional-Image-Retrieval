#!/usr/bin/env python3
"""Combine completed baseline summaries into one comparison CSV."""

from __future__ import annotations

import csv

from project_core import METHOD_FOLDERS, RESULTS_DIR


def main() -> None:
    rows = []
    for method, folder in METHOD_FOLDERS.items():
        path = RESULTS_DIR / folder / "summary.csv"
        if not path.exists():
            print(f"Missing: {path}")
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
            row["method"] = method
            rows.append(row)
    if not rows:
        raise SystemExit("No completed summaries found")
    output = RESULTS_DIR / "all_methods_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
