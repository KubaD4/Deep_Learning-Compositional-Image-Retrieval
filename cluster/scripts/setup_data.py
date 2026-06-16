#!/usr/bin/env python3
"""Extract the bundled CelebA archive into data/celeba."""

from __future__ import annotations

import argparse
import signal
import time
import zipfile

from project_core import CELEBA_DIR, DATA_ROOT


STOP_REQUESTED = False


def request_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-budget-seconds", type=int, default=480)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    archive = DATA_ROOT / "celeba.zip"
    image_dir = CELEBA_DIR / "img_align_celeba"
    if image_dir.is_dir() and sum(1 for _ in image_dir.glob("*.jpg")) == 202_599:
        print(f"CelebA is already extracted at {CELEBA_DIR}")
        return 0
    if not archive.is_file():
        raise FileNotFoundError(f"Missing dataset archive: {archive}")

    print(f"Resumable extraction of {archive} into {DATA_ROOT}")
    with zipfile.ZipFile(archive) as zipped:
        members = [
            member
            for member in zipped.infolist()
            if not member.filename.startswith("__MACOSX/")
            and not member.filename.endswith(".DS_Store")
        ]
        started = time.monotonic()
        for member_index, member in enumerate(members):
            destination = DATA_ROOT / member.filename
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif not destination.exists() or destination.stat().st_size != member.file_size:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                with zipped.open(member) as source, temporary.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                temporary.replace(destination)

            if member_index % 500 == 0:
                print(f"Archive members processed: {member_index}/{len(members)}")
            if STOP_REQUESTED or time.monotonic() - started >= args.time_budget_seconds:
                print("Extraction checkpoint reached. Submit the setup job again.")
                return 3

    image_count = sum(1 for _ in image_dir.glob("*.jpg"))
    if image_count != 202_599:
        raise RuntimeError(f"Expected 202599 images, found {image_count}")
    print(f"Dataset ready: {image_count} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
