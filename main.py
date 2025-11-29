from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from config.settings import Settings, load_settings, save_settings
from core.actions import move_photos, select_discarded
from core.grouping import group_bursts
from core.image_scan import scan_directory

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Burst photo filtering prototype. Scans a directory, groups similar "
            "captures, and moves discarded files."
        )
    )
    parser.add_argument(
        "directory", nargs="?", type=Path, help="Directory containing photos"
    )
    parser.add_argument("--gui", action="store_true", help="Launch the GUI prototype")
    parser.add_argument("--recursive", action="store_true", help="Scan recursively")
    parser.add_argument(
        "--time-threshold", type=int, default=None, help="Seconds between shots"
    )
    parser.add_argument(
        "--hash-threshold", type=int, default=None, help="Hamming distance"
    )
    parser.add_argument(
        "--min-group-size", type=int, default=None, help="Minimum burst size"
    )
    parser.add_argument(
        "--duplicate-dir",
        type=str,
        default=None,
        help="Directory name for moved duplicates",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List files without moving"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()

    effective = Settings(
        scan_recursive=args.recursive or settings.scan_recursive,
        time_threshold_seconds=args.time_threshold or settings.time_threshold_seconds,
        hash_threshold=args.hash_threshold or settings.hash_threshold,
        min_group_size=args.min_group_size or settings.min_group_size,
        duplicate_directory=args.duplicate_dir or settings.duplicate_directory,
    )

    if args.gui:
        from ui.app import run_gui  # noqa: PLC0415

        run_gui(effective, args.directory)
        save_settings(effective)
        return

    if not args.directory:
        raise SystemExit(
            "Please provide a directory or launch with --gui to open the interface."
        )

    run_cli(args, effective)
    save_settings(effective)


def run_cli(args: argparse.Namespace, settings: Settings) -> None:
    t1 = time.time()
    logger.info("Scanning %s (recursive=%s)", args.directory, settings.scan_recursive)
    photos = scan_directory(args.directory, recursive=settings.scan_recursive)
    logger.info("Found %s photos", len(photos))

    groups = group_bursts(
        photos,
        time_threshold_seconds=settings.time_threshold_seconds,
        hash_threshold=settings.hash_threshold,
        min_group_size=settings.min_group_size,
    )

    for group in groups:
        for idx, photo in enumerate(group.photos):
            photo.keep = idx == 0

    kept = [photo for photo in photos if photo.keep]
    discarded = select_discarded(photos)

    logger.info("Identified %s groups", len(groups))
    for group in groups:
        logger.info("Group %s: %s items", group.id, group.size())

    logger.info("Keeping %s photos, discarding %s", len(kept), len(discarded))

    duplicate_dir = args.directory / settings.duplicate_directory
    moved = move_photos(discarded, duplicate_dir, dry_run=args.dry_run)
    if args.dry_run:
        logger.info("[dry-run] Would move %s files to %s", moved, duplicate_dir)
    else:
        logger.info("Moved %s files to %s", moved, duplicate_dir)

    t2 = time.time()
    logger.info("Took %.4f seconds", t2 - t1)


if __name__ == "__main__":
    main()
