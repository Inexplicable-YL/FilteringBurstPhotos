from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config.settings import Settings, load_settings, save_settings
from ui.app import run_gui

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    settings = Settings(
        scan_recursive=args.recursive or settings.scan_recursive,
        time_threshold_seconds=args.time_threshold or settings.time_threshold_seconds,
        hash_threshold=args.hash_threshold or settings.hash_threshold,
        min_group_size=args.min_group_size or settings.min_group_size,
    )
    run_gui(settings, initial_directory=args.directory)
    save_settings(settings)


if __name__ == "__main__":
    main()
