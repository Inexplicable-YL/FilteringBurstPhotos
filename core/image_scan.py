from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from PIL import ExifTags, Image

from .image_hash import compute_phash
from .image_raw import RAW_EXTENSIONS, load_image_for_path
from .models import Photo

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def _discover_supported_extensions() -> set[str]:
    """Return a normalized set of file extensions supported by the scanner."""

    pil_extensions = {ext.lower() for ext in Image.registered_extensions()}
    return pil_extensions | {ext.lower() for ext in RAW_EXTENSIONS}


SUPPORTED_EXTENSIONS: Iterable[str] = _discover_supported_extensions()

logger = logging.getLogger(__name__)


EXIF_DATETIME_KEYS = {k: v for k, v in ExifTags.TAGS.items() if v == "DateTime"}


def scan_directory(
    directory: Path, *, recursive: bool = True, ignore_errors: bool = True
) -> list[Photo]:
    """Scan a directory for supported image files and return ``Photo`` items.

    Parameters:
        directory: Root directory to scan.
        recursive: Whether to recurse into subdirectories.
        ignore_errors: When ``True`` (default), files that fail to load or hash
            are skipped with a warning instead of aborting the scan.
    """

    if not directory.exists():
        raise FileNotFoundError(directory)

    paths = _collect_paths(directory, recursive)
    photos: list[Photo] = []
    for path in paths:
        try:
            image = load_image_for_path(path)
            taken_time = _resolve_taken_time(image, path)
            hash_hex = compute_phash(image)
        except Exception as exc:  # pragma: no cover - depends on file content
            if not ignore_errors:
                raise
            logger.warning("Skipping %s: %s", path, exc)
            continue

        photos.append(
            Photo(
                path=path,
                taken_time=taken_time,
                hash_hex=hash_hex,
                format=path.suffix.upper().lstrip("."),
            )
        )
    return photos


def _collect_paths(directory: Path, recursive: bool) -> list[Path]:
    def is_supported(path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_EXTENSIONS

    if recursive:
        return sorted(
            [
                path
                for path in directory.rglob("*")
                if path.is_file() and is_supported(path)
            ]
        )
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and is_supported(path)]
    )


def _resolve_taken_time(image: Image.Image, path: Path) -> datetime:
    try:
        exif = image.getexif()
        if exif:
            for key in EXIF_DATETIME_KEYS:
                value = exif.get(key)
                if value:
                    try:
                        return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")  # noqa: DTZ007
                    except ValueError:
                        logger.debug("Invalid EXIF date %s in %s", value, path)
    except Exception as exc:  # pragma: no cover - depends on file availability
        logger.debug("Failed to read EXIF from %s: %s", path, exc)

    stat = path.stat()
    timestamp = min(stat.st_mtime, stat.st_ctime)
    return datetime.fromtimestamp(timestamp)  # noqa: DTZ006
