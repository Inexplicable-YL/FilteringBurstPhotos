from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from PIL import ExifTags, Image

from .image_hash import compute_phash
from .models import Photo

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

SUPPORTED_EXTENSIONS: Iterable[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".raf",
    ".orf",
    ".rw2",
    ".dng",
}

logger = logging.getLogger(__name__)


EXIF_DATETIME_KEYS = {v: k for k, v in ExifTags.TAGS.items() if v == "DateTimeOriginal"}


def scan_directory(directory: Path, recursive: bool = True) -> list[Photo]:
    """Scan a directory for supported image files and return ``Photo`` items."""

    if not directory.exists():
        raise FileNotFoundError(directory)

    paths = _collect_paths(directory, recursive)
    photos: list[Photo] = []
    for path in paths:
        taken_time = _resolve_taken_time(path)
        hash_hex = compute_phash(path)
        photos.append(
            Photo(
                path=path,
                taken_time=taken_time,
                hash_hex=hash_hex,
                format=path.suffix.upper().lstrip("."),
            )
        )
    return photos


def _collect_paths(directory: Path, recursive: bool) -> Sequence[Path]:
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


def _resolve_taken_time(path: Path) -> datetime:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            if exif:
                for key in EXIF_DATETIME_KEYS:
                    value = exif.get(key)
                    if value:
                        try:
                            return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
                        except ValueError:
                            logger.debug("Invalid EXIF date %s in %s", value, path)
    except Exception as exc:  # pragma: no cover - depends on file availability
        logger.debug("Failed to read EXIF from %s: %s", path, exc)

    stat = path.stat()
    return datetime.fromtimestamp(stat.st_mtime)
