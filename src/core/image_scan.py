from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

import anyio
from anyio.to_thread import run_sync
from PIL import ExifTags, Image

from .image_hash import compute_phash_async
from .image_raw import RAW_EXTENSIONS, load_image_for_path_async
from .models import Photo

if TYPE_CHECKING:
    from pathlib import Path
logger = logging.getLogger(__name__)


def _discover_supported_extensions() -> set[str]:
    """Return a normalized set of file extensions supported by the scanner."""

    pil_extensions = {ext.lower() for ext in Image.registered_extensions()}
    return pil_extensions | {ext.lower() for ext in RAW_EXTENSIONS}


SUPPORTED_EXTENSIONS: set[str] = _discover_supported_extensions()

EXIF_DATETIME_KEYS = {k: v for k, v in ExifTags.TAGS.items() if v == "DateTime"}


async def scan_directory_async(
    directory: Path,
    recursive: bool = True,
    ignore_errors: bool = True,
    max_concurrency: int | None = None,
) -> list[Photo]:
    """Asynchronously scan a directory and hash supported image files.

    Work is parallelized with ``anyio`` to keep UI and event loops responsive.
    ``max_concurrency`` limits how many images are processed at once; by default
    it scales with available CPU cores but is capped to avoid overwhelming I/O.
    """

    if not directory.exists():
        raise FileNotFoundError(directory)

    paths = _collect_paths(directory, recursive)
    if not paths:
        return []

    concurrency = max_concurrency or min(16, (os.cpu_count() or 4) * 2)
    results: dict[int, Photo] = {}
    async with anyio.create_task_group() as tg:
        semaphore = anyio.Semaphore(concurrency)
        for idx, path in enumerate(paths):
            tg.start_soon(
                _process_path,
                idx,
                path,
                semaphore,
                results,
                ignore_errors,
            )

    return [results[i] for i in range(len(paths)) if i in results]


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


async def _process_path(
    index: int,
    path: Path,
    semaphore: anyio.Semaphore,
    results: dict[int, Photo],
    ignore_errors: bool,
) -> None:
    async with semaphore:
        try:
            image = await load_image_for_path_async(path)
            taken_time = await run_sync(_resolve_taken_time, image, path)
            hash_hex = await compute_phash_async(image)
        except Exception as exc:  # pragma: no cover - depends on file content
            if not ignore_errors:
                raise
            logger.warning("Skipping %s: %s", path, exc)
            return

        results[index] = Photo(
            path=path,
            image=image,
            taken_time=taken_time,
            hash_hex=hash_hex,
            format=path.suffix.upper().lstrip("."),
        )
