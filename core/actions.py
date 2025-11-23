from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .models import Photo


def select_discarded(photos: Iterable[Photo]) -> list[Photo]:
    """Return photos marked as not kept."""

    return [photo for photo in photos if not photo.keep]


def move_photos(
    photos: Iterable[Photo], destination: Path, *, dry_run: bool = False
) -> int:
    """Move discarded photos to ``destination``.

    Returns the number of files moved. Parent directories inside
    ``destination`` mirror the source structure to avoid clashes. When
    ``dry_run`` is ``True`` no filesystem changes are made, but the method still
    reports how many moves would have been executed.
    """

    photo_list = list(photos)
    if not photo_list:
        return 0

    destination.mkdir(parents=True, exist_ok=True)
    common_root = _infer_common_root(photo_list)
    moved = 0
    for photo in photo_list:
        if common_root is None:
            relative = photo.path.name
        else:
            try:
                relative = photo.path.relative_to(common_root)
            except ValueError:
                relative = photo.path.name
        target = destination / relative
        target = _ensure_unique_target(target)

        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(photo.path), target)
        moved += 1
    return moved


def _infer_common_root(photos: list[Photo]) -> Path | None:
    """Return the common parent directory for the provided photos if any."""

    if not photos:
        return None
    paths = [str(photo.path.parent) for photo in photos]
    try:
        common = os.path.commonpath(paths)
    except ValueError:
        return None
    return Path(common) if common else None


def _ensure_unique_target(target: Path) -> Path:
    """Generate a non-colliding destination path by appending counters."""

    if not target.exists():
        return target

    stem, suffix = target.stem, target.suffix
    parent = target.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
