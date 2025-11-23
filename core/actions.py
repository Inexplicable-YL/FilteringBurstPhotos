from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from .models import Photo


def select_discarded(photos: Iterable[Photo]) -> list[Photo]:
    """Return photos marked as not kept."""

    return [photo for photo in photos if not photo.keep]


def move_photos(
    photos: Iterable[Photo], destination: Path, dry_run: bool = False
) -> int:
    """Move discarded photos to ``destination``.

    Returns the number of files moved. Parent directories inside ``destination``
    mirror the source structure to avoid clashes.
    """

    destination.mkdir(parents=True, exist_ok=True)
    moved = 0
    for photo in photos:
        relative = photo.path.name
        target = destination / relative
        if not dry_run:
            shutil.move(str(photo.path), target)
        moved += 1
    return moved
