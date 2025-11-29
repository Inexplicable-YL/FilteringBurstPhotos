from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from PIL import Image


@dataclass
class Photo:
    """Represents a single photo on disk.

    Attributes:
        path: Full filesystem path.
        taken_time: Best-effort capture time.
        hash_hex: Hex representation of the perceptual hash.
        format: File extension in upper case (e.g. JPG, CR3).
        group_id: Identifier assigned during grouping.
        keep: Whether the user decided to keep the photo.
    """

    path: Path
    image: Image.Image
    taken_time: datetime
    hash_hex: str
    format: str
    group_id: int | None = None
    keep: bool = True


@dataclass
class Group:
    """Represents a burst group of photos."""

    id: int
    photos: list[Photo] = field(default_factory=list)

    @property
    def representative(self) -> Photo | None:
        return self.photos[0] if self.photos else None

    def size(self) -> int:
        return len(self.photos)
