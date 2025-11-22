from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


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
    taken_time: datetime
    hash_hex: str
    format: str
    group_id: Optional[int] = None
    keep: bool = True


@dataclass
class Group:
    """Represents a burst group of photos."""

    id: int
    photos: List[Photo] = field(default_factory=list)

    @property
    def representative(self) -> Optional[Photo]:
        return self.photos[0] if self.photos else None

    def size(self) -> int:
        return len(self.photos)
