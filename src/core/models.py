from datetime import datetime
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class Photo(BaseModel):
    """Represents a single photo on disk.

    Attributes:
        path: Full filesystem path.
        taken_time: Best-effort capture time.
        hash_hex: Hex representation of the perceptual hash.s
        format: File extension in upper case (e.g. JPG, CR3).
        group_id: Identifier assigned during grouping.
        keep: Whether the user decided to keep the photo.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    image: Image.Image
    taken_time: datetime
    hash_hex: str
    format: str
    group_id: int | None = None
    keep: bool = True


class Group(BaseModel):
    """Represents a burst group of photos."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int
    photos: list[Photo] = Field(default_factory=list)

    @property
    def representative(self) -> Photo | None:
        return self.photos[0] if self.photos else None

    def size(self) -> int:
        return len(self.photos)


class GroupingResult(BaseModel):
    """Snapshot returned by the streaming grouping pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    photos: list[Photo] = Field(default_factory=list)
    groups: list[Group] = Field(default_factory=list)
    done: bool = False
