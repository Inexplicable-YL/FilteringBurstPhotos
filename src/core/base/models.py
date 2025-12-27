from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar, get_args

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

    raw_image: Image.Image
    image: Image.Image

    taken_time: datetime
    hash_hex: str
    format: str
    group_id: int | None = None
    keep: bool = True


PhotoType = TypeVar("PhotoType", bound="Photo")
OtherPhotoType = TypeVar("OtherPhotoType", bound="Photo")

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


class PhotoResult(BaseModel, Generic[PhotoType]):
    """Snapshot returned by the streaming grouping pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    photos: list[PhotoType] = Field(default_factory=list)
    done: bool = False

    @property
    def PhotosType(self) -> type[PhotoType]:
        """The type of Photo this PhotoResult contains."""
        # First loop through bases -- this will help generic
        # any pydantic models.
        for base in self.__class__.mro():
            if hasattr(base, "__pydantic_generic_metadata__"):
                metadata = base.__pydantic_generic_metadata__
                if "args" in metadata and len(metadata["args"]) == 1:
                    return metadata["args"][0]

        for cls in self.__class__.__orig_bases__:  # type: ignore[attr-defined]
            type_args = get_args(cls)
            if type_args and len(type_args) == 1:
                return type_args[0]

        raise TypeError("Could not determine PhotoType for PhotoResult.")
