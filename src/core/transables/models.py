from datetime import datetime
from pathlib import Path
from typing import Self

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class Photo(BaseModel):
    """Represents a single photo on disk.

    Attributes:
        path: Full filesystem path.
        taken_time: Best-effort capture time.
        hash_hex: Hex representation of the perceptual hash.s
        format: File extension in upper case (e.g. JPG, CR3).
        keep: Whether the user decided to keep the photo.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    photo_type: str
    id: str
    path: Path
    raw_image: Image.Image
    image: Image.Image
    taken_time: datetime
    format: str
    keep: bool = True
    metadata: dict = Field(default_factory=dict)

    @property
    def PhotoType(self) -> type[Self]:
        """The type of Photo this Photo instance is."""
        return self.__class__

    def get_type(self, suffix: str | None = None, *, type: str | None = None) -> str:
        """Get the type of the Transable."""
        if type:
            type_ = type
        elif hasattr(self, "type") and self.photo_type:
            type_ = self.photo_type
        else:
            # Here we handle a case where the transable subclass is also a pydantic
            # model.
            cls = self.__class__
            # Then it's a pydantic sub-class, and we have to check
            # whether it's a generic, and if so recover the original type.
            if (
                hasattr(
                    cls,
                    "__pydantic_generic_metadata__",
                )
                and "origin" in cls.__pydantic_generic_metadata__  # pyright: ignore[reportAttributeAccessIssue]
                and cls.__pydantic_generic_metadata__["origin"] is not None  # pyright: ignore[reportAttributeAccessIssue]
            ):
                type_ = cls.__pydantic_generic_metadata__["origin"].__name__  # pyright: ignore[reportAttributeAccessIssue]
            else:
                type_ = cls.__name__

        if suffix:
            if type_[0].isupper():
                return type_ + suffix.title()
            return type_ + "_" + suffix.lower()
        return type_

    def get_unique_id(self) -> str:
        """Get a unique ID for this photo based on its path and ID."""
        return f"{self.get_type()}-{self.id}-{str(self.path)}"


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
