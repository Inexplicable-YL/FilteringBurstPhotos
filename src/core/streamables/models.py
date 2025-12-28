from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Generic, TypeVar, get_args

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

PhotoType = TypeVar("PhotoType", bound="Photo")
OtherPhotoType = TypeVar("OtherPhotoType", bound="Photo")


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

    type: str
    id: str
    path: Path
    raw_image: Image.Image
    image: Image.Image
    taken_time: datetime
    format: str
    keep: bool = True

    def get_type(self, suffix: str | None = None, *, type: str | None = None) -> str:
        """Get the type of the Runnable."""
        if type:
            type_ = type
        elif hasattr(self, "type") and self.type:
            type_ = self.type
        else:
            # Here we handle a case where the runnable subclass is also a pydantic
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


class PhotoResult(BaseModel, Generic[PhotoType]):
    """Snapshot returned by the streaming grouping pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    photos: list[PhotoType] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

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

    def get_unique_id(self, strict: bool = False) -> str | None:
        """Unique ID of the first photo, if available."""
        if self.photos:
            ids = [photo.get_unique_id() for photo in self.photos]
            if strict:
                return sha256(str(ids).encode("utf-8")).hexdigest()
            return sha256(str(sorted(ids)).encode("utf-8")).hexdigest()
        return None


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
