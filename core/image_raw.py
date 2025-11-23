from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from collections.abc import Iterable

RAW_EXTENSIONS: set[str] = {
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".raf",
    ".orf",
    ".rw2",
    ".dng",
}


class UnsupportedImageError(RuntimeError):
    """Raised when an image cannot be opened for hashing or preview."""


def load_image_for_hash(path: Path) -> Image.Image:
    """Return a Pillow ``Image`` suitable for hashing.

    RAW files are decoded with ``rawpy`` if available; otherwise a helpful
    ``UnsupportedImageError`` is raised. Standard image formats are loaded via
    Pillow. The returned image is fully loaded in memory, so the caller does not
    need to manage file handles.
    """

    suffix = path.suffix.lower()
    try:
        if suffix in RAW_EXTENSIONS:
            return _load_raw_as_image(path)
        return _load_standard_image(path)
    except FileNotFoundError as exc:  # pragma: no cover - relies on file system
        raise UnsupportedImageError(f"File not found: {path}") from exc
    except UnidentifiedImageError as exc:
        raise UnsupportedImageError(f"Unable to identify image file: {path}") from exc
    except UnsupportedImageError:
        raise
    except Exception as exc:  # pragma: no cover - unexpected runtime errors
        raise UnsupportedImageError(f"Failed to load image {path}: {exc}") from exc


def _load_standard_image(path: Path) -> Image.Image:
    """Load a non-RAW image and ensure it is an RGB pillow instance."""

    with Image.open(path) as image:
        return image.convert("RGB")


def _load_raw_as_image(path: Path) -> Image.Image:
    """Decode a RAW image using rawpy.

    A clear ``UnsupportedImageError`` is raised when ``rawpy`` is not available
    so callers can surface actionable feedback to the user.
    """

    try:
        import rawpy  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise UnsupportedImageError(
            "RAW file support requires the optional 'rawpy' dependency."
        ) from exc

    try:
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                no_auto_bright=True,
                output_bps=8,
                use_camera_wb=True,
                gamma=(1, 1),
            )
    except Exception as exc:  # pragma: no cover - depends on rawpy internals
        raise UnsupportedImageError(f"Failed to decode RAW file {path}: {exc}") from exc

    return Image.fromarray(rgb)
