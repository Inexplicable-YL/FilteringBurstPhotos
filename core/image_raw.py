from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

RAW_EXTENSIONS: Iterable[str] = {
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
    Pillow.
    """

    suffix = path.suffix.lower()
    try:
        if suffix in RAW_EXTENSIONS:
            return _load_raw_as_image(path)
        return Image.open(path).convert("RGB")
    except FileNotFoundError as exc:  # pragma: no cover - relies on file system
        raise UnsupportedImageError(f"File not found: {path}") from exc
    except UnidentifiedImageError as exc:
        raise UnsupportedImageError(f"Unable to identify image file: {path}") from exc


def _load_raw_as_image(path: Path) -> Image.Image:
    import rawpy  # type: ignore[import-not-found]  # noqa: PLC0415

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            no_auto_bright=True,
            output_bps=8,
            use_camera_wb=True,
            gamma=(1, 1),
        )

    return Image.fromarray(rgb)
