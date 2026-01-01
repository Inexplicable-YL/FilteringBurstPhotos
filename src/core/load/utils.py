from __future__ import annotations

import logging
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, Literal

import anyio
from anyio import to_thread
from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError

from core.transables.models import Photo
from core.hashing.phash import aphash

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "load_image",
    "aload_image",
    "collect_paths",
    "aload_photo",
    "UnsupportedImageError",
]


logger = logging.getLogger(__name__)


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


SUPPORTED_EXTENSIONS: set[str] = {
    ext.lower() for ext in Image.registered_extensions()
} | {ext.lower() for ext in RAW_EXTENSIONS}


EXIF_DATETIME_KEYS = {k: v for k, v in ExifTags.TAGS.items() if v == "DateTime"}


class UnsupportedImageError(RuntimeError):
    """Raised when an image cannot be opened for hashing or preview."""


def load_image(path: Path) -> Image.Image:
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
        return _load_image_from_path(path)
    except FileNotFoundError as exc:  # pragma: no cover - relies on file system
        raise UnsupportedImageError(f"File not found: {path}") from exc
    except UnidentifiedImageError as exc:
        raise UnsupportedImageError(f"Unable to identify image file: {path}") from exc
    except UnsupportedImageError:
        raise
    except Exception as exc:  # pragma: no cover - unexpected runtime errors
        raise UnsupportedImageError(f"Failed to load image {path}: {exc}") from exc


async def aload_image(path: Path) -> Image.Image:
    """Return a Pillow ``Image`` suitable for hashing asynchronously.

    RAW files are decoded with ``rawpy`` if available; otherwise a helpful
    ``UnsupportedImageError`` is raised. Standard image formats are loaded via
    Pillow. The returned image is fully loaded in memory, so the caller does not
    need to manage file handles.
    """

    return await to_thread.run_sync(load_image, path)


async def aload_photo(
    path: Path, *, semaphore: anyio.Semaphore | None = None, ignore_errors: bool = False
) -> Photo | None:
    """Load and hash a photo within optional concurrency limits.

    Args:
        path: File path to the image.
        semaphore: Optional semaphore that bounds concurrent hash operations.
        ignore_errors: Whether to skip photos that raise during processing.

    Returns:
        Photo instance when loading succeeds, otherwise None if skipped.
    """

    async def _load_photo() -> Photo | None:
        try:
            image = await aload_image(path)
            taken_time = await to_thread.run_sync(_resolve_taken_time, image, path)
            prep_image = await to_thread.run_sync(_preprocess_image, image)
            hash_hex = await aphash(prep_image)
        except Exception as exc:  # pragma: no cover - depends on file content
            if not ignore_errors:
                raise
            logger.warning("Skipping %s: %s", path, exc)
            return None

        return Photo(
            path=path,
            raw_image=image,
            image=prep_image,
            taken_time=taken_time,
            hash_hex=hash_hex,
            format=path.suffix.upper().lstrip("."),
        )

    if semaphore is not None:
        async with semaphore:
            return await _load_photo()
    return await _load_photo()


def collect_paths(directory: Path, *, recursive: bool = False) -> list[Path]:
    """Collect supported image files from a directory.

    Args:
        directory: Root path to scan.
        recursive: Whether to include subdirectories.

    Returns:
        Sorted list of supported image paths.
    """

    def is_supported(path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_EXTENSIONS

    if recursive:
        return sorted(
            [
                path
                for path in directory.rglob("*")
                if path.is_file() and is_supported(path)
            ]
        )
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and is_supported(path)]
    )


def _load_image_from_path(path: Path) -> Image.Image:
    """Load a non-RAW image and ensure it is an RGB pillow instance."""

    with Image.open(path) as image:
        image.load()
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


def _resolve_taken_time(image: Image.Image, path: Path) -> datetime:
    """Estimate the capture datetime from EXIF or filesystem metadata.

    Args:
        image: Decoded image whose EXIF data is inspected.
        path: Original file path used for fallback timestamps.

    Returns:
        Best effort capture datetime.
    """
    try:
        exif = image.getexif()
        if exif:
            for key in EXIF_DATETIME_KEYS:
                value = exif.get(key)
                if value:
                    try:
                        return datetime.strptime(  # noqa: DTZ007
                            str(value).strip(), "%Y:%m:%d %H:%M:%S"
                        )
                    except ValueError:
                        logger.debug("Invalid EXIF date %s in %s", value, path)
            for value in exif.values():
                if value:
                    try:
                        return datetime.strptime(  # noqa: DTZ007
                            str(value).strip(), "%Y:%m:%d %H:%M:%S"
                        )
                    except ValueError:
                        continue
            logger.debug("No valid EXIF date found in %s", path)
        logger.debug("No EXIF data found in %s", path)
    except Exception as exc:  # pragma: no cover - depends on file availability
        logger.debug("Failed to read EXIF from %s: %s", path, exc)

    stat = path.stat()
    timestamp = min(stat.st_mtime, stat.st_ctime)
    return datetime.fromtimestamp(timestamp)  # noqa: DTZ006


def _preprocess_image(
    image: Image.Image,
    *,
    max_long_side: int = 2048,
    max_file_size_bytes: int | None = None,
    output_format: Literal["JPEG", "WEBP", "PNG"] = "JPEG",
    background_rgb: tuple[int, int, int] = (255, 255, 255),
    max_quality: int = 92,
) -> Image.Image:
    """Preprocess an image with EXIF-aware orientation, safe RGB conversion, and size constraints.

    The function preserves aspect ratio (no distortion). It first caps the longest side to
    `max_long_side`, then (optionally) further downsizes until the image can be encoded within
    `max_file_size_bytes` using the specified `output_format` and `max_quality`.

    Args:
        image: Input Pillow image.
        max_long_side: Maximum length for the longer side in pixels.
        max_file_size_bytes: Optional maximum encoded file size in bytes.
        output_format: Encoding format used for file-size estimation (e.g., "JPEG", "WEBP", "PNG").
        background_rgb: Background color for alpha compositing when converting to RGB.
        max_quality: Quality used for file-size estimation for lossy formats (1-95 recommended).

    Returns:
        A processed Pillow image in RGB mode.
    """
    image = ImageOps.exif_transpose(image)

    if image.mode == "RGB":
        rgb = image.copy()
    elif image.mode in ("RGBA", "LA"):
        base = Image.new("RGB", image.size, background_rgb)
        base.paste(image, mask=image.split()[-1])
        rgb = base
    else:
        rgb = image.convert("RGB")

    w, h = rgb.size
    if max(w, h) > max_long_side:
        rgb.thumbnail((max_long_side, max_long_side), Image.Resampling.LANCZOS)

    if max_file_size_bytes is None:
        return rgb

    fmt = (output_format or "JPEG").upper()
    if fmt == "JPG":
        fmt = "JPEG"

    if fmt == "JPEG":
        save_kwargs = {
            "quality": int(max(1, min(95, max_quality))),
            "optimize": True,
            "progressive": True,
            "subsampling": "4:2:0",
        }
    elif fmt == "WEBP":
        save_kwargs = {
            "quality": int(max(1, min(100, max_quality))),
            "method": 6,
        }
    elif fmt == "PNG":
        save_kwargs = {
            "optimize": True,
            "compress_level": 9,
        }
    else:
        save_kwargs = {}

    buf = BytesIO()

    def _encoded_size(img: Image.Image) -> int:
        buf.seek(0)
        buf.truncate(0)
        img.save(buf, format=fmt, **save_kwargs)
        return buf.tell()

    size = _encoded_size(rgb)
    if size <= max_file_size_bytes:
        return rgb

    for _ in range(12):
        w, h = rgb.size
        if w <= 1 and h <= 1:
            break

        scale = (max_file_size_bytes / float(size)) ** 0.5
        scale *= 0.95

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        if new_w >= w and new_h >= h:
            new_w = max(1, w - 1)
            new_h = max(1, h - 1)

        rgb.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
        size = _encoded_size(rgb)
        if size <= max_file_size_bytes:
            break

    return rgb
