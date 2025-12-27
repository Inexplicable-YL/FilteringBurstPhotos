from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from ui.constants import MAX_BASE_PREVIEW_EDGE, PREVIEW_CACHE_LIMIT
from ui.image_utils import pil_to_qpixmap, scale_pixmap

if TYPE_CHECKING:
    from pathlib import Path

    from core.streamables.models import Photo

# Pillow renamed Resampling in newer versions; keep backward compatibility.
RESAMPLE_MODE = (
    Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS  # type: ignore
)


class PixmapCache:
    """Cache for base pixmaps and thumbnails to avoid repeated scaling."""

    def __init__(self) -> None:
        self._base_pixmap_cache: OrderedDict[Path, QPixmap] = OrderedDict()
        self._thumbnail_cache: dict[tuple[Path, int], QPixmap] = {}

    def clear_all(self) -> None:
        self._base_pixmap_cache.clear()
        self._thumbnail_cache.clear()

    def clear_thumbnails(self) -> None:
        self._thumbnail_cache.clear()

    def get_base_pixmap(self, photo: Photo) -> QPixmap:
        cached = self._base_pixmap_cache.get(photo.path)
        if cached:
            self._base_pixmap_cache.move_to_end(photo.path)
            return cached

        try:
            prepared = photo.raw_image
            if max(prepared.width, prepared.height) > MAX_BASE_PREVIEW_EDGE:
                prepared = prepared.copy()
                prepared.thumbnail(
                    (MAX_BASE_PREVIEW_EDGE, MAX_BASE_PREVIEW_EDGE), RESAMPLE_MODE
                )
            pixmap = pil_to_qpixmap(prepared)
        except RuntimeError:
            pixmap = QPixmap()

        if pixmap.isNull():
            # Fallback placeholder for failures.
            pixmap = QPixmap(1, 1)
            pixmap.fill(Qt.GlobalColor.darkGray)

        self._base_pixmap_cache[photo.path] = pixmap
        if len(self._base_pixmap_cache) > PREVIEW_CACHE_LIMIT:
            self._base_pixmap_cache.popitem(last=False)
        return pixmap

    def load_thumbnail(self, photo: Photo, size: int) -> QPixmap:
        key = (photo.path, size)
        cached = self._thumbnail_cache.get(key)
        if cached:
            return cached

        preview_cached = self._base_pixmap_cache.get(photo.path)
        if preview_cached:
            self._base_pixmap_cache.move_to_end(photo.path)
            scaled = scale_pixmap(preview_cached, size)
        else:
            try:
                thumbnail_image = photo.raw_image.copy()
                thumbnail_image.thumbnail((size * 2, size * 2), RESAMPLE_MODE)
                scaled = scale_pixmap(pil_to_qpixmap(thumbnail_image), size)
            except RuntimeError:
                scaled = QPixmap(size, size)
                scaled.fill(Qt.GlobalColor.darkGray)

        self._thumbnail_cache[key] = scaled
        return scaled
