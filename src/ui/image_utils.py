from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

if TYPE_CHECKING:
    from PIL import Image


def pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """Convert a Pillow image to a Qt pixmap."""

    prepared = _ensure_rgba(image)
    data = prepared.tobytes("raw", "RGBA")
    qimage = QImage(
        data, prepared.width, prepared.height, QImage.Format.Format_RGBA8888
    )
    return QPixmap.fromImage(qimage)


def scale_pixmap(pixmap: QPixmap, max_size: int) -> QPixmap:
    """Return a scaled pixmap preserving aspect ratio."""

    if pixmap.isNull():
        return pixmap
    return pixmap.scaled(
        max_size,
        max_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _ensure_rgba(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGBA")
    return image
