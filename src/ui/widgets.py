from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QLabel,
    QLayout,
    QLayoutItem,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.constants import (
    ACCENT_COLOR,
    BORDER_COLOR,
    MUTED_TEXT,
    PRIMARY_TEXT,
    SURFACE_BG,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from core.models import Group, Photo


class LeftContainer(QWidget):
    ctrl_scroll = Signal(QWheelEvent)

    @override
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: D401
        """Forward ctrl+滚轮事件用于缩放缩略图。"""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.ctrl_scroll.emit(event)
        else:
            super().wheelEvent(event)


class PreviewLabel(QLabel):
    size_changed = Signal(QSize)

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.size_changed.emit(self.size())


class FlowLayout(QLayout):
    """A simple flow layout that wraps children horizontally."""

    def __init__(self, margin: int = 12, spacing: int = 8) -> None:
        super().__init__()
        self._items: list[QLayoutItem] = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    @override
    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    @override
    def addWidget(self, widget: QWidget) -> None:
        widget.setParent(self.parentWidget())
        self.addItem(QLayoutItemWrapper(widget))

    @override
    def count(self) -> int:
        return len(self._items)

    @override
    def itemAt(self, index: int) -> QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    @override
    def takeAt(self, index: int) -> QLayoutItem:
        return self._items.pop(index)

    @override
    def sizeHint(self) -> QSize:
        width = self.parentWidget().width() if self.parentWidget() else 800
        return QSize(width, self._height_for_width(width))

    @override
    def minimumSize(self) -> QSize:
        width = 200
        return QSize(width, self._height_for_width(width))

    @override
    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    @override
    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect)

    def _do_layout(self, rect: QRect) -> None:
        x = rect.x() + self.contentsMargins().left()
        y = rect.y() + self.contentsMargins().top()
        line_height = 0
        spacing = self.spacing()
        max_width = rect.width() - self.contentsMargins().right()

        for item in self._items:
            widget = item.widget()
            if not widget:
                continue
            hint = widget.sizeHint()
            next_x = x + hint.width() + spacing
            if next_x - spacing > max_width and line_height > 0:
                x = rect.x() + self.contentsMargins().left()
                y = y + line_height + spacing
                next_x = x + hint.width() + spacing
                line_height = 0

            item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())

    def _height_for_width(self, width: int) -> int:
        x = self.contentsMargins().left()
        y = self.contentsMargins().top()
        line_height = 0
        spacing = self.spacing()
        max_width = width - self.contentsMargins().right()
        for item in self._items:
            widget = item.widget()
            if not widget:
                continue
            hint = widget.sizeHint()
            next_x = x + hint.width() + spacing
            if next_x - spacing > max_width and line_height > 0:
                x = self.contentsMargins().left()
                y = y + line_height + spacing
                next_x = x + hint.width() + spacing
                line_height = 0
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height + self.contentsMargins().bottom()

    def row_rects(self, available_width: int) -> list[QRect]:
        rects: list[QRect] = []
        x = self.contentsMargins().left()
        y = self.contentsMargins().top()
        line_height = 0
        spacing = self.spacing()
        max_width = available_width - self.contentsMargins().right()
        start_x = x
        for item in self._items:
            widget = item.widget()
            if not widget:
                continue
            w, h = widget.sizeHint().width(), widget.sizeHint().height()
            next_x = x + w + spacing
            if next_x - spacing > max_width and line_height > 0:
                rects.append(
                    QRect(
                        start_x - 6,
                        y - 6,
                        (x - spacing) - start_x + 12,
                        line_height + 12,
                    )
                )
                x = self.contentsMargins().left()
                y = y + line_height + spacing
                line_height = 0
                start_x = x
                next_x = x + w + spacing
            x = next_x
            line_height = max(line_height, h)
        if self._items:
            rects.append(
                QRect(
                    start_x - 6, y - 6, (x - spacing) - start_x + 12, line_height + 12
                )
            )
        return rects


class QLayoutItemWrapper(QLayoutItem):
    """Simple wrapper to let FlowLayout store widgets."""

    def __init__(self, widget: QWidget) -> None:
        super().__init__()
        self._widget = widget

    @override
    def geometry(self) -> QRect:
        return self._widget.geometry()

    @override
    def setGeometry(self, rect: QRect) -> None:
        self._widget.setGeometry(rect)

    @override
    def sizeHint(self) -> QSize:
        return self._widget.sizeHint()

    @override
    def minimumSize(self) -> QSize:
        return self._widget.minimumSizeHint()

    @override
    def maximumSize(self) -> QSize:
        return self._widget.maximumSize()

    @override
    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    @override
    def hasHeightForWidth(self) -> bool:
        return self._widget.hasHeightForWidth()

    @override
    def heightForWidth(self, width: int) -> int:
        return self._widget.heightForWidth(width)

    @override
    def widget(self) -> QWidget | None:
        return self._widget

    @override
    def isEmpty(self) -> bool:
        return False


class GroupWidget(QWidget):
    """Card-style container that groups thumbnails together."""

    def __init__(
        self,
        group: Group,
        thumbnail_size: int,
        thumbnail_loader: Callable[[Photo], QPixmap],
        on_photo_clicked: Callable[[Photo, bool], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.group = group
        self.thumbnail_loader = thumbnail_loader
        self.on_photo_clicked = on_photo_clicked

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.flow_layout = FlowLayout(margin=18, spacing=12)
        self.flow_layout.setContentsMargins(18, 44, 18, 18)
        self.setLayout(self.flow_layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.thumbnails: list[PhotoThumbnail] = []
        for photo in group.photos:
            thumb = PhotoThumbnail(
                photo, thumbnail_size, self.thumbnail_loader, self._on_thumb_clicked
            )
            self.thumbnails.append(thumb)
            self.flow_layout.addWidget(thumb)

    def _on_thumb_clicked(self, photo: Photo, selected: bool) -> None:
        self.on_photo_clicked(photo, selected)

    @override
    def sizeHint(self) -> QSize:
        # Match the flow layout's computed size to avoid extra vertical padding.
        return self.flow_layout.sizeHint()

    @override
    def minimumSizeHint(self) -> QSize:
        return self.flow_layout.minimumSize()

    def update_thumbnail_size(self, new_size: int) -> None:
        for thumb in self.thumbnails:
            thumb.update_size(new_size)
        self.updateGeometry()
        self.update()
        self.ensure_visible_thumbnails(self.rect())

    def clear_selection(self) -> None:
        for thumb in self.thumbnails:
            thumb.set_selected(False)
        self.update()

    def ensure_visible_thumbnails(self, visible_rect: QRect) -> None:
        margin = 64
        for thumb in self.thumbnails:
            rect = thumb.geometry().adjusted(-margin, -margin, margin, margin)
            if rect.intersects(visible_rect):
                thumb.ensure_loaded()

    @override
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        frame_rect = self.rect().adjusted(4, 4, -4, -4)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(SURFACE_BG))
        painter.drawRoundedRect(frame_rect, 12, 12)

        border_pen = QPen(QColor(BORDER_COLOR))
        border_pen.setWidthF(1.2)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(frame_rect, 12, 12)

        header_rect = QRect(
            frame_rect.left() + 16, frame_rect.top() + 10, frame_rect.width() - 32, 26
        )
        painter.setPen(QPen(QColor(MUTED_TEXT)))
        painter.drawText(
            header_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"连拍组 {self.group.id} · {len(self.group.photos)} 张",
        )

        stripe = QColor(BORDER_COLOR)
        stripe.setAlpha(80)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(stripe)
        rects = self.flow_layout.row_rects(self.width())
        for rect in rects:
            painter.drawRoundedRect(rect.adjusted(4, 8, -4, -4), 10, 10)
        super().paintEvent(event)


class PhotoThumbnail(QWidget):
    """Thumbnail widget supporting selection and dynamic sizing."""

    def __init__(
        self,
        photo: Photo,
        size: int,
        loader: Callable[[Photo], QPixmap],
        on_clicked: Callable[[Photo, bool], None],
    ) -> None:
        super().__init__()
        self.setObjectName("photoThumb")
        self.photo = photo
        self.loader = loader
        self.on_clicked = on_clicked
        self.is_selected = False
        self._size_hint: QSize
        self._needs_pixmap = True
        self._current_size = size

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label = QLabel(photo.path.name)
        self.text_label.setObjectName("photoName")
        self.text_label.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        self.text_label.setWordWrap(True)
        self.text_label.setToolTip(photo.path.name)
        self.text_label.setStyleSheet(f"color: {MUTED_TEXT}; padding: 0 4px;")

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 8)
        layout.setSpacing(6)
        layout.addWidget(self.image_label)
        layout.addWidget(self.text_label)
        self.setLayout(layout)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.update_size(size)
        self._apply_frame_style()

    @override
    def sizeHint(self) -> QSize:
        return self._size_hint

    @override
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.set_selected(not self.is_selected)
            self.on_clicked(self.photo, self.is_selected)
        super().mousePressEvent(event)

    def set_selected(self, value: bool) -> None:
        if value == self.is_selected:
            return
        self.is_selected = value
        self._apply_frame_style()

    def update_size(self, size: int) -> None:
        if size == self._current_size and hasattr(self, "_size_hint"):
            return
        self._current_size = size
        minimal = size < 70  # noqa: PLR2004
        self.image_label.setVisible(not minimal)
        if not minimal:
            self.image_label.setFixedSize(QSize(size, size))
        else:
            self.image_label.clear()
        self.text_label.setFixedWidth(size + 8)
        font = self.text_label.font()
        font.setPointSizeF(max(8.0, size / 14))
        self.text_label.setFont(font)
        self._size_hint = QSize(size + 12, (size if not minimal else 0) + 32)
        self._needs_pixmap = True
        self.update()

    def ensure_loaded(self) -> None:
        if not self.should_load():
            return
        self._load_pixmap()

    def should_load(self) -> bool:
        return self.image_label.isVisible() and self._needs_pixmap

    def _load_pixmap(self) -> None:
        scaled = self.loader(self.photo)
        self.image_label.setPixmap(scaled)
        self._needs_pixmap = False

    def _apply_frame_style(self) -> None:
        border = (
            f"2px solid {ACCENT_COLOR}"
            if self.is_selected
            else f"1px solid {BORDER_COLOR}"
        )
        self.setStyleSheet(
            f"""
            QWidget#photoThumb {{
                background: qlineargradient(
                    x1 0 y1 0, x2 0 y2 1,
                    stop: 0 #161d2c,
                    stop: 1 #0f1422
                );
                color: {PRIMARY_TEXT};
                border: {border};
                border-radius: 10px;
            }}
            QWidget#photoThumb:hover {{
                border-color: {ACCENT_COLOR};
                background: qlineargradient(
                    x1 0 y1 0, x2 0 y2 1,
                    stop: 0 #1c2435,
                    stop: 1 #121a28
                );
            }}
            """
        )


class ThumbnailScrollArea(QScrollArea):
    """Scroll area that forwards wheel events for zooming."""

    @override
    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            event.ignore()
            return
        super().wheelEvent(event)
