from __future__ import annotations

import logging
import threading
from pathlib import Path

import anyio
from anyio.to_thread import run_sync
from PIL import UnidentifiedImageError
from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QAction, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from config.settings import Settings
from core.grouping import group_bursts
from core.image_raw import load_image_for_hash
from core.image_scan import scan_directory
from core.models import Group, Photo
from ui.image_utils import pil_to_qpixmap, scale_pixmap

logger = logging.getLogger(__name__)

THUMBNAIL_MIN_SIZE = 48
THUMBNAIL_MAX_SIZE = 220
INITIAL_THUMBNAIL_SIZE = 140


class ScanWorker(QObject):
    """Background scanner using anyio threads to keep UI responsive."""

    finished = Signal(list, list)
    failed = Signal(str)

    def __init__(self, directory: Path, settings: Settings) -> None:
        super().__init__()
        self.directory = directory
        self.settings = settings

    def start(self) -> None:
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self) -> None:
        try:
            photos, groups = anyio.run(self._scan_and_group)
            self.finished.emit(photos, groups)
        except Exception as exc:  # pragma: no cover - background thread safety
            logger.exception("Failed during scan")
            self.failed.emit(str(exc))

    async def _scan_and_group(self) -> tuple[list[Photo], list[Group]]:
        photos = await run_sync(
            scan_directory,
            self.directory,
            self.settings.scan_recursive,
        )
        groups = await run_sync(
            group_bursts,
            photos,
            self.settings.time_threshold_seconds,
            self.settings.hash_threshold,
            self.settings.min_group_size,
        )
        return photos, groups


class MainWindow(QMainWindow):
    """GUI for browsing and filtering burst photos."""

    def __init__(self, settings: Settings, initial_directory: Path | None = None):
        super().__init__()
        self.settings = settings
        self.current_directory: Path | None = initial_directory
        self.photos: list[Photo] = []
        self.groups: list[Group] = []

        self.setWindowTitle("Filtering Burst Photos")
        self.resize(1280, 860)

        self._thumbnail_cache: dict[Path, QPixmap] = {}
        self.thumbnail_size = INITIAL_THUMBNAIL_SIZE
        self._selected_paths: set[Path] = set()
        self._group_widgets: list[GroupWidget] = []
        self._scan_worker: ScanWorker | None = None

        self.toolbar = self._build_toolbar()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.thumbnail_container = QWidget()
        self.thumbnail_container.setStyleSheet("background: black;")
        self.group_layout = QVBoxLayout()
        self.group_layout.setContentsMargins(12, 12, 12, 12)
        self.group_layout.setSpacing(16)
        self.thumbnail_container.setLayout(self.group_layout)

        self.thumbnail_scroll = ThumbnailScrollArea()
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_scroll.setWidget(self.thumbnail_container)
        self.thumbnail_scroll.viewport().installEventFilter(self)

        self.cancel_button = QPushButton("取消")
        self.save_button = QPushButton("保存")
        self.cancel_button.clicked.connect(self._cancel_selection)
        self.save_button.clicked.connect(self._save_selection)
        self._set_action_buttons_enabled(False)

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.cancel_button)
        buttons_layout.addWidget(self.save_button)
        buttons_container = QWidget()
        buttons_container.setLayout(buttons_layout)
        buttons_container.setStyleSheet("background: black; padding: 8px;")

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.thumbnail_scroll)
        left_layout.addWidget(buttons_container)
        left_container = QWidget()
        left_container.setLayout(left_layout)

        self.preview_label = QLabel("选择一张照片查看预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setStyleSheet("background: #111; color: #ccc;")

        splitter = QSplitter()
        splitter.addWidget(left_container)
        splitter.addWidget(self.preview_label)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        if initial_directory:
            self.load_directory(initial_directory)

    def _build_toolbar(self) -> QToolBar:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)

        open_action = QAction(QIcon.fromTheme("folder-open"), "打开目录", self)
        open_action.triggered.connect(self._pick_directory)
        toolbar.addAction(open_action)

        rescan_action = QAction("重新扫描", self)
        rescan_action.triggered.connect(self._rescan)
        toolbar.addAction(rescan_action)

        toolbar.addSeparator()

        quit_action = QAction("退出", self)
        quit_action.triggered.connect(QApplication.instance().quit)
        toolbar.addAction(quit_action)

        self.addToolBar(toolbar)
        return toolbar

    def _pick_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择照片目录")
        if selected:
            self.load_directory(Path(selected))

    def _rescan(self) -> None:
        if not self.current_directory:
            QMessageBox.information(self, "提示", "请先选择一个目录。")
            return
        self.load_directory(self.current_directory)

    def load_directory(self, directory: Path) -> None:
        if not directory.exists():
            QMessageBox.warning(self, "未找到目录", f"目录不存在: {directory}")
            return

        if self._scan_worker is not None:
            self.status_bar.showMessage("正在处理上一个请求，请稍候…", 3000)
            return

        self.status_bar.showMessage(f"扫描中: {directory}")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._scan_worker = ScanWorker(directory, self.settings)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.failed.connect(self._on_scan_failed)
        self._scan_worker.start()

    def _on_scan_finished(self, photos: list[Photo], groups: list[Group]) -> None:
        QApplication.restoreOverrideCursor()
        self.photos = photos
        self.groups = groups
        self.current_directory = (
            self._scan_worker.directory if self._scan_worker else self.current_directory
        )
        self._scan_worker = None
        self._populate_groups()
        self.status_bar.showMessage(
            f"找到 {len(self.photos)} 张照片，{len(self.groups)} 个连拍组", 5000
        )

    def _on_scan_failed(self, error: str) -> None:
        QApplication.restoreOverrideCursor()
        self._scan_worker = None
        logger.error("Scan failed: %s", error)
        QMessageBox.critical(self, "扫描失败", error)

    def _populate_groups(self) -> None:
        self._clear_layout(self.group_layout)
        self._group_widgets.clear()
        self._thumbnail_cache.clear()
        self._selected_paths.clear()
        self._set_action_buttons_enabled(False)

        if not self.groups:
            placeholder = QLabel("未找到连拍组")
            placeholder.setStyleSheet("color: #999; background: black;")
            placeholder.setAlignment(Qt.AlignCenter)
            self.group_layout.addWidget(placeholder)
            return

        for group in self.groups:
            widget = GroupWidget(
                group=group,
                thumbnail_size=self.thumbnail_size,
                thumbnail_loader=self._load_thumbnail,
                on_photo_clicked=self._on_photo_clicked,
            )
            self._group_widgets.append(widget)
            self.group_layout.addWidget(widget)

        self.group_layout.addStretch()

    def _load_thumbnail(self, photo: Photo) -> QPixmap:
        cached = self._thumbnail_cache.get(photo.path)
        if cached:
            return cached

        try:
            image = load_image_for_hash(photo.path)
            pixmap = scale_pixmap(pil_to_qpixmap(image), self.thumbnail_size)
        except (FileNotFoundError, UnidentifiedImageError, RuntimeError) as exc:
            logger.warning("Unable to load thumbnail for %s: %s", photo.path, exc)
            pixmap = QPixmap(self.thumbnail_size, self.thumbnail_size)
            pixmap.fill(Qt.darkGray)

        self._thumbnail_cache[photo.path] = pixmap
        return pixmap

    def _on_photo_clicked(self, photo: Photo, selected: bool) -> None:
        if selected:
            self._selected_paths.add(photo.path)
        else:
            self._selected_paths.discard(photo.path)
        self._update_preview(photo)
        self._set_action_buttons_enabled(bool(self._selected_paths))

    def _update_preview(self, photo: Photo) -> None:
        try:
            image = load_image_for_hash(photo.path)
            pixmap = pil_to_qpixmap(image)
            scaled = pixmap.scaled(
                self.preview_label.width(),
                self.preview_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.preview_label.setPixmap(scaled)
            self.preview_label.setText("")
            self.status_bar.showMessage(f"预览: {photo.path}")
        except Exception as exc:  # pragma: no cover - UI feedback
            logger.exception("Failed to load preview")
            QMessageBox.warning(self, "预览失败", str(exc))

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        style = "background-color: #444; color: #999;" if not enabled else ""
        self.cancel_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.cancel_button.setStyleSheet(style)
        self.save_button.setStyleSheet(style)

    def _cancel_selection(self) -> None:
        if not self._selected_paths:
            return
        self._selected_paths.clear()
        for widget in self._group_widgets:
            widget.clear_selection()
        self._set_action_buttons_enabled(False)
        self.preview_label.setText("选择一张照片查看预览")
        self.preview_label.setPixmap(QPixmap())

    def _save_selection(self) -> None:
        if not self._selected_paths:
            return

        new_groups: list[Group] = []
        kept_photos: list[Photo] = []
        for group in self.groups:
            selected_in_group = [
                photo for photo in group.photos if photo.path in self._selected_paths
            ]
            if not selected_in_group or len(selected_in_group) == len(group.photos):
                new_groups.append(group)
                kept_photos.extend(group.photos)
                continue

            group.photos = selected_in_group
            new_groups.append(group)
            kept_photos.extend(selected_in_group)
        self.groups = new_groups
        self.photos = kept_photos
        self._populate_groups()
        self.status_bar.showMessage("已保存选择，未选中的照片已移除显示", 4000)

    def _clear_layout(self, layout: QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if not item:
                continue
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self.thumbnail_scroll.viewport():
            if (
                event.type() == QEvent.Type.Wheel
                and QApplication.keyboardModifiers() & Qt.ControlModifier
            ):
                delta = event.angleDelta().y() // 120
                if delta:
                    new_size = max(
                        THUMBNAIL_MIN_SIZE,
                        min(THUMBNAIL_MAX_SIZE, self.thumbnail_size + delta * 12),
                    )
                    if new_size != self.thumbnail_size:
                        self.thumbnail_size = new_size
                        self._refresh_thumbnail_sizes()
                return True
        return super().eventFilter(obj, event)

    def _refresh_thumbnail_sizes(self) -> None:
        self._thumbnail_cache.clear()
        for widget in self._group_widgets:
            widget.update_thumbnail_size(self.thumbnail_size)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        # Refresh preview size when the window changes.
        if self._selected_paths:
            last_selected = next(iter(self._selected_paths))
            for group in self.groups:
                for photo in group.photos:
                    if photo.path == last_selected:
                        self._update_preview(photo)
                        return


class FlowLayout(QLayout):
    """A simple flow layout that wraps children horizontally."""

    def __init__(self, margin: int = 12, spacing: int = 8):
        super().__init__()
        self._items: list[QLayoutItem] = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def addItem(self, item: QLayoutItem) -> None:  # type: ignore[override]
        self._items.append(item)

    def addWidget(self, widget: QWidget) -> None:  # type: ignore[override]
        widget.setParent(self.parentWidget())
        self.addItem(QLayoutItemWrapper(widget))

    def count(self) -> int:  # type: ignore[override]
        return len(self._items)

    def itemAt(self, index: int):  # type: ignore[override]
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):  # type: ignore[override]
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def sizeHint(self):  # type: ignore[override]
        width = self.parentWidget().width() if self.parentWidget() else 800
        return QSize(width, self._height_for_width(width))

    def minimumSize(self) -> QSize:  # type: ignore[override]
        width = 200
        return QSize(width, self._height_for_width(width))

    def expandingDirections(self):  # type: ignore[override]
        return Qt.Orientations()

    def setGeometry(self, rect: QRect) -> None:  # type: ignore[override]
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

    def __init__(self, widget: QWidget):
        super().__init__()
        self._widget = widget

    def geometry(self) -> QRect:  # type: ignore[override]
        return self._widget.geometry()

    def setGeometry(self, rect: QRect) -> None:  # type: ignore[override]
        self._widget.setGeometry(rect)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return self._widget.sizeHint()

    def minimumSize(self) -> QSize:  # type: ignore[override]
        return self._widget.minimumSizeHint()

    def maximumSize(self) -> QSize:  # type: ignore[override]
        return self._widget.maximumSize()

    def expandingDirections(self):  # type: ignore[override]
        return Qt.Orientations()

    def hasHeightForWidth(self) -> bool:  # type: ignore[override]
        return self._widget.hasHeightForWidth()

    def heightForWidth(self, width: int) -> int:  # type: ignore[override]
        return self._widget.heightForWidth(width)

    def widget(self) -> QWidget | None:  # type: ignore[override]
        return self._widget

    def isEmpty(self) -> bool:  # type: ignore[override]
        return False


class GroupWidget(QWidget):
    """Widget that draws a dashed group frame behind thumbnails."""

    def __init__(
        self,
        group: Group,
        thumbnail_size: int,
        thumbnail_loader,
        on_photo_clicked,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.group = group
        self.thumbnail_loader = thumbnail_loader
        self.on_photo_clicked = on_photo_clicked

        self.setStyleSheet("background: black;")
        self.flow_layout = FlowLayout(margin=16, spacing=12)
        self.setLayout(self.flow_layout)
        self.thumbnails: list[PhotoThumbnail] = []
        for photo in group.photos:
            thumb = PhotoThumbnail(
                photo, thumbnail_size, self.thumbnail_loader, self._on_thumb_clicked
            )
            self.thumbnails.append(thumb)
            self.flow_layout.addWidget(thumb)

    def _on_thumb_clicked(self, photo: Photo, selected: bool) -> None:
        self.on_photo_clicked(photo, selected)

    def update_thumbnail_size(self, new_size: int) -> None:
        for thumb in self.thumbnails:
            thumb.update_size(new_size)
        self.updateGeometry()
        self.update()

    def clear_selection(self) -> None:
        for thumb in self.thumbnails:
            thumb.set_selected(False)
        self.update()

    def paintEvent(self, event):  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.black)

        pen = QPen(Qt.darkGray)
        pen.setStyle(Qt.DashLine)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.darkGray)
        rects = self.flow_layout.row_rects(self.width())
        for rect in rects:
            painter.fillRect(rect, Qt.darkGray)
            painter.drawRect(rect)
        super().paintEvent(event)


class PhotoThumbnail(QWidget):
    """Thumbnail widget supporting selection and dynamic sizing."""

    def __init__(self, photo: Photo, size: int, loader, on_clicked) -> None:
        super().__init__()
        self.photo = photo
        self.loader = loader
        self.on_clicked = on_clicked
        self.is_selected = False

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.text_label = QLabel(photo.path.name)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setStyleSheet("color: white;")

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.image_label)
        layout.addWidget(self.text_label)
        self.setLayout(layout)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("background: transparent; color: white;")

        self.update_size(size)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return self._size_hint

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self.set_selected(not self.is_selected)
            self.on_clicked(self.photo, self.is_selected)
        super().mousePressEvent(event)

    def set_selected(self, value: bool) -> None:
        self.is_selected = value
        border = "2px solid #4da3ff" if value else "1px solid transparent"
        self.setStyleSheet(
            f"background: rgba(0,0,0,0); color: white; border: {border}; border-radius: 6px;"
        )

    def update_size(self, size: int) -> None:
        pixmap = self.loader(self.photo)
        scaled = scale_pixmap(pixmap, size)
        minimal = size < 70
        self.image_label.setVisible(not minimal)
        if not minimal:
            self.image_label.setPixmap(scaled)
            self.image_label.setFixedSize(QSize(size, size))
        self.text_label.setFixedWidth(size + 8)
        font = self.text_label.font()
        font.setPointSizeF(max(8.0, size / 14))
        self.text_label.setFont(font)
        self._size_hint = QSize(size + 12, (size if not minimal else 0) + 32)
        self.update()


class ThumbnailScrollArea(QScrollArea):
    """Scroll area that forwards wheel events for zooming."""

    def wheelEvent(self, event):  # type: ignore[override]
        if event.modifiers() & Qt.ControlModifier:
            event.ignore()
            return
        super().wheelEvent(event)
