from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PIL import UnidentifiedImageError
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
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

THUMBNAIL_SIZE = 140


class MainWindow(QMainWindow):
    """Basic GUI prototype for browsing grouped burst photos."""

    def __init__(self, settings: Settings, initial_directory: Optional[Path] = None):
        super().__init__()
        self.settings = settings
        self.current_directory: Optional[Path] = initial_directory
        self.photos: List[Photo] = []
        self.groups: List[Group] = []

        self.setWindowTitle("Filtering Burst Photos")
        self.resize(1200, 800)

        self._thumbnail_cache: dict[Path, QPixmap] = {}

        self.toolbar = self._build_toolbar()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setViewMode(QListWidget.IconMode)
        self.thumbnail_list.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        self.thumbnail_list.setResizeMode(QListWidget.Adjust)
        self.thumbnail_list.setSelectionMode(QListWidget.SingleSelection)
        self.thumbnail_list.itemSelectionChanged.connect(self._on_selection_changed)

        self.preview_label = QLabel("选择一张照片查看预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        splitter = QSplitter()
        splitter.addWidget(self.thumbnail_list)
        splitter.addWidget(self.preview_label)
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

        self.status_bar.showMessage(f"扫描中: {directory}")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.photos = scan_directory(directory, recursive=self.settings.scan_recursive)
            self.groups = group_bursts(
                self.photos,
                time_threshold_seconds=self.settings.time_threshold_seconds,
                hash_threshold=self.settings.hash_threshold,
                min_group_size=self.settings.min_group_size,
            )
            self.current_directory = directory
            self._populate_thumbnails()
            self.status_bar.showMessage(
                f"找到 {len(self.photos)} 张照片，{len(self.groups)} 个连拍组", 5000
            )
        except Exception as exc:  # pragma: no cover - UI feedback
            logger.exception("Failed to scan directory")
            QMessageBox.critical(self, "扫描失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _populate_thumbnails(self) -> None:
        self.thumbnail_list.clear()
        self._thumbnail_cache.clear()

        for group in self.groups:
            for photo in group.photos:
                item = QListWidgetItem()
                item.setText(f"组 {group.id} · {photo.path.name}")
                pixmap = self._load_thumbnail(photo)
                item.setIcon(QIcon(pixmap))
                item.setData(Qt.UserRole, photo)
                self.thumbnail_list.addItem(item)

        if not self.groups:
            for photo in self.photos:
                item = QListWidgetItem(photo.path.name)
                pixmap = self._load_thumbnail(photo)
                item.setIcon(QIcon(pixmap))
                item.setData(Qt.UserRole, photo)
                self.thumbnail_list.addItem(item)

    def _load_thumbnail(self, photo: Photo) -> QPixmap:
        cached = self._thumbnail_cache.get(photo.path)
        if cached:
            return cached

        try:
            image = load_image_for_hash(photo.path)
            pixmap = scale_pixmap(pil_to_qpixmap(image), THUMBNAIL_SIZE)
        except (FileNotFoundError, UnidentifiedImageError, RuntimeError) as exc:
            logger.warning("Unable to load thumbnail for %s: %s", photo.path, exc)
            pixmap = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
            pixmap.fill(Qt.darkGray)

        self._thumbnail_cache[photo.path] = pixmap
        return pixmap

    def _on_selection_changed(self) -> None:
        selected = self.thumbnail_list.selectedItems()
        if not selected:
            self.preview_label.setText("选择一张照片查看预览")
            self.preview_label.setPixmap(QPixmap())
            return

        item = selected[0]
        photo = item.data(Qt.UserRole)
        if not isinstance(photo, Photo):
            return
        self._show_preview(photo)

    def _show_preview(self, photo: Photo) -> None:
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

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._on_selection_changed()


class DummyPreview(QWidget):
    """Placeholder widget for potential future overlay/flipbook."""

    def __init__(self) -> None:
        super().__init__()
        label = QLabel("叠加预览占位")
        label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)
