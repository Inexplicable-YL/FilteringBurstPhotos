from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast
from typing_extensions import override

from PySide6.QtCore import QCoreApplication, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QPixmap, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ui.constants import (
    ACCENT_COLOR,
    BORDER_COLOR,
    CANVAS_BG,
    INITIAL_THUMBNAIL_SIZE,
    MUTED_TEXT,
    PRIMARY_TEXT,
    SURFACE_BG,
    THUMBNAIL_MAX_SIZE,
    THUMBNAIL_MIN_SIZE,
)
from ui.pixmap_cache import PixmapCache
from ui.scan_worker import ScanWorker
from ui.widgets import (
    GroupWidget,
    LeftContainer,
    PhotoThumbnail,
    PreviewLabel,
    ThumbnailScrollArea,
)

if TYPE_CHECKING:
    from config.settings import Settings
    from core.base.models import Group, Photo

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """GUI for browsing and filtering burst photos."""

    def __init__(  # noqa: PLR0915
        self, settings: Settings, initial_directory: Path | None = None
    ) -> None:
        super().__init__()
        self.settings = settings
        self.current_directory: Path | None = initial_directory
        self.photos: list[Photo] = []
        self.groups: list[Group] = []

        self.setWindowTitle("Filtering Burst Photos")
        self.resize(1280, 860)

        self._pixmap_cache = PixmapCache()
        self.thumbnail_size = INITIAL_THUMBNAIL_SIZE
        self._selected_paths: set[Path] = set()
        self._last_preview_photo: Photo | None = None
        self._group_widgets: list[GroupWidget] = []
        self._scan_worker: ScanWorker | None = None
        self._streaming_updates = False
        self._photo_lookup: dict[Path, Photo] = {}
        self._background_timer = QTimer(self)
        self._background_timer.setSingleShot(True)
        self._background_timer.timeout.connect(self._background_load_step)
        self._snapshot_timer = QTimer(self)
        self._snapshot_timer.setSingleShot(True)
        self._snapshot_timer.timeout.connect(self._drain_pending_snapshot)
        self._pending_snapshot: tuple[list[Photo], list[Group], bool] | None = None
        self._thumb_resize_timer = QTimer(self)
        self._thumb_resize_timer.setSingleShot(True)
        self._thumb_resize_timer.timeout.connect(self._apply_pending_thumbnail_size)
        self._pending_thumbnail_size: int | None = None

        self.toolbar = self._build_toolbar()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.thumbnail_container = QWidget()
        self.thumbnail_container.setObjectName("thumbnailContainer")
        self.group_layout = QVBoxLayout()
        self.group_layout.setContentsMargins(14, 14, 14, 14)
        self.group_layout.setSpacing(16)
        self.thumbnail_container.setLayout(self.group_layout)

        self.thumbnail_scroll = ThumbnailScrollArea()
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_scroll.setWidget(self.thumbnail_container)
        self.thumbnail_scroll.viewport().installEventFilter(self)
        self.thumbnail_scroll.verticalScrollBar().valueChanged.connect(
            self._on_scrollbar_changed
        )
        self.thumbnail_scroll.horizontalScrollBar().valueChanged.connect(
            self._on_scrollbar_changed
        )

        self.cancel_button = QPushButton("取消")
        self.cancel_button.setObjectName("ghostButton")
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
        buttons_container.setObjectName("actionsBar")

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.thumbnail_scroll)
        left_layout.addWidget(buttons_container)
        left_container = LeftContainer()
        left_container.setLayout(left_layout)
        left_container.ctrl_scroll.connect(self._update_thumbnail_size)

        self.preview_label = PreviewLabel("选择一张照片查看预览")
        self.preview_label.setObjectName("previewArea")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self.preview_label.setScaledContents(False)
        self.preview_label.size_changed.connect(
            lambda _: self._update_preview(self._last_preview_photo)
        )

        splitter = QSplitter()
        splitter.addWidget(left_container)
        splitter.addWidget(self.preview_label)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)
        self._apply_theme()

        if initial_directory:
            self.load_directory(initial_directory)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: {CANVAS_BG};
                color: {PRIMARY_TEXT};
                font-family: 'Segoe UI Variable', 'Microsoft YaHei', 'Segoe UI', sans-serif;
            }}
            QLabel {{
                color: {PRIMARY_TEXT};
            }}
            QWidget#thumbnailContainer {{
                background: qlineargradient(
                    x1 0 y1 0, x2 0 y2 1,
                    stop: 0 #101625,
                    stop: 1 #0a0f18
                );
            }}
            QWidget#actionsBar {{
                background: rgba(12, 16, 24, 0.9);
                border-top: 1px solid {BORDER_COLOR};
                padding: 8px 12px;
            }}
            QLabel#previewArea {{
                background: qlineargradient(
                    x1 0 y1 0, x2 1 y2 1,
                    stop: 0 {SURFACE_BG},
                    stop: 1 #0f1523
                );
                color: {PRIMARY_TEXT};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
                padding: 18px;
            }}
            QToolBar {{
                background: {SURFACE_BG};
                border: none;
                padding: 8px 12px;
                spacing: 8px;
            }}
            QToolButton {{
                color: {PRIMARY_TEXT};
                padding: 6px 10px;
                border-radius: 8px;
            }}
            QToolButton:hover {{
                background: {BORDER_COLOR};
            }}
            QPushButton {{
                background: {ACCENT_COLOR};
                color: #0d1117;
                border-radius: 10px;
                padding: 10px 16px;
                font-weight: 600;
                border: none;
            }}
            QPushButton:hover:!disabled {{
                background: #55d8b3;
            }}
            QPushButton:pressed {{
                background: #34c499;
            }}
            QPushButton:disabled {{
                background: {BORDER_COLOR};
                color: {MUTED_TEXT};
            }}
            QPushButton#ghostButton {{
                background: transparent;
                color: {PRIMARY_TEXT};
                border: 1px solid {BORDER_COLOR};
            }}
            QPushButton#ghostButton:hover:!disabled {{
                border-color: {ACCENT_COLOR};
                color: {ACCENT_COLOR};
            }}
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QStatusBar {{
                background: {SURFACE_BG};
                color: {MUTED_TEXT};
                border-top: 1px solid {BORDER_COLOR};
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 12px;
                margin: 6px 0 6px 0;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_COLOR};
                border-radius: 6px;
                min-height: 24px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {ACCENT_COLOR};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QLabel.photoName {{
                color: {PRIMARY_TEXT};
            }}
            QLabel.photoMeta {{
                color: {MUTED_TEXT};
            }}
            """
        )

    def _build_toolbar(self) -> QToolBar:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        open_action = QAction(QIcon.fromTheme("folder-open"), "打开目录", self)
        open_action.triggered.connect(self._pick_directory)
        toolbar.addAction(open_action)

        rescan_action = QAction("重新扫描", self)
        rescan_action.triggered.connect(self._rescan)
        toolbar.addAction(rescan_action)

        toolbar.addSeparator()

        quit_action = QAction("退出", self)
        quit_action.triggered.connect(
            cast("QCoreApplication", QApplication.instance()).quit
        )
        toolbar.addAction(quit_action)

        self.addToolBar(toolbar)
        return toolbar

    def _pick_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择照片目录")
        if selected:
            self.load_directory(Path(selected))

    def _rescan(self) -> None:
        if not self.current_directory:
            QMessageBox.information(self, "提示", "请先选择一个目录")
            return
        self.load_directory(self.current_directory)

    def load_directory(self, directory: Path) -> None:
        if not directory.exists():
            QMessageBox.warning(self, "未找到目录", f"目录不存在: {directory}")
            return

        if self._scan_worker is not None:
            self.status_bar.showMessage("正在处理上一个请求，请稍候", 3000)
            return

        self.status_bar.showMessage(f"扫描: {directory}")
        self._streaming_updates = False
        self._pending_snapshot = None
        self._snapshot_timer.stop()
        self._background_timer.stop()
        self._selected_paths.clear()
        self._last_preview_photo = None
        self._pixmap_cache.clear_all()
        self._reset_preview()
        self._set_action_buttons_enabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._scan_worker = ScanWorker(directory, self.settings)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.failed.connect(self._on_scan_failed)
        self._scan_worker.start()

    def _queue_snapshot(
        self,
        photos: list[Photo],
        groups: list[Group],
        done: bool,
        *,
        force_now: bool = False,
    ) -> None:
        self._pending_snapshot = (photos, groups, done)
        self._snapshot_timer.stop()
        self._snapshot_timer.start(0 if force_now else 90)

    def _drain_pending_snapshot(self) -> None:
        if not self._pending_snapshot:
            return
        photos, groups, done = self._pending_snapshot
        self._pending_snapshot = None
        self._apply_snapshot(photos, groups, reset_selection=False, done=done)

    def _on_scan_finished(self, photos: list[Photo], groups: list[Group]) -> None:
        QApplication.restoreOverrideCursor()
        self.current_directory = (
            self._scan_worker.directory if self._scan_worker else self.current_directory
        )
        self._scan_worker = None
        if not self._streaming_updates:
            self._apply_snapshot(photos, groups, reset_selection=True, done=True)
        else:
            self._queue_snapshot(photos, groups, True, force_now=True)
            self.status_bar.showMessage(
                f"找到 {len(photos)} 张照片，{len(groups)} 个连拍组", 5000
            )

    def _on_scan_failed(self, error: str) -> None:
        QApplication.restoreOverrideCursor()
        self._scan_worker = None
        self._pending_snapshot = None
        self._snapshot_timer.stop()
        logger.error("Scan failed: %s", error)
        QMessageBox.critical(self, "扫描失败", error)

    def _on_scan_progress(
        self, photos: list[Photo], groups: list[Group], done: bool
    ) -> None:
        self._streaming_updates = True
        self._queue_snapshot(photos, groups, done)

    def _apply_snapshot(
        self,
        photos: list[Photo],
        groups: list[Group],
        *,
        reset_selection: bool,
        done: bool,
    ) -> None:
        self.photos = photos
        self.groups = groups
        self._photo_lookup = {photo.path: photo for photo in photos}

        available_paths = {photo.path for photo in photos}
        if reset_selection:
            self._selected_paths.clear()
            self._last_preview_photo = None
        else:
            self._selected_paths.intersection_update(available_paths)
            if self._last_preview_photo and (
                self._last_preview_photo.path not in available_paths
            ):
                self._last_preview_photo = None

        self._populate_groups(reset_selection=reset_selection)

        if self._last_preview_photo:
            self._update_preview(self._last_preview_photo)
        elif reset_selection or not self.groups:
            self._reset_preview()

        self._set_action_buttons_enabled(bool(self._selected_paths))
        message = f"已加载 {len(self.photos)} 张照片，{len(self.groups)} 个连拍组"
        self.status_bar.showMessage(message, 4000 if done else 1500)

    def _populate_groups(self, *, reset_selection: bool = True) -> None:
        self._clear_layout(self.group_layout)
        self._group_widgets.clear()
        if reset_selection:
            self._selected_paths.clear()

        if not self.groups:
            placeholder = QLabel("未找到连拍组")
            placeholder.setStyleSheet(
                f"color: {MUTED_TEXT}; background: transparent; padding: 18px;"
                f"border: 1px dashed {BORDER_COLOR}; border-radius: 12px;"
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.group_layout.addWidget(placeholder)
            return

        for group in self.groups:
            widget = GroupWidget(
                group=group,
                thumbnail_size=self.thumbnail_size,
                thumbnail_loader=self._load_thumbnail,
                on_photo_clicked=self._on_photo_clicked,
            )
            for thumb in widget.thumbnails:
                if thumb.photo.path in self._selected_paths:
                    thumb.set_selected(True)
            self._group_widgets.append(widget)
            self.group_layout.addWidget(widget)

        self.group_layout.addStretch()
        self._load_visible_thumbnails()
        self._schedule_background_loading(240)

    def _get_base_pixmap(self, photo: Photo) -> QPixmap:
        return self._pixmap_cache.get_base_pixmap(photo)

    def _load_thumbnail(self, photo: Photo) -> QPixmap:
        return self._pixmap_cache.load_thumbnail(photo, self.thumbnail_size)

    def _on_photo_clicked(self, photo: Photo, selected: bool) -> None:
        if selected:
            self._selected_paths.add(photo.path)
            self._last_preview_photo = photo
        else:
            self._selected_paths.discard(photo.path)
            if self._selected_paths:
                next_path = next(iter(self._selected_paths))
                self._last_preview_photo = self._photo_lookup.get(next_path)
            else:
                self._last_preview_photo = None

        self._update_preview(self._last_preview_photo)
        self._set_action_buttons_enabled(bool(self._selected_paths))

    def _update_preview(self, photo: Photo | None) -> None:
        if not photo:
            self._reset_preview()
            return

        try:
            target_size = self.preview_label.contentsRect().size()
            if target_size.width() <= 0 or target_size.height() <= 0:
                return
            scaled = self._get_base_pixmap(photo).scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.preview_label.setPixmap(scaled)
            self.preview_label.setText("")
            self.status_bar.showMessage(f"预览: {photo.path}")
        except Exception as exc:  # pragma: no cover - UI feedback
            logger.exception("Failed to load preview")
            QMessageBox.warning(self, "预览失败", str(exc))

    def _reset_preview(self) -> None:
        self.preview_label.setText("选择一张照片查看预览")
        self.preview_label.setPixmap(QPixmap())
        self._last_preview_photo = None

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        self.cancel_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)

    def _cancel_selection(self) -> None:
        if not self._selected_paths:
            return
        self._selected_paths.clear()
        for widget in self._group_widgets:
            widget.clear_selection()
        self._set_action_buttons_enabled(False)
        self._reset_preview()

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
        self._photo_lookup = {photo.path: photo for photo in kept_photos}
        self._selected_paths.clear()
        self._populate_groups(reset_selection=True)
        self._reset_preview()
        self._set_action_buttons_enabled(False)
        self.status_bar.showMessage("已保存选择，未选中的照片已移除显示", 4000)

    def _clear_layout(self, layout: QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if not item:
                continue
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()
            child_layout = item.layout()
            if child_layout:
                self._clear_layout(child_layout)

    def _update_thumbnail_size(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y() / 240
        if delta:
            new_size = int(
                max(
                    THUMBNAIL_MIN_SIZE,
                    min(THUMBNAIL_MAX_SIZE, self.thumbnail_size + delta * 12),
                )
            )
            if new_size == self.thumbnail_size:
                return
            self._pending_thumbnail_size = new_size
            self._thumb_resize_timer.start(60)

    def _apply_pending_thumbnail_size(self) -> None:
        if self._pending_thumbnail_size is None:
            return
        self.thumbnail_size = self._pending_thumbnail_size
        self._pending_thumbnail_size = None
        self._refresh_thumbnail_sizes()

    def _refresh_thumbnail_sizes(self) -> None:
        self._pixmap_cache.clear_thumbnails()
        for widget in self._group_widgets:
            widget.update_thumbnail_size(self.thumbnail_size)
        self._load_visible_thumbnails()
        self._schedule_background_loading(240)

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._last_preview_photo:
            self._update_preview(self._last_preview_photo)
        self._load_visible_thumbnails()
        self._schedule_background_loading(240)

    def _on_scrollbar_changed(self) -> None:
        self._load_visible_thumbnails()
        self._schedule_background_loading(240)

    def _visible_content_rect(self) -> QRect:
        viewport = self.thumbnail_scroll.viewport()
        return QRect(
            self.thumbnail_scroll.horizontalScrollBar().value(),
            self.thumbnail_scroll.verticalScrollBar().value(),
            viewport.width(),
            viewport.height(),
        )

    def _load_visible_thumbnails(self) -> None:
        visible = self._visible_content_rect()
        for widget in self._group_widgets:
            group_rect = widget.geometry()
            if not group_rect.intersects(visible):
                continue
            widget.ensure_visible_thumbnails(visible.translated(-group_rect.topLeft()))
        self._schedule_background_loading(240)

    def _distance_to_visible(self, rect: QRect, visible: QRect) -> int:
        if rect.intersects(visible):
            return 0
        dx = max(visible.left() - rect.right(), rect.left() - visible.right(), 0)
        dy = max(visible.top() - rect.bottom(), rect.top() - visible.bottom(), 0)
        return dx + dy

    def _schedule_background_loading(self, delay_ms: int = 80) -> None:
        self._background_timer.stop()
        self._background_timer.start(delay_ms)

    def _background_load_step(self) -> None:
        visible = self._visible_content_rect()
        candidates: list[tuple[int, int, PhotoThumbnail]] = []
        for widget in self._group_widgets:
            offset = widget.pos()
            for thumb in widget.thumbnails:
                if not thumb.should_load():
                    continue
                rect = thumb.geometry().translated(offset)
                distance = self._distance_to_visible(rect, visible)
                candidates.append((distance, rect.top(), thumb))

        if not candidates:
            return

        candidates.sort(key=lambda item: (item[0], item[1]))
        batch_size = 3
        for _, _, thumb in candidates[:batch_size]:
            thumb.ensure_loaded()

        if len(candidates) > batch_size:
            self._schedule_background_loading(40)
