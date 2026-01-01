from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import anyio
from PySide6.QtCore import QObject, Signal

from _count_time import timer
from core.grouping.streaming import StreamingBurstGrouper

if TYPE_CHECKING:
    from pathlib import Path

    from config.settings import Settings
    from core.transables.models import Group, Photo

logger = logging.getLogger(__name__)


class ScanWorker(QObject):
    """Background scanner using anyio threads to keep UI responsive."""

    finished = Signal(list, list)
    progress = Signal(list, list, bool)
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
        grouper = StreamingBurstGrouper(
            time_gap_seconds=2.0,
            hash_threshold=self.settings.hash_threshold,
            min_group_size=self.settings.min_group_size,
        )
        last_snapshot = None
        with timer("Scan Directory"):
            async for snapshot in grouper.iter_groups(
                self.directory,
                recursive=self.settings.scan_recursive,
            ):
                last_snapshot = snapshot
                self.progress.emit(snapshot.photos, snapshot.groups, snapshot.done)

        if last_snapshot is None:
            return [], []
        return last_snapshot.photos, last_snapshot.groups
