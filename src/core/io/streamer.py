from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import anyio

from core.io.loader import aload_photo, collect_paths
from core.models import Photo, PhotoResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime
    from pathlib import Path

    from anyio.abc import ObjectReceiveStream, ObjectSendStream


logger = logging.getLogger(__name__)


def _grouping_sort_key(photo: Photo) -> tuple[datetime, str]:
    """Return a deterministic sort key for snapshot batches."""

    return (photo.taken_time, photo.path.name)


class StreamLoader:
    """Stream photos while grouping incrementally by time and pHash."""

    def __init__(
        self,
        directory: Path,
        *,
        recursive: bool = True,
        batch_size: int = 32,
        max_concurrency: int | None = None,
    ) -> None:
        if not directory.exists():
            raise FileNotFoundError(directory)
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        self.directory = directory

        self.recursive = recursive

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size

        if max_concurrency is not None and max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self.max_concurrency = max_concurrency

        self._pending: list[Photo] = []
        self._paths: set[Path] = set()
        self._photos_by_path: dict[Path, Photo] = {}

    async def iter_photos(
        self,
        *,
        ignore_errors: bool = True,
    ) -> AsyncIterator[PhotoResult]:
        """Yield grouping snapshots while photos are loaded and hashed.

        Args:
            ignore_errors: Whether to continue when a photo fails to load.

        Yields:
            PhotoResult instances representing each processed batch.
        """
        paths = collect_paths(self.directory, recursive=self.recursive)
        paths = [p for p in paths if p not in self._paths]
        self._paths.update(paths)
        if not paths:
            yield PhotoResult(photos=[], done=True)
            return

        async for snapshot in self._iter_photos(paths, ignore_errors=ignore_errors):
            for photo in snapshot.photos:
                self._photos_by_path[photo.path] = photo
            yield snapshot

    async def _iter_photos(
        self,
        paths: list[Path],
        *,
        ignore_errors: bool = True,
    ) -> AsyncIterator[PhotoResult]:
        """Yield grouping snapshots while photos are loaded and hashed."""
        self._pending = []
        send, receive = anyio.create_memory_object_stream(self.batch_size * 4)
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    self._produce_photos,
                    paths,
                    send,
                    ignore_errors,
                )

                async with receive:
                    async for photo in receive:
                        self._pending.append(photo)
                        if len(self._pending) >= self.batch_size:
                            snapshot = self._flush_pending()
                            if snapshot:
                                yield snapshot

            if final_snapshot := self._flush_pending():
                final_snapshot.done = True
                yield final_snapshot
            else:
                yield PhotoResult(photos=[], done=True)
        finally:
            self._pending = []

    async def get_photos(self) -> list[Photo]:
        """Return all photos loaded from the directory."""

        async for _ in self.iter_photos():
            pass

        return sorted(self._photos_by_path.values(), key=_grouping_sort_key)

    async def _produce_photos(
        self,
        paths: list[Path],
        send: ObjectSendStream[Photo],
        ignore_errors: bool,
    ) -> None:
        """Create workers that load photos with bounded concurrency.

        Args:
            paths: Candidate file paths to load.
            send: Stream to emit loaded photos.
            ignore_errors: Whether to skip files that fail to load.
        """
        concurrency = self.max_concurrency or min(16, (os.cpu_count() or 4) * 2)
        semaphore = anyio.Semaphore(concurrency)
        path_send, path_receive = anyio.create_memory_object_stream(
            max(1, concurrency * 2)
        )

        try:
            async with anyio.create_task_group() as tg:
                for _ in range(concurrency):
                    tg.start_soon(
                        self._photo_worker,
                        path_receive,
                        semaphore,
                        ignore_errors,
                        send,
                    )
                async with path_send:
                    for path in paths:
                        await path_send.send(path)
        finally:
            await send.aclose()

    async def _photo_worker(
        self,
        receive: ObjectReceiveStream[Path],
        semaphore: anyio.Semaphore,
        ignore_errors: bool,
        send: ObjectSendStream[Photo],
    ) -> None:
        """Consume paths and load photos through the shared send stream.

        Args:
            receive: Stream supplying file paths.
            semaphore: Concurrency gate shared across workers.
            ignore_errors: Whether to suppress loader failures.
            send: Stream used to emit successfully loaded photos.
        """
        async with receive:
            async for path in receive:
                await self._load_and_send(
                    path,
                    semaphore,
                    ignore_errors,
                    send,
                )

    async def _load_and_send(
        self,
        path: Path,
        semaphore: anyio.Semaphore,
        ignore_errors: bool,
        send: ObjectSendStream[Photo],
    ) -> None:
        """Load a single photo and forward it if loading succeeded.

        Args:
            path: File path to process.
            semaphore: Semaphore guarding the async hash pipeline.
            ignore_errors: Whether to skip files that raise exceptions.
            send: Stream to emit the resulting Photo.
        """
        photo = await aload_photo(
            path, semaphore=semaphore, ignore_errors=ignore_errors
        )
        if photo is not None:
            await send.send(photo)

    def _flush_pending(self) -> PhotoResult | None:
        """Return a sorted snapshot of the pending batch."""

        if not self._pending:
            return None

        batch = self._pending
        self._pending = []

        return PhotoResult(
            photos=sorted(batch, key=_grouping_sort_key),
        )
