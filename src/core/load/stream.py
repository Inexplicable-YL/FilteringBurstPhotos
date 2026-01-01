from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing_extensions import override

import anyio

from core.load.utils import aload_photo, collect_paths
from core.transables.base import Transable, TransableConfig
from core.transables.models import Photo, PhotoResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

    from anyio.abc import ObjectReceiveStream, ObjectSendStream


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadInput:
    directory: Path
    recursive: bool = True
    ignore_errors: bool = True


def _grouping_sort_key(photo: Photo) -> tuple[datetime, str]:
    """Return a deterministic sort key for snapshot batches."""

    return (photo.taken_time, photo.path.name)


class StreamLoader(Transable[Photo, LoadInput]):
    """Stream photos while grouping incrementally by time and pHash."""

    def __init__(
        self,
        *,
        batch_size: int = 32,
        max_concurrency: int | None = None,
    ) -> None:
        super().__init__()

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size

        if max_concurrency is not None and max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self.max_concurrency = max_concurrency

        self._pending: list[Photo] = []
        self._paths: set[Path] = set()
        self._photos_by_path: dict[Path, Photo] = {}

    @override
    async def invoke(
        self,
        input: LoadInput,
        receive: PhotoResult[Photo] | None = None,
        config: TransableConfig | None = None,
        **kwargs: object,
    ) -> PhotoResult[Photo]:
        last: PhotoResult[Photo] | None = None
        async for snapshot in self.stream(input, None, config or {}, **kwargs):
            last = snapshot
        if last is None:
            return PhotoResult[Photo](photos=[], done=True)
        return last

    @override
    async def stream(
        self,
        input: LoadInput,
        receives: AsyncIterator[PhotoResult[Photo]] | None = None,
        config: TransableConfig | None = None,
        **kwargs: object,
    ) -> AsyncIterator[PhotoResult[Photo]]:
        """Yield grouping snapshots while photos are loaded and hashed."""
        if receives is not None:
            raise ValueError("StreamLoader does not accept upstream PhotoResult.")

        directory = input.directory
        if not directory.exists():
            raise FileNotFoundError(directory)
        if not directory.is_dir():
            raise NotADirectoryError(directory)

        paths = collect_paths(directory, recursive=input.recursive)
        paths = [p for p in paths if p not in self._paths]
        self._paths.update(paths)
        if not paths:
            yield PhotoResult[Photo](photos=[], done=True)
            return

        async for snapshot in self._iter(
            paths,
            ignore_errors=input.ignore_errors,
            max_concurrency=self._resolve_max_concurrency(config),
        ):
            for photo in snapshot.photos:
                self._photos_by_path[photo.path] = photo
            yield snapshot

    async def _iter(
        self,
        paths: list[Path],
        *,
        ignore_errors: bool = True,
        max_concurrency: int | None,
    ) -> AsyncIterator[PhotoResult[Photo]]:
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
                    max_concurrency,
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
                yield PhotoResult[Photo](photos=[], done=True)
        finally:
            self._pending = []

    async def _produce_photos(
        self,
        paths: list[Path],
        send: ObjectSendStream[Photo],
        ignore_errors: bool,
        max_concurrency: int | None,
    ) -> None:
        """Create workers that load photos with bounded concurrency.

        Args:
            paths: Candidate file paths to load.
            send: Stream to emit loaded photos.
            ignore_errors: Whether to skip files that fail to load.
        """
        concurrency = max_concurrency or min(16, (os.cpu_count() or 4) * 2)
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

    def _flush_pending(self) -> PhotoResult[Photo] | None:
        """Return a sorted snapshot of the pending batch."""

        if not self._pending:
            return None

        batch = self._pending
        self._pending = []

        return PhotoResult[Photo](
            photos=sorted(batch, key=_grouping_sort_key),
        )

    def _resolve_max_concurrency(self, config: TransableConfig | None) -> int | None:
        if config is None or "max_concurrency" not in config:
            return self.max_concurrency
        max_concurrency = config["max_concurrency"]
        if max_concurrency is None:
            return None
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        return max_concurrency
