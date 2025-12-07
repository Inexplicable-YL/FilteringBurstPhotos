from __future__ import annotations

import logging
import os
from bisect import bisect_left
from datetime import datetime
from typing import TYPE_CHECKING

import anyio
from anyio.to_thread import run_sync
from PIL import ExifTags, Image

from .image_hash import compute_phash_async, hamming_distance
from .image_raw import RAW_EXTENSIONS, load_image_for_path_async
from .models import Group, GroupingResult, Photo

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from anyio.abc import ObjectSendStream


logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS: set[str] = {
    ext.lower() for ext in Image.registered_extensions()
} | {ext.lower() for ext in RAW_EXTENSIONS}

EXIF_DATETIME_KEYS = {k: v for k, v in ExifTags.TAGS.items() if v == "DateTime"}


def _collect_paths(directory: Path, recursive: bool) -> list[Path]:
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


def _resolve_taken_time(image: Image.Image, path: Path) -> datetime:
    try:
        exif = image.getexif()
        if exif:
            for key in EXIF_DATETIME_KEYS:
                value = exif.get(key)
                if value:
                    try:
                        return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")  # noqa: DTZ007
                    except ValueError:
                        logger.debug("Invalid EXIF date %s in %s", value, path)
    except Exception as exc:  # pragma: no cover - depends on file availability
        logger.debug("Failed to read EXIF from %s: %s", path, exc)

    stat = path.stat()
    timestamp = min(stat.st_mtime, stat.st_ctime)
    return datetime.fromtimestamp(timestamp)  # noqa: DTZ006


def _grouping_sort_key(photo: Photo) -> tuple[datetime, str]:
    return (photo.taken_time, photo.path.name)


async def _load_photo(
    path: Path, semaphore: anyio.Semaphore, ignore_errors: bool
) -> Photo | None:
    async with semaphore:
        try:
            image = await load_image_for_path_async(path)
            taken_time = await run_sync(_resolve_taken_time, image, path)
            hash_hex = await compute_phash_async(image)
        except Exception as exc:  # pragma: no cover - depends on file content
            if not ignore_errors:
                raise
            logger.warning("Skipping %s: %s", path, exc)
            return None

        return Photo(
            path=path,
            image=image,
            taken_time=taken_time,
            hash_hex=hash_hex,
            format=path.suffix.upper().lstrip("."),
        )


class TimeCluster:
    """Maintains photos that are close in capture time."""

    def __init__(self, photo: Photo) -> None:
        self.photos: list[Photo] = [photo]
        self.start = photo.taken_time
        self.end = photo.taken_time

    def add_photo(self, photo: Photo) -> None:
        self.photos.append(photo)
        self.photos.sort(key=_grouping_sort_key)
        self.start = min(self.start, photo.taken_time)
        self.end = max(self.end, photo.taken_time)

    def merge(self, other: TimeCluster) -> None:
        self.photos.extend(other.photos)
        self.photos.sort(key=_grouping_sort_key)
        self.start = min(self.start, other.start)
        self.end = max(self.end, other.end)


class StreamingBurstGrouper:
    """Stream photos while grouping incrementally by time and pHash."""

    def __init__(
        self,
        *,
        time_gap_seconds: float = 2.0,
        hash_threshold: int = 5,
        min_group_size: int = 2,
        batch_size: int = 32,
    ) -> None:
        self.time_gap_seconds = time_gap_seconds
        self.hash_threshold = hash_threshold
        self.min_group_size = min_group_size
        self.batch_size = batch_size
        self.photos: list[Photo] = []
        self._pending: list[Photo] = []
        self._time_clusters: list[TimeCluster] = []

    async def iter_groups(
        self,
        directory: Path,
        *,
        recursive: bool = True,
        ignore_errors: bool = True,
        max_concurrency: int | None = None,
    ) -> AsyncIterator[GroupingResult]:
        """Yield grouping snapshots as photos are loaded and hashed."""

        if not directory.exists():
            raise FileNotFoundError(directory)

        paths = _collect_paths(directory, recursive)
        if not paths:
            yield GroupingResult(photos=[], groups=[], done=True)
            return

        send, receive = anyio.create_memory_object_stream(self.batch_size * 4)
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

        final_snapshot = self._flush_pending()
        if final_snapshot:
            final_snapshot.done = True
            yield final_snapshot
            return
        yield self._snapshot(done=True)

    async def _produce_photos(
        self,
        paths: list[Path],
        send: ObjectSendStream[Photo],
        ignore_errors: bool,
        max_concurrency: int | None,
    ) -> None:
        concurrency = max_concurrency or min(16, (os.cpu_count() or 4) * 2)
        semaphore = anyio.Semaphore(concurrency)

        try:
            async with anyio.create_task_group() as tg:
                for path in paths:
                    tg.start_soon(
                        self._load_and_send,
                        path,
                        semaphore,
                        ignore_errors,
                        send,
                    )
        finally:
            await send.aclose()

    async def _load_and_send(
        self,
        path: Path,
        semaphore: anyio.Semaphore,
        ignore_errors: bool,
        send: ObjectSendStream[Photo],
    ) -> None:
        photo = await _load_photo(path, semaphore, ignore_errors)
        if photo is not None:
            await send.send(photo)

    def _flush_pending(self) -> GroupingResult | None:
        if not self._pending:
            return None

        batch = self._pending
        self._pending = []
        for photo in batch:
            self._add_photo(photo)

        return self._snapshot()

    def _snapshot(self, *, done: bool = False) -> GroupingResult:
        groups = self._build_groups()
        return GroupingResult(
            photos=sorted(self.photos, key=_grouping_sort_key),
            groups=groups,
            done=done,
        )

    def _add_photo(self, photo: Photo) -> None:
        self.photos.append(photo)
        self._insert_into_time_clusters(photo)

    def _insert_into_time_clusters(self, photo: Photo) -> None:
        starts = [cluster.start for cluster in self._time_clusters]
        idx = bisect_left(starts, photo.taken_time)
        left = self._time_clusters[idx - 1] if idx > 0 else None
        right = self._time_clusters[idx] if idx < len(self._time_clusters) else None

        fits_left = (
            left is not None
            and (photo.taken_time - left.end).total_seconds() <= self.time_gap_seconds
        )
        fits_right = (
            right is not None
            and (right.start - photo.taken_time).total_seconds()
            <= self.time_gap_seconds
        )

        if fits_left and fits_right:
            assert left is not None
            assert right is not None
            left.add_photo(photo)
            left.merge(right)  # Merge bridging photo into a single cluster.
            self._time_clusters.pop(idx)
            return

        if fits_left:
            assert left is not None
            left.add_photo(photo)
            return

        if fits_right:
            assert right is not None
            right.add_photo(photo)
            return

        self._time_clusters.insert(idx, TimeCluster(photo))

    def _cluster_by_hash(
        self, photos: list[Photo], threshold: int
    ) -> list[list[Photo]]:
        """Split photos into hash-similar buckets using an anchor-based scan."""

        buckets: list[tuple[str, list[Photo]]] = []
        for photo in photos:
            target = None
            for anchor_hash, bucket in buckets:
                if hamming_distance(anchor_hash, photo.hash_hex) <= threshold:
                    target = bucket
                    break

            if target is None:
                buckets.append((photo.hash_hex, [photo]))
            else:
                target.append(photo)

        return [bucket for _, bucket in buckets]

    def _build_groups(self) -> list[Group]:
        groups: list[Group] = []
        next_id = 1
        for time_cluster in self._time_clusters:
            for photos in self._cluster_by_hash(
                sorted(time_cluster.photos, key=_grouping_sort_key),
                self.hash_threshold,
            ):
                if len(photos) < self.min_group_size:
                    for photo in photos:
                        photo.group_id = None
                    continue

                group = Group(id=next_id, photos=list(photos))
                for photo in group.photos:
                    photo.group_id = group.id
                groups.append(group)
                next_id += 1

        return groups
