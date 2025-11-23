from __future__ import annotations

from typing import TYPE_CHECKING

from .image_hash import hamming_distance
from .models import Group, Photo

if TYPE_CHECKING:
    from collections.abc import Sequence


def group_bursts(
    photos: Sequence[Photo],
    time_threshold_seconds: int = 3,
    hash_threshold: int = 5,
    min_group_size: int = 2,
) -> list[Group]:
    """Group photos into burst sequences using time and hash similarity.

    Photos are first sorted by capture time then filename to ensure stable
    ordering. A new group is started when either the capture time gap or the
    hash distance exceeds the configured thresholds. Groups smaller than
    ``min_group_size`` are returned as singletons so the UI can still display
    them consistently.
    """

    sorted_photos = sorted(photos, key=lambda p: (p.taken_time, p.path.name))
    groups: list[Group] = []
    current_group: list[Photo] = []
    group_id = 1

    def commit_group():
        nonlocal group_id
        if not current_group:
            return
        group = Group(id=group_id, photos=list(current_group))
        for photo in group.photos:
            photo.group_id = group.id
        groups.append(group)
        group_id += 1
        current_group.clear()

    for photo in sorted_photos:
        if not current_group:
            current_group.append(photo)
            continue

        prev = current_group[-1]
        time_gap = (photo.taken_time - prev.taken_time).total_seconds()
        distance = hamming_distance(photo.hash_hex, prev.hash_hex)

        if time_gap <= time_threshold_seconds and distance <= hash_threshold:
            current_group.append(photo)
        else:
            commit_group()
            current_group.append(photo)

    commit_group()

    # Reclassify small groups as singletons
    for group in groups:
        if group.size() < min_group_size:
            for photo in group.photos:
                photo.group_id = None

    return groups
