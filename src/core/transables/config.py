from __future__ import annotations

from typing import (
    Any,
    TypedDict,
    TypeVar,
)

Input = TypeVar("Input")


class TransableConfig(TypedDict, total=False):
    """Runtime hints for stream execution."""

    run_name: str
    tags: list[str]
    metadata: dict[str, Any]
    max_concurrency: int
    stream_buffer: int
