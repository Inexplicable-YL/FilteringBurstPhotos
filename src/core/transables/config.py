from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import Context, ContextVar, copy_context
from typing import (
    Any,
    ParamSpec,
    TypedDict,
    TypeVar,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field

P = ParamSpec("P")
Output = TypeVar("Output")


class TransableConfig(TypedDict, total=False):
    """Runtime hints for stream execution."""

    run_name: str
    tags: list[str]
    metadata: dict[str, Any]
    max_concurrency: int | None
    stream_buffer: int
    recursion_limit: int


CONFIG_KEYS = [
    "run_name",
    "tags",
    "metadata",
    "max_concurrency",
    "stream_buffer",
    "recursion_limit",
]

COPIABLE_KEYS = [
    "tags",
    "metadata",
]

DEFAULT_RECURSION_LIMIT = 25
DEFAULT_STREAM_BUFFER = 64

var_child_transable_config: ContextVar[TransableConfig | None] = ContextVar(
    "child_transable_config",
    default=None,
)


class TransableConfigModel(BaseModel):
    """Pydantic model for TransableConfig validation and schemas."""

    model_config = ConfigDict(extra="allow")

    run_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_concurrency: int | None = None
    stream_buffer: int = DEFAULT_STREAM_BUFFER
    recursion_limit: int = DEFAULT_RECURSION_LIMIT


def _copy_config_values(config: TransableConfig) -> TransableConfig:
    copied: TransableConfig = {}
    for key, value in config.items():
        if value is None:
            continue
        if key == "tags":
            copied[key] = list(cast("list[str]", value))
        elif key == "metadata":
            copied[key] = dict(cast("dict[str, Any]", value))
        else:
            copied[key] = value
    return copied


@contextmanager
def set_config_context(config: TransableConfig) -> Generator[Context, None, None]:
    """Set the child Transable config for nested calls."""
    ctx = copy_context()
    token = ctx.run(var_child_transable_config.set, config)
    try:
        yield ctx
    finally:
        ctx.run(var_child_transable_config.reset, token)


def ensure_config(config: TransableConfig | None = None) -> TransableConfig:
    """Ensure a config has defaults for Transable execution."""
    empty: TransableConfig = {
        "tags": [],
        "metadata": {},
        "recursion_limit": DEFAULT_RECURSION_LIMIT,
        "stream_buffer": DEFAULT_STREAM_BUFFER,
    }
    if var_config := var_child_transable_config.get():
        empty.update(_copy_config_values(var_config))
    if config is not None:
        empty.update(_copy_config_values(config))
    return empty


def get_config_list(
    config: TransableConfig | Sequence[TransableConfig] | None,
    length: int,
) -> list[TransableConfig]:
    """Get a list of configs from a single config or a list of configs."""
    if length < 0:
        raise ValueError(f"length must be >= 0, but got {length}")
    if isinstance(config, Sequence) and len(config) != length:
        msg = (
            "config must be a list of the same length as inputs, "
            f"but got {len(config)} configs for {length} inputs"
        )
        raise ValueError(msg)

    if isinstance(config, Sequence):
        return [ensure_config(item) for item in config]
    return [ensure_config(config) for _ in range(length)]


def patch_config(
    config: TransableConfig | None,
    *,
    run_name: str | None = None,
    max_concurrency: int | None = None,
    stream_buffer: int | None = None,
    recursion_limit: int | None = None,
) -> TransableConfig:
    """Patch a config with new values."""
    config = ensure_config(config)
    if run_name is not None:
        config["run_name"] = run_name
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    if stream_buffer is not None:
        config["stream_buffer"] = stream_buffer
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    return config


def merge_configs(*configs: TransableConfig | None) -> TransableConfig:
    """Merge multiple configs into one."""
    base: TransableConfig = {}
    for config in (ensure_config(c) for c in configs if c is not None):
        for key, value in config.items():
            if key == "metadata":
                base["metadata"] = {
                    **base.get("metadata", {}),
                    **(cast("dict[str, Any]", value) or {}),
                }
            elif key == "tags":
                base["tags"] = sorted(
                    set(base.get("tags", []) + (cast("list[str]", value) or []))
                )
            elif key == "recursion_limit":
                if value != DEFAULT_RECURSION_LIMIT:
                    base["recursion_limit"] = int(cast("int", value))
            elif key == "stream_buffer":
                if value != DEFAULT_STREAM_BUFFER:
                    base["stream_buffer"] = int(cast("int", value))
            else:
                base[key] = value
    return base


def run_in_context(
    config: TransableConfig,
    func: Callable[P, Output],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Output:
    """Run a callable with a normalized config bound to the context."""
    with set_config_context(config) as context:
        return context.run(func, *args, **kwargs)


async def arun_in_context(
    config: TransableConfig,
    func: Callable[P, Awaitable[Output]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Output:
    """Run an async callable with a normalized config bound to the context."""
    with set_config_context(config) as context:
        return await context.run(func, *args, **kwargs)
