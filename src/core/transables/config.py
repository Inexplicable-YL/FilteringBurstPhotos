from __future__ import annotations

import asyncio
import sys
import warnings
from collections.abc import Awaitable, Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from contextvars import Context, ContextVar, copy_context
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    TypedDict,
    TypeVar,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from uuid import UUID

    from core.transables.tracing import TransableCallback

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
    callbacks: list[TransableCallback] | TransableCallbackManager | None
    trace: bool
    run_id: UUID | None


CONFIG_KEYS = [
    "run_name",
    "tags",
    "metadata",
    "max_concurrency",
    "stream_buffer",
    "recursion_limit",
    "callbacks",
    "trace",
    "run_id",
]

COPIABLE_KEYS = [
    "tags",
    "metadata",
    "callbacks",
]

DEFAULT_RECURSION_LIMIT = 25
DEFAULT_STREAM_BUFFER = 64
_METADATA_SKIP_KEYS = {"api_key"}
_METADATA_VALUE_TYPES = (str, int, float, bool)

var_child_transable_config: ContextVar[TransableConfig | None] = ContextVar(
    "child_transable_config",
    default=None,
)


class TransableCallbackManager:
    def __init__(
        self,
        handlers: Sequence[TransableCallback] | None = None,
        *,
        inheritable_handlers: Sequence[TransableCallback] | None = None,
    ) -> None:
        self._handlers = list(handlers or [])
        if inheritable_handlers is None:
            self._inheritable_handlers = list(self._handlers)
        else:
            self._inheritable_handlers = list(inheritable_handlers)

    def add_handler(self, handler: TransableCallback, *, inherit: bool = True) -> None:
        self._handlers.append(handler)
        if inherit:
            self._inheritable_handlers.append(handler)

    def append(self, handler: TransableCallback) -> None:
        self.add_handler(handler, inherit=True)

    def extend(
        self,
        handlers: Sequence[TransableCallback],
        *,
        inherit: bool = True,
    ) -> None:
        for handler in handlers:
            self.add_handler(handler, inherit=inherit)

    def get_child(self) -> TransableCallbackManager:
        return TransableCallbackManager(
            handlers=list(self._inheritable_handlers),
            inheritable_handlers=list(self._inheritable_handlers),
        )

    def copy(self) -> TransableCallbackManager:
        return TransableCallbackManager(
            handlers=list(self._handlers),
            inheritable_handlers=list(self._inheritable_handlers),
        )

    def merge(self, other: TransableCallbackManager) -> TransableCallbackManager:
        return TransableCallbackManager(
            handlers=[*self._handlers, *other._handlers],
            inheritable_handlers=[
                *self._inheritable_handlers,
                *other._inheritable_handlers,
            ],
        )

    @property
    def handlers(self) -> list[TransableCallback]:
        return list(self._handlers)

    @property
    def inheritable_handlers(self) -> list[TransableCallback]:
        return list(self._inheritable_handlers)

    def __iter__(self) -> Iterator[TransableCallback]:
        return iter(self._handlers)

    def __getitem__(self, index: int) -> TransableCallback:
        return self._handlers[index]

    def __len__(self) -> int:
        return len(self._handlers)

    def __bool__(self) -> bool:
        return bool(self._handlers)


def get_current_config() -> TransableConfig | None:
    """Return the current config from the context if set."""
    return var_child_transable_config.get()


class TransableConfigModel(BaseModel):
    """Pydantic model for TransableConfig validation and schemas."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    run_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_concurrency: int | None = None
    stream_buffer: int = DEFAULT_STREAM_BUFFER
    recursion_limit: int = DEFAULT_RECURSION_LIMIT
    callbacks: list[Any] | TransableCallbackManager = Field(default_factory=list)
    trace: bool = False
    run_id: UUID | None = None


def _copy_callbacks(
    callbacks: list[TransableCallback] | TransableCallbackManager,
) -> list[TransableCallback] | TransableCallbackManager:
    if isinstance(callbacks, TransableCallbackManager):
        return callbacks.copy()
    return list(callbacks)


def _copy_config_values(config: TransableConfig) -> TransableConfig:
    copied: TransableConfig = {}
    for key, value in config.items():
        if value is None:
            continue
        if key == "tags":
            copied[key] = list(cast("list[str]", value))
        elif key == "metadata":
            copied[key] = dict(cast("dict[str, Any]", value))
        elif key == "callbacks":
            copied[key] = _copy_callbacks(
                cast("list[TransableCallback] | TransableCallbackManager", value)
            )
        else:
            copied[key] = value
    return copied


def _merge_callbacks(
    base_callbacks: list[TransableCallback] | TransableCallbackManager | None,
    new_callbacks: list[TransableCallback] | TransableCallbackManager | None,
) -> list[TransableCallback] | TransableCallbackManager | None:
    if new_callbacks is None:
        return base_callbacks
    if base_callbacks is None:
        return _copy_callbacks(new_callbacks)
    if isinstance(base_callbacks, TransableCallbackManager):
        merged = base_callbacks.copy()
        if isinstance(new_callbacks, TransableCallbackManager):
            return merged.merge(new_callbacks)
        merged.extend(new_callbacks, inherit=True)
        return merged
    if isinstance(new_callbacks, TransableCallbackManager):
        return TransableCallbackManager(base_callbacks).merge(new_callbacks)
    return list(base_callbacks) + list(new_callbacks)


def _enrich_metadata(config: TransableConfig) -> None:
    metadata = dict(config.get("metadata") or {})
    for key, value in config.items():
        if key in CONFIG_KEYS or key in metadata:
            continue
        if key.startswith("__") or key in _METADATA_SKIP_KEYS:
            continue
        if isinstance(value, _METADATA_VALUE_TYPES):
            metadata[key] = value
    config["metadata"] = metadata


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
    empty = TransableConfig(
        tags=[],
        metadata={},
        recursion_limit=DEFAULT_RECURSION_LIMIT,
        stream_buffer=DEFAULT_STREAM_BUFFER,
        callbacks=[],
        trace=False,
        run_id=None,
    )
    if var_config := var_child_transable_config.get():
        empty.update(_copy_config_values(var_config))
    if config is not None:
        empty.update(_copy_config_values(config))
    _enrich_metadata(empty)
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
    if length > 1 and isinstance(config, dict) and config.get("run_id") is not None:
        warnings.warn(
            "Provided run_id will be used only for the first element of the batch.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        subsequent = cast(
            "TransableConfig", {k: v for k, v in config.items() if k != "run_id"}
        )
        return [
            ensure_config(subsequent) if i else ensure_config(config)
            for i in range(length)
        ]
    return [ensure_config(config) for _ in range(length)]


def patch_config(
    config: TransableConfig | None,
    *,
    run_name: str | None = None,
    max_concurrency: int | None = None,
    stream_buffer: int | None = None,
    recursion_limit: int | None = None,
    callbacks: list[TransableCallback] | TransableCallbackManager | None = None,
    trace: bool | None = None,
    run_id: UUID | None = None,
    child: bool = False,
) -> TransableConfig:
    """Patch a config with new values.

    When child is True, callbacks inherit and run identifiers are cleared.
    """
    config = ensure_config(config)
    if child and callbacks is None:
        callbacks = config.get("callbacks")
        if isinstance(callbacks, TransableCallbackManager):
            callbacks = callbacks.get_child()
        elif callbacks is not None:
            callbacks = _copy_callbacks(callbacks)
    if run_name is not None:
        config["run_name"] = run_name
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    if stream_buffer is not None:
        config["stream_buffer"] = stream_buffer
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    if callbacks is not None:
        config["callbacks"] = callbacks
        config.pop("run_name", None)
        config.pop("run_id", None)
    if trace is not None:
        config["trace"] = trace
    if run_id is not None:
        config["run_id"] = run_id
    if child:
        config.pop("run_name", None)
        config.pop("run_id", None)
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
            elif key == "callbacks":
                base["callbacks"] = _merge_callbacks(
                    base.get("callbacks"),
                    cast("list[TransableCallback] | TransableCallbackManager", value),
                )
            elif key == "trace":
                base["trace"] = bool(value) or bool(base.get("trace", False))
            elif key == "run_id":
                if value is not None:
                    base["run_id"] = cast("UUID", value)
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
        coro = context.run(func, *args, **kwargs)
        return await coro_with_context(coro, context, create_task=True)


def asyncio_accepts_context() -> bool:
    """Return True if asyncio.create_task accepts a context argument."""
    return sys.version_info >= (3, 11)


def coro_with_context(
    coro: Awaitable[Output],
    context: Context,
    *,
    create_task: bool = False,
) -> Awaitable[Output]:
    """Await a coroutine with an explicit context."""
    if asyncio_accepts_context():
        return asyncio.create_task(coro, context=context)  # type: ignore[arg-type,call-arg,unused-ignore]
    if create_task:
        return context.run(asyncio.create_task, coro)  # type: ignore[arg-type,call-arg,unused-ignore]
    return coro
