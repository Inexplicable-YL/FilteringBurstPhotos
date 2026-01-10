from __future__ import annotations

import asyncio
import contextlib
import inspect
import sys
import warnings
from collections.abc import Awaitable, Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import Context, ContextVar, copy_context
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    Self,
    TypedDict,
    TypeVar,
    cast,
)
from typing_extensions import override

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from concurrent.futures import Executor
    from uuid import UUID

    from core.transables.tracing import AsyncTransableCallback, TransableCallback
P = ParamSpec("P")
T = TypeVar("T")
Output = TypeVar("Output")


class TransableConfig(TypedDict, total=False):
    """Runtime hints for stream execution."""

    run_name: str
    tags: list[str]
    metadata: dict[str, Any]
    max_concurrency: int | None
    stream_buffer: int
    recursion_limit: int
    callbacks: (
        list[TransableCallback | AsyncTransableCallback]
        | AsyncCallbackManager
        | CallbackManager
        | None
    )
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


class BaseCallbackManager:
    def __init__(
        self,
        handlers: Sequence[TransableCallback | AsyncTransableCallback] | None = None,
        *,
        inheritable_handlers: Sequence[TransableCallback | AsyncTransableCallback]
        | None = None,
    ) -> None:
        self._handlers = list(handlers or [])
        if inheritable_handlers is None:
            self._inheritable_handlers = list(self._handlers)
        else:
            self._inheritable_handlers = list(inheritable_handlers)

    def add_handler(
        self,
        handler: TransableCallback | AsyncTransableCallback,
        *,
        inherit: bool = True,
    ) -> None:
        self._handlers.append(handler)
        if inherit:
            self._inheritable_handlers.append(handler)

    def append(self, handler: TransableCallback | AsyncTransableCallback) -> None:
        self.add_handler(handler, inherit=True)

    def extend(
        self,
        handlers: Sequence[TransableCallback | AsyncTransableCallback],
        *,
        inherit: bool = True,
    ) -> None:
        for handler in handlers:
            self.add_handler(handler, inherit=inherit)

    def get_child(self) -> Self:
        return self.__class__(
            handlers=list(self._inheritable_handlers),
            inheritable_handlers=list(self._inheritable_handlers),
        )

    def copy(self) -> Self:
        return self.__class__(
            handlers=list(self._handlers),
            inheritable_handlers=list(self._inheritable_handlers),
        )

    def merge(self, other: BaseCallbackManager) -> Self:
        return self.__class__(
            handlers=[*self._handlers, *other._handlers],
            inheritable_handlers=[
                *self._inheritable_handlers,
                *other._inheritable_handlers,
            ],
        )

    @staticmethod
    def _safe_call(func: Callable[..., Any], *args: Any) -> Any:
        with contextlib.suppress(Exception):
            return func(*args)
        return None

    @staticmethod
    def _call_handler(
        listener: Callable[..., Any],
        run: Any,
        config: TransableConfig,
    ) -> Any:
        try:
            params = inspect.signature(listener).parameters
        except (TypeError, ValueError):
            return BaseCallbackManager._safe_call(listener, run, config)
        if any(
            param.kind == inspect.Parameter.VAR_POSITIONAL for param in params.values()
        ):
            return BaseCallbackManager._safe_call(listener, run, config)
        positional_params = [
            param
            for param in params.values()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(positional_params) >= 2:
            return BaseCallbackManager._safe_call(listener, run, config)
        if len(positional_params) == 1:
            return BaseCallbackManager._safe_call(listener, run)
        return BaseCallbackManager._safe_call(listener)

    @staticmethod
    def _call_stream_handler(
        listener: Callable[..., Any],
        run: Any,
        chunk: Any,
        config: TransableConfig,
    ) -> Any:
        try:
            params = inspect.signature(listener).parameters
        except (TypeError, ValueError):
            return BaseCallbackManager._safe_call(listener, run, chunk, config)
        if any(
            param.kind == inspect.Parameter.VAR_POSITIONAL for param in params.values()
        ):
            return BaseCallbackManager._safe_call(listener, run, chunk, config)
        positional_params = [
            param
            for param in params.values()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(positional_params) >= 3:
            return BaseCallbackManager._safe_call(listener, run, chunk, config)
        if len(positional_params) == 2:
            return BaseCallbackManager._safe_call(listener, run, chunk)
        if len(positional_params) == 1:
            return BaseCallbackManager._safe_call(listener, chunk)
        return BaseCallbackManager._safe_call(listener)

    @property
    def handlers(self) -> list[TransableCallback | AsyncTransableCallback]:
        return list(self._handlers)

    @property
    def inheritable_handlers(self) -> list[TransableCallback | AsyncTransableCallback]:
        return list(self._inheritable_handlers)

    def __iter__(self) -> Iterator[TransableCallback | AsyncTransableCallback]:
        return iter(self._handlers)

    def __getitem__(self, index: int) -> TransableCallback | AsyncTransableCallback:
        return self._handlers[index]

    def __len__(self) -> int:
        return len(self._handlers)

    def __bool__(self) -> bool:
        return bool(self._handlers)


class CallbackManager(BaseCallbackManager):
    def on_start(self, run: Any, config: TransableConfig) -> None:
        self._notify("on_start", run, config)

    def on_end(self, run: Any, config: TransableConfig) -> None:
        self._notify("on_end", run, config)

    def on_error(self, run: Any, config: TransableConfig) -> None:
        self._notify("on_error", run, config)

    def on_stream_chunk(self, run: Any, chunk: Any, config: TransableConfig) -> None:
        self._notify_stream("on_stream_chunk", run, chunk, config)

    def _notify(self, method: str, run: Any, config: TransableConfig) -> None:
        if not self._handlers:
            return
        for callback in self._handlers:
            handler = getattr(callback, method, None)
            if handler is None:
                continue
            self._call_handler(handler, run, config)

    def _notify_stream(
        self,
        method: str,
        run: Any,
        chunk: Any | None,
        config: TransableConfig,
    ) -> None:
        if not self._handlers:
            return
        for callback in self._handlers:
            handler = getattr(callback, method, None)
            if handler is None:
                continue
            if chunk is None:
                self._call_handler(handler, run, config)
            else:
                self._call_stream_handler(handler, run, chunk, config)


class AsyncCallbackManager(BaseCallbackManager):
    async def on_start(self, run: Any, config: TransableConfig) -> None:
        await self._notify("on_start", run, config)

    async def on_end(self, run: Any, config: TransableConfig) -> None:
        await self._notify("on_end", run, config)

    async def on_error(self, run: Any, config: TransableConfig) -> None:
        await self._notify("on_error", run, config)

    async def on_stream_chunk(
        self,
        run: Any,
        chunk: Any,
        config: TransableConfig,
    ) -> None:
        await self._notify_stream("on_stream_chunk", run, chunk, config)

    async def _notify(self, method: str, run: Any, config: TransableConfig) -> None:
        if not self._handlers:
            return
        for callback in self._handlers:
            handler = getattr(callback, method, None)
            if handler is None:
                continue
            result = self._call_handler(handler, run, config)
            await self._maybe_await(result)

    async def _notify_stream(
        self,
        method: str,
        run: Any,
        chunk: Any | None,
        config: TransableConfig,
    ) -> None:
        if not self._handlers:
            return
        for callback in self._handlers:
            handler = getattr(callback, method, None)
            if handler is None:
                continue
            if chunk is None:
                result = self._call_handler(handler, run, config)
            else:
                result = self._call_stream_handler(handler, run, chunk, config)
            await self._maybe_await(result)

    @staticmethod
    async def _maybe_await(result: Any) -> None:
        if inspect.isawaitable(result):
            with contextlib.suppress(Exception):
                await result


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
    callbacks: list[Any] | AsyncCallbackManager | CallbackManager = Field(
        default_factory=list
    )
    trace: bool = False
    run_id: UUID | None = None


def _copy_callbacks(
    callbacks: list[TransableCallback | AsyncTransableCallback]
    | AsyncCallbackManager
    | CallbackManager,
) -> (
    list[TransableCallback | AsyncTransableCallback]
    | AsyncCallbackManager
    | CallbackManager
):
    if isinstance(callbacks, (AsyncCallbackManager, CallbackManager)):
        return callbacks.copy()
    return list(callbacks)


def _extend_manager(
    target: BaseCallbackManager,
    source: BaseCallbackManager,
) -> None:
    inheritable = list(source.inheritable_handlers)
    for handler in source.handlers:
        inherit = False
        if handler in inheritable:
            inherit = True
            inheritable.remove(handler)
        target.add_handler(handler, inherit=inherit)


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
                cast(
                    "list[TransableCallback | AsyncTransableCallback] | AsyncCallbackManager | CallbackManager",
                    value,
                )
            )
        else:
            copied[key] = value
    return copied


def _merge_callbacks(  # noqa: PLR0911
    base_callbacks: (
        list[TransableCallback | AsyncTransableCallback]
        | AsyncCallbackManager
        | CallbackManager
        | None
    ),
    new_callbacks: (
        list[TransableCallback | AsyncTransableCallback]
        | AsyncCallbackManager
        | CallbackManager
        | None
    ),
) -> (
    list[TransableCallback | AsyncTransableCallback]
    | AsyncCallbackManager
    | CallbackManager
    | None
):
    if new_callbacks is None:
        return base_callbacks
    if base_callbacks is None:
        return _copy_callbacks(new_callbacks)
    if isinstance(base_callbacks, AsyncCallbackManager):
        merged = base_callbacks.copy()
        if isinstance(new_callbacks, AsyncCallbackManager):
            return merged.merge(new_callbacks)
        if isinstance(new_callbacks, CallbackManager):
            _extend_manager(merged, new_callbacks)
            return merged
        merged.extend(new_callbacks, inherit=True)
        return merged
    if isinstance(base_callbacks, CallbackManager):
        merged = base_callbacks.copy()
        if isinstance(new_callbacks, CallbackManager):
            return merged.merge(new_callbacks)
        if isinstance(new_callbacks, AsyncCallbackManager):
            _extend_manager(merged, new_callbacks)
            return merged
        merged.extend(new_callbacks, inherit=True)
        return merged
    if isinstance(new_callbacks, AsyncCallbackManager):
        return AsyncCallbackManager(base_callbacks).merge(new_callbacks)
    if isinstance(new_callbacks, CallbackManager):
        return CallbackManager(base_callbacks).merge(new_callbacks)
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
    callbacks: (
        list[TransableCallback | AsyncTransableCallback]
        | AsyncCallbackManager
        | CallbackManager
        | None
    ) = None,
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
        if isinstance(callbacks, (AsyncCallbackManager, CallbackManager)):
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
                    cast(
                        "list[TransableCallback | AsyncTransableCallback] | AsyncCallbackManager | CallbackManager",
                        value,
                    ),
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


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that copies the context to the child thread."""

    @override
    def submit(
        self,
        func: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[T]:
        """Submit a function to the executor.

        Args:
            func: The function to submit.
            *args: The positional arguments to the function.
            **kwargs: The keyword arguments to the function.

        Returns:
            The future for the function.
        """
        return super().submit(
            cast("Callable[..., T]", partial(copy_context().run, func, *args, **kwargs))
        )

    @override
    def map(
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        **kwargs: Any,
    ) -> Iterator[T]:
        """Map a function to multiple iterables.

        Args:
            fn: The function to map.
            *iterables: The iterables to map over.
            timeout: The timeout for the map.
            chunksize: The chunksize for the map.

        Returns:
            The iterator for the mapped function.
        """
        contexts = [copy_context() for _ in range(len(iterables[0]))]  # type: ignore[arg-type]

        def _wrapped_fn(*args: Any) -> T:
            return contexts.pop().run(fn, *args)

        return super().map(
            _wrapped_fn,
            *iterables,
            **kwargs,
        )


@contextmanager
def get_executor_for_config(
    config: TransableConfig | None,
) -> Generator[Executor, None, None]:
    """Get an executor for a config.

    Args:
        config: The config.

    Yields:
        The executor.
    """
    config = config or {}
    with ContextThreadPoolExecutor(
        max_workers=config.get("max_concurrency")
    ) as executor:
        yield executor


async def run_in_executor(
    executor_or_config: Executor | TransableConfig | None,
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run a function in an executor.

    Args:
        executor_or_config: The executor or config to run in.
        func: The function.
        *args: The positional arguments to the function.
        **kwargs: The keyword arguments to the function.

    Returns:
        The output of the function.
    """

    def wrapper() -> T:
        try:
            return func(*args, **kwargs)
        except StopIteration as exc:
            # StopIteration can't be set on an asyncio.Future
            # it raises a TypeError and leaves the Future pending forever
            # so we need to convert it to a RuntimeError
            raise RuntimeError from exc

    if executor_or_config is None or isinstance(executor_or_config, dict):
        # Use default executor with context copied from current context
        return await asyncio.get_running_loop().run_in_executor(
            None,
            cast("Callable[..., T]", partial(copy_context().run, wrapper)),
        )

    return await asyncio.get_running_loop().run_in_executor(executor_or_config, wrapper)
