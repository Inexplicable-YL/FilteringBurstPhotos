from __future__ import annotations

import contextlib
import inspect
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from core.transables.config import (
    TransableConfig,
    ensure_config,
    set_config_context,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class TransableRun:
    id: uuid.UUID
    root_id: uuid.UUID
    parent_id: uuid.UUID | None
    depth: int
    path: list[uuid.UUID]
    name: str
    run_type: str
    input: Any
    receive: Any | None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=_now)
    end_time: datetime | None = None
    output: Any | None = None
    error: BaseException | None = None
    stream_count: int = 0
    last_output: Any | None = None


class TransableCallback(Protocol):
    def on_start(self, run: TransableRun, config: TransableConfig) -> None: ...

    def on_end(self, run: TransableRun, config: TransableConfig) -> None: ...

    def on_error(self, run: TransableRun, config: TransableConfig) -> None: ...

    def on_stream_start(self, run: TransableRun, config: TransableConfig) -> None: ...

    def on_stream_chunk(
        self, run: TransableRun, chunk: Any, config: TransableConfig
    ) -> None: ...

    def on_stream_end(self, run: TransableRun, config: TransableConfig) -> None: ...

    def on_stream_error(self, run: TransableRun, config: TransableConfig) -> None: ...


var_current_run: ContextVar[TransableRun | None] = ContextVar(
    "current_transable_run",
    default=None,
)


@contextmanager
def set_run_context(run: TransableRun) -> Any:
    token = var_current_run.set(run)
    try:
        yield
    finally:
        var_current_run.reset(token)


def get_current_run() -> TransableRun | None:
    return var_current_run.get()


def _resolve_name(transable: object, config: TransableConfig) -> str:
    name = config.get("run_name")
    if name:
        return name
    get_name = getattr(transable, "get_name", None)
    if callable(get_name):
        return str(get_name())
    return type(transable).__name__


def _get_callbacks(config: TransableConfig) -> list[TransableCallback]:
    callbacks = config.get("callbacks") or []
    return list(callbacks)


def _notify(
    callbacks: list[TransableCallback],
    method: str,
    run: TransableRun,
    config: TransableConfig,
) -> None:
    for callback in callbacks:
        handler = getattr(callback, method, None)
        if handler is None:
            continue
        _call_handler(handler, run, config)


def _notify_stream(
    callbacks: list[TransableCallback],
    method: str,
    run: TransableRun,
    chunk: Any | None,
    config: TransableConfig,
) -> None:
    for callback in callbacks:
        handler = getattr(callback, method, None)
        if handler is None:
            continue
        if chunk is None:
            _call_handler(handler, run, config)
        else:
            _call_stream_handler(handler, run, chunk, config)


def _should_trace(config: TransableConfig) -> bool:
    return bool(config.get("trace")) or bool(config.get("callbacks"))


def _create_run(
    transable: object,
    config: TransableConfig,
    input_value: Any,
    receive_value: Any | None,
    run_type: str,
) -> TransableRun:
    parent = get_current_run()
    if parent is None:
        run_id = config.get("run_id") or uuid.uuid4()
        root_id = run_id
        parent_id = None
        depth = 0
        path = [run_id]
    else:
        run_id = uuid.uuid4()
        root_id = parent.root_id
        parent_id = parent.id
        depth = parent.depth + 1
        path = [*parent.path, run_id]
    return TransableRun(
        id=run_id,
        root_id=root_id,
        parent_id=parent_id,
        depth=depth,
        path=path,
        name=_resolve_name(transable, config),
        run_type=run_type,
        input=input_value,
        receive=receive_value,
        tags=list(config.get("tags") or []),
        metadata=dict(config.get("metadata") or {}),
    )


async def arun_with_tracing(
    transable: object,
    func: Callable[..., Any],
    input_value: Any,
    receive_value: Any | None,
    config: TransableConfig,
    run_type: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    config = ensure_config(config)
    if not _should_trace(config):
        with set_config_context(config) as context:
            return await context.run(func, *args, **kwargs)

    callbacks = _get_callbacks(config)
    run = _create_run(transable, config, input_value, receive_value, run_type)
    _notify(callbacks, "on_start", run, config)
    try:
        with set_run_context(run), set_config_context(config) as context:
            result = await context.run(func, *args, **kwargs)
    except BaseException as exc:
        run.error = exc
        run.end_time = _now()
        _notify(callbacks, "on_error", run, config)
        raise
    run.output = result
    run.end_time = _now()
    _notify(callbacks, "on_end", run, config)
    return result


async def aiter_with_tracing(
    transable: object,
    iterator: AsyncIterator[Any],
    input_value: Any,
    receive_value: Any | None,
    config: TransableConfig,
    run_type: str,
) -> AsyncIterator[Any]:
    config = ensure_config(config)
    if not _should_trace(config):
        with set_config_context(config):
            async for item in iterator:
                yield item
        return

    callbacks = _get_callbacks(config)
    run = _create_run(transable, config, input_value, receive_value, run_type)
    _notify(callbacks, "on_start", run, config)
    _notify_stream(callbacks, "on_stream_start", run, None, config)
    try:
        with set_run_context(run), set_config_context(config):
            async for item in iterator:
                run.stream_count += 1
                run.last_output = item
                _notify_stream(callbacks, "on_stream_chunk", run, item, config)
                yield item
    except BaseException as exc:
        run.error = exc
        run.end_time = _now()
        _notify_stream(callbacks, "on_stream_error", run, None, config)
        _notify(callbacks, "on_error", run, config)
        raise
    run.output = run.last_output
    run.end_time = _now()
    _notify_stream(callbacks, "on_stream_end", run, None, config)
    _notify(callbacks, "on_end", run, config)


class TransableListener:
    def __init__(
        self,
        *,
        on_start: Callable[[TransableRun, TransableConfig], None] | None = None,
        on_end: Callable[[TransableRun, TransableConfig], None] | None = None,
        on_error: Callable[[TransableRun, TransableConfig], None] | None = None,
        on_stream_start: Callable[[TransableRun, TransableConfig], None] | None = None,
        on_stream_chunk: Callable[[TransableRun, Any, TransableConfig], None]
        | None = None,
        on_stream_end: Callable[[TransableRun, TransableConfig], None] | None = None,
        on_stream_error: Callable[[TransableRun, TransableConfig], None] | None = None,
    ) -> None:
        self._on_start = on_start
        self._on_end = on_end
        self._on_error = on_error
        self._on_stream_start = on_stream_start
        self._on_stream_chunk = on_stream_chunk
        self._on_stream_end = on_stream_end
        self._on_stream_error = on_stream_error

    def on_start(self, run: TransableRun, config: TransableConfig) -> None:
        if self._on_start is None:
            return
        _call_handler(self._on_start, run, config)

    def on_end(self, run: TransableRun, config: TransableConfig) -> None:
        if self._on_end is None:
            return
        _call_handler(self._on_end, run, config)

    def on_error(self, run: TransableRun, config: TransableConfig) -> None:
        if self._on_error is None:
            return
        _call_handler(self._on_error, run, config)

    def on_stream_start(self, run: TransableRun, config: TransableConfig) -> None:
        if self._on_stream_start is None:
            return
        _call_handler(self._on_stream_start, run, config)

    def on_stream_chunk(
        self,
        run: TransableRun,
        chunk: Any,
        config: TransableConfig,
    ) -> None:
        if self._on_stream_chunk is None:
            return
        _call_stream_handler(self._on_stream_chunk, run, chunk, config)

    def on_stream_end(self, run: TransableRun, config: TransableConfig) -> None:
        if self._on_stream_end is None:
            return
        _call_handler(self._on_stream_end, run, config)

    def on_stream_error(self, run: TransableRun, config: TransableConfig) -> None:
        if self._on_stream_error is None:
            return
        _call_handler(self._on_stream_error, run, config)


def _call_handler(
    listener: Callable[..., None],
    run: TransableRun,
    config: TransableConfig,
) -> None:
    try:
        params = inspect.signature(listener).parameters
    except (TypeError, ValueError):
        with contextlib.suppress(Exception):
            listener(run, config)
        return
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params.values()):
        with contextlib.suppress(Exception):
            listener(run, config)
        return
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
        with contextlib.suppress(Exception):
            listener(run, config)
        return
    if len(positional_params) == 1:
        with contextlib.suppress(Exception):
            listener(run)
        return
    with contextlib.suppress(Exception):
        listener()


def _call_stream_handler(
    listener: Callable[..., None],
    run: TransableRun,
    chunk: Any,
    config: TransableConfig,
) -> None:
    try:
        params = inspect.signature(listener).parameters
    except (TypeError, ValueError):
        with contextlib.suppress(Exception):
            listener(run, chunk, config)
        return
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params.values()):
        with contextlib.suppress(Exception):
            listener(run, chunk, config)
        return
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
        with contextlib.suppress(Exception):
            listener(run, chunk, config)
        return
    if len(positional_params) == 2:
        with contextlib.suppress(Exception):
            listener(run, chunk)
        return
    if len(positional_params) == 1:
        with contextlib.suppress(Exception):
            listener(chunk)
        return
    with contextlib.suppress(Exception):
        listener()
