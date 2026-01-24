from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from core.transables.config import (
    AsyncCallbackManager,
    CallbackManager,
    TransableConfig,
    coro_with_context,
    ensure_config,
    set_config_context,
)
from core.transables.utils import (
    Input,
    Output,
    PhotoType,
    acall_func_with_variable_args,
    accepts_any,
    call_func_with_variable_args,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence


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
    run_type: str | None
    input_value: Any
    receive_value: Any | None
    output: Any | None = None
    count: int = 0

    start_time: datetime = field(default_factory=_now)
    end_time: datetime | None = None

    errors: list[BaseException] | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TransableCallback(Protocol):
    def on_start(self, run: TransableRun, config: TransableConfig) -> None: ...

    def on_end(self, run: TransableRun, config: TransableConfig) -> None: ...

    def on_error(self, run: TransableRun, config: TransableConfig) -> None: ...

    def on_stream_chunk(
        self, run: TransableRun, chunk: Any, config: TransableConfig
    ) -> None: ...


class AsyncTransableCallback(Protocol):
    async def on_start(self, run: TransableRun, config: TransableConfig) -> None: ...

    async def on_end(self, run: TransableRun, config: TransableConfig) -> None: ...

    async def on_error(self, run: TransableRun, config: TransableConfig) -> None: ...

    async def on_stream_chunk(
        self, run: TransableRun, chunk: Any, config: TransableConfig
    ) -> None: ...


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


def get_callback_manager(config: TransableConfig) -> CallbackManager:
    callbacks = config.get("callbacks")
    if isinstance(callbacks, CallbackManager):
        return callbacks
    if isinstance(callbacks, AsyncCallbackManager):
        return CallbackManager(
            handlers=callbacks.handlers,
            inheritable_handlers=callbacks.inheritable_handlers,
        )
    if callbacks is None:
        return CallbackManager()
    return CallbackManager(callbacks)


def get_async_callback_manager(config: TransableConfig) -> AsyncCallbackManager:
    callbacks = config.get("callbacks")
    if isinstance(callbacks, AsyncCallbackManager):
        return callbacks
    if isinstance(callbacks, CallbackManager):
        return AsyncCallbackManager(
            handlers=callbacks.handlers,
            inheritable_handlers=callbacks.inheritable_handlers,
        )
    if callbacks is None:
        return AsyncCallbackManager()
    return AsyncCallbackManager(callbacks)


def _should_trace(config: TransableConfig) -> bool:
    return bool(config.get("trace")) or bool(config.get("callbacks"))


def create_run(
    transable: object,
    config: TransableConfig,
    input_value: Any,
    receive_value: Any | None,
    run_type: str | None = None,
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
        input_value=input_value,
        receive_value=receive_value,
        tags=list(config.get("tags") or []),
        metadata=dict(config.get("metadata") or {}),
    )


def run_with_tracing(
    transable: object,
    func: Callable[[Input], Output]
    | Callable[[Input, TransableConfig], Output]
    | Callable[[Input, PhotoType, TransableConfig], Output]
    | Callable[[Input, Iterator[PhotoType]], Output]
    | Callable[[Input, Iterator[PhotoType], TransableConfig], Output],
    input_value: Any,
    receive_value: Any | None,
    config: TransableConfig,
    *,
    run_type: str | None = None,
    **kwargs: Any,
) -> Output:
    config = ensure_config(config)
    if not _should_trace(config):
        with set_config_context(config) as context:
            return context.run(
                call_func_with_variable_args,
                func,
                input_value,
                receive_value,
                config,
                **kwargs,
            )

    callback_manager = get_callback_manager(config)
    run = create_run(transable, config, input_value, receive_value, run_type)
    callback_manager.on_start(run, config)
    try:
        with set_run_context(run), set_config_context(config) as context:
            result = context.run(
                call_func_with_variable_args,
                func,
                input_value,
                receive_value,
                config,
                **kwargs,
            )
    except BaseException as exc:
        run.errors = [exc]
        run.end_time = _now()
        callback_manager.on_error(run, config)
        raise
    run.output = result
    run.count = 1
    run.end_time = _now()
    callback_manager.on_end(run, config)
    return result


async def arun_with_tracing(
    transable: object,
    func: Callable[[Input], Awaitable[Output]]
    | Callable[[Input, TransableConfig], Awaitable[Output]]
    | Callable[[Input, PhotoType, TransableConfig], Awaitable[Output]]
    | Callable[[Input, AsyncIterator[PhotoType]], Awaitable[Output]]
    | Callable[[Input, AsyncIterator[PhotoType], TransableConfig], Awaitable[Output]],
    input_value: Any,
    receive_value: Any | None,
    config: TransableConfig,
    *,
    run_type: str | None = None,
    **kwargs: Any,
) -> Output:
    config = ensure_config(config)
    if not _should_trace(config):
        with set_config_context(config) as context:
            coro = context.run(
                acall_func_with_variable_args,
                func,
                input_value,
                receive_value,
                config,
                **kwargs,
            )
            return await coro_with_context(coro, context, create_task=True)

    callback_manager = get_async_callback_manager(config)
    run = create_run(transable, config, input_value, receive_value, run_type)
    await callback_manager.on_start(run, config)
    try:
        with set_run_context(run), set_config_context(config) as context:
            coro = context.run(
                acall_func_with_variable_args,
                func,
                input_value,
                receive_value,
                config,
                **kwargs,
            )
            result = await coro_with_context(coro, context, create_task=True)
    except BaseException as exc:
        run.errors = [exc]
        run.end_time = _now()
        await callback_manager.on_error(run, config)
        raise
    run.output = result
    run.count = 1
    run.end_time = _now()
    await callback_manager.on_end(run, config)
    return result


def iter_with_tracing(
    transable: object,
    func: Callable[[Input], Iterator[Output]]
    | Callable[[Input, TransableConfig], Iterator[Output]]
    | Callable[[Input, PhotoType, TransableConfig], Iterator[Output]]
    | Callable[[Input, Iterator[PhotoType]], Iterator[Output]]
    | Callable[[Input, Iterator[PhotoType], TransableConfig], Iterator[Output]],
    input_value: Any,
    receive_value: Any | None,
    config: TransableConfig,
    *,
    keep_order: bool = False,
    return_exceptions: bool = False,
    run_type: str | None = None,
    **kwargs: Any,
) -> Iterator[Output]:
    config = ensure_config(config)
    if not _should_trace(config):
        with set_config_context(config) as context:
            iterator = context.run(
                call_func_with_variable_args,
                func,
                input_value,
                receive_value,
                config,
                keep_order=keep_order,
                return_exceptions=return_exceptions,
                **kwargs,
            )
            for item in iterator:
                yield item
        return

    results = []
    callback_manager = get_callback_manager(config)
    run = create_run(transable, config, input_value, receive_value, run_type)
    callback_manager.on_start(run, config)
    try:
        with set_run_context(run), set_config_context(config) as context:
            iterator = context.run(
                call_func_with_variable_args,
                func,
                input_value,
                receive_value,
                config,
                keep_order=keep_order,
                return_exceptions=return_exceptions,
                **kwargs,
            )
            for item in iterator:
                run.count += 1
                callback_manager.on_stream_chunk(run, item, config)
                yield item
                results.append(item)
    except BaseException as exc:
        run.errors = [exc]
        run.end_time = _now()
        callback_manager.on_error(run, config)
        raise
    run.output = results
    run.end_time = _now()
    callback_manager.on_end(run, config)


async def aiter_with_tracing(
    transable: object,
    func: Callable[[Input], AsyncIterator[Output]]
    | Callable[[Input, TransableConfig], AsyncIterator[Output]]
    | Callable[[Input, PhotoType, TransableConfig], AsyncIterator[Output]]
    | Callable[[Input, AsyncIterator[PhotoType]], AsyncIterator[Output]]
    | Callable[
        [Input, AsyncIterator[PhotoType], TransableConfig], AsyncIterator[Output]
    ],
    input_value: Any,
    receive_value: Any | None,
    config: TransableConfig,
    *,
    keep_order: bool = False,
    return_exceptions: bool = False,
    run_type: str | None = None,
    **kwargs: Any,
) -> AsyncIterator[Output]:
    config = ensure_config(config)
    if not _should_trace(config):
        with set_config_context(config) as context:
            iterator = context.run(
                call_func_with_variable_args,
                func,
                input_value,
                receive_value,
                config,
                keep_order=keep_order,
                return_exceptions=return_exceptions,
                **kwargs,
            )
            while True:
                try:
                    item = await coro_with_context(anext(iterator), context)
                except StopAsyncIteration:
                    break
                yield item
        return

    results = []
    callback_manager = get_async_callback_manager(config)
    run = create_run(transable, config, input_value, receive_value, run_type)
    await callback_manager.on_start(run, config)
    try:
        with set_run_context(run), set_config_context(config) as context:
            iterator = context.run(
                call_func_with_variable_args,
                func,
                input_value,
                receive_value,
                config,
                keep_order=keep_order,
                return_exceptions=return_exceptions,
                **kwargs,
            )
            while True:
                try:
                    item = await coro_with_context(anext(iterator), context)
                except StopAsyncIteration:
                    break
                run.count += 1
                await callback_manager.on_stream_chunk(run, item, config)
                yield item
    except BaseException as exc:
        run.errors = [exc]
        run.end_time = _now()
        await callback_manager.on_error(run, config)
        raise
    run.output = results
    run.end_time = _now()
    await callback_manager.on_end(run, config)


def batch_with_tracing(
    transable: object,
    func: Callable[[Input, list[PhotoType]], Sequence[Exception | Output]]
    | Callable[[Input, list[PhotoType], TransableConfig], Sequence[Exception | Output]],
    input_value: Any,
    receive_value: list[PhotoType] | None,
    config: TransableConfig,
    run_type: str,
    *,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> Sequence[Output | Exception]:
    config = ensure_config(config)
    receive_value = receive_value or []
    if accepts_any(func, "return_exceptions"):
        kwargs["return_exceptions"] = return_exceptions

    if not _should_trace(config):
        try:
            with set_config_context(config) as context:
                results = list(
                    context.run(
                        call_func_with_variable_args,
                        func,
                        input_value,
                        receive_value,
                        config,
                        **kwargs,
                    )
                )
        except Exception as exc:
            if not return_exceptions:
                raise
            results = [exc]
        return results

    callback_manager = get_callback_manager(config)
    run = create_run(transable, config, input_value, receive_value, run_type)
    callback_manager.on_start(run, config)
    try:
        with set_run_context(run), set_config_context(config) as context:
            results = list(
                context.run(
                    call_func_with_variable_args,
                    func,
                    input_value,
                    receive_value,
                    config,
                    **kwargs,
                )
            )
    except Exception as exc:
        run.errors = [exc]
        run.end_time = _now()
        callback_manager.on_error(run, config)
        if not return_exceptions:
            raise
        results = [exc]

    for item in results:
        run.count += 1
        if isinstance(item, Exception):
            run.errors = run.errors or []
            run.errors.append(item)

    run.output = results
    run.end_time = _now()
    callback_manager.on_end(run, config)
    return results


async def abatch_with_tracing(
    transable: object,
    func: Callable[[Input, list[PhotoType]], Awaitable[Sequence[Exception | Output]]]
    | Callable[
        [Input, list[PhotoType], TransableConfig],
        Awaitable[Sequence[Exception | Output]],
    ],
    input_value: Any,
    receive_value: list[PhotoType] | None,
    config: TransableConfig,
    run_type: str,
    *,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> Sequence[Output | Exception]:
    config = ensure_config(config)
    receive_value = receive_value or []
    if accepts_any(func, "return_exceptions"):
        kwargs["return_exceptions"] = return_exceptions

    if not _should_trace(config):
        try:
            with set_config_context(config) as context:
                coro = context.run(
                    acall_func_with_variable_args,
                    func,
                    input_value,
                    receive_value,
                    config,
                    **kwargs,
                )
                results = list(await coro_with_context(coro, context, create_task=True))
        except Exception as exc:
            if not return_exceptions:
                raise
            results = [exc]
        return results

    callback_manager = get_async_callback_manager(config)
    run = create_run(transable, config, input_value, receive_value, run_type)
    await callback_manager.on_start(run, config)
    try:
        with set_run_context(run), set_config_context(config) as context:
            coro = context.run(
                acall_func_with_variable_args,
                func,
                input_value,
                receive_value,
                config,
                **kwargs,
            )
            results = list(await coro_with_context(coro, context, create_task=True))
    except Exception as exc:
        run.errors = [exc]
        run.end_time = _now()
        await callback_manager.on_error(run, config)
        if not return_exceptions:
            raise
        results = [exc]

    for item in results:
        run.count += 1
        if isinstance(item, Exception):
            run.errors = run.errors or []
            run.errors.append(item)

    run.output = results
    run.end_time = _now()
    await callback_manager.on_end(run, config)
    return results


class TransableListener:
    def __init__(
        self,
        *,
        on_start: (
            Callable[[TransableRun], None]
            | Callable[[TransableRun, TransableConfig], None]
            | None
        ) = None,
        on_end: (
            Callable[[TransableRun], None]
            | Callable[[TransableRun, TransableConfig], None]
            | None
        ) = None,
        on_error: (
            Callable[[TransableRun], None]
            | Callable[[TransableRun, TransableConfig], None]
            | None
        ) = None,
        on_stream_chunk: Callable[[TransableRun, Any, TransableConfig], None]
        | None = None,
    ) -> None:
        self._on_start = on_start
        self._on_end = on_end
        self._on_error = on_error
        self._on_stream_chunk = on_stream_chunk

    def on_start(self, run: TransableRun, config: TransableConfig) -> None:
        if self._on_start is None:
            return
        CallbackManager._call_handler(self._on_start, run, config)

    def on_end(self, run: TransableRun, config: TransableConfig) -> None:
        if self._on_end is None:
            return
        CallbackManager._call_handler(self._on_end, run, config)

    def on_error(self, run: TransableRun, config: TransableConfig) -> None:
        if self._on_error is None:
            return
        CallbackManager._call_handler(self._on_error, run, config)

    def on_stream_chunk(
        self,
        run: TransableRun,
        chunk: Any,
        config: TransableConfig,
    ) -> None:
        if self._on_stream_chunk is None:
            return
        CallbackManager._call_stream_handler(
            self._on_stream_chunk,
            run,
            chunk,
            config,
        )


class AsyncTransableListener:
    def __init__(
        self,
        *,
        on_start: (
            Callable[[TransableRun], Awaitable[None]]
            | Callable[[TransableRun, TransableConfig], Awaitable[None]]
            | None
        ) = None,
        on_end: (
            Callable[[TransableRun], Awaitable[None]]
            | Callable[[TransableRun, TransableConfig], Awaitable[None]]
            | None
        ) = None,
        on_error: (
            Callable[[TransableRun], Awaitable[None]]
            | Callable[[TransableRun, TransableConfig], Awaitable[None]]
            | None
        ) = None,
        on_stream_chunk: Callable[[TransableRun, Any, TransableConfig], Awaitable[None]]
        | None = None,
    ) -> None:
        self._on_start = on_start
        self._on_end = on_end
        self._on_error = on_error
        self._on_stream_chunk = on_stream_chunk

    async def on_start(self, run: TransableRun, config: TransableConfig) -> None:
        if self._on_start is None:
            return
        result = AsyncCallbackManager._call_handler(self._on_start, run, config)
        await AsyncCallbackManager._maybe_await(result)

    async def on_end(self, run: TransableRun, config: TransableConfig) -> None:
        if self._on_end is None:
            return
        result = AsyncCallbackManager._call_handler(self._on_end, run, config)
        await AsyncCallbackManager._maybe_await(result)

    async def on_error(self, run: TransableRun, config: TransableConfig) -> None:
        if self._on_error is None:
            return
        result = AsyncCallbackManager._call_handler(self._on_error, run, config)
        await AsyncCallbackManager._maybe_await(result)

    async def on_stream_chunk(
        self,
        run: TransableRun,
        chunk: Any,
        config: TransableConfig,
    ) -> None:
        if self._on_stream_chunk is None:
            return
        result = AsyncCallbackManager._call_stream_handler(
            self._on_stream_chunk,
            run,
            chunk,
            config,
        )
        await AsyncCallbackManager._maybe_await(result)
