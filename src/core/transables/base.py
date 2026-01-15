from __future__ import annotations

import contextlib
import inspect
import queue
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import FIRST_COMPLETED, Future, wait
from contextvars import copy_context
from functools import wraps
from operator import itemgetter
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Protocol,
    cast,
    get_args,
    overload,
)
from typing_extensions import override

import anyio
from anyio import BrokenResourceError, ClosedResourceError
from pydantic import BaseModel, ConfigDict, Field, create_model

from core.transables.config import (
    TransableConfig,
    TransableConfigModel,
    arun_in_context,
    ensure_config,
    get_executor_for_config,
    merge_configs,
    patch_config,
    run_in_context,
    run_in_executor,
)
from core.transables.serializable import (
    Serializable,
    SerializedConstructor,
    SerializedNotImplemented,
)
from core.transables.tracing import (
    AsyncTransableListener,
    TransableListener,
    TransableRun,
    abatch_with_tracing,
    aiter_with_tracing,
    arun_with_tracing,
    batch_with_tracing,
    iter_with_tracing,
    run_with_tracing,
)
from core.transables.utils import (
    Input,
    OtherInput,
    OtherPhotoType,
    Output,
    PhotoType,
    acall_func_with_variable_args,
    accepts_receive,
    accepts_receives,
    call_func_with_variable_args,
    clone_photo,
    coerce_photo,
    ensure_subclass,
    get_function_first_arg_dict_keys,
    get_lambda_source,
    indent_lines_after_first,
    is_async_callable,
    is_async_generator,
    is_generator,
    merge_photos,
    stream_buffer,
)

if TYPE_CHECKING:
    from anyio.abc import ObjectReceiveStream, ObjectSendStream

Listener = (
    Callable[[TransableRun], None] | Callable[[TransableRun, TransableConfig], None]
)
AsyncListener = (
    Callable[[TransableRun], Awaitable[None]]
    | Callable[[TransableRun, TransableConfig], Awaitable[None]]
)


class Transable(ABC, Generic[Input, PhotoType]):
    """Base class for streaming pipeline components."""

    name: str | None

    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        """Get the name of the Transable."""
        if name:
            name_ = name
        elif hasattr(self, "name") and self.name:
            name_ = self.name
        else:
            # Here we handle a case where the transable subclass is also a pydantic
            # model.
            cls = self.__class__
            # Then it's a pydantic sub-class, and we have to check
            # whether it's a generic, and if so recover the original name.
            if (
                hasattr(
                    cls,
                    "__pydantic_generic_metadata__",
                )
                and "origin" in cls.__pydantic_generic_metadata__  # pyright: ignore[reportAttributeAccessIssue]
                and cls.__pydantic_generic_metadata__["origin"] is not None  # pyright: ignore[reportAttributeAccessIssue]
            ):
                name_ = cls.__pydantic_generic_metadata__["origin"].__name__  # pyright: ignore[reportAttributeAccessIssue]
            else:
                name_ = cls.__name__

        if suffix:
            if name_[0].isupper():
                return name_ + suffix.title()
            return name_ + "_" + suffix.lower()
        return name_

    @property
    def InputType(self) -> type[Input]:  # noqa: N802
        """The type of input this Transable accepts specified as a type annotation."""
        # First loop through all parent classes and if any of them is
        # a pydantic model, we will pick up the generic parameterization
        # from that model via the __pydantic_generic_metadata__ attribute.
        for base in self.__class__.mro():
            if hasattr(base, "__pydantic_generic_metadata__"):
                metadata = base.__pydantic_generic_metadata__
                if "args" in metadata and len(metadata["args"]) == 2:
                    return metadata["args"][0]

        # If we didn't find a pydantic model in the parent classes,
        # then loop through __orig_bases__. This corresponds to
        # Transables that are not pydantic models.
        for cls in self.__class__.__orig_bases__:  # type: ignore[attr-defined]
            type_args = get_args(cls)
            if type_args and len(type_args) == 2:
                return type_args[0]

        msg = (
            f"Transable {self.get_name()} doesn't have an inferable InputType. "
            "Override the InputType property to specify the input type."
        )
        raise TypeError(msg)

    @property
    def PhotoType(self) -> type[PhotoType]:  # noqa: N802
        """The type of Photo this Transable processes specified as a type annotation."""
        # First loop through bases -- this will help generic
        # any pydantic models.
        for base in self.__class__.mro():
            if hasattr(base, "__pydantic_generic_metadata__"):
                metadata = base.__pydantic_generic_metadata__
                if "args" in metadata and len(metadata["args"]) == 2:
                    return metadata["args"][1]

        for cls in self.__class__.__orig_bases__:  # type: ignore[attr-defined]
            type_args = get_args(cls)
            if type_args and len(type_args) == 2:
                return type_args[1]

        msg = (
            f"Transable {self.get_name()} doesn't have an inferable PhotoType. "
            "Override the PhotoType property to specify the photo type."
        )
        raise TypeError(msg)

    @property
    def input_schema(self) -> type[BaseModel]:
        """The type of input this Runnable accepts specified as a pydantic model."""
        return self.get_input_schema()

    def get_input_schema(
        self,
        config: TransableConfig | None = None,  # noqa: ARG002
    ) -> type[BaseModel]:
        """Get a pydantic model that can be used to validate input to the Runnable.

        Runnables that leverage the configurable_fields and configurable_alternatives
        methods will have a dynamic input schema that depends on which
        configuration the Runnable is invoked with.

        This method allows to get an input schema for a specific configuration.

        Args:
            config: A config to use when generating the schema.

        Returns:
            A pydantic model that can be used to validate input.
        """
        root_type = self.InputType

        if (
            inspect.isclass(root_type)
            and not isinstance(root_type, GenericAlias)
            and issubclass(root_type, BaseModel)
        ):
            return root_type

        return create_model(
            self.get_name("Input"),
            root=root_type,
            # create model needs access to appropriate type annotations to be
            # able to construct the pydantic model.
            # When we create the model, we pass information about the namespace
            # where the model is being created, so the type annotations can
            # be resolved correctly as well.
            # self.__class__.__module__ handles the case when the Runnable is
            # being sub-classed in a different module.
            module_name=self.__class__.__module__,
        )

    def get_input_jsonschema(
        self, config: TransableConfig | None = None
    ) -> dict[str, Any]:
        """Get a JSON schema that represents the input to the Runnable.

        Args:
            config: A config to use when generating the schema.

        Returns:
            A JSON schema that represents the input to the Runnable.

        Example:

            .. code-block:: python

                from langchain_core.runnables import RunnableLambda

                def add_one(x: int) -> int:
                    return x + 1

                runnable = RunnableLambda(add_one)

                print(runnable.get_input_jsonschema())

        .. versionadded:: 0.3.0
        """
        return self.get_input_schema(config).model_json_schema()

    def __or__(
        self,
        other: Transable[Any, OtherPhotoType]
        | Callable[[Any], OtherPhotoType]
        | Callable[[Any], Awaitable[OtherPhotoType]]
        | Callable[[Any, Iterator[OtherPhotoType]], Iterator[OtherPhotoType]]
        | Callable[[Any, AsyncIterator[OtherPhotoType]], AsyncIterator[OtherPhotoType]],
    ) -> TransableSerializable[Input, OtherPhotoType]:
        return TransableSequence(self, other)

    def __ror__(
        self,
        other: Transable[OtherInput, Any]
        | Callable[[OtherInput], Any]
        | Callable[[OtherInput], Awaitable[Any]]
        | Callable[[OtherInput, Iterator[Any]], Iterator[Any]]
        | Callable[[OtherInput, AsyncIterator[Any]], AsyncIterator[Any]],
    ) -> TransableSerializable[OtherInput, PhotoType]:
        return TransableSequence(other, self)

    def pipe(
        self,
        *others: Transable[Any, OtherPhotoType] | Callable[[Any], OtherPhotoType],
        name: str | None = None,
    ) -> TransableSerializable[Input, OtherPhotoType]:
        """"""
        return TransableSequence(self, *others, name=name)

    def __and__(
        self,
        other: Transable[Any, Any]
        | Callable[[Any], Any]
        | Callable[[Any], Awaitable[Any]]
        | Callable[[Any, Iterator[Any]], Iterator[Any]]
        | Callable[[Any, AsyncIterator[Any]], AsyncIterator[Any]],
    ) -> TransableSerializable[Input, PhotoType]:
        return TransableParallel(self, other)

    def __rand__(
        self,
        other: Transable[Any, Any]
        | Callable[[Any], Any]
        | Callable[[Any], Awaitable[Any]]
        | Callable[[Any, Iterator[Any]], Iterator[Any]]
        | Callable[[Any, AsyncIterator[Any]], AsyncIterator[Any]],
    ) -> TransableSerializable[Input, PhotoType]:
        return TransableParallel(other, self)

    def parallel(
        self,
        *others: Transable[Input, PhotoType] | Callable[[Any], OtherPhotoType],
    ) -> TransableSerializable[Input, PhotoType]:
        """"""
        return TransableParallel(self, *others)

    @abstractmethod
    def invoke(
        self,
        input: Input,
        receive: PhotoType | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoType:
        """Consume a single input and yield a single result."""

    async def ainvoke(
        self,
        input: Input,
        receive: PhotoType | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoType:
        """Consume a single input and yield streaming results."""
        return await run_in_executor(
            config, self.invoke, input, receive, config, **kwargs
        )

    def stream(
        self,
        input: Input,
        receives: Iterator[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> Iterator[PhotoType]:
        """Consume a stream of inputs and yield streaming results."""
        config = ensure_config(config)

        if receives is None:
            child_config = patch_config(config, child=True)
            try:
                yield run_in_context(
                    child_config,
                    self.invoke,
                    input,
                    None,
                    child_config,
                    **kwargs,
                )
            except Exception as e:
                if not ignore_exceptions:
                    raise
                raise ValueError(
                    "Exceptions cannot be ignored when only one output can be generated."
                ) from e
            return

        max_concurrency = config.get("max_concurrency")
        max_in_flight = int(max_concurrency) if max_concurrency else None

        def _run_one(item: PhotoType) -> PhotoType:
            child_config = patch_config(config, child=True)
            return run_in_context(
                child_config,
                self.invoke,
                input,
                item,
                child_config,
                **kwargs,
            )

        with get_executor_for_config(config) as executor:
            in_flight: set[Future[PhotoType]] = set()
            idx_by_future: dict[Future[PhotoType], int] = {}
            pending_results: dict[int, PhotoType | None] = {}
            next_idx = 0

            def _collect_done(done: set[Future[PhotoType]]) -> None:
                nonlocal next_idx
                for fut in done:
                    i = idx_by_future.pop(fut)
                    try:
                        res = fut.result()
                    except Exception:
                        if not ignore_exceptions:
                            for pf in in_flight:
                                pf.cancel()
                            raise
                        pending_results[i] = None
                    else:
                        pending_results[i] = res

            def _yield_ready_in_order() -> Iterator[PhotoType]:
                nonlocal next_idx
                while next_idx in pending_results:
                    out = pending_results.pop(next_idx)
                    next_idx += 1
                    if out is not None:
                        yield out

            def _yield_done_unordered(
                done: set[Future[PhotoType]],
            ) -> Iterator[PhotoType]:
                for fut in done:
                    idx_by_future.pop(fut, None)
                    try:
                        out = fut.result()
                    except Exception:
                        if not ignore_exceptions:
                            for pf in in_flight:
                                pf.cancel()
                            raise
                        continue
                    else:
                        if out is not None:
                            yield out

            for idx, item in enumerate(receives):
                if max_in_flight is not None:
                    while len(in_flight) >= max_in_flight:
                        done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                        if keep_order:
                            _collect_done(done)
                            yield from _yield_ready_in_order()
                        else:
                            yield from _yield_done_unordered(done)

                fut = cast("Future[PhotoType]", executor.submit(_run_one, item))
                in_flight.add(fut)
                idx_by_future[fut] = idx

            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                if keep_order:
                    _collect_done(done)
                    yield from _yield_ready_in_order()
                else:
                    yield from _yield_done_unordered(done)

    async def astream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoType]:
        """Consume a stream of inputs and yield streaming results."""
        config = ensure_config(config)
        # If receives is None, we just invoke once.
        if receives is None:
            child_config = patch_config(config, child=True)
            try:
                yield await arun_in_context(
                    child_config,
                    self.ainvoke,
                    input,
                    None,
                    child_config,
                    **kwargs,
                )
            except Exception as e:
                if not ignore_exceptions:
                    raise
                else:
                    raise ValueError(
                        "Exceptions cannot be ignored when only one output can be generated."
                    ) from e
            return

        send, recv = anyio.create_memory_object_stream[PhotoType](
            max_buffer_size=stream_buffer(config)
        )
        out_send, out_recv = anyio.create_memory_object_stream[
            tuple[int, PhotoType | None]
        ](max_buffer_size=stream_buffer(config))

        max_concurrency = config.get("max_concurrency")
        limiter = (
            anyio.CapacityLimiter(max_concurrency)
            if max_concurrency
            else contextlib.nullcontext()
        )

        async def _broadcast() -> None:
            try:
                async for item in receives:
                    await send.send(item)
            finally:
                await send.aclose()

        async def _run_one(idx: int, item: PhotoType) -> None:
            async with limiter:
                child_config = patch_config(config, child=True)
                try:
                    result = await arun_in_context(
                        child_config,
                        self.ainvoke,
                        input,
                        item,
                        child_config,
                        **kwargs,
                    )
                except Exception:
                    if not ignore_exceptions:
                        raise
                    await out_send.send((idx, None))
                else:
                    await out_send.send((idx, result))

        async def _process_items() -> None:
            idx = 0
            try:
                async with anyio.create_task_group() as tg, recv:
                    async for item in recv:
                        tg.start_soon(_run_one, idx, item)
                        idx += 1
            finally:
                await out_send.aclose()

        async def _collect_outputs() -> AsyncIterator[PhotoType]:
            async with out_recv:
                # don't keep order
                if not keep_order:
                    async for payload in out_recv:
                        if payload[1] is None:
                            continue
                        yield cast("PhotoType", payload[1])
                    return

                # keep order
                cond = anyio.Condition()
                pending: dict[int, PhotoType] = {}
                done = False

                async def _ingest() -> None:
                    nonlocal done
                    async for payload in out_recv:
                        i, v = cast("tuple[int, PhotoType]", payload)
                        async with cond:
                            pending[i] = v
                            cond.notify_all()
                    async with cond:
                        done = True
                        cond.notify_all()

                next_idx = 0
                async with anyio.create_task_group() as tg:
                    tg.start_soon(_ingest)
                    while True:
                        async with cond:
                            while next_idx not in pending:
                                if done:
                                    return
                                await cond.wait()
                            out = pending.pop(next_idx)
                            next_idx += 1
                        if out is not None:
                            yield out

        async with anyio.create_task_group() as tg:
            tg.start_soon(_broadcast)
            tg.start_soon(_process_items)
            async for out in _collect_outputs():
                yield out

    @overload
    def batch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[False],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType]: ...

    @overload
    def batch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]: ...

    def batch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]:
        """Consume a batch of inputs and return a list of results."""
        config = ensure_config(config)
        if receives is None or len(receives) == 1:
            receive = receives[0] if receives else None
            child_config = patch_config(config, child=True)
            if return_exceptions:
                try:
                    result = run_in_context(
                        child_config,
                        self.invoke,
                        input,
                        receive,
                        child_config,
                        **kwargs,
                    )
                except Exception as e:
                    return [e]
                else:
                    return [result]
            else:
                result = run_in_context(
                    child_config,
                    self.invoke,
                    input,
                    receive,
                    child_config,
                    **kwargs,
                )
                return [result]

        results: dict[int, PhotoType | Exception | None] = dict.fromkeys(
            range(len(receives))
        )

        def _process_item(idx: int, item: PhotoType) -> None:
            child_config = patch_config(config, child=True)
            if return_exceptions:
                try:
                    results[idx] = run_in_context(
                        child_config,
                        self.invoke,
                        input,
                        item,
                        child_config,
                        **kwargs,
                    )
                except Exception as e:
                    results[idx] = e
            else:
                results[idx] = run_in_context(
                    child_config,
                    self.invoke,
                    input,
                    item,
                    child_config,
                    **kwargs,
                )

        with get_executor_for_config(config) as executor:
            executor.map(_process_item, range(len(receives)), receives)

        if any(res is None for res in results.values()):
            raise RuntimeError("Parallel step did not return a PhotoResult.")

        return list(cast("dict[int, PhotoType | Exception]", results).values())

    @overload
    async def abatch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[False],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType]: ...

    @overload
    async def abatch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]: ...

    async def abatch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]:
        """Consume a batch of inputs and return a list of results."""
        config = ensure_config(config)

        if receives is None or len(receives) == 1:
            receive = receives[0] if receives else None
            child_config = patch_config(config, child=True)
            if return_exceptions:
                try:
                    result = await arun_in_context(
                        child_config,
                        self.ainvoke,
                        input,
                        receive,
                        child_config,
                        **kwargs,
                    )
                except Exception as e:
                    return [e]
                else:
                    return [result]
            else:
                result = await arun_in_context(
                    child_config,
                    self.ainvoke,
                    input,
                    receive,
                    child_config,
                    **kwargs,
                )
                return [result]

        results: dict[int, PhotoType | Exception | None] = dict.fromkeys(
            range(len(receives))
        )

        max_concurrency = config.get("max_concurrency")
        limiter = (
            anyio.CapacityLimiter(max_concurrency)
            if max_concurrency
            else contextlib.nullcontext()
        )

        async def _process_item(idx: int, item: PhotoType) -> None:
            async with limiter:
                child_config = patch_config(config, child=True)
                if return_exceptions:
                    try:
                        results[idx] = await arun_in_context(
                            child_config,
                            self.ainvoke,
                            input,
                            item,
                            child_config,
                            **kwargs,
                        )
                    except Exception as e:
                        results[idx] = e
                else:
                    results[idx] = await arun_in_context(
                        child_config,
                        self.ainvoke,
                        input,
                        item,
                        child_config,
                        **kwargs,
                    )

        async with anyio.create_task_group() as tg:
            for idx, item in enumerate(receives):
                tg.start_soon(_process_item, idx, item)

        if any(res is None for res in results.values()):
            raise RuntimeError("Parallel step did not return a PhotoResult.")

        return list(cast("dict[int, PhotoType | Exception]", results).values())

    def bind(self, **kwargs: Any) -> Transable[Input, PhotoType]:
        """Return a bound transable with pre-applied kwargs."""
        return TransableBinding(bound=self, kwargs=kwargs, config={})

    def config_schema(self) -> type[BaseModel]:
        """Return the Pydantic model for TransableConfig."""
        return TransableConfigModel

    def get_config_jsonschema(self) -> dict[str, Any]:
        """Return the JSON schema for TransableConfig."""
        return self.config_schema().model_json_schema()

    def with_listeners(
        self,
        *,
        on_start: Listener | None = None,
        on_end: Listener | None = None,
        on_error: Listener | None = None,
        on_stream_chunk: Callable[[TransableRun, Any, TransableConfig], None]
        | None = None,
    ) -> Transable[Input, PhotoType]:
        """Bind lifecycle listeners to a Transable."""

        listener = TransableListener(
            on_start=on_start,
            on_end=on_end,
            on_error=on_error,
            on_stream_chunk=on_stream_chunk,
        )

        def listener_factory(_config: TransableConfig) -> TransableConfig:
            return {
                "callbacks": [listener],
                "trace": True,
            }

        return TransableBinding(
            bound=self,
            config_factories=[listener_factory],
        )

    def with_alisteners(
        self,
        *,
        on_start: AsyncListener | None = None,
        on_end: AsyncListener | None = None,
        on_error: AsyncListener | None = None,
        on_stream_chunk: Callable[[TransableRun, Any, TransableConfig], Awaitable[None]]
        | None = None,
    ) -> Transable[Input, PhotoType]:
        """Bind async lifecycle listeners to a Transable."""

        listener = AsyncTransableListener(
            on_start=on_start,
            on_end=on_end,
            on_error=on_error,
            on_stream_chunk=on_stream_chunk,
        )

        def listener_factory(_config: TransableConfig) -> TransableConfig:
            return {
                "callbacks": [listener],
                "trace": True,
            }

        return TransableBinding(
            bound=self,
            config_factories=[listener_factory],
        )

    def with_config(
        self,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> Transable[Input, PhotoType]:
        """Bind config to a Transable, returning a new Transable."""
        return TransableBinding(
            bound=self,
            config=cast("TransableConfig", {**(config or {}), **kwargs}),
            kwargs={},
        )

    def with_types(
        self,
        input_type: type[Input] | None = None,
        photo_type: type[PhotoType] | None = None,
    ) -> Transable[Input, PhotoType]:
        return TransableBinding(
            bound=self,
            custom_input_type=input_type,
            custom_photo_type=photo_type,
            kwargs={},
        )

    # tools for subclasses to call functions with config

    def _call_with_config(
        self,
        func: Callable[[Input], Output]
        | Callable[[Input, TransableConfig], Output]
        | Callable[[Input, PhotoType, TransableConfig], Output]
        | Callable[[Input, Iterator[PhotoType]], Output]
        | Callable[[Input, Iterator[PhotoType], TransableConfig], Output],
        input: Input,
        receive: Any | None,
        config: TransableConfig | None,
        *,
        run_type: str | None = None,
        **kwargs: Any,
    ) -> Output:
        config = ensure_config(config)
        return run_with_tracing(
            self,
            func,
            input,
            receive,
            config,
            run_type=run_type,
            **kwargs,
        )

    async def _acall_with_config(
        self,
        func: Callable[[Input], Awaitable[Output]]
        | Callable[[Input, TransableConfig], Awaitable[Output]]
        | Callable[[Input, PhotoType, TransableConfig], Awaitable[Output]]
        | Callable[[Input, AsyncIterator[PhotoType]], Awaitable[Output]]
        | Callable[
            [Input, AsyncIterator[PhotoType], TransableConfig], Awaitable[Output]
        ],
        input: Input,
        receive: Any | None,
        config: TransableConfig | None,
        *,
        run_type: str | None = None,
        **kwargs: Any,
    ) -> Output:
        config = ensure_config(config)
        return await arun_with_tracing(
            self,
            func,
            input,
            receive,
            config,
            run_type=run_type,
            **kwargs,
        )

    def _stream_with_config(
        self,
        func: Callable[[Input], Iterator[Output]]
        | Callable[[Input, TransableConfig], Iterator[Output]]
        | Callable[[Input, PhotoType, TransableConfig], Iterator[Output]]
        | Callable[[Input, Iterator[PhotoType]], Iterator[Output]]
        | Callable[[Input, Iterator[PhotoType], TransableConfig], Iterator[Output]],
        input: Input,
        receives: Any | None,
        config: TransableConfig | None,
        *,
        run_type: str | None = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        config = ensure_config(config)
        yield from iter_with_tracing(
            self,
            func,
            input,
            receives,
            config,
            run_type=run_type,
            **kwargs,
        )

    async def _astream_with_config(
        self,
        func: Callable[[Input], AsyncIterator[Output]]
        | Callable[[Input, TransableConfig], AsyncIterator[Output]]
        | Callable[[Input, PhotoType, TransableConfig], AsyncIterator[Output]]
        | Callable[[Input, AsyncIterator[PhotoType]], AsyncIterator[Output]]
        | Callable[
            [Input, AsyncIterator[PhotoType], TransableConfig], AsyncIterator[Output]
        ],
        input: Input,
        receives: Any | None,
        config: TransableConfig | None,
        *,
        run_type: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        config = ensure_config(config)
        async for item in aiter_with_tracing(
            self,
            func,
            input,
            receives,
            config,
            run_type=run_type,
            **kwargs,
        ):
            yield item

    def _batch_with_config(
        self,
        func: Callable[[Input, list[PhotoType]], Sequence[Exception | Output]]
        | Callable[
            [Input, list[PhotoType], TransableConfig], Sequence[Exception | Output]
        ],
        input: Any,
        receives: list[PhotoType] | None,
        config: TransableConfig,
        run_type: str,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> Sequence[Output | Exception]:
        config = ensure_config(config)
        return batch_with_tracing(
            self,
            func,
            input,
            receives,
            config,
            return_exceptions=return_exceptions,
            run_type=run_type,
            **kwargs,
        )

    async def _abatch_with_config(
        self,
        func: Callable[
            [Input, list[PhotoType]], Awaitable[Sequence[Exception | Output]]
        ]
        | Callable[
            [Input, list[PhotoType], TransableConfig],
            Awaitable[Sequence[Exception | Output]],
        ],
        input: Any,
        receives: list[PhotoType] | None,
        config: TransableConfig,
        run_type: str,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> Sequence[Output | Exception]:
        config = ensure_config(config)
        return await abatch_with_tracing(
            self,
            func,
            input,
            receives,
            config,
            return_exceptions=return_exceptions,
            run_type=run_type,
            **kwargs,
        )


class TransableSerializable(Serializable, Transable[Input, PhotoType]):
    """Transable that can be serialized to JSON."""

    name: str | None = None

    model_config = ConfigDict(
        # Suppress warnings from pydantic protected namespaces
        # (e.g., `model_`)
        protected_namespaces=(),
    )

    @override
    def to_json(self) -> SerializedConstructor | SerializedNotImplemented:
        """Serialize the Transable to JSON.

        Returns:
            A JSON-serializable representation of the Transable.
        """
        dumped = super().to_json()
        with contextlib.suppress(Exception):
            dumped["name"] = self.get_name()
        return dumped


class TransableSequence(TransableSerializable[Input, PhotoType]):
    start: Transable[Input, Any]
    middle: list[Transable[Any, Any]] = Field(default_factory=list)
    end: Transable[Any, PhotoType]

    def __init__(
        self,
        *chains: TransableLike,
        name: str | None = None,
        start: Transable[Input, Any] | None = None,
        middle: list[Transable[Any, Any]] | None = None,
        end: Transable[Any, PhotoType] | None = None,
    ) -> None:
        chains_flat: list[Transable[Any, Any]] = []
        if not chains and start is not None and end is not None:
            chains_flat = [start] + (middle or []) + [end]
        for chain in chains:
            if isinstance(chain, TransableSequence):
                chains_flat.extend(chain.chains)
            else:
                chains_flat.append(coerce_to_transable(chain))
        if len(chains_flat) < 2:
            msg = (
                f"TransableSequence must have at least 2 steps, got {len(chains_flat)}"
            )
            raise ValueError(msg)

        datas = {
            "start": chains_flat[0],
            "middle": list(chains_flat[1:-1]),
            "end": chains_flat[-1],
        }
        super().__init__(**datas, name=name)
        self._validate_chain()

    @property
    @override
    def InputType(self) -> type[Input]:
        """The type of the input to the Transable."""
        return self.start.InputType

    @property
    @override
    def PhotoType(self) -> type[PhotoType]:
        """The type of the output of the Transable."""
        return self.end.PhotoType

    @override
    def get_input_schema(
        self, config: TransableConfig | None = None
    ) -> type[BaseModel]:
        first = self.chains[0]
        if len(self.chains) == 1:
            return first.get_input_schema(config)
        return first.get_input_schema(config)

    @property
    def chains(self) -> list[Transable[Any, Any]]:
        """All the Transables that make up the sequence in order.

        Returns:
            A list of Transables.
        """
        return [self.start, *self.middle, self.end]

    @override
    def __or__(
        self,
        other: Transable[Any, OtherPhotoType]
        | Callable[[Any], OtherPhotoType]
        | Callable[[Any], Awaitable[OtherPhotoType]]
        | Callable[[Any, Iterator[OtherPhotoType]], Iterator[OtherPhotoType]]
        | Callable[[Any, AsyncIterator[OtherPhotoType]], AsyncIterator[OtherPhotoType]],
    ) -> TransableSerializable[Input, OtherPhotoType]:
        if isinstance(other, TransableSequence):
            return TransableSequence(
                self.start,
                *self.middle,
                self.end,
                other.start,
                *other.middle,
                other.end,
                name=self.name or other.name,
            )
        return TransableSequence(
            self.start,
            *self.middle,
            self.end,
            coerce_to_transable(other),
            name=self.name,
        )

    @override
    def __ror__(
        self,
        other: Transable[OtherInput, Any]
        | Callable[[OtherInput], Any]
        | Callable[[OtherInput], Awaitable[Any]]
        | Callable[[OtherInput, Iterator[Any]], Iterator[Any]]
        | Callable[[OtherInput, AsyncIterator[Any]], AsyncIterator[Any]],
    ) -> TransableSerializable[OtherInput, PhotoType]:
        if isinstance(other, TransableSequence):
            return TransableSequence(
                other.start,
                *other.middle,
                other.end,
                self.start,
                *self.middle,
                self.end,
                name=self.name or other.name,
            )
        return TransableSequence(
            coerce_to_transable(other),
            self.start,
            *self.middle,
            self.end,
            name=self.name,
        )

    @override
    def __repr__(self) -> str:
        return "\n| ".join(
            repr(step) if i == 0 else indent_lines_after_first(repr(step), "| ")
            for i, step in enumerate(self.chains)
        )

    def _invoke(
        self,
        input: Input,
        receive: PhotoType | None,
        config: TransableConfig,
        **kwargs: Any,
    ) -> PhotoType:
        current = receive
        for transable in self.chains:
            child_config = patch_config(config, child=True)
            current = transable.invoke(
                input,
                current,
                config=child_config,
                **kwargs,
            )
        return cast("PhotoType", current)

    @override
    def invoke(
        self,
        input: Input,
        receive: PhotoType | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoType:
        config = ensure_config(config)

        return self._call_with_config(
            self._invoke,
            input,
            receive,
            config,
            run_type="invoke",
            **kwargs,
        )

    async def _ainvoke(
        self,
        input: Input,
        receive: PhotoType | None,
        config: TransableConfig,
        **kwargs: Any,
    ) -> PhotoType:
        current = receive
        for transable in self.chains:
            child_config = patch_config(config, child=True)
            current = await transable.ainvoke(
                input,
                current,
                config=child_config,
                **kwargs,
            )
        return cast("PhotoType", current)

    @override
    async def ainvoke(
        self,
        input: Input,
        receive: PhotoType | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoType:
        config = ensure_config(config)

        return await self._acall_with_config(
            self._ainvoke,
            input,
            receive,
            config,
            run_type="ainvoke",
            **kwargs,
        )

    def _stream(
        self,
        input: Input,
        receives: Iterator[PhotoType] | None,
        config: TransableConfig,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> Iterator[PhotoType]:
        child_config = patch_config(config, child=True)
        upstream: Iterator[Any] = run_in_context(
            child_config,
            self.start.stream,
            input,
            receives,
            child_config,
            keep_order=keep_order,
            ignore_exceptions=ignore_exceptions,
            **kwargs,
        )
        for transable in self.chains[1:]:
            upstream = self._pipe_stream(transable, upstream, input, config, **kwargs)
        yield from upstream

    @override
    def stream(
        self,
        input: Input,
        receives: Iterator[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> Iterator[PhotoType]:
        config = ensure_config(config)

        yield from self._stream_with_config(
            self._stream,
            input,
            receives,
            config,
            keep_order=keep_order,
            ignore_exceptions=ignore_exceptions,
            run_type="stream",
            **kwargs,
        )

    async def _astream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoType] | None,
        config: TransableConfig,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        child_config = patch_config(config, child=True)
        upstream: AsyncIterator[Any] = run_in_context(
            child_config,
            self.start.astream,
            input,
            receives,
            child_config,
            keep_order=keep_order,
            ignore_exceptions=ignore_exceptions,
            **kwargs,
        )
        for transable in self.chains[1:]:
            upstream = self._pipe_async_stream(
                transable, upstream, input, config, **kwargs
            )
        async for result in upstream:
            yield result

    @override
    async def astream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoType]:
        config = ensure_config(config)

        async for result in self._astream_with_config(
            self._astream,
            input,
            receives,
            config,
            keep_order=keep_order,
            ignore_exceptions=ignore_exceptions,
            run_type="astream",
            **kwargs,
        ):
            yield cast("PhotoType", result)

    def _batch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]:
        total = 1 if receives is None else len(receives)
        if total == 0:
            return []

        failed: dict[int, Exception] = {}
        current: list[PhotoType] | None = receives

        for transable in self.chains:
            child_config = patch_config(config, child=True)
            if return_exceptions:
                remaining_idxs = [i for i in range(total) if i not in failed]
                if not remaining_idxs:
                    break
                step_receives = current if current is not None else None
                try:
                    step_results = list(
                        run_in_context(
                            child_config,
                            transable.batch,
                            input,
                            step_receives,
                            child_config,
                            return_exceptions=True,
                            **kwargs,
                        )
                    )
                except Exception as exc:
                    for idx in remaining_idxs:
                        failed[idx] = exc
                    current = []
                    break

                failed = failed | {
                    idx: res
                    for idx, res in zip(remaining_idxs, step_results, strict=False)
                    if isinstance(res, Exception)
                }
                current = [
                    cast("PhotoType", res)
                    for res in step_results
                    if not isinstance(res, Exception)
                ]
            else:
                current = list(
                    cast(
                        "Sequence[PhotoType]",
                        run_in_context(
                            child_config,
                            transable.batch,
                            input,
                            current,
                            child_config,
                            return_exceptions=False,
                            **kwargs,
                        ),
                    )
                )

        if return_exceptions:
            outputs: list[PhotoType | Exception] = []
            successes = list(cast("list[PhotoType]", current or []))
            for i in range(total):
                if i in failed:
                    outputs.append(failed[i])
                else:
                    if not successes:
                        raise RuntimeError(
                            "Sequence batch did not return results for all inputs."
                        )
                    outputs.append(successes.pop(0))
            return outputs

        return current or []

    @overload
    def batch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[False],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType]: ...

    @overload
    def batch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]: ...

    @override
    def batch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]:
        config = ensure_config(config)

        return self._batch_with_config(
            self._batch,
            input,
            receives,
            config,
            return_exceptions=return_exceptions,
            run_type="batch",
            **kwargs,
        )

    async def _abatch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]:
        total = 1 if receives is None else len(receives)
        if total == 0:
            return []

        failed: dict[int, Exception] = {}
        current: list[PhotoType] | None = receives

        for transable in self.chains:
            child_config = patch_config(config, child=True)
            if return_exceptions:
                remaining_idxs = [i for i in range(total) if i not in failed]
                if not remaining_idxs:
                    break
                step_receives = current if current is not None else None
                try:
                    step_results = list(
                        await arun_in_context(
                            child_config,
                            transable.abatch,
                            input,
                            step_receives,
                            child_config,
                            return_exceptions=True,
                            **kwargs,
                        )
                    )
                except Exception as exc:
                    for idx in remaining_idxs:
                        failed[idx] = exc
                    current = []
                    break

                failed = failed | {
                    idx: res
                    for idx, res in zip(remaining_idxs, step_results, strict=False)
                    if isinstance(res, Exception)
                }
                current = [
                    cast("PhotoType", res)
                    for res in step_results
                    if not isinstance(res, Exception)
                ]
            else:
                current = list(
                    cast(
                        "Sequence[PhotoType]",
                        await arun_in_context(
                            child_config,
                            transable.abatch,
                            input,
                            current,
                            child_config,
                            return_exceptions=False,
                            **kwargs,
                        ),
                    )
                )

        if return_exceptions:
            outputs: list[PhotoType | Exception] = []
            successes = list(cast("list[PhotoType]", current or []))
            for i in range(total):
                if i in failed:
                    outputs.append(failed[i])
                else:
                    if not successes:
                        raise RuntimeError(
                            "Sequence batch did not return results for all inputs."
                        )
                    outputs.append(successes.pop(0))
            return outputs

        return current or []

    @overload
    async def abatch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[False],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType]: ...

    @overload
    async def abatch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]: ...

    @override
    async def abatch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]:
        config = ensure_config(config)

        return await self._abatch_with_config(
            self._abatch,
            input,
            receives,
            config,
            return_exceptions=return_exceptions,
            run_type="abatch",
            **kwargs,
        )

    def _validate_chain(self) -> None:
        previous = self.start
        sequence_input_type = self.start.InputType
        for transable in self.chains[1:]:
            ensure_subclass(transable.PhotoType, previous.PhotoType)
            self._ensure_input_compatibility(transable, sequence_input_type)
            previous = transable

    def _pipe_stream(
        self,
        downstream: Transable[Any, Any],
        upstream: Iterator[Any],
        input: Input,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        ensured_config = ensure_config(config)
        child_config = patch_config(ensured_config, child=True)

        def _gen() -> Iterator[Any]:
            def _coerced() -> Iterator[Any]:
                for result in upstream:
                    yield coerce_photo(result, downstream.PhotoType)

            stream_iter = run_in_context(
                child_config,
                downstream.stream,
                input,
                _coerced(),
                child_config,
                **kwargs,
            )
            yield from stream_iter

        return _gen()

    def _pipe_async_stream(
        self,
        downstream: Transable[Any, Any],
        upstream: AsyncIterator[Any],
        input: Input,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        ensured_config = ensure_config(config)
        child_config = patch_config(ensured_config, child=True)

        async def _gen() -> AsyncIterator[Any]:
            send, recv = anyio.create_memory_object_stream[Any](
                max_buffer_size=stream_buffer(ensured_config)
            )

            async def _produce() -> None:
                try:
                    async for result in upstream:
                        await send.send(coerce_photo(result, downstream.PhotoType))
                finally:
                    await send.aclose()

            async with anyio.create_task_group() as tg, recv:
                tg.start_soon(_produce)
                stream_iter = run_in_context(
                    child_config,
                    downstream.astream,
                    input,
                    recv,
                    child_config,
                    **kwargs,
                )
                async for result in stream_iter:
                    yield result

        return _gen()

    def _ensure_input_compatibility(
        self,
        transable: Transable[Any, Any],
        sequence_input_type: type[Input],
    ) -> None:
        """Ensure each step can accept the sequence input type."""
        target_input_type = transable.InputType
        if sequence_input_type is Any or target_input_type is Any:
            return
        if not (
            inspect.isclass(sequence_input_type) and inspect.isclass(target_input_type)
        ):
            return
        if not issubclass(sequence_input_type, target_input_type):
            raise TypeError(
                f"InputType mismatch for {transable.get_name()}: "
                f"{target_input_type.__name__} expected, "
                f"got {sequence_input_type.__name__}."
            )


class TransableParallel(TransableSerializable[Input, PhotoType]):
    steps: list[Transable[Input, PhotoType]] = Field(default_factory=list)

    def __init__(
        self,
        *steps: TransableLike,
        name: str | None = None,
    ) -> None:
        flat: list[Transable[Input, PhotoType]] = []
        for step in steps:
            if isinstance(step, TransableParallel):
                flat.extend(step.steps)
            else:
                flat.append(coerce_to_transable(step))
        if len(flat) < 2:
            raise ValueError(
                f"TransableParallel must have at least 2 steps, got {len(flat)}"
            )

        first_type = flat[0].PhotoType
        for step in flat[1:]:
            if step.PhotoType is not first_type:
                raise TypeError(
                    "Parallel Transables require identical PhotoType values."
                )

        super().__init__(steps=flat, name=name)  # pyright: ignore[reportCallIssue]

    @property
    @override
    def InputType(self) -> type[Input]:
        """The type of the input to the Transable."""
        return self.steps[0].InputType

    @override
    def get_input_schema(
        self, config: TransableConfig | None = None
    ) -> type[BaseModel]:
        if all(
            s.get_input_schema(config).model_json_schema().get("type", "object")
            == "object"
            for s in self.steps
        ):
            # This is correct, but pydantic typings/mypy don't think so.
            return create_model(
                self.get_name("Input"),
                field_definitions={
                    k: (v.annotation, v.default)
                    for step in self.steps
                    for k, v in step.get_input_schema(config).model_fields.items()
                    if k != "__root__"
                },
            )

        return super().get_input_schema(config)

    @override
    def __repr__(self) -> str:
        return "\n& ".join(
            repr(step) if i == 0 else indent_lines_after_first(repr(step), "& ")
            for i, step in enumerate(self.steps)
        )

    def _invoke(
        self,
        input: Input,
        receive: PhotoType | None,
        config: TransableConfig,
        **kwargs: Any,
    ) -> PhotoType:
        if receive is None:
            raise ValueError(
                "Parallel Transables require an input PhotoResult when invoked."
            )
        upstream = coerce_photo(receive, self.PhotoType)
        results: list[PhotoType | None] = [None] * len(self.steps)

        def _run(idx: int, step: Transable[Input, PhotoType]) -> None:
            child_config = patch_config(config, child=True)
            res = step.invoke(
                input,
                clone_photo(upstream),
                config=child_config,
                **kwargs,
            )
            results[idx] = coerce_photo(res, self.PhotoType)

        executor_config = patch_config(config, max_concurrency=len(self.steps) + 1)
        with get_executor_for_config(executor_config) as executor:
            executor.map(_run, range(len(self.steps)), self.steps)

        if any(res is None for res in results):
            raise RuntimeError("Parallel step did not return a PhotoResult.")

        return merge_photos(receive, [res for res in results if res is not None])

    @override
    def invoke(
        self,
        input: Input,
        receive: PhotoType | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoType:
        config = ensure_config(config)

        return self._call_with_config(
            self._invoke,
            input,
            receive,
            config,
            run_type="parallel_invoke",
            **kwargs,
        )

    async def _ainvoke(
        self,
        input: Input,
        receive: PhotoType | None,
        config: TransableConfig,
        **kwargs: Any,
    ) -> PhotoType:
        if receive is None:
            raise ValueError(
                "Parallel Transables require an input PhotoResult when invoked."
            )
        upstream = coerce_photo(receive, self.PhotoType)
        results: list[PhotoType | None] = [None] * len(self.steps)

        async def _run(idx: int, step: Transable[Input, PhotoType]) -> None:
            child_config = patch_config(config, child=True)
            res = await step.ainvoke(
                input,
                clone_photo(upstream),
                config=child_config,
                **kwargs,
            )
            results[idx] = coerce_photo(res, self.PhotoType)

        async with anyio.create_task_group() as tg:
            for idx, step in enumerate(self.steps):
                tg.start_soon(_run, idx, step)

        if any(res is None for res in results):
            raise RuntimeError("Parallel step did not return a PhotoResult.")

        return merge_photos(receive, [res for res in results if res is not None])

    @override
    async def ainvoke(
        self,
        input: Input,
        receive: PhotoType | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoType:
        config = ensure_config(config)

        return await self._acall_with_config(
            self._ainvoke,
            input,
            receive,
            config,
            run_type="parallel_ainvoke",
            **kwargs,
        )

    def _stream(
        self,
        input: Input,
        receives: Iterator[PhotoType] | None,
        config: TransableConfig,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> Iterator[PhotoType]:
        if receives is None:
            raise ValueError(
                "Parallel Transables require an input PhotoResult stream when streamed."
            )

        buffer_size = stream_buffer(config)
        sentinel = object()
        cond = threading.Condition()
        step_outputs: dict[str, list[PhotoType | None]] = defaultdict(
            lambda: [None] * len(self.steps)
        )
        base_items: dict[str, PhotoType] = {}
        pending: set[str] | None = set() if not keep_order else None
        step_done: list[bool] = [False] * len(self.steps)
        base_done = False
        errors: list[Exception] = []
        fatal_error: Exception | None = None
        arrived_count: dict[str, int] = {}
        ready_queue: deque[str] = deque()
        stop_event = threading.Event()

        base_queue: queue.Queue[object] | None = None
        if keep_order:
            base_queue = queue.Queue()

        child_queues: list[queue.Queue[object]] = [
            queue.Queue(maxsize=buffer_size) for _ in self.steps
        ]

        def _queue_iter(q: queue.Queue[object]) -> Iterator[PhotoType]:
            while True:
                if stop_event.is_set() and q.empty():
                    return
                try:
                    item = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is sentinel or stop_event.is_set():
                    return
                yield cast("PhotoType", item)

        def _put_queue(q: queue.Queue[object], item: object) -> None:
            while True:
                if stop_event.is_set():
                    return
                try:
                    q.put(item, timeout=0.1)
                except queue.Full:
                    continue
                else:
                    return

        def _broadcast() -> None:
            nonlocal base_done, fatal_error
            try:
                for item in receives:
                    if stop_event.is_set():
                        break
                    coerced = coerce_photo(item, self.PhotoType)
                    base_item = clone_photo(coerced)
                    unique_id = base_item.get_unique_id()
                    with cond:
                        if any(step_done):
                            if not ignore_exceptions:
                                fatal_error = ValueError(
                                    "A parallel step finished early; cannot continue streaming."
                                )
                                cond.notify_all()
                                break
                            return
                        base_items[unique_id] = base_item
                        arrived_count[unique_id] = 0
                        if pending is not None:
                            pending.add(unique_id)
                        cond.notify_all()
                    if base_queue is not None:
                        _put_queue(base_queue, base_item)
                    for q in child_queues:
                        _put_queue(q, clone_photo(coerced))
            except Exception as exc:
                with cond:
                    fatal_error = exc
                    cond.notify_all()
            finally:
                with cond:
                    base_done = True
                    cond.notify_all()
                if base_queue is not None:
                    _put_queue(base_queue, sentinel)
                for q in child_queues:
                    _put_queue(q, sentinel)

        def _run_step(idx: int, step: Transable[Input, PhotoType]) -> None:
            try:
                child_config = patch_config(config, child=True)
                stream_iter = run_in_context(
                    child_config,
                    step.stream,
                    input,
                    _queue_iter(child_queues[idx]),
                    child_config,
                    keep_order=keep_order,
                    ignore_exceptions=ignore_exceptions,
                    **kwargs,
                )
                for result in stream_iter:
                    coerced = coerce_photo(result, self.PhotoType)
                    unique_id = coerced.get_unique_id()
                    with cond:
                        if (
                            unique_id not in base_items
                            and unique_id not in step_outputs
                        ):
                            if ignore_exceptions:
                                continue
                            errors.append(
                                ValueError(
                                    "Parallel step produced results with no matching base item: "
                                    f"{unique_id}"
                                )
                            )
                            cond.notify_all()
                            return
                        outputs = step_outputs[unique_id]
                        if outputs[idx] is None:
                            arrived_count[unique_id] = (
                                arrived_count.get(unique_id, 0) + 1
                            )
                        outputs[idx] = clone_photo(coerced)
                        if (
                            pending is not None
                            and unique_id in pending
                            and arrived_count.get(unique_id, 0) == len(self.steps)
                        ):
                            ready_queue.append(unique_id)
                        cond.notify_all()
            except Exception as exc:
                if not ignore_exceptions:
                    with cond:
                        errors.append(exc)
                        cond.notify_all()
                    stop_event.set()
            finally:
                with cond:
                    step_done[idx] = True
                    if pending is not None and pending:
                        missing_ids = [
                            uid for uid in pending if step_outputs[uid][idx] is None
                        ]
                        if missing_ids:
                            if ignore_exceptions:
                                for unique_id in missing_ids:
                                    pending.remove(unique_id)
                                    base_items.pop(unique_id, None)
                                    step_outputs.pop(unique_id, None)
                                    arrived_count.pop(unique_id, None)
                            else:
                                errors.append(
                                    ValueError(
                                        "Parallel step results missing for unique ID(s): "
                                        f"{', '.join(missing_ids)}."
                                    )
                                )
                    cond.notify_all()

        def _collect_ordered() -> Iterator[PhotoType]:
            assert base_queue is not None
            while True:
                base_item = base_queue.get()
                if base_item is sentinel:
                    break
                base_item = cast("PhotoType", base_item)
                unique_id = base_item.get_unique_id()
                while True:
                    with cond:
                        if fatal_error is not None:
                            raise fatal_error
                        if errors and not ignore_exceptions:
                            raise errors[0]
                        outputs = step_outputs[unique_id]
                        missing = [
                            idx for idx, res in enumerate(outputs) if res is None
                        ]
                        if not missing:
                            step_outputs.pop(unique_id, None)
                            base_items.pop(unique_id, None)
                            merged = merge_photos(
                                base_item,
                                cast("list[PhotoType]", list(outputs)),
                            )
                            arrived_count.pop(unique_id, None)
                            break
                        if any(step_done[idx] for idx in missing):
                            step_outputs.pop(unique_id, None)
                            base_items.pop(unique_id, None)
                            arrived_count.pop(unique_id, None)
                            if ignore_exceptions:
                                merged = None
                                break
                            raise ValueError(
                                "Parallel step results missing for unique ID "
                                f"{unique_id}."
                            )
                        cond.wait()
                if merged is not None:
                    yield merged

            with cond:
                if fatal_error is not None:
                    raise fatal_error
                if errors and not ignore_exceptions:
                    raise errors[0]
                if step_outputs and not ignore_exceptions:
                    raise ValueError(
                        "Parallel step produced results with no matching base item: "
                        f"{', '.join(step_outputs.keys())}"
                    )

        def _collect_unordered() -> Iterator[PhotoType]:
            assert pending is not None
            while True:
                with cond:
                    if fatal_error is not None:
                        raise fatal_error
                    if errors and not ignore_exceptions:
                        raise errors[0]
                    while not ready_queue:
                        if base_done and not pending:
                            break
                        cond.wait()
                        if fatal_error is not None:
                            raise fatal_error
                        if errors and not ignore_exceptions:
                            raise errors[0]
                    if base_done and not pending and not ready_queue:
                        break
                    unique_id = ready_queue.popleft()
                    if unique_id not in pending:
                        continue
                    base_item = base_items.pop(unique_id)
                    outputs = step_outputs.pop(unique_id, None)
                    pending.remove(unique_id)
                    arrived_count.pop(unique_id, None)
                if outputs is not None:
                    outs = cast("list[PhotoType]", list(outputs))
                    yield merge_photos(base_item, outs)

            with cond:
                if fatal_error is not None:
                    raise fatal_error
                if errors and not ignore_exceptions:
                    raise errors[0]
                if step_outputs and not ignore_exceptions:
                    raise ValueError(
                        "Parallel step produced results with no matching base item: "
                        f"{', '.join(step_outputs.keys())}"
                    )

        executor_config = patch_config(config, max_concurrency=len(self.steps) + 1)
        with get_executor_for_config(executor_config) as executor:
            futures = [executor.submit(_broadcast)]
            futures.extend(
                executor.submit(_run_step, idx, step)
                for idx, step in enumerate(self.steps)
            )
            try:
                if keep_order:
                    yield from _collect_ordered()
                else:
                    yield from _collect_unordered()
            finally:
                stop_event.set()
                if base_queue is not None:
                    _put_queue(base_queue, sentinel)
                for q in child_queues:
                    _put_queue(q, sentinel)
                wait(futures)

    @override
    def stream(
        self,
        input: Input,
        receives: Iterator[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> Iterator[PhotoType]:
        config = ensure_config(config)

        yield from self._stream_with_config(
            self._stream,
            input,
            receives,
            config,
            keep_order=keep_order,
            ignore_exceptions=ignore_exceptions,
            run_type="parallel_stream",
            **kwargs,
        )

    async def _astream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoType] | None,
        config: TransableConfig,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoType]:
        if receives is None:
            raise ValueError(
                "Parallel Transables require an input PhotoResult stream when streamed."
            )

        buffer_size = stream_buffer(config)
        base_items: dict[str, PhotoType] = {}
        pending: set[str] | None = set() if not keep_order else None
        base_done = False
        step_outputs: dict[str, list[PhotoType | None]] = defaultdict(
            lambda: [None] * len(self.steps)
        )

        arrived_count: dict[str, int] = {}
        ready_queue: deque[str] = deque()

        base_send: ObjectSendStream[PhotoType] | None = None
        base_recv: ObjectReceiveStream[PhotoType] | None = None
        if keep_order:
            base_send, base_recv = anyio.create_memory_object_stream[PhotoType](
                max_buffer_size=buffer_size
            )

        child_sends: list[ObjectSendStream[PhotoType]] = []
        child_recvs: list[ObjectReceiveStream[PhotoType]] = []
        for _ in self.steps:
            s, r = anyio.create_memory_object_stream[PhotoType](
                max_buffer_size=buffer_size
            )
            child_sends.append(s)
            child_recvs.append(r)

        output_cond: anyio.Condition = anyio.Condition()
        step_done: list[bool] = [False] * len(self.steps)

        async def _broadcast() -> None:
            nonlocal base_done
            active_child_sends: list[ObjectSendStream[PhotoType]] = list(child_sends)
            try:
                async for item in receives:
                    coerced = coerce_photo(item, self.PhotoType)
                    base_item = clone_photo(coerced)
                    unique_id = base_item.get_unique_id()
                    async with output_cond:
                        if any(step_done):
                            if not ignore_exceptions:
                                raise ValueError(
                                    "A parallel step finished early; cannot continue streaming."
                                )
                            return
                        base_items[unique_id] = base_item
                        arrived_count[unique_id] = 0
                        if pending is not None:
                            pending.add(unique_id)
                        output_cond.notify_all()
                    if base_send is not None:
                        await base_send.send(base_item)

                    to_remove: list[ObjectSendStream[PhotoType]] = []
                    for send in list(active_child_sends):
                        try:
                            await send.send(clone_photo(coerced))
                        except (BrokenResourceError, ClosedResourceError):
                            if not ignore_exceptions:
                                raise
                            to_remove.append(send)
                    if to_remove:
                        active_child_sends = [
                            send for send in active_child_sends if send not in to_remove
                        ]
            finally:
                if base_send is not None:
                    await base_send.aclose()
                for send in child_sends:
                    await send.aclose()
                async with output_cond:
                    base_done = True
                    output_cond.notify_all()

        async def _run_step(idx: int, step: Transable[Input, PhotoType]) -> None:
            try:
                async with child_recvs[idx]:
                    child_config = patch_config(config, child=True)
                    stream_iter = run_in_context(
                        child_config,
                        step.astream,
                        input,
                        child_recvs[idx],
                        child_config,
                        keep_order=keep_order,
                        ignore_exceptions=ignore_exceptions,
                        **kwargs,
                    )
                    async for result in stream_iter:
                        coerced = coerce_photo(result, self.PhotoType)
                        unique_id = coerced.get_unique_id()
                        async with output_cond:
                            if (
                                unique_id not in base_items
                                and unique_id not in step_outputs
                            ):
                                if ignore_exceptions:
                                    continue
                                raise ValueError(
                                    "Parallel step produced results with no matching base item: "
                                    f"{unique_id}"
                                )
                            outputs = step_outputs[unique_id]
                            if outputs[idx] is None:
                                arrived_count[unique_id] = (
                                    arrived_count.get(unique_id, 0) + 1
                                )
                            outputs[idx] = clone_photo(coerced)
                            if (
                                pending is not None
                                and unique_id in pending
                                and arrived_count.get(unique_id, 0) == len(self.steps)
                            ):
                                ready_queue.append(unique_id)
                            output_cond.notify_all()
            except Exception:
                if not ignore_exceptions:
                    raise
            finally:
                async with output_cond:
                    step_done[idx] = True

                    if pending is not None and pending:
                        missing_ids = [
                            uid for uid in pending if step_outputs[uid][idx] is None
                        ]
                        if missing_ids:
                            if ignore_exceptions:
                                for unique_id in missing_ids:
                                    pending.remove(unique_id)
                                    base_items.pop(unique_id, None)
                                    step_outputs.pop(unique_id, None)
                                    arrived_count.pop(unique_id, None)
                            else:
                                raise ValueError(
                                    "Parallel step results missing for unique ID(s): "
                                    f"{', '.join(missing_ids)}."
                                )

                    output_cond.notify_all()

        async def _collect_outputs_ordered() -> AsyncIterator[PhotoType]:
            assert base_recv is not None
            async with base_recv:
                async for base_item in base_recv:
                    unique_id = base_item.get_unique_id()

                    while True:
                        async with output_cond:
                            outputs = step_outputs[unique_id]
                            missing = [
                                i for i, res in enumerate(outputs) if res is None
                            ]
                            if not missing:
                                outs = cast("list[PhotoType]", list(outputs))
                                step_outputs.pop(unique_id, None)
                                base_items.pop(unique_id, None)
                                arrived_count.pop(unique_id, None)
                                break
                            if any(step_done[i] for i in missing):
                                step_outputs.pop(unique_id, None)
                                base_items.pop(unique_id, None)
                                arrived_count.pop(unique_id, None)
                                if ignore_exceptions:
                                    outs = None
                                    break
                                raise ValueError(
                                    "Parallel step results missing for unique ID "
                                    f"{unique_id}."
                                )
                            await output_cond.wait()
                    if outs is not None:
                        yield merge_photos(base_item, outs)
                async with output_cond:
                    if step_outputs and not ignore_exceptions:
                        raise ValueError(
                            "Parallel step produced results with no matching base item: "
                            f"{', '.join(step_outputs.keys())}"
                        )

        async def _collect_outputs_unordered() -> AsyncIterator[PhotoType]:
            assert pending is not None
            while True:
                async with output_cond:
                    while not ready_queue:
                        if base_done and not pending:
                            break
                        await output_cond.wait()
                    if base_done and not pending and not ready_queue:
                        break
                    uid = ready_queue.popleft()
                    if uid not in pending:
                        continue
                    base_item = base_items.pop(uid)
                    outputs = step_outputs.pop(uid, None)
                    pending.remove(uid)
                    arrived_count.pop(uid, None)
                if outputs is not None:
                    outs = cast("list[PhotoType]", list(outputs))
                    yield merge_photos(base_item, outs)

            async with output_cond:
                if step_outputs and not ignore_exceptions:
                    raise ValueError(
                        "Parallel step produced results with no matching base item: "
                        f"{', '.join(step_outputs.keys())}"
                    )

        async with anyio.create_task_group() as tg:
            tg.start_soon(_broadcast)
            for idx, step in enumerate(self.steps):
                tg.start_soon(_run_step, idx, step)
            if keep_order:
                async for output in _collect_outputs_ordered():
                    yield output
            else:
                async for output in _collect_outputs_unordered():
                    yield output

    @override
    async def astream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoType]:
        config = ensure_config(config)

        async for result in self._astream_with_config(
            self._astream,
            input,
            receives,
            config,
            keep_order=keep_order,
            ignore_exceptions=ignore_exceptions,
            run_type="parallel_astream",
            **kwargs,
        ):
            yield cast("PhotoType", result)

    def _batch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]:
        if receives is None:
            raise ValueError(
                "Parallel Transables require an input PhotoResult batch when batched."
            )
        if not receives:
            return []

        base_items = [coerce_photo(item, self.PhotoType) for item in receives]
        step_results: list[Sequence[PhotoType | Exception] | None] = [None] * len(
            self.steps
        )

        def _run_step(idx: int, step: Transable[Input, PhotoType]) -> None:
            child_config = patch_config(config, child=True)
            step_inputs = [clone_photo(item) for item in base_items]
            try:
                results = run_in_context(
                    child_config,
                    step.batch,
                    input,
                    step_inputs,
                    child_config,
                    return_exceptions=True,
                    **kwargs,
                )
            except Exception as exc:
                if not return_exceptions:
                    raise
                step_results[idx] = [exc] * len(base_items)
                return

            coerced_results: list[PhotoType | Exception] = []
            for res in results:
                if isinstance(res, Exception):
                    coerced_results.append(res)
                else:
                    coerced_results.append(coerce_photo(res, self.PhotoType))
            step_results[idx] = coerced_results

        executor_config = patch_config(config, max_concurrency=len(self.steps) + 1)
        with get_executor_for_config(executor_config) as executor:
            executor.map(_run_step, range(len(self.steps)), self.steps)

        outputs: list[PhotoType | Exception] = []
        for i, base_item in enumerate(base_items):
            item_results = [
                step_result[i] for step_result in step_results if step_result
            ]
            if excs := [res for res in item_results if isinstance(res, Exception)]:
                exc = (
                    ExceptionGroup(
                        f"Errors occurred in parallel Transable steps for photo {i}: {base_item}.",
                        excs,
                    )
                    if len(excs) > 1
                    else excs[0]
                )
                outputs.append(exc)
                continue
            results = cast("list[PhotoType]", item_results)
            outputs.append(merge_photos(base_item, results))

        if (not return_exceptions) and (
            excs := [res for res in outputs if isinstance(res, Exception)]
        ):
            raise (
                ExceptionGroup(
                    "Errors occurred in parallel Transable steps.",
                    excs,
                )
                if len(excs) > 1
                else excs[0]
            )
        return outputs

    @overload
    def batch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[False],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType]: ...

    @overload
    def batch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]: ...

    @override
    def batch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]:
        config = ensure_config(config)

        return self._batch_with_config(
            self._batch,
            input,
            receives,
            config,
            return_exceptions=return_exceptions,
            run_type="parallel_batch",
            **kwargs,
        )

    async def _abatch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]:
        if receives is None:
            raise ValueError(
                "Parallel Transables require an input PhotoResult batch when batched."
            )
        if not receives:
            return []

        base_items = [coerce_photo(item, self.PhotoType) for item in receives]
        step_results: list[Sequence[PhotoType | Exception] | None] = [None] * len(
            self.steps
        )

        async def _run_step(idx: int, step: Transable[Input, PhotoType]) -> None:
            child_config = patch_config(config, child=True)
            step_inputs = [clone_photo(item) for item in base_items]
            try:
                results = await arun_in_context(
                    child_config,
                    step.abatch,
                    input,
                    step_inputs,
                    child_config,
                    return_exceptions=True,
                    **kwargs,
                )
            except Exception as exc:
                if not return_exceptions:
                    raise
                step_results[idx] = [exc] * len(base_items)
                return

            coerced_results: list[PhotoType | Exception] = []
            for res in results:
                if isinstance(res, Exception):
                    coerced_results.append(res)
                else:
                    coerced_results.append(coerce_photo(res, self.PhotoType))
            step_results[idx] = coerced_results

        async with anyio.create_task_group() as tg:
            for idx, step in enumerate(self.steps):
                tg.start_soon(_run_step, idx, step)

        outputs: list[PhotoType | Exception] = []
        for i, base_item in enumerate(base_items):
            item_results = [
                step_result[i] for step_result in step_results if step_result
            ]
            if excs := [res for res in item_results if isinstance(res, Exception)]:
                exc = (
                    ExceptionGroup(
                        f"Errors occurred in parallel Transable steps for photo {i}: {base_item}.",
                        excs,
                    )
                    if len(excs) > 1
                    else excs[0]
                )
                outputs.append(exc)
                continue
            results = cast("list[PhotoType]", item_results)
            outputs.append(merge_photos(base_item, results))

        if (not return_exceptions) and (
            excs := [res for res in outputs if isinstance(res, Exception)]
        ):
            raise (
                ExceptionGroup(
                    "Errors occurred in parallel Transable steps.",
                    excs,
                )
                if len(excs) > 1
                else excs[0]
            )
        return outputs

    @overload
    async def abatch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[False],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType]: ...

    @overload
    async def abatch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]: ...

    @override
    async def abatch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]:
        config = ensure_config(config)

        return await self._abatch_with_config(
            self._abatch,
            input,
            receives,
            config,
            return_exceptions=return_exceptions,
            run_type="parallel_abatch",
            **kwargs,
        )

    @override
    def __and__(
        self,
        other: Transable[Any, Any]
        | Callable[[Any], Any]
        | Callable[[Any], Awaitable[Any]]
        | Callable[[Any, Iterator[Any]], Iterator[Any]]
        | Callable[[Any, AsyncIterator[Any]], AsyncIterator[Any]],
    ) -> TransableSerializable[Input, PhotoType]:
        if isinstance(other, TransableParallel):
            return TransableParallel(
                *(self.steps + other.steps),
                name=self.name or other.name,
            )
        return TransableParallel(
            *self.steps,
            coerce_to_transable(other),
            name=self.name,
        )

    @override
    def __rand__(
        self,
        other: Transable[Any, Any]
        | Callable[[Any], Any]
        | Callable[[Any], Awaitable[Any]]
        | Callable[[Any, Iterator[Any]], Iterator[Any]]
        | Callable[[Any, AsyncIterator[Any]], AsyncIterator[Any]],
    ) -> TransableSerializable[Input, PhotoType]:
        if isinstance(other, TransableParallel):
            return TransableParallel(
                *(other.steps + self.steps),
                name=other.name or self.name,
            )
        return TransableParallel(
            coerce_to_transable(other),
            *self.steps,
            name=self.name,
        )


# Incomplete revisions
class TransableLambda(Transable[Input, PhotoType]):  # noqa: PLW1641
    """Transable defined from a pair of functions."""

    def __init__(
        self,
        func: (
            Callable[[Input], PhotoType]
            | Callable[[Input], Transable[Input, PhotoType]]
            | Callable[[Input, TransableConfig], PhotoType]
            | Callable[[Input, PhotoType], PhotoType]
            | Callable[[Input, PhotoType, TransableConfig], PhotoType]
        )
        | (
            Callable[[Input], Awaitable[PhotoType]]
            | Callable[[Input, TransableConfig], Awaitable[PhotoType]]
            | Callable[[Input, PhotoType], Awaitable[PhotoType]]
            | Callable[[Input, PhotoType, TransableConfig], Awaitable[PhotoType]]
        )
        | (
            Callable[[Input, Iterator[PhotoType]], Iterator[PhotoType]]
            | Callable[
                [Input, Iterator[PhotoType], TransableConfig], Iterator[PhotoType]
            ]
            | Callable[[Input, AsyncIterator[PhotoType]], AsyncIterator[PhotoType]]
            | Callable[
                [Input, AsyncIterator[PhotoType], TransableConfig],
                AsyncIterator[PhotoType],
            ]
        ),
        afunc: (
            Callable[[Input], Awaitable[PhotoType]]
            | Callable[[Input, TransableConfig], Awaitable[PhotoType]]
            | Callable[[Input, PhotoType], Awaitable[PhotoType]]
            | Callable[[Input, PhotoType, TransableConfig], Awaitable[PhotoType]]
        )
        | (
            Callable[[Input, AsyncIterator[PhotoType]], AsyncIterator[PhotoType]]
            | Callable[
                [Input, AsyncIterator[PhotoType], TransableConfig],
                AsyncIterator[PhotoType],
            ]
        )
        | None = None,
        stream_func: (
            Callable[[Input, Iterator[PhotoType]], Iterator[PhotoType]]
            | Callable[
                [Input, Iterator[PhotoType], TransableConfig], Iterator[PhotoType]
            ]
            | Callable[[Input, AsyncIterator[PhotoType]], AsyncIterator[PhotoType]]
            | Callable[
                [Input, AsyncIterator[PhotoType], TransableConfig],
                AsyncIterator[PhotoType],
            ]
        )
        | None = None,
        astream_func: (
            Callable[[Input, AsyncIterator[PhotoType]], AsyncIterator[PhotoType]]
            | Callable[
                [Input, AsyncIterator[PhotoType], TransableConfig],
                AsyncIterator[PhotoType],
            ]
        )
        | None = None,
        name: str | None = None,
    ) -> None:
        """"""
        func_for_name = func

        if afunc is not None:
            if is_async_callable(func) or is_async_generator(func):
                raise TypeError(
                    "Func was provided as a coroutine function, but afunc was "
                    "also provided. If providing both, func should be a regular "
                    "function to avoid ambiguity."
                )
            self.afunc = afunc
            func_for_name = afunc

        if astream_func is not None:
            self.astream_func = astream_func
            func_for_name = astream_func

        if stream_func is not None:
            if is_generator(stream_func):
                self.stream_func = stream_func
                func_for_name = stream_func
            elif is_async_generator(stream_func):
                if astream_func is not None:
                    raise ValueError(
                        "stream_func was provided as an async generator, but "
                        "astream_func was also provided. Only one of stream_func, "
                        "astream_func should be provided in this case."
                    )
                self.astream_func = stream_func
                func_for_name = stream_func
            else:
                raise TypeError("stream_func must be a generator or async generator.")

        if is_generator(func):
            if stream_func is not None:
                raise ValueError(
                    "Func was provided along with stream_func, but "
                    "func is a generator. Only one of func, "
                    "stream_func should be provided in this case."
                )
            self.func = func
            self.stream_func = func
            func_for_name = func
        elif is_async_generator(func):
            if astream_func is not None:
                raise ValueError(
                    "Func was provided along with astream_func, but "
                    "func is an async generator. Only one of func, "
                    "astream_func should be provided in this case."
                )
            self.afunc = func
            self.astream_func = func
            func_for_name = func
        elif is_async_callable(func):
            self.afunc = func
            func_for_name = func
        elif callable(func):
            self.func = func
            func_for_name = func
        else:
            raise TypeError("func must be a callable, generator, or async generator.")

        if (
            accepts_receives(func)
            and not hasattr(self, "stream_func")
            and not hasattr(self, "astream_func")
        ):
            if is_async_callable(func) or is_async_generator(func):
                self.astream_func = func
            else:
                self.stream_func = func
        if (
            hasattr(self, "afunc")
            and not hasattr(self, "stream_func")
            and not hasattr(self, "astream_func")
        ):
            afunc_ref = self.afunc
            if (
                accepts_receives(afunc_ref)
                and not accepts_receive(afunc_ref)
                or is_async_generator(afunc_ref)
            ):
                self.astream_func = afunc_ref
            elif is_async_callable(afunc_ref):
                try:
                    return_annotation = inspect.signature(afunc_ref).return_annotation
                except ValueError:
                    return_annotation = inspect.Signature.empty
                if return_annotation is not inspect.Signature.empty:
                    origin = getattr(return_annotation, "__origin__", None)
                    if return_annotation is AsyncIterator or origin is AsyncIterator:
                        self.astream_func = afunc_ref

        try:
            if name is not None:
                self.name = name
            elif func_for_name.__name__ != "<lambda>":
                self.name = func_for_name.__name__
        except AttributeError:
            pass

        self._repr: str | None = None

    @property
    @override
    def InputType(self) -> Any:
        """The type of the input to this Transable."""
        func = (
            getattr(self, "func", None)
            or getattr(self, "afunc", None)
            or getattr(self, "stream_func", None)
            or self.astream_func
        )
        try:
            params = inspect.signature(func).parameters
            first_param = next(iter(params.values()), None)
            if first_param and first_param.annotation != inspect.Parameter.empty:
                return first_param.annotation
        except ValueError:
            pass
        return Any

    @property
    @override
    def PhotoType(self) -> Any:
        """The type of the output of this Transable as a type annotation.

        Returns:
            The type of the output of this Transable.
        """
        func = (
            getattr(self, "func", None)
            or getattr(self, "afunc", None)
            or getattr(self, "stream_func", None)
            or self.astream_func
        )
        try:
            sig = inspect.signature(func)
            if sig.return_annotation != inspect.Signature.empty:
                # unwrap iterator types
                if getattr(sig.return_annotation, "__origin__", None) in (
                    Iterator,
                    AsyncIterator,
                ):
                    return getattr(sig.return_annotation, "__args__", (Any,))[0]
                return sig.return_annotation
        except ValueError:
            pass
        return Any

    @override
    def get_input_schema(
        self, config: TransableConfig | None = None
    ) -> type[BaseModel]:
        """The pydantic schema for the input to this Runnable.

        Args:
            config: The config to use. Defaults to None.

        Returns:
            The input schema for this Runnable.
        """
        func = (
            getattr(self, "func", None)
            or getattr(self, "afunc", None)
            or getattr(self, "stream_func", None)
            or self.astream_func
        )

        if isinstance(func, itemgetter):
            # This is terrible, but afaict it's not possible to access _items
            # on itemgetter objects, so we have to parse the repr
            items = str(func).replace("operator.itemgetter(", "")[:-1].split(", ")
            if all(
                item[0] == "'" and item[-1] == "'" and len(item) > 2 for item in items
            ):
                fields = {item[1:-1]: (Any, ...) for item in items}
                # It's a dict, lol
                return create_model(self.get_name("Input"), field_definitions=fields)
            module = getattr(func, "__module__", None)
            return create_model(
                self.get_name("Input"),
                root=list[Any],
                # To create the schema, we need to provide the module
                # where the underlying function is defined.
                # This allows pydantic to resolve type annotations appropriately.
                module_name=module,
            )

        if self.InputType != Any:
            return super().get_input_schema(config)

        if dict_keys := get_function_first_arg_dict_keys(func):
            return create_model(
                self.get_name("Input"),
                field_definitions=dict.fromkeys(dict_keys, (Any, ...)),
            )

        return super().get_input_schema(config)

    @override
    def __repr__(self) -> str:
        if self._repr is None:
            if hasattr(self, "func"):
                self._repr = f"TransableLambda({get_lambda_source(self.func) or '...'})"
            elif hasattr(self, "afunc"):
                self._repr = (
                    f"TransableLambda(afunc={get_lambda_source(self.afunc) or '...'})"
                )
            elif hasattr(self, "stream_func"):
                self._repr = (
                    "TransableLambda(stream_func="
                    f"{get_lambda_source(self.stream_func) or '...'})"
                )
            elif hasattr(self, "astream_func"):
                self._repr = (
                    "TransableLambda(astream_func="
                    f"{get_lambda_source(self.astream_func) or '...'})"
                )
            else:
                self._repr = "TransableLambda(...)"
        return self._repr

    def _merge_results(
        self,
        receive: PhotoType | None,
        results: list[PhotoType],
    ) -> PhotoType:
        if not results:
            if receive is not None:
                return coerce_photo(receive, self.PhotoType)
            raise ValueError(
                "TransableLambda generator produced no results and no prior receive value."
            )
        try:
            base = (
                coerce_photo(receive, self.PhotoType)
                if receive is not None
                else results[0]
            )
            return merge_photos(base, results)
        except Exception:
            return results[-1]

    def _invoke(
        self,
        input: Input,
        receive: PhotoType | None,
        config: TransableConfig,
        **kwargs: Any,
    ) -> PhotoType:
        if inspect.isgeneratorfunction(self.func):
            receives_iter: Iterator[PhotoType] = (
                iter((coerce_photo(receive, self.PhotoType),)) if receive else iter([])
            )
            output: PhotoType | None = None
            for chunk in call_func_with_variable_args(
                cast("Callable[[Input], Iterator[PhotoType]]", self.func),
                input,
                receives_iter,
                config,
                **kwargs,
            ):
                if output is None:
                    output = chunk
                else:
                    try:
                        output = merge_photos(receive or output, [output, chunk])
                    except (ValueError, TypeError):
                        output = chunk
        else:
            output = call_func_with_variable_args(
                cast("Callable[[Input], Any]", self.func),
                input,
                receive,
                config,
                **kwargs,
            )
        if isinstance(output, Transable):
            recursion_limit = config["recursion_limit"]  # type: ignore
            if recursion_limit <= 0:
                raise RecursionError(
                    f"Recursion limit reached when invoking {self} with input {input}."
                )
            child_config = patch_config(config, child=True)
            output = output.invoke(
                input,
                receive,
                patch_config(child_config, recursion_limit=recursion_limit - 1),
                **kwargs,
            )
        if isinstance(output, AsyncIterator):
            raise TypeError(
                "Cannot invoke an async iterator synchronously. Use `ainvoke` instead."
            )
        if isinstance(output, Iterator):
            results = [
                coerce_photo(res, self.PhotoType)
                for res in cast("Iterator[Any]", output)
                if res is not None
            ]
            if not results:
                raise ValueError(
                    "TransableLambda generator produced no results and no prior receive value."
                )
            if receive is not None:
                output = merge_photos(receive, results)
            else:
                _output: PhotoType | None = None
                for res in results:
                    if _output is None:
                        _output = res
                    else:
                        try:
                            _output = merge_photos(_output, [_output, res])
                        except (ValueError, TypeError):
                            _output = res
                output = _output
        return cast("PhotoType", output)

    @override
    def invoke(
        self,
        input: Input,
        receive: PhotoType | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoType:
        config = ensure_config(config)
        if hasattr(self, "func"):
            return self._call_with_config(
                self._invoke,
                input,
                receive,
                config,
                run_type="invoke",
                **kwargs,
            )
        raise TypeError(
            "Cannot invoke a coroutine function synchronously. Use `ainvoke` instead."
        )

    async def _ainvoke(
        self,
        input: Input,
        receive: PhotoType | None,
        config: TransableConfig,
        **kwargs: Any,
    ) -> PhotoType:
        if hasattr(self, "afunc"):
            afunc = self.afunc
        else:

            @wraps(self._invoke)
            async def f(*args: Any, **kwargs: Any) -> Any:
                return await run_in_executor(config, self._invoke, *args, **kwargs)

            afunc = f
        if inspect.isasyncgenfunction(afunc):

            async def _receives_iter() -> AsyncIterator[PhotoType]:
                if receive is not None:
                    yield coerce_photo(receive, self.PhotoType)

            output: PhotoType | None = None
            async for chunk in call_func_with_variable_args(
                cast("Callable[[Input], AsyncIterator[PhotoType]]", afunc),
                input,
                _receives_iter(),
                config,
                **kwargs,
            ):
                if output is None:
                    output = chunk
                else:
                    try:
                        output = merge_photos(receive or output, [output, chunk])
                    except (ValueError, TypeError):
                        output = chunk
        else:
            output = await acall_func_with_variable_args(
                cast("Callable[[Input], Awaitable[Any]]", afunc),
                input,
                receive,
                config,
                **kwargs,
            )
        if isinstance(output, Transable):
            recursion_limit = config["recursion_limit"]  # type: ignore
            if recursion_limit <= 0:
                raise RecursionError(
                    f"Recursion limit reached when invoking {self} with input {input}."
                )
            child_config = patch_config(config, child=True)
            output = await output.ainvoke(
                input,
                receive,
                patch_config(child_config, recursion_limit=recursion_limit - 1),
                **kwargs,
            )
        results: list[PhotoType] | None = None
        if isinstance(output, AsyncIterator):
            results = [
                coerce_photo(res, self.PhotoType)
                async for res in cast("AsyncIterator[Any]", output)
                if res is not None
            ]
        elif isinstance(output, Iterator):
            results = [
                coerce_photo(res, self.PhotoType)
                for res in cast("Iterator[Any]", output)
                if res is not None
            ]
        if results is not None:
            if not results:
                raise ValueError(
                    "TransableLambda generator produced no results and no prior receive value."
                )
            if receive is not None:
                output = merge_photos(receive, results)
            else:
                _output: PhotoType | None = None
                for res in results:
                    if _output is None:
                        _output = res
                    else:
                        try:
                            _output = merge_photos(_output, [_output, res])
                        except (ValueError, TypeError):
                            _output = res
                output = _output
        return cast("PhotoType", output)

    @override
    async def ainvoke(
        self,
        input: Input,
        receive: PhotoType | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoType:
        config = ensure_config(config)
        return await self._acall_with_config(
            self._ainvoke,
            input,
            receive,
            config,
            run_type="ainvoke",
            **kwargs,
        )

    def _stream(
        self,
        input: Input,
        receives: Iterator[PhotoType] | None,
        config: TransableConfig,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> Iterator[PhotoType]:
        if not hasattr(self, "stream_func"):
            child_config = patch_config(config, child=True)
            yield from super().stream(
                input,
                receives,
                child_config,
                keep_order=keep_order,
                ignore_exceptions=ignore_exceptions,
                **kwargs,
            )
            return

        receives_iter: Iterator[PhotoType] = receives or iter([])

        if is_generator(self.stream_func):
            stream_output = call_func_with_variable_args(
                self.stream_func,
                input,
                receives_iter,
                config,
                **kwargs,
            )
            if isinstance(stream_output, AsyncIterator):
                raise TypeError(
                    "Cannot stream an async iterator synchronously. Use `astream` instead."
                )
            if not isinstance(stream_output, Iterator):
                raise TypeError("stream_func must return an iterator.")

            for res in stream_output:
                if res is None:
                    continue
                yield coerce_photo(res, self.PhotoType)
        else:
            raise TypeError("stream_func must be a generator.")

    @override
    def stream(
        self,
        input: Input,
        receives: Iterator[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> Iterator[PhotoType]:
        config = ensure_config(config)

        yield from self._stream_with_config(
            self._stream,
            input,
            receives,
            config,
            keep_order=keep_order,
            ignore_exceptions=ignore_exceptions,
            run_type="stream",
            **kwargs,
        )

    async def _astream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoType] | None,
        config: TransableConfig,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoType]:
        stream_func: Callable[..., Any] | None = None
        if hasattr(self, "astream_func"):
            stream_func = self.astream_func
        elif hasattr(self, "stream_func"):
            stream_func = self.stream_func

        if stream_func is None:
            child_config = patch_config(config, child=True)
            async for item in super().astream(
                input,
                receives,
                child_config,
                keep_order=keep_order,
                ignore_exceptions=ignore_exceptions,
                **kwargs,
            ):
                yield item
            return

        async def _empty_stream() -> AsyncIterator[PhotoType]:
            if False:
                yield None

        receives_for_call: AsyncIterator[PhotoType] = receives or _empty_stream()

        if is_async_generator(stream_func):
            async for res in call_func_with_variable_args(
                cast(
                    "Callable[[Input, AsyncIterator[PhotoType]], AsyncIterator[PhotoType]]",
                    stream_func,
                ),
                input,
                receives_for_call,
                config,
                **kwargs,
            ):
                if res is None:
                    continue
                yield coerce_photo(res, self.PhotoType)
        elif is_async_callable(stream_func):
            output = await acall_func_with_variable_args(
                cast(
                    "Callable[[Input, AsyncIterator[PhotoType]], Awaitable[Any]]",
                    stream_func,
                ),
                input,
                receives_for_call,
                config,
                **kwargs,
            )
            if isinstance(output, AsyncIterator):
                async for res in output:
                    if res is None:
                        continue
                    yield coerce_photo(res, self.PhotoType)
            else:
                raise TypeError("stream_func must return an async iterator.")
        else:
            recv_queue: queue.Queue[tuple[str, PhotoType] | tuple[str, object]] = (
                queue.Queue()
            )
            out_queue: queue.Queue[tuple[str, object]] = queue.Queue()
            recv_sentinel = object()

            def _iter_receives() -> Iterator[PhotoType]:
                while True:
                    kind, payload = recv_queue.get()
                    if kind == "done":
                        break
                    yield cast("PhotoType", payload)

            def _run_stream() -> None:
                try:
                    stream_iter = call_func_with_variable_args(
                        cast(
                            "Callable[[Input, Iterator[PhotoType]], Iterator[PhotoType]]",
                            stream_func,
                        ),
                        input,
                        _iter_receives(),
                        config,
                        **kwargs,
                    )
                    if isinstance(stream_iter, AsyncIterator):
                        out_queue.put(
                            (
                                "error",
                                TypeError(
                                    "Cannot stream an async iterator synchronously. "
                                    "Use `astream` instead."
                                ),
                            )
                        )
                        return
                    if not isinstance(stream_iter, Iterator):
                        out_queue.put(
                            ("error", TypeError("stream_func must return an iterator."))
                        )
                        return
                    for res in stream_iter:
                        out_queue.put(("item", res))
                except Exception as exc:
                    out_queue.put(("error", exc))
                finally:
                    out_queue.put(("done", None))

            ctx = copy_context()
            thread = threading.Thread(target=lambda: ctx.run(_run_stream), daemon=True)
            thread.start()

            async def _feed_receives() -> None:
                try:
                    if receives_for_call is not None:
                        async for item in receives_for_call:
                            recv_queue.put(("item", item))
                finally:
                    recv_queue.put(("done", recv_sentinel))

            async with anyio.create_task_group() as tg:
                tg.start_soon(_feed_receives)
                while True:
                    kind, payload = await run_in_executor(config, out_queue.get)
                    if kind == "done":
                        return
                    if kind == "error":
                        raise cast("Exception", payload)
                    if payload is None:
                        continue
                    yield coerce_photo(cast("PhotoType", payload), self.PhotoType)

    @override
    async def astream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoType]:
        config = ensure_config(config)

        async for item in self._astream_with_config(
            self._astream,
            input,
            receives,
            config,
            keep_order=keep_order,
            ignore_exceptions=ignore_exceptions,
            run_type="astream",
            **kwargs,
        ):
            yield item

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, TransableLambda):
            if hasattr(self, "func") and hasattr(other, "func"):
                return self.func == other.func
            if hasattr(self, "afunc") and hasattr(other, "afunc"):
                return self.afunc == other.afunc
            if hasattr(self, "stream_func") and hasattr(other, "stream_func"):
                return self.stream_func == other.stream_func
            if hasattr(self, "astream_func") and hasattr(other, "astream_func"):
                return self.astream_func == other.astream_func
            return False
        return False


class TransableBinding(TransableSerializable[Input, PhotoType]):
    """Transable that delegates to another Transable with bound kwargs/config."""

    bound: Transable[Input, PhotoType]
    kwargs: Mapping[str, Any] = Field(default_factory=dict)
    config: TransableConfig = Field(default_factory=dict)  # type: ignore[arg-type]
    config_factories: list[Callable[[TransableConfig], TransableConfig]] = Field(
        default_factory=list
    )
    custom_input_type: type[Input] | None = None
    custom_photo_type: type[PhotoType] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        *,
        bound: Transable[Input, PhotoType],
        kwargs: Mapping[str, Any] | None = None,
        config: TransableConfig | None = None,
        config_factories: Sequence[Callable[[TransableConfig], TransableConfig]]
        | None = None,
        custom_input_type: type[Input] | None = None,
        custom_photo_type: type[PhotoType] | None = None,
        **other_kwargs: Any,
    ) -> None:
        params = {
            "bound": bound,
            "kwargs": kwargs or {},
            "config": config or {},
            "config_factories": list(config_factories or []),
            "custom_input_type": custom_input_type,
            "custom_photo_type": custom_photo_type,
        }
        super().__init__(
            **params,
            **other_kwargs,
        )
        self.config = config or {}

    @override
    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        return self.bound.get_name(suffix, name=name)

    @property
    @override
    def InputType(self) -> type[Input]:
        return (
            cast("type[Input]", self.custom_input_type)
            if self.custom_input_type is not None
            else self.bound.InputType
        )

    @property
    @override
    def PhotoType(self) -> type[PhotoType]:
        return (
            cast("type[PhotoType]", self.custom_photo_type)
            if self.custom_photo_type is not None
            else self.bound.PhotoType
        )

    @override
    def get_input_schema(
        self, config: TransableConfig | None = None
    ) -> type[BaseModel]:
        if self.custom_input_type is not None:
            return super().get_input_schema(config)
        return self.bound.get_input_schema(merge_configs(self.config, config))

    def _merge_configs(self, *configs: TransableConfig | None) -> TransableConfig:
        config = merge_configs(self.config, *configs)
        return merge_configs(
            config,
            *(factory(config) for factory in self.config_factories),
        )

    @override
    def invoke(
        self,
        input: Input,
        receive: PhotoType | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoType:
        merged_config = self._merge_configs(config)
        return self.bound.invoke(
            input,
            receive,
            merged_config,
            **{**self.kwargs, **kwargs},
        )

    @override
    async def ainvoke(
        self,
        input: Input,
        receive: PhotoType | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoType:
        merged_config = self._merge_configs(config)
        return await self.bound.ainvoke(
            input,
            receive,
            merged_config,
            **{**self.kwargs, **kwargs},
        )

    @override
    def stream(
        self,
        input: Input,
        receives: Iterator[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> Iterator[PhotoType]:
        merged_config = self._merge_configs(config)

        yield from self.bound.stream(
            input,
            receives,
            merged_config,
            keep_order=keep_order,
            ignore_exceptions=ignore_exceptions,
            **{**self.kwargs, **kwargs},
        )

    @override
    async def astream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        keep_order: bool = False,
        ignore_exceptions: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoType]:
        merged_config = self._merge_configs(config)

        async for item in self.bound.astream(
            input,
            receives,
            merged_config,
            keep_order=keep_order,
            ignore_exceptions=ignore_exceptions,
            **{**self.kwargs, **kwargs},
        ):
            yield item

    @overload
    def batch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[False],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType]: ...

    @overload
    def batch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]: ...

    @override
    def batch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]:
        merged_config = self._merge_configs(config)
        return self.bound.batch(
            input,
            receives,
            merged_config,
            return_exceptions=return_exceptions,
            **{**self.kwargs, **kwargs},
        )

    @overload
    async def abatch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[False],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType]: ...

    @overload
    async def abatch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]: ...

    @override
    async def abatch(
        self,
        input: Input,
        receives: list[PhotoType] | None = None,
        config: TransableConfig | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> Sequence[PhotoType | Exception]:
        merged_config = self._merge_configs(config)
        return await self.bound.abatch(
            input,
            receives,
            merged_config,
            return_exceptions=return_exceptions,
            **{**self.kwargs, **kwargs},
        )

    @override
    def bind(self, **kwargs: Any) -> Transable[Input, PhotoType]:
        """Bind additional kwargs to a Transable, returning a new Transable."""
        return self.__class__(
            bound=self.bound,
            config=self.config,
            config_factories=self.config_factories,
            kwargs={**self.kwargs, **kwargs},
            custom_input_type=self.custom_input_type,
            custom_photo_type=self.custom_photo_type,
        )

    @override
    def with_types(
        self,
        input_type: type[Input] | None = None,
        photo_type: type[PhotoType] | None = None,
    ) -> Transable[Input, PhotoType]:
        return self.__class__(
            bound=self.bound,
            kwargs=self.kwargs,
            config=self.config,
            config_factories=self.config_factories,
            custom_input_type=(
                input_type if input_type is not None else self.custom_input_type
            ),
            custom_photo_type=(
                photo_type if photo_type is not None else self.custom_photo_type
            ),
        )

    @override
    def with_config(
        self,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> Transable[Input, PhotoType]:
        return self.__class__(
            bound=self.bound,
            kwargs=self.kwargs,
            config=cast("TransableConfig", {**self.config, **(config or {}), **kwargs}),
            config_factories=self.config_factories,
            custom_input_type=self.custom_input_type,
            custom_photo_type=self.custom_photo_type,
        )

    @override
    def with_listeners(
        self,
        *,
        on_start: Listener | None = None,
        on_end: Listener | None = None,
        on_error: Listener | None = None,
        on_stream_chunk: Callable[[TransableRun, Any, TransableConfig], None]
        | None = None,
    ) -> Transable[Input, PhotoType]:
        """Bind lifecycle listeners to a Transable."""

        listener = TransableListener(
            on_start=on_start,
            on_end=on_end,
            on_error=on_error,
            on_stream_chunk=on_stream_chunk,
        )

        def listener_factory(_config: TransableConfig) -> TransableConfig:
            return {
                "callbacks": [listener],
                "trace": True,
            }

        return self.__class__(
            bound=self.bound,
            kwargs=self.kwargs,
            config=self.config,
            config_factories=[listener_factory, *self.config_factories],
            custom_input_type=self.custom_input_type,
            custom_photo_type=self.custom_photo_type,
        )

    @override
    def with_alisteners(
        self,
        *,
        on_start: AsyncListener | None = None,
        on_end: AsyncListener | None = None,
        on_error: AsyncListener | None = None,
        on_stream_chunk: Callable[[TransableRun, Any, TransableConfig], Awaitable[None]]
        | None = None,
    ) -> Transable[Input, PhotoType]:
        """Bind async lifecycle listeners to a Transable."""

        listener = AsyncTransableListener(
            on_start=on_start,
            on_end=on_end,
            on_error=on_error,
            on_stream_chunk=on_stream_chunk,
        )

        def listener_factory(_config: TransableConfig) -> TransableConfig:
            return {
                "callbacks": [listener],
                "trace": True,
            }

        return self.__class__(
            bound=self.bound,
            kwargs=self.kwargs,
            config=self.config,
            config_factories=[listener_factory, *self.config_factories],
            custom_input_type=self.custom_input_type,
            custom_photo_type=self.custom_photo_type,
        )

    @override
    def __getattr__(self, name: str) -> Any:  # type: ignore[misc]
        attr = getattr(self.bound, name)

        if callable(attr) and (
            config_param := inspect.signature(attr).parameters.get("config")
        ):
            if config_param.kind == inspect.Parameter.KEYWORD_ONLY:

                @wraps(attr)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    return attr(
                        *args,
                        config=merge_configs(self.config, kwargs.pop("config", None)),
                        **kwargs,
                    )

                return wrapper
            if config_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                idx = list(inspect.signature(attr).parameters).index("config")

                @wraps(attr)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    if len(args) >= idx + 1:
                        argsl = list(args)
                        argsl[idx] = merge_configs(self.config, argsl[idx])
                        return attr(*argsl, **kwargs)
                    return attr(
                        *args,
                        config=merge_configs(self.config, kwargs.pop("config", None)),
                        **kwargs,
                    )

                return wrapper

        return attr


class _TransableCallableSync(Protocol[Input, PhotoType]):
    def __call__(
        self, _in: Input, /, *, receive: PhotoType, config: TransableConfig
    ) -> PhotoType: ...


class _TransableCallableAsync(Protocol[Input, PhotoType]):
    def __call__(
        self, _in: Input, /, *, receive: PhotoType, config: TransableConfig
    ) -> Awaitable[PhotoType]: ...


class _TransableCallableIterator(Protocol[Input, PhotoType]):
    def __call__(
        self, _in: Input, /, *, receives: Iterator[PhotoType], config: TransableConfig
    ) -> Iterator[PhotoType]: ...


class _TransableCallableAsyncIterator(Protocol[Input, PhotoType]):
    def __call__(
        self,
        _in: Input,
        /,
        *,
        receives: AsyncIterator[PhotoType],
        config: TransableConfig,
    ) -> AsyncIterator[PhotoType]: ...


TransableLike = (
    Transable[Input, PhotoType]
    | Callable[[Input], PhotoType]
    | Callable[[Input], Awaitable[PhotoType]]
    | Callable[[Input, Iterator[PhotoType]], Iterator[PhotoType]]
    | Callable[[Input, AsyncIterator[PhotoType]], AsyncIterator[PhotoType]]
    | _TransableCallableSync[Input, PhotoType]
    | _TransableCallableAsync[Input, PhotoType]
    | _TransableCallableIterator[Input, PhotoType]
    | _TransableCallableAsyncIterator[Input, PhotoType]
)


def coerce_to_transable(
    obj: TransableLike,
) -> Transable[Input, PhotoType]:
    """Coerce a Transable-like object into a Transable.

    Args:
        obj: A Transable-like object.

    Returns:
        A Transable.

    Raises:
        TypeError: If the object is not Transable-like.
    """
    if isinstance(obj, Transable):
        return obj
    if is_async_generator(obj) or inspect.isgeneratorfunction(obj):
        return TransableLambda(obj)
    if callable(obj):
        return TransableLambda(cast("Callable[[Input], PhotoType]", obj))
    msg = (
        f"Expected a Transable, callable or dict."
        f"Instead got an unsupported type: {type(obj)}"
    )
    raise TypeError(msg)
