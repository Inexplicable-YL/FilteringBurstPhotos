from __future__ import annotations

import contextlib
import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import FIRST_COMPLETED, Future, wait
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
        raise NotImplementedError

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
        raise NotImplementedError

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


# Incomplete revisions
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

        with get_executor_for_config(config) as executor:
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
        raise NotImplementedError

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
        # raise NotImplementedError
        if receives is None:
            raise ValueError(
                "Parallel Transables require an input PhotoResult stream when streamed."
            )
        buffer_size = stream_buffer(config)
        step_outputs: dict[str, list[PhotoType | None]] = defaultdict(
            lambda: [None] * len(self.steps)
        )

        # Use an unbounded base stream to avoid backpressure deadlocks when downstream
        # steps need to buffer multiple items before yielding.
        base_send, base_recv = anyio.create_memory_object_stream[PhotoType](
            max_buffer_size=buffer_size
        )
        child_sends: list[ObjectSendStream[PhotoType]] = []
        child_recvs: list[ObjectReceiveStream[PhotoType]] = []
        for _ in self.steps:
            s, r = anyio.create_memory_object_stream(max_buffer_size=buffer_size)
            child_sends.append(s)
            child_recvs.append(r)

        outpot_cond: anyio.Condition = anyio.Condition()
        step_done: list[bool] = [False] * len(self.steps)

        async def _broadcast() -> None:
            try:
                async for item in receives:
                    coerced = coerce_photo(item, self.PhotoType)
                    await base_send.send(clone_photo(coerced))
                    for send in child_sends:
                        await send.send(clone_photo(coerced))
            finally:
                await base_send.aclose()
                for send in child_sends:
                    await send.aclose()

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
                        **kwargs,
                    )
                    async for result in stream_iter:
                        coerced = coerce_photo(result, self.PhotoType)
                        async with outpot_cond:
                            step_outputs[coerced.get_unique_id()][idx] = clone_photo(
                                coerced
                            )
                            outpot_cond.notify_all()
            finally:
                async with outpot_cond:
                    step_done[idx] = True
                    outpot_cond.notify_all()

        async def _collect_outputs() -> AsyncIterator[PhotoType]:
            async with base_recv:
                async for base_item in base_recv:
                    unique_id = base_item.get_unique_id()
                    while True:
                        async with outpot_cond:
                            outputs = step_outputs[unique_id]
                            missing = [
                                idx for idx, res in enumerate(outputs) if res is None
                            ]
                            if not missing:
                                merged = merge_photos(
                                    base_item,
                                    cast(
                                        "list[PhotoType]",
                                        outputs,
                                    ),
                                )
                                step_outputs.pop(unique_id, None)
                                yield merged
                                break
                            if any(step_done[idx] for idx in missing):
                                step_outputs.pop(unique_id, None)
                                raise ValueError(
                                    "Parallel step results missing for unique ID "
                                    f"{unique_id}."
                                )
                            await outpot_cond.wait()
                # Base stream exhausted; if steps emitted unmatched IDs, surface them.
                if step_outputs:
                    raise ValueError(
                        "Parallel step produced results with no matching base item: "
                        f"{', '.join(step_outputs.keys())}"
                    )

        async with anyio.create_task_group() as tg:
            tg.start_soon(_broadcast)
            for idx, step in enumerate(self.steps):
                tg.start_soon(_run_step, idx, step)
            async for output in _collect_outputs():
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
            run_type="astream",
            **kwargs,
        ):
            yield cast("PhotoType", result)


FuncType = (
    Callable[[Input], PhotoType]
    | Callable[[Input], Transable[Input, PhotoType]]
    | Callable[[Input], Iterator[PhotoType]]
    | Callable[[Input, TransableConfig], PhotoType]
    | Callable[[Input, TransableConfig], Iterator[PhotoType]]
    | Callable[[Input, PhotoType], PhotoType]
    | Callable[[Input, PhotoType, TransableConfig], PhotoType]
)
AsyncFuncType = (
    Callable[[Input], Awaitable[PhotoType]]
    | Callable[[Input], AsyncIterator[PhotoType]]
    | Callable[[Input, TransableConfig], Awaitable[PhotoType]]
    | Callable[[Input, TransableConfig], AsyncIterator[PhotoType]]
    | Callable[[Input, PhotoType], Awaitable[PhotoType]]
    | Callable[[Input, PhotoType, TransableConfig], Awaitable[PhotoType]]
)
TransableFuncType = (
    Callable[[Input, Iterator[PhotoType]], Iterator[PhotoType]]
    | Callable[[Input, Iterator[PhotoType]], AsyncIterator[PhotoType]]
    | Callable[[Input, Iterator[PhotoType], TransableConfig], Iterator[PhotoType]]
    | Callable[[Input, Iterator[PhotoType], TransableConfig], AsyncIterator[PhotoType]]
    | Callable[[Input, AsyncIterator[PhotoType]], AsyncIterator[PhotoType]]
    | Callable[
        [Input, AsyncIterator[PhotoType], TransableConfig],
        AsyncIterator[PhotoType],
    ]
)


# Incomplete revisions
class TransableLambda(Transable[Input, PhotoType]):  # noqa: PLW1641
    """Transable defined from a pair of functions."""

    def __init__(
        self,
        func: FuncType | AsyncFuncType | TransableFuncType,
        stream_func: TransableFuncType | None = None,
        name: str | None = None,
    ) -> None:
        """"""
        if stream_func is not None:
            self.stream_func = stream_func
            func_for_name = stream_func

        if is_generator(func) or is_async_generator(func):
            if stream_func is not None:
                raise ValueError(
                    "Func was provided along with stream_func, but func is async or a generator. "
                    "Only one of func or stream_func should be provided in this case."
                )
            self.func = func
            self.stream_func = func
            func_for_name = func
        elif is_async_callable(func) or callable(func):
            self.func = func
            func_for_name = func
        else:
            raise TypeError("func must be a callable, generator, or async generator.")

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
        func = getattr(self, "func", None) or self.stream_func
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
        func = getattr(self, "func", None) or self.stream_func
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
        func = getattr(self, "func", None) or self.stream_func

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
            elif hasattr(self, "stream_func"):
                self._repr = (
                    "TransableLambda(stream_func="
                    f"{get_lambda_source(self.stream_func) or '...'})"
                )
            else:
                self._repr = "TransableLambda(...)"
        return self._repr

    async def _ainvoke(
        self,
        input: Input,
        receive: PhotoType | None,
        config: TransableConfig,
        **kwargs: Any,
    ) -> Any:
        config = ensure_config(config)
        if is_generator(self.func) or is_async_generator(self.func):
            output: PhotoType | None = None
            if is_generator(self.func):
                results = list(
                    call_func_with_variable_args(
                        cast("Callable[[Input], Iterator[PhotoType]]", self.func),
                        input,
                        receive,
                        config,
                        **kwargs,
                    )
                )
            else:
                results = [
                    res
                    async for res in call_func_with_variable_args(
                        cast("Callable[[Input], AsyncIterator[PhotoType]]", self.func),
                        input,
                        receive,
                        config,
                        **kwargs,
                    )
                ]
            results = [
                coerce_photo(res, self.PhotoType) for res in results if res is not None
            ]
            if not results:
                if receive is not None:
                    return coerce_photo(receive, self.PhotoType)
                raise ValueError(
                    "TransableLambda generator produced no results and no prior receive value."
                )
            if len(results) == 1:
                output = results[0]
            try:
                output = merge_photos(
                    receive if receive is not None else results[0], results
                )
            except Exception:
                output = results[-1]
        elif is_async_callable(self.func):
            output = await call_func_with_variable_args(
                cast("Callable[[Input], Awaitable[PhotoType]]", self.func),
                input,
                receive,
                config,
                **kwargs,
            )
        else:
            output = call_func_with_variable_args(
                cast("Callable[[Input], PhotoType]", self.func),
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
        if hasattr(self, "func"):
            return await self._acall_with_config(
                self._ainvoke,
                input,
                receive,
                config,
                "invoke",
                **kwargs,
            )

        raise TypeError(
            "Cannot use invoke when only stream_func is defined.Please use stream instead."
        )

    async def _astream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoType] | None,
        config: TransableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoType]:
        if is_generator(self.stream_func):
            for res in call_func_with_variable_args(
                cast(
                    "Callable[[Input, Iterator[PhotoType]], Iterator[PhotoType]]",
                    self.stream_func,
                ),
                input,
                receives,
                config,
                **kwargs,
            ):
                yield coerce_photo(res, self.PhotoType)
        elif is_async_generator(self.stream_func):
            async for res in call_func_with_variable_args(
                cast(
                    "Callable[[Input, AsyncIterator[PhotoType]], AsyncIterator[PhotoType]]",
                    self.stream_func,
                ),
                input,
                receives,
                config,
                **kwargs,
            ):
                yield coerce_photo(res, self.PhotoType)
        else:
            raise TypeError("stream_func must be a generator or async generator.")

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

        if hasattr(self, "stream_func"):
            async for item in self._astream_with_config(
                self._astream,
                input,
                receives,
                config,
                "stream",
                **kwargs,
            ):
                yield item
            return
        if receives is None:
            yield await self.ainvoke(input, None, config, **kwargs)
            return
        async for item in receives:
            yield await self.ainvoke(input, item, config, **kwargs)

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, TransableLambda):
            if hasattr(self, "func") and hasattr(other, "func"):
                return self.func == other.func
            if hasattr(self, "stream_func") and hasattr(other, "stream_func"):
                return self.stream_func == other.stream_func
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
        return TransableLambda(cast("TransableFuncType", obj))
    if callable(obj):
        return TransableLambda(cast("Callable[[Input], PhotoType]", obj))
    msg = (
        f"Expected a Transable, callable or dict."
        f"Instead got an unsupported type: {type(obj)}"
    )
    raise TypeError(msg)
