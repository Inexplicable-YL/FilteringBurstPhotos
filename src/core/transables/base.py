from __future__ import annotations

import contextlib
import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    cast,
    get_args,
)
from typing_extensions import override

import anyio
from pydantic import ConfigDict, Field

from core.transables.config import TransableConfig
from core.transables.serializable import (
    Serializable,
    SerializedConstructor,
    SerializedNotImplemented,
)
from core.transables.utils import (
    Input,
    OtherInput,
    OtherPhotoType,
    PhotoType,
    call_func_with_variable_args,
    clone_photo,
    coerce_photo,
    ensure_subclass,
    is_async_callable,
    is_async_generator,
    is_generator,
    merge_photos,
    stream_buffer,
)

if TYPE_CHECKING:
    from anyio.abc import ObjectReceiveStream, ObjectSendStream


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
        *others: Transable[Input, PhotoType],
    ) -> TransableSerializable[Input, PhotoType]:
        """"""
        return TransableParallel(self, *others)

    @abstractmethod
    async def invoke(
        self,
        input: Input,
        receive: PhotoType | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoType:
        """Consume a single input and yield streaming results."""

    @abstractmethod
    async def stream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoType] | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoType]:
        """Consume a stream of inputs and yield streaming results."""
        if receives is None:
            yield await self.invoke(input, config=config, **kwargs)
            return
        async for item in receives:
            yield await self.invoke(input, item, config=config, **kwargs)


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
    async def invoke(
        self,
        input: Input,
        receive: PhotoType | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoType:
        current = receive
        for transable in self.chains:
            current = await transable.invoke(input, current, config=config, **kwargs)
        return cast("PhotoType", current)

    @override
    async def stream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoType] | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoType]:
        upstream: AsyncIterator[Any] = self.start.stream(input, receives, config)
        for transable in self.chains[1:]:
            upstream = self._pipe(transable, upstream, input, config)
        async for result in upstream:
            yield cast("PhotoType", result)

    def _validate_chain(self) -> None:
        previous = self.start
        for transable in self.chains[1:]:
            ensure_subclass(transable.PhotoType, previous.PhotoType)
            previous = transable

    def _pipe(
        self,
        downstream: Transable[Any, Any],
        upstream: AsyncIterator[Any],
        input: Input,
        config: TransableConfig | None = None,
    ) -> AsyncIterator[Any]:
        buffer_size = stream_buffer(config)

        async def _gen() -> AsyncIterator[Any]:
            send, recv = anyio.create_memory_object_stream[Any](
                max_buffer_size=buffer_size
            )

            async def _produce() -> None:
                try:
                    async for result in upstream:
                        await send.send(coerce_photo(result, downstream.PhotoType))
                finally:
                    await send.aclose()

            async with anyio.create_task_group() as tg, recv:
                tg.start_soon(_produce)
                async for result in downstream.stream(input, recv, config):
                    yield result

        return _gen()


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
    async def invoke(
        self,
        input: Input,
        receive: PhotoType | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoType:
        if receive is None:
            raise ValueError(
                "Parallel Transables require an input PhotoResult when invoked."
            )
        upstream = coerce_photo(receive, self.PhotoType)
        results: list[PhotoType | None] = [None] * len(self.steps)

        async def _run(idx: int, step: Transable[Input, PhotoType]) -> None:
            res = await step.invoke(
                input, clone_photo(upstream), config=config, **kwargs
            )
            results[idx] = coerce_photo(res, self.PhotoType)

        async with anyio.create_task_group() as tg:
            for idx, step in enumerate(self.steps):
                tg.start_soon(_run, idx, step)

        if any(res is None for res in results):
            raise RuntimeError("Parallel step did not return a PhotoResult.")

        return merge_photos(receive, [res for res in results if res is not None])

    @override
    async def stream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoType] | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoType]:
        if receives is None:
            raise ValueError(
                "Parallel Transables require an input PhotoResult stream when streamed."
            )

        buffer_size = stream_buffer(config)
        step_outputs: dict[str, list[PhotoType | None]] = defaultdict(
            lambda: [None] * len(self.steps)
        )

        base_send, base_recv = anyio.create_memory_object_stream[PhotoType](buffer_size)
        child_sends: list[ObjectSendStream[PhotoType]] = []
        child_recvs: list[ObjectReceiveStream[PhotoType]] = []
        for _ in self.steps:
            s, r = anyio.create_memory_object_stream(buffer_size)
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
                    async for result in step.stream(input, child_recvs[idx], config):
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

        async with anyio.create_task_group() as tg:
            tg.start_soon(_broadcast)
            for idx, step in enumerate(self.steps):
                tg.start_soon(_run_step, idx, step)
            async for output in _collect_outputs():
                yield output


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
    async def invoke(
        self,
        input: Input,
        receive: PhotoType | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoType:
        if hasattr(self, "func"):
            return await self._invoke(input, receive, config, **kwargs)

        raise TypeError(
            "Cannot use invoke when only stream_func is defined.Please use stream instead."
        )

    @override
    async def stream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoType] | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoType]:
        if hasattr(self, "stream_func"):
            async for res in self._stream(input, receives, config, **kwargs):
                yield res
            return
        if receives is None:
            yield await self.invoke(input, None, config, **kwargs)
            return
        async for item in receives:
            yield await self.invoke(input, item, config, **kwargs)

    async def _invoke(
        self,
        input: Input,
        receive: PhotoType | None,
        config: TransableConfig | None,
        **kwargs: Any,
    ) -> Any:
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
            output = await output.invoke(
                input,
                receive,
                config,
                **kwargs,
            )

        return cast("PhotoType", output)

    async def _stream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoType] | None,
        config: TransableConfig | None,
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
    def __eq__(self, other: object) -> bool:
        if isinstance(other, TransableLambda):
            if hasattr(self, "func") and hasattr(other, "func"):
                return self.func == other.func
            if hasattr(self, "stream_func") and hasattr(other, "stream_func"):
                return self.stream_func == other.stream_func
            return False
        return False


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
