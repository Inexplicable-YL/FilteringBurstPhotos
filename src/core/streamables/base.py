from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypedDict,
    TypeVar,
    cast,
    get_args,
)
from typing_extensions import override

import anyio
from pydantic import ConfigDict, Field

from core.streamables.models import OtherPhotoType, Photo, PhotoResult, PhotoType
from core.streamables.serializable import (
    Serializable,
    SerializedConstructor,
    SerializedNotImplemented,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from anyio.abc import ObjectReceiveStream, ObjectSendStream

Input = TypeVar("Input")


class StreamableConfig(TypedDict, total=False):
    """Runtime hints for stream execution."""

    run_name: str
    tags: list[str]
    metadata: dict[str, Any]
    max_concurrency: int
    stream_buffer: int


def _stream_buffer(config: StreamableConfig | None) -> int:
    if config is None:
        return 64
    return max(1, int(config.get("stream_buffer", 64)))


def _ensure_subclass(child: type[Photo], parent: type[Photo]) -> None:
    if not issubclass(child, parent):
        raise TypeError(
            f"PhotoType mismatch: {child.__name__} is not a subclass of {parent.__name__}"
        )


def _coerce_photo_result(
    result: PhotoResult[Any],
    target_type: type[PhotoType],
) -> PhotoResult[PhotoType]:
    _ensure_subclass(target_type, result.PhotosType)
    if result.PhotosType is target_type:
        return cast("PhotoResult[PhotoType]", result)
    photos = [
        photo if isinstance(photo, target_type) else target_type(**photo.model_dump())
        for photo in result.photos
    ]
    return PhotoResult[target_type](photos=photos)


class Streamable(ABC, Generic[Input, PhotoType]):
    """Base class for streaming pipeline components."""

    name: str | None

    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        """Get the name of the Runnable."""
        if name:
            name_ = name
        elif hasattr(self, "name") and self.name:
            name_ = self.name
        else:
            # Here we handle a case where the runnable subclass is also a pydantic
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
        """The type of input this Streamable accepts specified as a type annotation."""
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
        # Runnables that are not pydantic models.
        for cls in self.__class__.__orig_bases__:  # type: ignore[attr-defined]
            type_args = get_args(cls)
            if type_args and len(type_args) == 2:
                return type_args[0]

        msg = (
            f"Streamable {self.get_name()} doesn't have an inferable InputType. "
            "Override the InputType property to specify the input type."
        )
        raise TypeError(msg)

    @property
    def PhotoType(self) -> type[PhotoType]:  # noqa: N802
        """The type of Photo this Streamable processes specified as a type annotation."""
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
            f"Streamable {self.get_name()} doesn't have an inferable PhotoType. "
            "Override the PhotoType property to specify the photo type."
        )
        raise TypeError(msg)

    @abstractmethod
    async def invoke(
        self,
        input: Input,
        receive: PhotoResult[PhotoType] | None = None,
        config: StreamableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoResult[PhotoType]:
        """Consume a single input and yield streaming results."""

    @abstractmethod
    async def stream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoResult[PhotoType]] | None = None,
        config: StreamableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoResult[PhotoType]]:
        """Consume a stream of inputs and yield streaming results."""
        if receives is None:
            yield await self.invoke(input, config=config, **kwargs)
            return
        async for item in receives:
            yield await self.invoke(input, item, config=config, **kwargs)

    def __or__(
        self, other: Streamable[Input, OtherPhotoType]
    ) -> Streamable[Input, OtherPhotoType]:
        _ensure_subclass(other.PhotoType, self.PhotoType)
        return StreamableSequence(self, other)

    def __ror__(
        self, other: Streamable[Input, PhotoType]
    ) -> Streamable[Input, PhotoType]:
        _ensure_subclass(self.PhotoType, other.PhotoType)
        return StreamableSequence(other, self)

    def pipe(
        self,
        *others: Streamable[Any, OtherPhotoType],
        name: str | None = None,
    ) -> Streamable[Input, OtherPhotoType]:
        """"""
        return StreamableSequence(self, *others, name=name)

    def __and__(
        self, other: Streamable[Input, PhotoType]
    ) -> Streamable[Input, PhotoType]:
        if other.PhotoType is not self.PhotoType:
            raise TypeError("Parallel Streamables require identical PhotoType values.")
        return StreamableParallel(self, other)

    def __rand__(
        self, other: Streamable[Input, PhotoType]
    ) -> Streamable[Input, PhotoType]:
        if other.PhotoType is not self.PhotoType:
            raise TypeError("Parallel Streamables require identical PhotoType values.")
        return StreamableParallel(other, self)

    def parallel(
        self,
        *others: Streamable[Input, PhotoType],
    ) -> Streamable[Input, PhotoType]:
        """"""
        return StreamableParallel(self, *others)


class StreamableSerializable(Serializable, Streamable[Input, PhotoType]):
    """Runnable that can be serialized to JSON."""

    name: str | None = None

    model_config = ConfigDict(
        # Suppress warnings from pydantic protected namespaces
        # (e.g., `model_`)
        protected_namespaces=(),
    )

    @override
    def to_json(self) -> SerializedConstructor | SerializedNotImplemented:
        """Serialize the Runnable to JSON.

        Returns:
            A JSON-serializable representation of the Runnable.
        """
        dumped = super().to_json()
        with contextlib.suppress(Exception):
            dumped["name"] = self.get_name()
        return dumped


class StreamableSequence(StreamableSerializable[Input, PhotoType]):
    start: Streamable[Input, Any]
    middle: list[Streamable[Any, Any]] = Field(default_factory=list)
    end: Streamable[Any, PhotoType]

    def __init__(
        self,
        *chains: Streamable[Any, Any],
        name: str | None = None,
        start: Streamable[Input, Any] | None = None,
        middle: list[Streamable[Any, Any]] | None = None,
        end: Streamable[Any, PhotoType] | None = None,
    ) -> None:
        chains_flat: list[Streamable[Any, Any]] = []
        if not chains and start is not None and end is not None:
            chains_flat = [start] + (middle or []) + [end]
        for chain in chains:
            if isinstance(chain, StreamableSequence):
                chains_flat.extend(chain.chains)
            else:
                chains_flat.append(chain)
        if len(chains_flat) < 2:
            msg = (
                f"StreamableSequence must have at least 2 steps, got {len(chains_flat)}"
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
    def chains(self) -> list[Streamable[Any, Any]]:
        """All the Runnables that make up the sequence in order.

        Returns:
            A list of Runnables.
        """
        return [self.start, *self.middle, self.end]

    @override
    async def invoke(
        self,
        input: Input,
        receive: PhotoResult[PhotoType] | None = None,
        config: StreamableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoResult[PhotoType]:
        current: PhotoResult[Any] | None = receive
        for streamable in self.chains:
            current = await streamable.invoke(input, current, config=config, **kwargs)
        return cast("PhotoResult[PhotoType]", current)

    @override
    async def stream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoResult[PhotoType]] | None = None,
        config: StreamableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoResult[PhotoType]]:
        upstream: AsyncIterator[PhotoResult[Any]] = self.start.stream(
            input, receives, config
        )
        for streamable in self.chains[1:]:
            upstream = self._pipe(streamable, upstream, input, config)
        async for result in upstream:
            yield cast("PhotoResult[PhotoType]", result)

    def _validate_chain(self) -> None:
        previous = self.start
        for streamable in self.chains[1:]:
            _ensure_subclass(streamable.PhotoType, previous.PhotoType)
            previous = streamable

    def _pipe(
        self,
        downstream: Streamable[Any, Any],
        upstream: AsyncIterator[PhotoResult[Any]],
        input: Input,
        config: StreamableConfig | None = None,
    ) -> AsyncIterator[PhotoResult[Any]]:
        async def _gen() -> AsyncIterator[PhotoResult[Any]]:
            async def _produce() -> AsyncIterator[PhotoResult[Any]]:
                async for result in upstream:
                    yield _coerce_photo_result(result, downstream.PhotoType)

            async for result in downstream.stream(input, _produce(), config):
                yield result

        return _gen()


class StreamableParallel(StreamableSerializable[Input, PhotoType]):
    steps: list[Streamable[Input, PhotoType]] = Field(default_factory=list)

    def __init__(
        self,
        *steps: Streamable[Input, PhotoType],
        name: str | None = None,
    ) -> None:
        flat: list[Streamable[Input, PhotoType]] = []
        for step in steps:
            if isinstance(step, StreamableParallel):
                flat.extend(step.steps)
            else:
                flat.append(step)
        if len(flat) < 2:
            raise ValueError(
                f"StreamableParallel must have at least 2 steps, got {len(flat)}"
            )

        first_type = flat[0].PhotoType
        for step in flat[1:]:
            if step.PhotoType is not first_type:
                raise TypeError(
                    "Parallel Streamables require identical PhotoType values."
                )

        super().__init__(steps=flat, name=name)  # pyright: ignore[reportCallIssue]

    @override
    async def invoke(
        self,
        input: Input,
        receive: PhotoResult[PhotoType] | None = None,
        config: StreamableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoResult[PhotoType]:
        if receive is None:
            raise ValueError(
                "Parallel Streamables require an input PhotoResult when invoked."
            )
        upstream = _coerce_photo_result(receive, self.PhotoType)
        results: list[PhotoResult[PhotoType] | None] = [None] * len(self.steps)

        async def _run(idx: int, step: Streamable[Input, PhotoType]) -> None:
            res = await step.invoke(input, upstream.model_copy(), config=config, **kwargs)
            results[idx] = _coerce_photo_result(res, self.PhotoType)

        async with anyio.create_task_group() as tg:
            for idx, step in enumerate(self.steps):
                tg.start_soon(_run, idx, step)

        if any(res is None for res in results):
            raise RuntimeError("Parallel step did not return a PhotoResult.")

        return self._merge_step_results(
            receive, [res for res in results if res is not None]
        )

    @override
    async def stream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoResult[PhotoType]] | None = None,
        config: StreamableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoResult[PhotoType]]:
        if receives is None:
            raise ValueError(
                "Parallel Streamables require an input PhotoResult stream when streamed."
            )

        buffer_size = _stream_buffer(config)
        step_outputs: dict[str, list[PhotoResult[PhotoType] | None]] = defaultdict(
            lambda: [None] * len(self.steps)
        )

        base_send, base_recv = anyio.create_memory_object_stream[
            PhotoResult[PhotoType]
        ](buffer_size)
        child_sends: list[ObjectSendStream[PhotoResult[PhotoType]]] = []
        child_recvs: list[ObjectReceiveStream[PhotoResult[PhotoType]]] = []
        for _ in self.steps:
            s, r = anyio.create_memory_object_stream(buffer_size)
            child_sends.append(s)
            child_recvs.append(r)

        outpot_cond: anyio.Condition = anyio.Condition()

        async def _broadcast() -> None:
            try:
                async for item in receives:
                    coerced = _coerce_photo_result(item, self.PhotoType)
                    await base_send.send(coerced)
                    for send in child_sends:
                        await send.send(coerced)
            finally:
                await base_send.aclose()
                for send in child_sends:
                    await send.aclose()

        async def _run_step(idx: int, step: Streamable[Input, PhotoType]) -> None:
            async with child_recvs[idx]:
                async for result in step.stream(input, child_recvs[idx], config):
                    coerced = _coerce_photo_result(result, self.PhotoType)
                    if unique_id := coerced.get_unique_id():
                        async with outpot_cond:
                            step_outputs[unique_id][idx] = coerced
                            outpot_cond.notify_all()

        async def _collect_outputs() -> AsyncIterator[PhotoResult[PhotoType]]:
            def _all_steps_done(_unique_id: str) -> Callable[[], bool]:
                return lambda: all(
                    step_outputs[_unique_id][idx] is not None
                    for idx in range(len(self.steps))
                )

            async with base_recv:
                async for base_item in base_recv:
                    unique_id = base_item.get_unique_id()
                    if unique_id is None:
                        continue
                    async with outpot_cond:
                        await outpot_cond.wait_for(
                            _all_steps_done(unique_id),
                        )
                    merged = self._merge_step_results(
                        base_item,
                        cast("list[PhotoResult[PhotoType]]", step_outputs[unique_id]),
                    )
                    step_outputs.pop(unique_id, None)
                    yield merged

        async with anyio.create_task_group() as tg:
            tg.start_soon(_broadcast)
            for idx, step in enumerate(self.steps):
                tg.start_soon(_run_step, idx, step)
            async for output in _collect_outputs():
                yield output

    def _merge_step_results(
        self,
        base: PhotoResult[PhotoType],
        step_results: list[PhotoResult[PhotoType]],
    ) -> PhotoResult[PhotoType]:
        if not step_results:
            raise ValueError("No step results to merge.")

        def _merge_photos(photos: list[PhotoType]) -> PhotoType:
            merged_data = photos[0].model_dump()
            for photo in photos[1:]:
                update = {
                    key: value
                    for key, value in photo.model_dump().items()
                    if value is not None
                }
                merged_data.update(update)
            return photos[0].__class__(**merged_data)

        base_strict_id = base.get_unique_id(strict=True)
        shared_strict_ids = {
            result.get_unique_id(strict=True) for result in step_results
        }
        shared_loose_ids = {result.get_unique_id() for result in step_results}

        photo_map: dict[str, list[PhotoType]] = {}
        if len(shared_strict_ids) == 1 and base_strict_id == next(
            iter(shared_strict_ids)
        ):
            expected_len = len(base.photos)
            for step_result in step_results:
                if len(step_result.photos) != expected_len:
                    raise ValueError(
                        "Parallel step results have mismatched photo counts."
                    )
            for idx, photo in enumerate(base.photos):
                photo_map[photo.get_unique_id()] = [
                    photo,
                    *[step_result.photos[idx] for step_result in step_results],
                ]
        elif len(shared_loose_ids) == 1 and base.get_unique_id() == next(
            iter(shared_loose_ids)
        ):
            step_dicts = [
                {p.get_unique_id(): p for p in step_result.photos}
                for step_result in step_results
            ]
            for base_photo in base.photos:
                uid = base_photo.get_unique_id()
                merged_photos = [base_photo]
                for step_dict in step_dicts:
                    if uid not in step_dict:
                        raise ValueError(
                            "Parallel step results have mismatched unique IDs; cannot merge."
                        )
                    merged_photos.append(step_dict[uid])
                photo_map[uid] = merged_photos
        else:
            raise ValueError(
                "Parallel step results have mismatched unique IDs; cannot merge."
            )
        merged_photos: list[PhotoType] = []
        for photos in photo_map.values():
            merged_photos.append(_merge_photos(photos))

        merged_metadata: dict[str, Any] = dict(base.metadata)
        for result in step_results:
            merged_metadata.update(result.metadata)
        return PhotoResult[PhotoType](photos=merged_photos, metadata=merged_metadata)


'''
FuncType = (
    Callable[[Input], PhotoResult[PhotoType]]
    | Callable[[Input], Streamable[Input, PhotoType]]
    | Callable[[Input], Iterator[PhotoResult[PhotoType]]]
    | Callable[[Input, StreamableConfig], PhotoResult[PhotoType]]
    | Callable[[Input, StreamableConfig], Streamable[Input, PhotoType]]
    | Callable[[Input, StreamableConfig], Iterator[PhotoResult[PhotoType]]]
    | Callable[[Input, PhotoResult[PhotoType]], PhotoResult[PhotoType]]
    | Callable[[Input, PhotoResult[PhotoType]], Streamable[Input, PhotoType]]
    | Callable[[Input, PhotoResult[PhotoType]], Iterator[PhotoResult[PhotoType]]]
    | Callable[
        [Input, PhotoResult[PhotoType], StreamableConfig], PhotoResult[PhotoType]
    ]
    | Callable[
        [Input, PhotoResult[PhotoType], StreamableConfig], Streamable[Input, PhotoType]
    ]
    | Callable[
        [Input, PhotoResult[PhotoType], StreamableConfig],
        Iterator[PhotoResult[PhotoType]],
    ]
    | Callable[[Input], Awaitable[PhotoResult[PhotoType]]]
    | Callable[[Input], AsyncIterator[PhotoResult[PhotoType]]]
    | Callable[[Input, StreamableConfig], Awaitable[PhotoResult[PhotoType]]]
    | Callable[[Input, StreamableConfig], AsyncIterator[PhotoResult[PhotoType]]]
)


class StreamableLambda(
    Streamable[Input, PhotoType],
):
    """Streamable defined from a pair of functions."""

    def __init__(
        self,
        func: FuncType,
        name: str | None = None,
    ) -> None:
        """"""
        self.func = func
        self.name = name

    @override
    async def invoke(
        self,
        input: Input,
        receive: PhotoResult[PhotoType] | None = None,
        config: StreamableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoResult[PhotoType]:
        result = await self._call_func(input, receive, config, **kwargs)

        if isinstance(result, Streamable):
            return await result.invoke(input, receive, config=config, **kwargs)

        if isinstance(result, PhotoResult):
            return _coerce_photo_result(result, self.PhotoType)

        if isinstance(result, Iterator):
            last: PhotoResult[Any] | None = None
            for item in result:
                last = item
            if last is None:
                raise ValueError("StreamableLambda iterator returned no PhotoResult.")
            return _coerce_photo_result(last, self.PhotoType)

        if isinstance(result, AsyncIterator):
            last_async: PhotoResult[Any] | None = None
            async for item in result:
                last_async = item
            if last_async is None:
                raise ValueError(
                    "StreamableLambda async iterator returned no PhotoResult."
                )
            return _coerce_photo_result(last_async, self.PhotoType)

        raise TypeError(
            f"Unsupported return type from StreamableLambda: {type(result)!r}"
        )

    @override
    async def stream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoResult[PhotoType]] | None = None,
        config: StreamableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoResult[PhotoType]]:
        if receives is None:
            yield await self.invoke(input, None, config=config, **kwargs)
            return
        async for item in receives:
            yield await self.invoke(input, item, config=config, **kwargs)

    async def _call_func(
        self,
        input: Input,
        receive: PhotoResult[PhotoType] | None,
        config: StreamableConfig | None,
        **kwargs: Any,
    ) -> Any:
        attempts = [
            (input, receive, config),
            (input, receive),
            (input, config),
            (input,),
        ]
        last_error: Exception | None = None
        for args in attempts:
            try:
                value = self.func(*args, **kwargs)
                break
            except TypeError as exc:
                last_error = exc
                continue
        else:
            raise TypeError(
                "StreamableLambda func signature not supported."
            ) from last_error

        if inspect.isawaitable(value):
            return await value
        return value
'''
