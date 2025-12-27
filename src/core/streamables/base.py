from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
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
    from collections.abc import AsyncIterator

    from anyio.abc import ObjectReceiveStream


Input = TypeVar("Input")


class StreamableConfig(TypedDict, total=False):
    """Runtime hints for stream execution."""

    run_name: str
    tags: list[str]
    metadata: dict[str, Any]
    max_concurrency: int
    stream_buffer: int


def _stream_buffer(config: StreamableConfig) -> int:
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
    return PhotoResult[target_type](photos=photos, done=result.done)


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
    async def iter(
        self,
        input: Input,
        config: StreamableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoResult[PhotoType]]:
        """Consume a single input and yield streaming results."""
        if False:
            yield

    @abstractmethod
    async def stream(
        self,
        input: Input,
        receive: ObjectReceiveStream[PhotoResult[PhotoType]],
        config: StreamableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoResult[PhotoType]]:
        """Consume a stream of inputs and yield streaming results."""
        if False:
            yield

    def __or__(
        self, other: Streamable[Input, OtherPhotoType]
    ) -> Streamable[Input, OtherPhotoType]:
        _ensure_subclass(other.PhotoType, self.PhotoType)
        return StreamableChain(self, other)

    def __ror__(
        self, other: Streamable[Input, PhotoType]
    ) -> Streamable[Input, PhotoType]:
        _ensure_subclass(self.PhotoType, other.PhotoType)
        return StreamableChain(other, self)

    def pipe(
        self,
        *others: Streamable[Any, OtherPhotoType],
        name: str | None = None,
    ) -> Streamable[Input, OtherPhotoType]:
        """"""
        return StreamableChain(self, *others, name=name)

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


class StreamableChain(StreamableSerializable[Input, PhotoType]):
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
            if isinstance(chain, StreamableChain):
                chains_flat.extend(chain.chains)
            else:
                chains_flat.append(chain)
        if len(chains_flat) < 2:
            msg = f"StreamableChain must have at least 2 steps, got {len(chains_flat)}"
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
    async def iter(
        self,
        input: Input,
        config: StreamableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoResult[PhotoType]]:
        upstream: AsyncIterator[PhotoResult[Any]] = self.start.iter(input, config)
        for runnable in [*self.middle, self.end]:
            upstream = self._pipe(runnable, upstream, input, config)
        async for result in upstream:
            yield cast("PhotoResult[PhotoType]", result)

    @override
    async def stream(
        self,
        input: Input,
        receive: ObjectReceiveStream[PhotoResult[PhotoType]],
        config: StreamableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoResult[PhotoType]]:
        upstream: AsyncIterator[PhotoResult[Any]] = self.start.stream(
            input, receive, config
        )
        for runnable in [*self.middle, self.end]:
            upstream = self._pipe(runnable, upstream, input, config)
        async for result in upstream:
            yield cast("PhotoResult[PhotoType]", result)

    def _validate_chain(self) -> None:
        previous = self.start
        for runnable in [*self.middle, self.end]:
            _ensure_subclass(runnable.PhotoType, previous.PhotoType)
            previous = runnable

    def _pipe(
        self,
        downstream: Streamable[Any, Any],
        upstream: AsyncIterator[PhotoResult[Any]],
        input: Input,
        config: StreamableConfig,
    ) -> AsyncIterator[PhotoResult[Any]]:
        buffer_size = _stream_buffer(config)

        async def _gen() -> AsyncIterator[PhotoResult[Any]]:
            send, recv = anyio.create_memory_object_stream(buffer_size)

            async def _produce() -> None:
                try:
                    async for result in upstream:
                        coerced = _coerce_photo_result(result, downstream.PhotoType)
                        await send.send(coerced)
                finally:
                    await send.aclose()

            async with anyio.create_task_group() as tg:
                tg.start_soon(_produce)
                async for result in downstream.stream(input, recv, config):
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
    async def iter(
        self,
        input: Input,
        config: StreamableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoResult[PhotoType]]:
        step_results: list[PhotoResult[PhotoType] | None] = [None] * len(self.steps)

        async def _collect(idx: int, step: Streamable[Input, PhotoType]) -> None:
            last: PhotoResult[PhotoType] | None = None
            async for result in step.iter(input, config):
                last = _coerce_photo_result(result, self.PhotoType)
            if last is not None:
                step_results[idx] = last

        async with anyio.create_task_group() as tg:
            for idx, step in enumerate(self.steps):
                tg.start_soon(_collect, idx, step)

        if any(result is None for result in step_results):
            return

        base = step_results[0]
        assert base is not None
        merged = self._merge_step_results(
            base, [res for res in step_results[1:] if res is not None]
        )
        yield merged

    @override
    async def stream(
        self,
        input: Input,
        receive: ObjectReceiveStream[PhotoResult[PhotoType]],
        config: StreamableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoResult[PhotoType]]:
        buffer_size = _stream_buffer(config)

        async with receive:
            async for base in receive:
                base_coerced = _coerce_photo_result(base, self.PhotoType)
                step_results: list[PhotoResult[PhotoType] | None] = [
                    None for _ in self.steps
                ]

                async def _collect(
                    idx: int,
                    step: Streamable[Input, PhotoType],
                    _base_coerced: PhotoResult[PhotoType],
                    _step_results: list[PhotoResult[PhotoType] | None],
                ) -> None:
                    send, recv = anyio.create_memory_object_stream(buffer_size)
                    results: list[PhotoResult[PhotoType]] = []

                    async def _feed() -> None:
                        async with send:
                            await send.send(_base_coerced)

                    async def _run_step() -> None:
                        results.extend(
                            [
                                _coerce_photo_result(r, self.PhotoType)
                                async for r in step.stream(input, recv, config)
                            ]
                        )

                    async with anyio.create_task_group() as tg:
                        tg.start_soon(_feed)
                        tg.start_soon(_run_step)

                    if results:
                        _step_results[idx] = results[-1]

                async with anyio.create_task_group() as tg:
                    for idx, step in enumerate(self.steps):
                        tg.start_soon(_collect, idx, step, base_coerced, step_results)

                if any(result is None for result in step_results):
                    continue

                merged = self._merge_step_results(
                    base_coerced,
                    [res for res in step_results if res is not None],
                )
                yield merged

    def _merge_step_results(
        self,
        base: PhotoResult[PhotoType],
        step_results: list[PhotoResult[PhotoType]],
    ) -> PhotoResult[PhotoType]:
        base_map: dict[Any, PhotoType] = {p.path: p for p in base.photos}
        merged_data = {p.path: p.model_dump() for p in base.photos}
        changed_keys: dict[Any, set[str]] = {p.path: set() for p in base.photos}

        for result in step_results:
            for photo in result.photos:
                key = photo.path
                base_photo = base_map.get(key)
                if base_photo is None:
                    merged_data[key] = photo.model_dump()
                    changed_keys[key] = set(photo.model_dump().keys())
                    continue
                base_dump = base_photo.model_dump()
                new_dump = photo.model_dump()
                for field, value in new_dump.items():
                    if base_dump.get(field) == value:
                        continue
                    seen = changed_keys.setdefault(key, set())
                    if field in seen:
                        raise ValueError(
                            f"Conflict on field '{field}' for photo {key}: multiple steps modified this key."
                        )
                    seen.add(field)
                    merged_data[key][field] = value

        merged_photos = [
            self.PhotoType(**data)
            for _, data in sorted(merged_data.items(), key=lambda x: str(x[0]))
        ]
        done_flag = base.done and all(result.done for result in step_results)
        return PhotoResult[self.PhotoType](photos=merged_photos, done=done_flag)
