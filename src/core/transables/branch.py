"""Transable that selects which branch to run based on a condition."""

from __future__ import annotations

import inspect
import types
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)
from typing_extensions import override

import anyio
from pydantic import BaseModel, ConfigDict

from core.transables.base import (
    Transable,
    TransableLike,
    TransableSerializable,
    coerce_to_transable,
)
from core.transables.config import (
    TransableConfig,
    arun_in_context,
    ensure_config,
    run_in_context,
)
from core.transables.models import Photo
from core.transables.utils import (
    Input,
    PhotoType,
    accepts_config,
    coerce_photo,
    is_async_callable,
    stream_buffer,
)

if TYPE_CHECKING:
    from anyio.abc import ObjectReceiveStream, ObjectSendStream

_MIN_BRANCHES = 2
ConditionMode = Literal["input", "photo", "input_photo"]
ConditionCallable = Callable[..., bool] | Callable[..., Awaitable[bool]]


def _annotation_is_photo(annotation: Any) -> bool:  # noqa: PLR0911
    if annotation is inspect.Signature.empty or annotation is Any:
        return False
    if annotation is PhotoType:
        return True
    if isinstance(annotation, TypeVar):
        bound = annotation.__bound__
        return bool(isinstance(bound, type) and issubclass(bound, Photo))
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            return issubclass(annotation, Photo)
        return False
    if origin is Annotated:
        return _annotation_is_photo(get_args(annotation)[0])
    if origin is Union or origin is getattr(types, "UnionType", Union):
        return any(_annotation_is_photo(arg) for arg in get_args(annotation))
    return False


def _resolve_condition_mode(condition: ConditionCallable) -> ConditionMode:
    try:
        signature = inspect.signature(condition)
    except (TypeError, ValueError):
        return "input"

    positional_params = [
        param
        for param in signature.parameters.values()
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional_params) >= 2:
        return "input_photo"
    if len(positional_params) == 1:
        param = positional_params[0]
        annotation = param.annotation
        try:
            hints = get_type_hints(condition)
        except Exception:
            hints = {}
        if param.name in hints:
            annotation = hints[param.name]
        if _annotation_is_photo(annotation):
            return "photo"
        if annotation is inspect.Signature.empty and param.name in {
            "photo",
            "receive",
            "item",
        }:
            return "photo"
        return "input"
    return "input"


async def _call_condition(
    condition: ConditionCallable,
    mode: ConditionMode,
    input: Input,
    photo: PhotoType | None,
    config: TransableConfig,
) -> bool:
    if mode == "input":
        args = (input,)
    elif mode == "photo":
        args = (photo,)
    else:
        args = (input, photo)

    kwargs: dict[str, Any] = {}
    if accepts_config(condition):
        kwargs["config"] = config

    if is_async_callable(condition):
        result = await arun_in_context(config, condition, *args, **kwargs)
    else:
        result = run_in_context(config, condition, *args, **kwargs)
        if inspect.isawaitable(result):
            result = await result

    return bool(result)


class TransableBranch(TransableSerializable[Input, PhotoType]):
    """Transable that selects which branch to run based on a condition."""

    branches: Sequence[tuple[ConditionCallable, Transable[Input, PhotoType]]]
    default: Transable[Input, PhotoType]
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    def __init__(
        self,
        *branches: tuple[ConditionCallable, TransableLike] | TransableLike,
        name: str | None = None,
    ) -> None:
        if len(branches) < 2:
            raise ValueError("TransableBranch requires at least two branches.")

        default = branches[-1]
        default_transable = cast(
            "Transable[Input, PhotoType]",
            coerce_to_transable(cast("TransableLike", default)),
        )

        branches_list: list[tuple[ConditionCallable, Transable[Input, PhotoType]]] = []
        for branch in branches[:-1]:
            if not isinstance(branch, (tuple, list)):
                raise TypeError(
                    "TransableBranch branches must be tuples or lists, "
                    f"not {type(branch)}"
                )
            if len(branch) != _MIN_BRANCHES:
                raise ValueError(
                    "TransableBranch branches must be tuples or lists of length 2, "
                    f"not {len(branch)}"
                )
            condition, transable = branch
            if not callable(condition):
                raise TypeError("TransableBranch condition must be callable.")
            branches_list.append(
                (cast("ConditionCallable", condition), coerce_to_transable(transable))
            )

        first_type = branches_list[0][1].PhotoType
        for _, transable in branches_list:
            if transable.PhotoType is not first_type:
                raise TypeError(
                    "TransableBranch requires identical PhotoType values for all "
                    "branches."
                )
        if default_transable.PhotoType is not first_type:
            raise TypeError(
                "TransableBranch requires identical PhotoType values for all branches."
            )

        super().__init__(
            branches=branches_list,  # pyright: ignore[reportCallIssue]
            default=default_transable,  # pyright: ignore[reportCallIssue]
            name=name,
        )

        self._condition_modes = [
            _resolve_condition_mode(condition) for condition, _ in branches_list
        ]
        self._has_photo_conditions = any(
            mode != "input" for mode in self._condition_modes
        )

    @property
    @override
    def InputType(self) -> type[Input]:
        return self.branches[0][1].InputType

    @property
    @override
    def PhotoType(self) -> type[PhotoType]:
        return self.default.PhotoType

    @override
    def get_input_schema(
        self, config: TransableConfig | None = None
    ) -> type[BaseModel]:
        transables = [self.default] + [branch for _, branch in self.branches]
        for transable in transables:
            schema = transable.get_input_schema(config)
            if schema.model_json_schema().get("type") is not None:
                return schema
        return super().get_input_schema(config)

    async def _select_branch(
        self,
        input: Input,
        photo: PhotoType | None,
        config: TransableConfig,
        cache: dict[int, bool],
    ) -> Transable[Input, PhotoType]:
        for idx, (condition, transable) in enumerate(self.branches):
            mode = self._condition_modes[idx]
            if mode == "input":
                if idx not in cache:
                    cache[idx] = await _call_condition(
                        condition,
                        mode,
                        input,
                        photo,
                        config,
                    )
                if cache[idx]:
                    return transable
            else:
                matched = await _call_condition(
                    condition,
                    cast("ConditionMode", mode),
                    input,
                    photo,
                    config,
                )
                if matched:
                    return transable
        return self.default

    @override
    async def invoke(
        self,
        input: Input,
        receive: PhotoType | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> PhotoType:
        config = ensure_config(config)

        async def _invoke_branch() -> PhotoType:
            cache: dict[int, bool] = {}
            receive_value = (
                coerce_photo(receive, self.PhotoType) if receive is not None else None
            )
            branch = await self._select_branch(
                input,
                receive_value,
                config,
                cache,
            )
            result = await branch.invoke(
                input,
                receive_value,
                config=config,
                **kwargs,
            )
            return coerce_photo(result, self.PhotoType)

        return await self._call_with_config(
            _invoke_branch,
            input,
            receive,
            config,
            "invoke",
        )

    @override
    async def stream(
        self,
        input: Input,
        receives: AsyncIterator[PhotoType] | None = None,
        config: TransableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[PhotoType]:
        config = ensure_config(config)

        async def _stream_impl() -> AsyncIterator[PhotoType]:
            cache: dict[int, bool] = {}

            async def _select(photo: PhotoType | None) -> Transable[Input, PhotoType]:
                return await self._select_branch(
                    input,
                    photo,
                    config,
                    cache,
                )

            if receives is None:
                branch = await _select(None)
                stream_iter = run_in_context(
                    config,
                    branch.stream,
                    input,
                    None,
                    config,
                    **kwargs,
                )
                async for result in stream_iter:
                    yield coerce_photo(result, self.PhotoType)
                return

            if not self._has_photo_conditions:
                branch = await _select(None)
                stream_iter = run_in_context(
                    config,
                    branch.stream,
                    input,
                    receives,
                    config,
                    **kwargs,
                )
                async for result in stream_iter:
                    yield coerce_photo(result, self.PhotoType)
                return

            async def _empty_stream() -> AsyncIterator[PhotoType]:
                if False:
                    yield None

            async def _prepend(
                item: PhotoType, iterator: AsyncIterator[PhotoType]
            ) -> AsyncIterator[PhotoType]:
                yield item
                async for value in iterator:
                    yield value

            async def _coerce_stream(
                iterator: AsyncIterator[PhotoType],
            ) -> AsyncIterator[PhotoType]:
                async for value in iterator:
                    yield coerce_photo(value, self.PhotoType)

            try:
                first_raw = await anext(receives)
            except StopAsyncIteration:
                # No photos upstream; still evaluate once for photo-based conditions.
                branch = await _select(None)
                stream_iter = run_in_context(
                    config,
                    branch.stream,
                    input,
                    _empty_stream(),
                    config,
                    **kwargs,
                )
                async for result in stream_iter:
                    yield coerce_photo(result, self.PhotoType)
                return

            first = coerce_photo(first_raw, self.PhotoType)
            upstream = _prepend(first, _coerce_stream(receives))

            buffer_size = stream_buffer(config)
            target_branches = [branch for _, branch in self.branches] + [self.default]
            branch_map = {id(branch): idx for idx, branch in enumerate(target_branches)}
            branch_sends: list[ObjectSendStream[PhotoType]] = []
            branch_recvs: list[ObjectReceiveStream[PhotoType]] = []
            for _ in target_branches:
                send, recv = anyio.create_memory_object_stream[PhotoType](buffer_size)
                branch_sends.append(send)
                branch_recvs.append(recv)

            output_send, output_recv = anyio.create_memory_object_stream[
                tuple[int, PhotoType]
            ](buffer_size)
            id_to_index: dict[str, int] = {}
            seen_indices: set[int] = set()
            total_inputs = 0
            remaining_branches = len(target_branches)
            remaining_lock = anyio.Lock()

            async def _dispatch() -> None:
                nonlocal total_inputs
                try:
                    async for item in upstream:
                        coerced = coerce_photo(item, self.PhotoType)
                        unique_id = coerced.get_unique_id()
                        if unique_id in id_to_index:
                            raise ValueError(
                                "Branch stream received duplicate photo unique ID "
                                f"{unique_id}."
                            )
                        index = total_inputs
                        total_inputs += 1
                        id_to_index[unique_id] = index
                        branch = await _select(coerced)
                        branch_idx = branch_map.get(id(branch))
                        if branch_idx is None:
                            raise ValueError(
                                "Branch selection returned an unknown branch."
                            )
                        await branch_sends[branch_idx].send(coerced)
                finally:
                    for send in branch_sends:
                        await send.aclose()

            async def _run_branch(
                idx: int, branch: Transable[Input, PhotoType]
            ) -> None:
                nonlocal remaining_branches
                try:
                    async with branch_recvs[idx]:
                        stream_iter = run_in_context(
                            config,
                            branch.stream,
                            input,
                            branch_recvs[idx],
                            config,
                            **kwargs,
                        )
                        async for result in stream_iter:
                            coerced = coerce_photo(result, self.PhotoType)
                            unique_id = coerced.get_unique_id()
                            if unique_id not in id_to_index:
                                raise ValueError(
                                    f"Branch output missing for unique ID {unique_id}."
                                )
                            index = id_to_index.pop(unique_id)
                            if index in seen_indices:
                                raise ValueError(
                                    f"Branch output duplicated for input index {index}."
                                )
                            seen_indices.add(index)
                            await output_send.send((index, coerced))
                finally:
                    async with remaining_lock:
                        remaining_branches -= 1
                        if remaining_branches == 0:
                            await output_send.aclose()

            async def _collect_outputs() -> AsyncIterator[PhotoType]:
                next_index = 0
                pending: dict[int, PhotoType] = {}
                async with output_recv:
                    async for index, photo in output_recv:
                        pending[index] = photo
                        while next_index in pending:
                            yield pending.pop(next_index)
                            next_index += 1
                if pending or id_to_index:
                    missing = sorted(id_to_index.values())
                    raise ValueError(
                        "Branch outputs missing for input indexes: "
                        f"{', '.join(str(idx) for idx in missing)}"
                    )
                if next_index != total_inputs:
                    raise ValueError(
                        "Branch outputs missing for input indexes: "
                        f"{', '.join(str(idx) for idx in range(next_index, total_inputs))}"
                    )

            async with anyio.create_task_group() as tg:
                tg.start_soon(_dispatch)
                for idx, branch in enumerate(target_branches):
                    tg.start_soon(_run_branch, idx, branch)
                async for output in _collect_outputs():
                    yield coerce_photo(output, self.PhotoType)

        async for result in self._stream_with_config(
            _stream_impl(),
            input,
            receives,
            config,
            "stream",
        ):
            yield result
