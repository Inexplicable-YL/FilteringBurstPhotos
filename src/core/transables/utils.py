from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Iterator
from typing import (
    TYPE_CHECKING,
    Any,
    TypeGuard,
    TypeVar,
    cast,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from core.transables.config import TransableConfig
    from core.transables.models import Photo


Input = TypeVar("Input")
Output = TypeVar("Output")
PhotoType = TypeVar("PhotoType", bound="Photo")
OtherPhotoType = TypeVar("OtherPhotoType", bound="Photo")


def stream_buffer(config: TransableConfig | None) -> int:
    if config is None:
        return 64
    return max(1, int(config.get("stream_buffer", 64)))


def ensure_subclass(child: type[Photo], parent: type[Photo]) -> None:
    if not issubclass(child, parent):
        raise TypeError(
            f"PhotoType mismatch: {child.__name__} is not a subclass of {parent.__name__}"
        )


def coerce_photo(
    photo: Photo,
    target_type: type[PhotoType],
) -> PhotoType:
    ensure_subclass(target_type, photo.PhotoType)
    if photo.PhotoType is target_type:
        return cast("PhotoType", photo)
    return target_type(**photo.model_dump())


def clone_photo(photo: PhotoType) -> PhotoType:
    """Create a deep copy to isolate mutations across parallel branches."""
    return photo.model_copy(deep=True)


def merge_photos(
    base: PhotoType,
    photos: list[PhotoType],
) -> PhotoType:
    if not photos:
        raise ValueError("No step results to merge.")

    if {base.get_unique_id()} != {res.get_unique_id() for res in photos}:
        raise ValueError("Parallel step results must from the same photo.")

    base_data = base.model_dump(exclude={"metadata"})
    merged_data = base_data.copy()
    changed_keys: set[str] = set()

    for photo in photos:
        photo_data = photo.model_dump(exclude={"metadata"})
        for key, value in photo_data.items():
            base_value = base_data.get(key)
            if value == base_value:
                continue
            if key in changed_keys:
                raise ValueError(
                    "Parallel steps modified the same field "
                    f"'{key}' for photo {base.get_unique_id()}."
                )
            changed_keys.add(key)
            merged_data[key] = value

    merged_metadata: dict[str, Any] = dict(base.metadata)
    changed_meta_keys: set[str] = set()
    for result in photos:
        for key, value in result.metadata.items():
            base_value = base.metadata.get(key)
            if value == base_value:
                continue
            if key in changed_meta_keys:
                raise ValueError(
                    "Parallel steps modified the same metadata field "
                    f"'{key}' for PhotoResult {base.get_unique_id()}."
                )
            changed_meta_keys.add(key)
            merged_metadata[key] = value

    return base.PhotoType(**merged_data, metadata=merged_metadata)


def accepts_config(callable: Callable[..., Any]) -> bool:  # noqa: A002
    """Check if a callable accepts a config argument.

    Args:
        callable: The callable to check.

    Returns:
        bool: True if the callable accepts a config argument, False otherwise.
    """
    try:
        return inspect.signature(callable).parameters.get("config") is not None
    except ValueError:
        return False


def accepts_receive(callable: Callable[..., Any]) -> bool:  # noqa: A002
    """Check if a callable accepts a receive argument.

    Args:
        callable: The callable to check.

    Returns:
        bool: True if the callable accepts a receive argument, False otherwise.
    """
    try:
        return inspect.signature(callable).parameters.get("receive") is not None
    except ValueError:
        return False


def accepts_receives(callable: Callable[..., Any]) -> bool:  # noqa: A002
    """Check if a callable accepts a receives argument.

    Args:
        callable: The callable to check.

    Returns:
        bool: True if the callable accepts a receives argument, False otherwise.
    """
    try:
        return inspect.signature(callable).parameters.get("receives") is not None
    except ValueError:
        return False


def call_func_with_variable_args(
    func: Callable[[Input], Output]
    | Callable[[Input, TransableConfig], Output]
    | Callable[[Input, PhotoType, TransableConfig], Output]
    | Callable[[Input, Iterator[PhotoType]], Output]
    | Callable[[Input, AsyncIterator[PhotoType]], Output]
    | Callable[[Input, Iterator[PhotoType], TransableConfig], Output]
    | Callable[[Input, AsyncIterator[PhotoType], TransableConfig], Output],
    input: Input,
    all_receive: PhotoType | Iterator[PhotoType] | AsyncIterator[PhotoType] | None,
    config: TransableConfig | None,
    **kwargs: Any,
) -> Output:
    """Call a function that may or may not accept a TransableConfig.

    Args:
        func: The function to call.
        input: The input to the function.
        config: The TransableConfig to pass if accepted.
        **kwargs: Additional keyword arguments to pass to the function.
    Returns:
        PhotoType: The result of the function call.
    """
    if accepts_config(func):
        kwargs["config"] = config
    if accepts_receive(func) and not isinstance(all_receive, Iterator | AsyncIterator):
        kwargs["receive"] = all_receive
    elif accepts_receives(func) and (
        isinstance(all_receive, Iterator | AsyncIterator) or all_receive is None
    ):
        if not isinstance(all_receive, AsyncIterator) and all_receive is not None:

            async def async_wrapper(
                iterator: Iterator[PhotoType],
            ) -> AsyncIterator[PhotoType]:
                for item in iterator:
                    yield item

            all_receive = async_wrapper(all_receive)
        kwargs["receives"] = all_receive
    return func(input, **kwargs)  # type: ignore[arg-type]


async def acall_func_with_variable_args(
    func: Callable[[Input], Awaitable[Output]]
    | Callable[[Input, TransableConfig], Awaitable[Output]]
    | Callable[[Input, PhotoType, TransableConfig], Awaitable[Output]]
    | Callable[[Input, Iterator[PhotoType]], Awaitable[Output]]
    | Callable[[Input, AsyncIterator[PhotoType]], Awaitable[Output]]
    | Callable[[Input, Iterator[PhotoType], TransableConfig], Awaitable[Output]]
    | Callable[[Input, AsyncIterator[PhotoType], TransableConfig], Awaitable[Output]],
    input: Input,
    all_receive: PhotoType | Iterator[PhotoType] | AsyncIterator[PhotoType] | None,
    config: TransableConfig | None,
    **kwargs: Any,
) -> Output:
    """Asynchronously call a function that may or may not accept a TransableConfig.

    Args:
        func: The function to call.
        input: The input to the function.
        config: The TransableConfig to pass if accepted.
        **kwargs: Additional keyword arguments to pass to the function.
    Returns:
        PhotoType: The result of the function call.
    """
    if accepts_config(func):
        kwargs["config"] = config
    if accepts_receive(func) and not isinstance(all_receive, Iterator):
        kwargs["receive"] = all_receive
    elif accepts_receives(func) and (
        isinstance(all_receive, Iterator | AsyncIterator) or all_receive is None
    ):
        if not isinstance(all_receive, AsyncIterator) and all_receive is not None:

            async def async_wrapper(
                iterator: Iterator[PhotoType],
            ) -> AsyncIterator[PhotoType]:
                for item in iterator:
                    yield item

            all_receive = async_wrapper(all_receive)
        kwargs["receives"] = all_receive
    return await func(input, **kwargs)  # type: ignore[arg-type]


def is_generator(func: Any) -> TypeGuard[Callable[..., Iterator]]:
    """Check if a function is a generator.

    Args:
        func: The function to check.
    Returns:
        TypeGuard[Callable[..., Iterator]: True if the function is a
            generator, False otherwise.
    """
    return inspect.isgeneratorfunction(func) or (
        hasattr(func, "__call__")  # noqa: B004
        and inspect.isgeneratorfunction(func.__call__)
    )


def is_async_generator(
    func: Any,
) -> TypeGuard[Callable[..., AsyncIterator]]:
    """Check if a function is an async generator.

    Args:
        func: The function to check.

    Returns:
        TypeGuard[Callable[..., AsyncIterator]: True if the function is
            an async generator, False otherwise.
    """
    return inspect.isasyncgenfunction(func) or (
        hasattr(func, "__call__")  # noqa: B004
        and inspect.isasyncgenfunction(func.__call__)
    )


def is_async_callable(
    func: Any,
) -> TypeGuard[Callable[..., Awaitable]]:
    """Check if a function is async.

    Args:
        func: The function to check.

    Returns:
        TypeGuard[Callable[..., Awaitable]: True if the function is async,
            False otherwise.
    """
    return asyncio.iscoroutinefunction(func) or (
        hasattr(func, "__call__")  # noqa: B004
        and asyncio.iscoroutinefunction(func.__call__)
    )
