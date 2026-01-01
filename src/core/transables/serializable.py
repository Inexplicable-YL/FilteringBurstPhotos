from __future__ import annotations

import contextlib
import logging
from abc import ABC
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    TypedDict,
)
from typing_extensions import override

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

logger = logging.getLogger(__name__)


class BaseSerialized(TypedDict):
    """Base class for serialized objects.

    Parameters:
        id: The unique identifier of the object.
        name: The name of the object. Optional.
    """

    id: list[str]
    name: NotRequired[str]


class SerializedConstructor(BaseSerialized):
    """Serialized constructor.

    Parameters:
        type: The type of the object. Must be "constructor".
        kwargs: The constructor arguments.
    """

    type: Literal["constructor"]
    kwargs: dict[str, Any]


class SerializedNotImplemented(BaseSerialized):
    """Serialized not implemented.

    Parameters:
        type: The type of the object. Must be "not_implemented".
        repr: The representation of the object. Optional.
    """

    type: Literal["not_implemented"]
    repr: str | None


def try_neq_default(value: Any, key: str, model: BaseModel) -> bool:
    """Try to determine if a value is different from the default.

    Args:
        value: The value.
        key: The key.
        model: The pydantic model.

    Returns:
        Whether the value is different from the default.

    Raises:
        Exception: If the key is not in the model.
    """
    field = type(model).model_fields[key]
    return _try_neq_default(value, field)


def _try_neq_default(value: Any, field: FieldInfo) -> bool:
    # Handle edge case: inequality of two objects does not evaluate to a bool (e.g. two
    # Pandas DataFrames).
    try:
        return bool(field.get_default() != value)
    except Exception as _:
        try:
            return all(field.get_default() != value)
        except Exception as _:
            try:
                return value is not field.default
            except Exception as _:
                return False


class Serializable(BaseModel, ABC):
    """Serializable base class.

    This class is used to serialize objects to JSON.

    It relies on the following methods and properties:

    - `is_lc_serializable`: Is this class serializable?
        By design, even if a class inherits from Serializable, it is not serializable by
        default. This is to prevent accidental serialization of objects that should not
        be serialized.
    - `get_lc_namespace`: Get the namespace of the langchain object.
        During deserialization, this namespace is used to identify
        the correct class to instantiate.
        Please see the `Reviver` class in `langchain_core.load.load` for more details.
        During deserialization an additional mapping is handle
        classes that have moved or been renamed across package versions.
    - `lc_secrets`: A map of constructor argument names to secret ids.
    - `lc_attributes`: List of additional attribute names that should be included
        as part of the serialized representation.
    """

    # Remove default BaseModel init docstring.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """"""  # noqa: D419
        super().__init__(*args, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Is this class serializable?

        By design, even if a class inherits from Serializable, it is not serializable by
        default. This is to prevent accidental serialization of objects that should not
        be serialized.

        Returns:
            Whether the class is serializable. Default is False.
        """
        return False

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.

        For example, if the class is `langchain.llms.openai.OpenAI`, then the
        namespace is ["langchain", "llms", "openai"]
        """
        return cls.__module__.split(".")

    @classmethod
    def lc_id(cls) -> list[str]:
        """A unique identifier for this class for serialization purposes.

        The unique identifier is a list of strings that describes the path
        to the object.
        """
        # Pydantic generics change the class name. So we need to do the following
        if (
            "origin" in cls.__pydantic_generic_metadata__
            and cls.__pydantic_generic_metadata__["origin"] is not None
        ):
            original_name = cls.__pydantic_generic_metadata__["origin"].__name__
        else:
            original_name = cls.__name__
        return [*cls.get_lc_namespace(), original_name]

    model_config = ConfigDict(
        extra="ignore",
    )

    @override
    def __repr_args__(self) -> Any:
        return [
            (k, v)
            for k, v in super().__repr_args__()
            if (k not in type(self).model_fields or try_neq_default(v, k, self))
        ]

    def to_json(self) -> SerializedConstructor | SerializedNotImplemented:
        """Serialize the object to JSON.

        Returns:
            A json serializable object or a SerializedNotImplemented object.
        """
        if not self.is_lc_serializable():
            return self.to_json_not_implemented()

        model_fields = type(self).model_fields
        # Get latest values for kwargs if there is an attribute with same name
        lc_kwargs = {}
        for k, v in self:
            if not _is_field_useful(self, k, v):
                continue
            # Do nothing if the field is excluded
            if k in model_fields and model_fields[k].exclude:
                continue

            lc_kwargs[k] = getattr(self, k, v)
        return {"type": "constructor", "id": self.lc_id(), "kwargs": lc_kwargs}

    def to_json_not_implemented(self) -> SerializedNotImplemented:
        """Serialize a "not implemented" object."""
        return to_json_not_implemented(self)


def _is_field_useful(inst: Serializable, key: str, value: Any) -> bool:
    """Check if a field is useful as a constructor argument.

    Args:
        inst: The instance.
        key: The key.
        value: The value.

    Returns:
        Whether the field is useful. If the field is required, it is useful.
        If the field is not required, it is useful if the value is not None.
        If the field is not required and the value is None, it is useful if the
        default value is different from the value.
    """
    field = type(inst).model_fields.get(key)
    if not field:
        return False

    if field.is_required():
        return True

    # Handle edge case: a value cannot be converted to a boolean (e.g. a
    # Pandas DataFrame).
    try:
        value_is_truthy = bool(value)
    except Exception as _:
        value_is_truthy = False

    if value_is_truthy:
        return True

    # Value is still falsy here!
    if field.default_factory is dict and isinstance(value, dict):
        return False

    # Value is still falsy here!
    if field.default_factory is list and isinstance(value, list):
        return False

    value_neq_default = _try_neq_default(value, field)

    # If value is falsy and does not match the default
    return value_is_truthy or value_neq_default


def to_json_not_implemented(obj: object) -> SerializedNotImplemented:
    """Serialize a "not implemented" object.

    Args:
        obj: object to serialize.

    Returns:
        SerializedNotImplemented
    """
    _id: list[str] = []
    try:
        if hasattr(obj, "__name__"):
            _id = [*obj.__module__.split("."), obj.__name__]  # pyright: ignore[reportAttributeAccessIssue]
        elif hasattr(obj, "__class__"):
            _id = [*obj.__class__.__module__.split("."), obj.__class__.__name__]
    except Exception:
        logger.debug("Failed to serialize object", exc_info=True)

    result: SerializedNotImplemented = {
        "type": "not_implemented",
        "id": _id,
        "repr": None,
    }
    with contextlib.suppress(Exception):
        result["repr"] = repr(obj)
    return result
