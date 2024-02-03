import copy
from dataclasses import fields
from functools import wraps
from typing import Any, Dict

from tarzan import features
from tarzan.features.sequence import Sequence
from tarzan.utils import asdict, zip_dict


def encode_nested_example(schema, obj, level=0):
    """Encode a nested example.
    This is used since some features (in particular ClassLabel) have some logic during encoding.

    To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or
    empty (if it is a sequence) has to be encoded. If the first element needs to be encoded, then all the elements of
    the list will be encoded, otherwise they'll stay the same.
    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(schema, dict):
        if level == 0 and obj is None:
            raise ValueError("Got None but expected a dictionary instead")
        return (
            {
                k: encode_nested_example(sub_schema, sub_obj, level=level + 1)
                for k, (sub_schema, sub_obj) in zip_dict(schema, obj)
            }
            if obj is not None
            else None
        )

    elif isinstance(schema, (list, tuple)):
        sub_schema = schema[0]
        if obj is None:
            return None
        else:
            if len(obj) > 0:
                return [
                    encode_nested_example(sub_schema, o, level=level + 1) for o in obj
                ]
            return list(obj)
    elif isinstance(schema, Sequence):
        if obj is None:
            return None
        # We allow to reverse list of dict => dict of list for compatibility with tfds
        if isinstance(schema.feature, dict):
            # dict of list to fill
            list_dict = {}
            if isinstance(obj, (list, tuple)):
                # obj is a list of dict
                for k, dict_tuples in zip_dict(schema.feature, *obj):
                    list_dict[k] = [
                        encode_nested_example(dict_tuples[0], o, level=level + 1)
                        for o in dict_tuples[1:]
                    ]
                return list_dict
            else:
                # obj is a single dict
                for k, (sub_schema, sub_objs) in zip_dict(schema.feature, obj):
                    list_dict[k] = [
                        encode_nested_example(sub_schema, o, level=level + 1)
                        for o in sub_objs
                    ]
                return list_dict
        # schema.feature is not a dict
        if isinstance(obj, str):  # don't interpret a string as a list
            raise ValueError(f"Got a string but expected a list instead: '{obj}'")
        else:
            if len(obj) > 0:
                return [
                    encode_nested_example(schema.feature, o, level=level + 1)
                    for o in obj
                ]
            return list(obj)
    # Object with special encoding:
    else:
        return schema.encode_example(obj) if obj is not None else None


def decode_nested_example(schema, obj):
    """Decode a nested example.
    This is used since some features (in particular Audio and Image) have some logic during decoding.

    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(schema, dict):
        return (
            {
                k: decode_nested_example(sub_schema, sub_obj)
                for k, (sub_schema, sub_obj) in zip_dict(schema, obj)
            }
            if obj is not None
            else None
        )
    elif isinstance(schema, (list, tuple)):
        sub_schema = schema[0]
        if obj is None:
            return None
        else:
            if len(obj) > 0:
                return [decode_nested_example(sub_schema, o) for o in obj]
            return list(obj)
    elif isinstance(schema, Sequence):
        # We allow to reverse list of dict => dict of list for compatibility with tfds
        if isinstance(schema.feature, dict):
            return {
                k: decode_nested_example([schema.feature[k]], obj[k])
                for k in schema.feature
            }
        else:
            return decode_nested_example([schema.feature], obj)
    # Object with special decoding:
    else:
        # we pass the token to read and decode files from private repositories in streaming mode
        return schema.decode_example(obj) if obj is not None else None


def generate_from_dict(obj: Any):
    """Regenerate the nested feature object from a deserialized dict.
    We use the '_type' fields to get the dataclass name to load.

    generate_from_dict is the recursive helper for Features.from_dict, and allows for a convenient constructor syntax
    to define features from deserialized JSON dictionaries. This function is used in particular when deserializing
    a :class:`DatasetInfo` that was dumped to a JSON object.
    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(obj, list):
        return [generate_from_dict(value) for value in obj]
    # Otherwise we have a dict or a dataclass
    if "_type" not in obj or isinstance(obj["_type"], dict):
        return {key: generate_from_dict(value) for key, value in obj.items()}
    class_type = getattr(features, obj.pop("_type"))

    if class_type == Sequence:
        return Sequence(
            feature=generate_from_dict(obj["feature"]), length=obj.get("length", -1)
        )

    field_names = {f.name for f in fields(class_type)}
    return class_type(**{k: v for k, v in obj.items() if k in field_names})


def keep_features_dicts_synced(func):
    """
    Wrapper to keep the secondary dictionary, which tracks whether keys are decodable, of the :class:`datasets.Features`
    object in sync with the main dictionary.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            self: "Features" = args[0]
            args = args[1:]
        else:
            self: "Features" = kwargs.pop("self")
        out = func(self, *args, **kwargs)
        return out

    wrapper._decorator_name_ = "_keep_dicts_synced"
    return wrapper


class Features(dict):
    """A special dictionary that defines the internal structure of a dataset."""

    def __init__(*args, **kwargs):
        # self not in the signature to allow passing self as a kwarg
        if not args:
            raise TypeError(
                "descriptor '__init__' of 'Features' object needs an argument"
            )
        self, *args = args
        super(Features, self).__init__(*args, **kwargs)

    __setitem__ = keep_features_dicts_synced(dict.__setitem__)
    __delitem__ = keep_features_dicts_synced(dict.__delitem__)
    update = keep_features_dicts_synced(dict.update)
    setdefault = keep_features_dicts_synced(dict.setdefault)
    pop = keep_features_dicts_synced(dict.pop)
    popitem = keep_features_dicts_synced(dict.popitem)
    clear = keep_features_dicts_synced(dict.clear)

    def __reduce__(self):
        return Features, (dict(self),)

    @classmethod
    def from_dict(cls, dic) -> "Features":
        """
        Construct [`Features`] from dict.

        Regenerate the nested feature object from a deserialized dict.
        We use the `_type` key to infer the dataclass name of the feature `FieldType`.

        It allows for a convenient constructor syntax
        to define features from deserialized JSON dictionaries. This function is used in particular when deserializing
        a [`DatasetInfo`] that was dumped to a JSON object.

        Args:
            dic (`dict[str, Any]`):
                Python dictionary.

        Returns:
            `Features`

        Example::
            >>> Features.from_dict({'_type': {'dtype': 'string', 'id': None, '_type': 'Value'}})
            {'_type': Value(dtype='string', id=None)}
        """
        obj = generate_from_dict(dic)
        return cls(**obj)

    def to_dict(self):
        return asdict(self)

    def encode_example(self, example):
        """
        Encode example into a format for storage.

        Args:
            example (`dict[str, Any]`):
                Data in a Dataset row.

        Returns:
            `dict[str, Any]`
        """
        return encode_nested_example(self, example)

    def encode_column(self, column, column_name: str):
        """
        Encode column into a format for storage.

        Args:
            column (`list[Any]`):
                Data in a Dataset column.
            column_name (`str`):
                Dataset column name.

        Returns:
            `list[Any]`
        """
        return [encode_nested_example(self[column_name], obj) for obj in column]

    def encode_batch(self, batch):
        """
        Encode batch into a format for storage.

        Args:
            batch (`dict[str, list[Any]]`):
                Data in a Dataset batch.

        Returns:
            `dict[str, list[Any]]`
        """
        encoded_batch = {}
        if set(batch) != set(self):
            raise ValueError(
                f"Column mismatch between batch {set(batch)} and features {set(self)}"
            )
        for key, column in batch.items():
            encoded_batch[key] = [
                encode_nested_example(self[key], obj) for obj in column
            ]
        return encoded_batch

    def decode_example(self, example: Dict):
        """Decode example with custom feature decoding.

        Args:
            example (`dict[str, Any]`):
                Dataset row data.

        Returns:
            `dict[str, Any]`
        """

        return {
            column_name: decode_nested_example(feature, value)
            for column_name, (feature, value) in zip_dict(
                {key: value for key, value in self.items() if key in example}, example
            )
        }

    def decode_column(self, column: list, column_name: str):
        """Decode column with custom feature decoding.

        Args:
            column (`list[Any]`):
                Dataset column data.
            column_name (`str`):
                Dataset column name.

        Returns:
            `list[Any]`
        """
        return [
            (
                decode_nested_example(self[column_name], value)
                if value is not None
                else None
            )
            for value in column
        ]

    def decode_batch(self, batch: Dict):
        """Decode batch with custom feature decoding.

        Args:
            batch (`dict[str, list[Any]]`):
                Dataset batch data.

        Returns:
            `dict[str, list[Any]]`
        """
        decoded_batch = {}
        for column_name, column in batch.items():
            decoded_batch[column_name] = [
                (
                    decode_nested_example(self[column_name], value)
                    if value is not None
                    else None
                )
                for value in column
            ]
        return decoded_batch

    def copy(self) -> "Features":
        """
        Make a deep copy of [`Features`].

        Returns:
            [`Features`]
        """
        return copy.deepcopy(self)
