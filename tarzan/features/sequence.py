from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class Sequence:
    """Construct a list of feature from a single type or a dict of types.
    Mostly here for compatiblity with tfds.

    Args:
        feature:
            A list of features of a single type or a dictionary of types.
        length (`int`):
            Length of the sequence.

    Example:

    ```py
    >>> from torchtts.data.v2.features import Features, Sequence, Text, Scalar
    >>> features = Features({'post': Sequence(feature={'text': Text(), 'upvotes': Scalar(dtype='int32')})})
    >>> features
    {'post': Sequence(feature={'text': Text(), 'upvotes': Scalar(shape=(), dtype='int32')}, length=-1)}
    ```
    """

    feature: Any
    length: int = -1
    # Automatically constructed
    dtype: ClassVar[str] = "list"
    _type: str = field(default="Sequence", init=False, repr=False)
