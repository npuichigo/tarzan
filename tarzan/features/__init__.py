from typing import Union

from tarzan.features.audio import Audio
from tarzan.features.json import Json
from tarzan.features.scalar import Scalar
from tarzan.features.sequence import Sequence
from tarzan.features.tensor import Tensor
from tarzan.features.text import Text

FeatureType = Union[
    dict,
    list,
    tuple,
    Audio,
    Json,
    Scalar,
    Sequence,
    Text,
    Tensor,
]

from tarzan.features.features import Features
