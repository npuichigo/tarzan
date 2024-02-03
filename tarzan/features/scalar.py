from dataclasses import dataclass, field

from tarzan.features.tensor import Tensor


@dataclass
class Scalar(Tensor):
    # Automatically constructed
    _type: str = field(default="Scalar", init=False, repr=False)

    def __init__(self, dtype, **kwargs):
        super().__init__(shape=(), dtype=dtype)
