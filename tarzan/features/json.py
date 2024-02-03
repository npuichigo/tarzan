import json
from dataclasses import dataclass, field

from tarzan.features.text import Text


@dataclass
class Json(Text):
    # Automatically constructed
    _type: str = field(default="Json", init=False, repr=False)

    def encode_example(self, example):
        return super().encode_example(json.dumps(example))

    def decode_example(self, example):
        return json.loads(super().decode_example(example))
