from dataclasses import dataclass, field

from tarzan.utils import StreamWrapper


@dataclass
class Text:
    # Automatically constructed
    _type: str = field(default="Text", init=False, repr=False)

    def encode_example(self, example):
        return example.encode("utf-8")

    def decode_example(self, example):
        if isinstance(example, StreamWrapper):
            fobj = example
            example = example.read()
            fobj.close()
            if not example:
                return None
        return example.decode("utf-8")
