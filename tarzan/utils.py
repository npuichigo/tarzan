import collections
import copy
import itertools
from dataclasses import fields, is_dataclass
from typing import Dict, Any


def zip_dict(*dicts):
    """Iterate over items of dictionaries grouped by their keys."""
    for key in set(itertools.chain(*dicts)):  # set merge all keys
        # Will raise KeyError if the dict don't have the same keys
        yield key, tuple(d[key] for d in dicts)


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def asdict(obj):
    """Convert an object to its dictionary representation recursively.

    <Added version="2.4.0"/>
    """

    # Implementation based on https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict

    def _is_dataclass_instance(obj):
        # https://docs.python.org/3/library/dataclasses.html#dataclasses.is_dataclass
        return is_dataclass(obj) and not isinstance(obj, type)

    def _asdict_inner(obj):
        if _is_dataclass_instance(obj):
            result = {}
            for f in fields(obj):
                value = _asdict_inner(getattr(obj, f.name))
                if (
                    not f.init
                    or value != f.default
                    or f.metadata.get("include_in_asdict_even_if_is_default", False)
                ):
                    result[f.name] = value
            return result
        elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
            # obj is a namedtuple
            return type(obj)(*[_asdict_inner(v) for v in obj])
        elif isinstance(obj, (list, tuple)):
            # Assume we can create an object of this type by passing in a
            # generator (which is not true for namedtuples, handled
            # above).
            return type(obj)(_asdict_inner(v) for v in obj)
        elif isinstance(obj, dict):
            return {_asdict_inner(k): _asdict_inner(v) for k, v in obj.items()}
        else:
            return copy.deepcopy(obj)

    if not isinstance(obj, dict) and not _is_dataclass_instance(obj):
        raise TypeError(f"{obj} is not a dict or a dataclass")

    return _asdict_inner(obj)


class StreamWrapper:
    """
    StreamWrapper is introduced to wrap file handler generated by DataPipe operation like `FileOpener`.

    StreamWrapper would guarantee the wrapped file handler is closed when it's out of scope.
    """

    session_streams: Dict[Any, int] = {}
    debug_unclosed_streams: bool = False

    def __init__(self, file_obj, parent_stream=None, name=None):
        self.file_obj = file_obj
        self.child_counter = 0
        self.parent_stream = parent_stream
        self.close_on_last_child = False
        self.name = name
        self.closed = False
        if parent_stream is not None:
            if not isinstance(parent_stream, StreamWrapper):
                raise RuntimeError(f'Parent stream should be StreamWrapper, {type(parent_stream)} was given')
            parent_stream.child_counter += 1
            self.parent_stream = parent_stream
        if StreamWrapper.debug_unclosed_streams:
            StreamWrapper.session_streams[self] = 1

    @classmethod
    def close_streams(cls, v, depth=0):
        """Traverse structure and attempts to close all found StreamWrappers on best effort basis."""
        if depth > 10:
            return
        if isinstance(v, StreamWrapper):
            v.close()
        else:
            # Traverse only simple structures
            if isinstance(v, dict):
                for vv in v.values():
                    cls.close_streams(vv, depth=depth + 1)
            elif isinstance(v, (list, tuple)):
                for vv in v:
                    cls.close_streams(vv, depth=depth + 1)

    def __getattr__(self, name):
        file_obj = self.__dict__['file_obj']
        return getattr(file_obj, name)

    def close(self, *args, **kwargs):
        if self.closed:
            return
        if StreamWrapper.debug_unclosed_streams:
            del StreamWrapper.session_streams[self]
        if hasattr(self, "parent_stream") and self.parent_stream is not None:
            self.parent_stream.child_counter -= 1
            if not self.parent_stream.child_counter and self.parent_stream.close_on_last_child:
                self.parent_stream.close()
        try:
            self.file_obj.close(*args, **kwargs)
        except AttributeError:
            pass
        self.closed = True

    def autoclose(self):
        """Automatically close stream when all child streams are closed or if there are none."""
        self.close_on_last_child = True
        if self.child_counter == 0:
            self.close()

    def __dir__(self):
        attrs = list(self.__dict__.keys()) + list(StreamWrapper.__dict__.keys())
        attrs += dir(self.file_obj)
        return list(set(attrs))

    def __del__(self):
        if not self.closed:
            self.close()

    def __iter__(self):
        yield from self.file_obj

    def __next__(self):
        return next(self.file_obj)

    def __repr__(self):
        if self.name is None:
            return f"StreamWrapper<{self.file_obj!r}>"
        else:
            return f"StreamWrapper<{self.name},{self.file_obj!r}>"

    def __getstate__(self):
        return self.file_obj

    def __setstate__(self, obj):
        self.file_obj = obj