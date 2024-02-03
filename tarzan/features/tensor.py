from dataclasses import dataclass, field

import numpy as np
from tarzan.utils import StreamWrapper


@dataclass
class Tensor:
    """Generic data of arbitrary shape and type."""

    shape: tuple
    dtype: str
    # Automatically constructed
    _type: str = field(default="Tensor", init=False, repr=False)

    def __post_init__(self):
        self.shape = tuple(self.shape)
        if not isinstance(self.dtype, str):
            raise TypeError("dtype must be a string")
        if not is_valid_dtype(self.dtype):
            raise ValueError("dtype must be a valid dtype for numpy")

    def encode_example(self, example):
        if not isinstance(example, np.ndarray):
            example = np.asarray(example, dtype=self.dtype)
        # Ensure the shape and dtype match
        if example.dtype != self.dtype:
            raise ValueError(
                "Dtype {} do not match {}".format(example.dtype, self.dtype)
            )
        assert_shape_match(example.shape, self.shape)
        return example.tobytes()

    def decode_example(self, example):
        if isinstance(example, StreamWrapper):
            fobj = example
            example = example.read()
            fobj.close()
            if not example:
                return None
        shape = [-1 if dim is None else dim for dim in self.shape]
        return np.frombuffer(example, dtype=self.dtype).reshape(shape)


class Dimension:
    __slots__ = ["_value"]

    def __init__(self, value):
        """Creates a new Dimension with the given value."""
        if isinstance(value, int):  # Most common case.
            if value < 0:
                raise ValueError("Dimension %d must be >= 0" % value)
            self._value = value
        elif value is None:
            self._value = None
        elif isinstance(value, Dimension):
            self._value = value._value
        else:
            try:
                self._value = value.__index__()
            except AttributeError:
                raise TypeError(
                    "Dimension value must be integer or None or have "
                    "an __index__ method, got value '{0!r}' with type '{1!r}'".format(
                        value, type(value)
                    )
                )
            if self._value < 0:
                raise ValueError("Dimension %d must be >= 0" % self._value)

    def __repr__(self):
        return "Dimension(%s)" % repr(self._value)

    def __str__(self):
        value = self._value
        return "?" if value is None else str(value)

    def __eq__(self, other):
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return None
        return self._value == other.value

    def __ne__(self, other):
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return None
        return self._value != other.value

    def __int__(self):
        return self._value

    def __index__(self):
        return self._value

    @property
    def value(self):
        return self._value

    def is_compatible_with(self, other):
        other = as_dimension(other)
        return self._value is None or other.value is None or self._value == other.value

    def assert_is_compatible_with(self, other):
        if not self.is_compatible_with(other):
            raise ValueError("Dimensions %s and %s are not compatible" % (self, other))

    def merge_with(self, other):
        other = as_dimension(other)
        self.assert_is_compatible_with(other)
        if self._value is None:
            return Dimension(other.value)
        else:
            return Dimension(self._value)

    def __add__(self, other):
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(self._value + other.value)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(self._value - other.value)

    def __rsub__(self, other):
        other = as_dimension(other)
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(other.value - self._value)

    def __mul__(self, other):
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented

        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(self._value * other.value)

    def __rmul__(self, other):
        return self * other

    def __floordiv__(self, other):
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(self._value // other.value)

    def __rfloordiv__(self, other):
        other = as_dimension(other)
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(other.value // self._value)

    def __truediv__(self, other):
        raise TypeError(
            "unsupported operand type(s) for /: 'Dimension' and '{}', "
            "please use // instead".format(type(other).__name__)
        )

    def __rtruediv__(self, other):
        raise TypeError(
            "unsupported operand type(s) for /: '{}' and 'Dimension', "
            "please use // instead".format(type(other).__name__)
        )

    def __mod__(self, other):
        other = as_dimension(other)
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(self._value % other.value)

    def __rmod__(self, other):
        other = as_dimension(other)
        return other % self

    def __lt__(self, other):
        other = as_dimension(other)
        if self._value is None or other.value is None:
            return None
        else:
            return self._value < other.value

    def __le__(self, other):
        other = as_dimension(other)
        if self._value is None or other.value is None:
            return None
        else:
            return self._value <= other.value

    def __gt__(self, other):
        other = as_dimension(other)
        if self._value is None or other.value is None:
            return None
        else:
            return self._value > other.value

    def __ge__(self, other):
        other = as_dimension(other)
        if self._value is None or other.value is None:
            return None
        else:
            return self._value >= other.value

    def __reduce__(self):
        return Dimension, (self._value,)


def as_dimension(value):
    if isinstance(value, Dimension):
        return value
    else:
        return Dimension(value)


class TensorShape:
    def __init__(self, dims):
        if isinstance(dims, (tuple, list)):  # Most common case.
            self._dims = [Dimension(d) for d in dims]
        elif dims is None:
            self._dims = None
        elif isinstance(dims, TensorShape):
            self._dims = dims.dims
        else:
            try:
                dims_iter = iter(dims)
            except TypeError:
                # Treat as a singleton dimension
                self._dims = [as_dimension(dims)]
            else:
                self._dims = []
                for d in dims_iter:
                    try:
                        self._dims.append(as_dimension(d))
                    except TypeError as e:
                        raise TypeError(
                            "Failed to convert '{0!r}' to a shape: '{1!r}'"
                            "could not be converted to a dimension. A shape should "
                            "either be single dimension (e.g. 10), or an iterable of "
                            "dimensions (e.g. [1, 10, None]).".format(dims, d)
                        ) from e

    def __repr__(self):
        return "TensorShape(%r)" % self._dims

    def __str__(self):
        if self.rank is None:
            return "<unknown>"
        elif self.rank == 1:
            return "(%s,)" % self._dims[0]
        else:
            return "(%s)" % ", ".join(str(d) for d in self._dims)

    @property
    def rank(self):
        if self._dims is not None:
            return len(self._dims)
        return None

    @property
    def dims(self):
        return self._dims

    @property
    def ndims(self):
        return self.rank

    def __len__(self):
        if self._dims is None:
            raise ValueError("Cannot take the length of shape with unknown rank.")
        return len(self._dims)

    def __bool__(self):
        return self._dims is not None

    def __iter__(self):
        if self._dims is None:
            raise ValueError("Cannot iterate over a shape with unknown rank.")
        else:
            return iter(d for d in self._dims)

    def __getitem__(self, key):
        if self._dims is not None:
            if isinstance(key, slice):
                return TensorShape(self._dims[key])
            else:
                return self._dims[key]
        else:
            if isinstance(key, slice):
                start = key.start if key.start is not None else 0
                stop = key.stop

                if key.step is not None:
                    raise ValueError("Steps are not yet handled")
                if stop is None:
                    return unknown_shape()
                elif start < 0 or stop < 0:
                    return unknown_shape()
                else:
                    return unknown_shape(rank=stop - start)
            else:
                return Dimension(None)

    def num_elements(self):
        if self.is_fully_defined():
            size = 1
            for dim in self._dims:
                size *= dim.value
            return size
        else:
            return None

    def merge_with(self, other):
        other = as_shape(other)
        if self._dims is None:
            return other
        else:
            try:
                self.assert_same_rank(other)
                new_dims = []
                for i, dim in enumerate(self._dims):
                    new_dims.append(dim.merge_with(other[i]))
                return TensorShape(new_dims)
            except ValueError:
                raise ValueError("Shapes %s and %s are not compatible" % (self, other))

    def __add__(self, other):
        if not isinstance(other, TensorShape):
            other = TensorShape(other)
        return self.concatenate(other)

    def __radd__(self, other):
        if not isinstance(other, TensorShape):
            other = TensorShape(other)
        return other.concatenate(self)

    def concatenate(self, other):
        other = as_shape(other)
        if self._dims is None or other.dims is None:
            return unknown_shape()
        else:
            return TensorShape(self._dims + other.dims)

    def assert_same_rank(self, other):
        other = as_shape(other)
        if self.rank is not None and other.rank is not None:
            if self.rank != other.rank:
                raise ValueError(
                    "Shapes %s and %s must have the same rank" % (self, other)
                )

    def assert_has_rank(self, rank):
        if self.rank not in (None, rank):
            raise ValueError("Shape %s must have rank %d" % (self, rank))

    def with_rank(self, rank):
        try:
            return self.merge_with(unknown_shape(rank=rank))
        except ValueError:
            raise ValueError("Shape %s must have rank %d" % (self, rank))

    def with_rank_at_least(self, rank):
        if self.rank is not None and self.rank < rank:
            raise ValueError("Shape %s must have rank at least %d" % (self, rank))
        else:
            return self

    def with_rank_at_most(self, rank):
        if self.rank is not None and self.rank > rank:
            raise ValueError("Shape %s must have rank at most %d" % (self, rank))
        else:
            return self

    def is_compatible_with(self, other):
        other = as_shape(other)
        if self._dims is not None and other.dims is not None:
            if self.rank != other.rank:
                return False
            for x_dim, y_dim in zip(self._dims, other.dims):
                if not x_dim.is_compatible_with(y_dim):
                    return False
        return True

    def assert_is_compatible_with(self, other):
        if not self.is_compatible_with(other):
            raise ValueError("Shapes %s and %s are incompatible" % (self, other))

    def most_specific_compatible_shape(self, other):
        other = as_shape(other)
        if self._dims is None or other.dims is None or self.rank != other.rank:
            return unknown_shape()

        dims = [(Dimension(None))] * self.rank
        for i, (d1, d2) in enumerate(zip(self._dims, other.dims)):
            if d1 is not None and d2 is not None and d1 == d2:
                dims[i] = d1
        return TensorShape(dims)

    def is_fully_defined(self):
        return self._dims is not None and all(
            dim.value is not None for dim in self._dims
        )

    def assert_is_fully_defined(self):
        if not self.is_fully_defined():
            raise ValueError("Shape %s is not fully defined" % self)

    def as_list(self):
        if self._dims is None:
            raise ValueError("as_list() is not defined on an unknown TensorShape.")
        return [dim.value for dim in self._dims]

    def __eq__(self, other):
        """Returns True if `self` is equivalent to `other`."""
        try:
            other = as_shape(other)
        except TypeError:
            return NotImplemented
        return self._dims == other.dims

    def __ne__(self, other):
        try:
            other = as_shape(other)
        except TypeError:
            return NotImplemented
        if self.rank is None or other.rank is None:
            raise ValueError("The inequality of unknown TensorShapes is undefined.")
        if self.rank != other.rank:
            return True
        return self._dims != other.dims

    def __reduce__(self):
        return TensorShape, (self._dims,)

    def __concat__(self, other):
        return self.concatenate(other)


def as_shape(shape):
    if isinstance(shape, TensorShape):
        return shape
    else:
        return TensorShape(shape)


def unknown_shape(rank=None, **kwargs):
    if rank is None and "ndims" in kwargs:
        rank = kwargs.pop("ndims")
    if kwargs:
        raise TypeError("Unknown argument: %s" % kwargs)
    if rank is None:
        return TensorShape(None)
    else:
        return TensorShape([Dimension(None)] * rank)


def assert_shape_match(shape1, shape2):
    """Ensure the shape1 match the pattern given by shape2.

    Args:
      shape1 (tuple): Static shape
      shape2 (tuple): Dynamic shape (can contain None)
    """
    shape1 = TensorShape(shape1)
    shape2 = TensorShape(shape2)
    if shape1.ndims is None or shape2.ndims is None:
        raise ValueError(
            "Shapes must have known rank. Got %s and %s." % (shape1.ndims, shape2.ndims)
        )
    shape1.assert_same_rank(shape2)
    shape1.assert_is_compatible_with(shape2)


def is_valid_dtype(dtype_str):
    try:
        np.dtype(dtype_str)
        return True
    except TypeError:
        return False
