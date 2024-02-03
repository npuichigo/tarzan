import pytest
from tarzan.features import Scalar


@pytest.fixture
def feature():
    return Scalar(dtype='int32')


def test_scalar(feature):
    example = 1
    assert feature.decode_example(feature.encode_example(example)) == example
