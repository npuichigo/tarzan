import pytest
from tarzan.features import Text


@pytest.fixture
def feature():
    return Text()


def test_scalar(feature):
    example = "hello, world"
    assert feature.decode_example(feature.encode_example(example)) == example
