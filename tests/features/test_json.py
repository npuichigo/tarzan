import pytest
from tarzan.features import Json


@pytest.fixture
def feature():
    return Json()


def test_scalar(feature):
    example = {'a': 1, 'b': 2}
    assert feature.decode_example(feature.encode_example(example)) == example
