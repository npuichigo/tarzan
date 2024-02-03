import numpy as np
import pytest
from tarzan.features import Tensor


@pytest.fixture
def feature():
    return Tensor(shape=(3, 4), dtype='float32')


def test_tensor(feature):
    example = np.random.rand(3, 4).astype('float32')
    np.testing.assert_allclose(feature.decode_example(feature.encode_example(example)), example)
