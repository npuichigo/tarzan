import pytest

from tarzan.features import Features, Text
from tarzan.info import DatasetInfo


@pytest.fixture
def info():
    return DatasetInfo(
        description="A test dataset",
        features=Features({"text": Text()}),
        metadata={"key": "value"},
    )


def test_save_to_json(info, tmpdir):
    info.write_to_json(tmpdir / "dataset_info.json", pretty_print=True)
    assert (tmpdir / "dataset_info.json").exists()

    assert DatasetInfo.from_json(tmpdir / "dataset_info.json") == info
