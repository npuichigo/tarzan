import pytest
import tarfile

from tarzan.features import Features, Text
from tarzan.info import DatasetInfo
from tarzan.writers import TarWriter


@pytest.fixture
def info():
    return DatasetInfo(
        description="A test dataset",
        features=Features({"text": Text()}),
        metadata={"key": "value"},
    )


def test_chunk_writer(info, tmpdir):
    with TarWriter(f"{tmpdir}/fake.tar", info.features) as writer:
        for i in range(3):
            writer.write(str(i), {"text": f"hello_{i}"})

    assert (tmpdir / "fake.tar").exists()

    with tarfile.open(f"{tmpdir}/fake.tar", "r") as tar:
        print(tar.getnames())
        assert sorted(tar.getnames()) == ['0', '0/text', '1', '1/text', '2', '2/text']
        for i in range(3):
            content = tar.extractfile(f"{i}/text").read()
            assert info.features.decode_column([content], 'text')[0] == f"hello_{i}"
