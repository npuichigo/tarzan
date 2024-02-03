import io
import logging
import os.path
import tarfile
from typing import Any, Dict

from tarzan.features import Features
from tarzan.writers.base_writer import Writer

logger = logging.getLogger(__name__)


def _add_dir_to_tar(tar, name):
    tar_info = tarfile.TarInfo(name=name)
    tar_info.type = tarfile.DIRTYPE
    tar_info.mode = 0o755
    tar.addfile(tar_info)


def _write_to_tar(tar, prefix, data):
    size = 0
    if isinstance(data, dict):
        _add_dir_to_tar(tar, prefix)
        for key, value in data.items():
            if key.isdigit():
                raise ValueError(
                    "Keys cannot be integers since they are reserved for indexing."
                )
            size += _write_to_tar(tar, f"{prefix}/{key}", value)
    elif isinstance(data, list):
        _add_dir_to_tar(tar, prefix)
        for i, value in enumerate(data):
            size += _write_to_tar(tar, f"{prefix}/{i}", value)
    else:
        # Assumed to be bytes or None
        tarinfo = tarfile.TarInfo(name=prefix)
        if data is None:
            size = 0
            tar.addfile(tarinfo)
        else:
            size = len(data)
            tarinfo.size = len(data)
            tar.addfile(tarinfo, fileobj=io.BytesIO(data))
    return size


class TarWriter(Writer):
    """TarWriter writes data to tar files with nested directory structure."""

    def __init__(
        self,
        path: str,
        features: Features,
    ):
        self.tar_stream = tarfile.open(path, "w|")
        self.features = features
        self.written_idx = set()

    def write(self, idx: str, objects: Dict[str, Any]):
        if objects.keys() != self.features.keys():
            raise ValueError(
                f"Keys {objects.keys()} do not match specified features {self.features}"
            )
        if idx in self.written_idx:
            raise ValueError(f"Index {idx} already written")
        written_obj = self.features.encode_example(objects)
        size = _write_to_tar(self.tar_stream, idx, written_obj)
        self.written_idx.add(idx)
        return size

    def close(self):
        self.tar_stream.close()
