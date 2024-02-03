import logging
import os
import tarfile
from typing import Dict, Iterator, List, Optional, Tuple

from more_itertools import peekable

from tarzan.features import Features
from tarzan.info import DatasetInfo
from tarzan.utils import StreamWrapper

logger = logging.getLogger(__name__)


def _get_tar_index(tarinfo: tarfile.TarInfo):
    return tarinfo.name.split("/")[0]


def _feature_stream(stream: StreamWrapper):
    peekable_tar = peekable(stream)

    feature_group = []

    while True:
        # Group successive features with same index prefix
        try:
            tarinfo = next(peekable_tar)
        except StopIteration:
            break
        index = _get_tar_index(tarinfo)
        feature_group.append(tarinfo)

        try:
            tarinfo = peekable_tar.peek()
        # Last index case
        except StopIteration:
            yield index, _compose_feature(stream, feature_group)
            break

        # Successive index case
        if _get_tar_index(tarinfo) != index:
            yield index, _compose_feature(stream, feature_group)
            feature_group.clear()


def _get_inner_fobj(tar_stream: StreamWrapper, tarinfo: tarfile.TarInfo):
    extracted_fobj = tar_stream.extractfile(tarinfo)
    if extracted_fobj is None:
        raise tarfile.ExtractError(
            f"failed to extract file {tarinfo.name} from {tar_stream.name}"
        )
    inner_pathname = os.path.normpath(os.path.join(tar_stream.name, tarinfo.name))
    return StreamWrapper(extracted_fobj, tar_stream, name=inner_pathname)


def _compose_feature(tar_stream: StreamWrapper, group: List[tarfile.TarInfo]):
    """Compose feature (maybe nested) with same index prefix."""
    # Single feature case
    if len(group) == 1:
        tarinfo = group[0]
        # Single feature must be a file
        assert tarinfo.isfile()
        return _get_inner_fobj(tar_stream, tarinfo)
    # Nested feature case
    else:
        nested = {}
        for tarinfo in group:
            if tarinfo.isdir():
                continue
            parts = tarinfo.name.split("/")
            current_dict = nested
            for part in parts[1:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            current_dict[parts[-1]] = _get_inner_fobj(tar_stream, tarinfo)
        return _transform_dict(nested)


def _transform_dict(input_dict):
    """Transform a nested dictionary to a list if all keys are integers."""
    result_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            value = _transform_dict(value)
        result_dict[key] = value
    if all(key.isdigit() for key in result_dict.keys()):
        result_dict = [
            value
            for key, value in sorted(result_dict.items(), key=lambda item: int(item[0]))
        ]

    return result_dict


class TarReader:
    def __init__(
        self,
        tar_files: List[str],
        features: Features,
        mode: str = "r:*",
    ) -> None:
        super().__init__()
        self.tar_files = tar_files
        self.features = features
        self.mode = mode

    @classmethod
    def from_dataset_info(cls, dataset_info_file: str) -> "TarReader":
        info = DatasetInfo.from_json(dataset_info_file)
        data_dir = os.path.dirname(dataset_info_file)
        tar_files = [os.path.join(data_dir, f) for f in info.file_list]
        return cls(tar_files=tar_files, features=info.features)

    def __iter__(self) -> Iterator[Tuple[str, str, Dict]]:
        for tar_file in self.tar_files:
            tar_stream = StreamWrapper(
                tarfile.open(tar_file, self.mode),
                name=tar_file,
            )
            example_stream = _feature_stream(tar_stream)

            try:
                while True:
                    try:
                        index, example = next(example_stream)
                    except StopIteration:
                        break
                    decoded_example = self.features.decode_example(example)
                    yield tar_file, index, decoded_example
            finally:
                tar_stream.autoclose()
