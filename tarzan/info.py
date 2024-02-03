import copy
import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from tarzan.features.features import Features
from tarzan.utils import asdict, update_dict

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a dataset."""

    description: str = dataclasses.field(default_factory=str)
    file_list: List[str] = dataclasses.field(default_factory=list)
    features: Optional[Features] = None
    size_in_bytes: Optional[int] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        # Convert back to the correct classes when we reload from dict
        if self.features is not None and not isinstance(self.features, Features):
            self.features = Features.from_dict(self.features)

    def write_to_json(self, dataset_info_file, pretty_print=False):
        """Write `DatasetInfo` and license (if present) as JSON files to `dataset_info_dir`.

        Args:
            dataset_info_file (`str`):
                Destination json file.
            pretty_print (`bool`, defaults to `False`):
                If `True`, the JSON will be pretty-printed with the indent level of 4.
        """
        if self.file_list is None:
            logger.warning(
                "No file list provided, the dataset info will be incomplete."
            )
        with open(dataset_info_file, "wb") as f:
            self._dump_info(f, pretty_print=pretty_print)

    def _dump_info(self, file, pretty_print=False):
        """Dump info in `file` file-like object open in bytes mode (to support remote files)"""
        file.write(
            json.dumps(asdict(self), indent=4 if pretty_print else None).encode("utf-8")
        )

    @classmethod
    def from_json(cls, dataset_info_file: str) -> "DatasetInfo":
        """Create [`DatasetInfo`] from the JSON file `dataset_info_file`.

        This function updates all the dynamically generated fields (num_examples,
        hash, time of creation,...) of the [`DatasetInfo`].

        This will overwrite all previous metadata.

        Args:
            dataset_info_file (`str`):
                The Json file of dataset info.
        """
        logger.info(f"Loading Dataset info from {dataset_info_file}")
        with open(dataset_info_file, "r", encoding="utf-8") as f:
            dataset_info_dict = json.load(f)
        return cls.from_dict(dataset_info_dict)

    @classmethod
    def from_dict(cls, dataset_info_dict: Dict) -> "DatasetInfo":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dataset_info_dict.items() if k in field_names})

    def update(self, other_dataset_info: "DatasetInfo", ignore_none=True):
        self_dict = self.__dict__
        self.__dict__ = update_dict(
            self_dict,
            {
                k: copy.deepcopy(v)
                for k, v in other_dataset_info.__dict__.items()
                if (v is not None or not ignore_none)
            },
        )

    def copy(self) -> "DatasetInfo":
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})
