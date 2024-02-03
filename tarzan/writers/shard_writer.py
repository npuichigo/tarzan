import logging
import os

from tarzan.info import DatasetInfo
from tarzan.writers.base_writer import Writer
from tarzan.writers.tar_writer import TarWriter

logger = logging.getLogger(__name__)


class ShardWriter(Writer):
    """Writer wrapper to split data into multiple shards."""
    DATASET_INFO_FILENAME = "dataset_info.json"

    def __init__(self,
                 path: str,
                 info: DatasetInfo,
                 pattern: str = "%05d",
                 max_count: int = 1000,
                 max_size: int = 3e9):
        self.path = path
        os.makedirs(path, exist_ok=True)

        self.pattern = f"{path}/{pattern}.tar"
        self.info = info
        self.max_count = max_count
        self.max_size = max_size

        self.writer_stream = None
        self.shard = 0
        self.count = 0
        self.size = 0
        self.total_count = 0
        self.fname = None
        self.next_stream()

    def next_stream(self):
        self.finish()
        self.fname = self.pattern % self.shard
        self.shard += 1
        self.writer_stream = TarWriter(self.fname, features=self.info.features)
        self.count = 0
        self.size = 0

    def write(self, obj):
        if self.writer_stream is None or self.count >= self.max_count or self.size > self.max_size:
            self.next_stream()
        size = self.writer_stream.write(str(self.count), obj)
        self.count += 1
        self.size += size
        self.total_count += 1

    def finish(self):
        if self.writer_stream is not None:
            self.writer_stream.close()
            assert self.fname is not None
            self.info.file_list.append(os.path.basename(self.fname))
            self.writer_stream = None

    def close(self):
        logger.info(f"{self.total_count} examples have been written to {self.shard} shards")
        self.finish()
        self.info.write_to_json(f"{self.path}/{self.DATASET_INFO_FILENAME}", pretty_print=True)
