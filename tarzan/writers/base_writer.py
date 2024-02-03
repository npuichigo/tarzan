from abc import ABC, abstractmethod
from typing import Any


class Writer(ABC):
    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    @abstractmethod
    def write(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
