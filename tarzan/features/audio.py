import io
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import librosa
import numpy as np

from tarzan.features.tensor import Tensor

logger = logging.getLogger(__name__)


class AudioDecoder:
    """Read audio during decoding."""

    def __init__(self, fobj, dtype, shape, sample_rate):
        self._fobj = fobj
        self._dtype = dtype
        self._shape = shape
        self._sample_rate = sample_rate
        self._channels = shape[1] if len(shape) > 1 else 1

    def read_all(self):
        self._fobj.seek(0)
        try:
            return librosa.load(
                self._fobj, sr=self._sample_rate, mono=self._channels == 1, dtype=self._dtype
            )
        except Exception as e:
            logger.error(f"Error reading audio: {e}")
            return None, None

    def read_range(self, start: float, end: float):
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if start > end:
            raise ValueError(f"end must be >= start, got {end} < {start}")
        duration = end - start
        self._fobj.seek(0)
        try:
            return librosa.load(
                self._fobj,
                sr=self._sample_rate,
                mono=self._channels == 1,
                dtype=self._dtype,
                offset=start,
                duration=duration,
            )
        except Exception as e:
            logger.error(f"Error reading audio: {e}")
            return None, None


@dataclass
class Audio(Tensor):
    """`FeatureConnector` for audio."""

    sample_rate: Optional[int]
    # Automatically constructed
    _type: str = field(default="Audio", init=False, repr=False)

    def __init__(
        self, shape=(None,), dtype="float32", sample_rate=None, lazy_decode=True
    ):
        super().__init__(shape=shape, dtype=dtype)
        self.sample_rate = sample_rate

    def encode_example(self, audio_or_path_or_fobj):
        if isinstance(audio_or_path_or_fobj, (str, os.PathLike)):
            filename = os.fspath(audio_or_path_or_fobj)
            with open(filename, "rb") as audio_f:
                audio = audio_f.read()
        elif isinstance(audio_or_path_or_fobj, np.ndarray):
            raise ValueError("Audio must be a path or file-like object.")
        else:
            audio = audio_or_path_or_fobj.read()
        return audio

    def decode_example(self, audio_data):
        if isinstance(audio_data, bytes):
            audio_fobj = io.BytesIO(audio_data)
        else:
            audio_fobj = audio_data
        return AudioDecoder(audio_fobj, self.dtype, self.shape, self.sample_rate)
