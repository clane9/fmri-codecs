import numpy as np


class Codec:
    def encode(self, series: np.ndarray) -> bytes: ...

    def decode(self, buf: bytes) -> np.ndarray: ...
