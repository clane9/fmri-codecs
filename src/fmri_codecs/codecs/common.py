import io

import numpy as np


def encode_numpy(x: np.ndarray) -> bytes:
    with io.BytesIO() as f:
        np.save(f, x)
        x = f.getvalue()
    return x


def decode_numpy(x: bytes) -> np.ndarray:
    with io.BytesIO(x) as f:
        x = np.load(f)
    return x
