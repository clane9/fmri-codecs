import zlib
from typing import Literal

import numpy as np
import zstd

from fmri_codecs.codecs.registry import register_codec
from fmri_codecs.codecs.common import encode_numpy, decode_numpy


class QuantizeCodec:
    def __init__(
        self,
        n_bins: int = 4096,
        compression: Literal["gzip", "zstd", "none"] = "zstd",
        vmax: float = 5.0,
    ):
        self.n_bins = n_bins
        self.vmax = vmax
        self.compression = compression

        self.bin_width = 2 * self.vmax / self.n_bins
        self.compress = {
            "gzip": zlib.compress,
            "zstd": zstd.compress,
            "none": _noop,
        }[compression]
        self.decompress = {
            "gzip": zlib.decompress,
            "zstd": zstd.decompress,
            "none": _noop,
        }[compression]

    def hparams(self) -> dict[str, int | str | float]:
        return {
            "n_bins": self.n_bins,
            "compression": self.compression,
        }

    def __str__(self):
        return f"quantize_nb-{self.n_bins}_comp-{self.compression}"

    def encode(self, x: np.ndarray) -> bytes:
        x = np.round(x / self.bin_width)
        info = np.iinfo(np.int16)
        x = np.clip(x, info.min, info.max)
        x = x.astype(np.int16)
        x = encode_numpy(x)
        x = self.compress(x)
        return x

    def decode(self, x: bytes) -> np.ndarray:
        x = self.decompress(x)
        x = decode_numpy(x)
        x = x * self.bin_width
        return x


def _noop(x):
    return x


@register_codec
def quantize(**kwargs):
    return QuantizeCodec(**kwargs)
