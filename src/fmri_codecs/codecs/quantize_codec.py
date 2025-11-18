import io
import zlib
from typing import Literal

import numpy as np
import zstd

from fmri_codecs import register_codec


class QuantizeCodec:
    def __init__(
        self,
        n_bins: int = 4096,
        vmax: float = 5.0,
        compression: Literal["gzip", "zstd", "none"] = "gzip",
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

    def encode(self, x: np.ndarray) -> bytes:
        x = np.round(x / self.bin_width)
        info = np.iinfo(np.int16)
        x = np.clip(x, info.min, info.max)
        x = x.astype(np.int16)
        x = _encode_numpy(x)
        x = self.compress(x)
        return x

    def decode(self, x: bytes) -> np.ndarray:
        x = self.decompress(x)
        x = _decode_numpy(x)
        x = x * self.bin_width
        return x


def _encode_numpy(x: np.ndarray) -> bytes:
    with io.BytesIO() as f:
        np.save(f, x)
        x = f.getvalue()
    return x


def _decode_numpy(x: bytes) -> np.ndarray:
    with io.BytesIO(x) as f:
        x = np.load(f)
    return x


def _noop(x):
    return x


@register_codec
def quantize_nb256_none():
    return QuantizeCodec(n_bins=256, compression="none")


@register_codec
def quantize_nb512_none():
    return QuantizeCodec(n_bins=512, compression="none")


@register_codec
def quantize_nb1024_none():
    return QuantizeCodec(n_bins=1024, compression="none")


@register_codec
def quantize_nb2048_none():
    return QuantizeCodec(n_bins=2048, compression="none")


@register_codec
def quantize_nb256_gzip():
    return QuantizeCodec(n_bins=256, compression="gzip")


@register_codec
def quantize_nb512_gzip():
    return QuantizeCodec(n_bins=512, compression="gzip")


@register_codec
def quantize_nb1024_gzip():
    return QuantizeCodec(n_bins=1024, compression="gzip")


@register_codec
def quantize_nb2048_gzip():
    return QuantizeCodec(n_bins=2048, compression="gzip")


@register_codec
def quantize_nb256_zstd():
    return QuantizeCodec(n_bins=256, compression="zstd")


@register_codec
def quantize_nb512_zstd():
    return QuantizeCodec(n_bins=512, compression="zstd")


@register_codec
def quantize_nb1024_zstd():
    return QuantizeCodec(n_bins=1024, compression="zstd")


@register_codec
def quantize_nb2048_zstd():
    return QuantizeCodec(n_bins=2048, compression="zstd")
