import importlib
import pkgutil
from typing import Callable

import numpy as np

import fmri_codecs.codecs


class Codec:
    """Base interface for fmri codecs."""

    __codec_name__: str

    def encode(self, x: np.ndarray) -> bytes: ...

    def decode(self, buf: bytes) -> np.ndarray: ...

    def fit(self, X: list[np.ndarray]) -> "Codec": ...


_CODEC_REGISTRY: dict[str, Callable[..., Codec]] = {}


def register_codec(name_or_func: str | Callable | None = None):
    def _decorator(func: Callable):
        name = name_or_func if isinstance(name_or_func, str) else func.__name__
        _CODEC_REGISTRY[name] = func
        return func

    if isinstance(name_or_func, Callable):
        return _decorator(name_or_func)
    return _decorator


def create_codec(name: str) -> Codec:
    if name not in _CODEC_REGISTRY:
        raise ValueError(f"Codec {name} not registered")
    codec = _CODEC_REGISTRY[name]()
    codec.__codec_name__ = name
    return codec


def list_codecs() -> list[str]:
    return list(_CODEC_REGISTRY)


def import_codecs():
    # https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-namespace-packages
    codec_modules = {
        name: importlib.import_module(name)
        for finder, name, ispkg in pkgutil.iter_modules(
            fmri_codecs.codecs.__path__, fmri_codecs.codecs.__name__ + "."
        )
    }
    return codec_modules
