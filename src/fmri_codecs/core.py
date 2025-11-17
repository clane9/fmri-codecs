import importlib
import pkgutil
from typing import Callable

import numpy as np

import fmri_codecs.codecs


class Codec:
    """Base interface for fmri codecs."""

    def fit(self, x: np.ndarray) -> "Codec": ...

    def encode(self, x: np.ndarray) -> bytes: ...

    def decode(self, x: bytes) -> np.ndarray: ...


_CODEC_REGISTRY: dict[str, Callable[..., Codec]] = {}


def register_codec(name_or_func: str | Callable | None = None):
    def _decorator(func: Callable):
        name = name_or_func if isinstance(name_or_func, str) else func.__name__
        if name in _CODEC_REGISTRY:
            raise ValueError(f"Codec {name} already registered")
        _CODEC_REGISTRY[name] = func
        return func

    if isinstance(name_or_func, Callable):
        return _decorator(name_or_func)
    return _decorator


def create_codec(name: str) -> Codec:
    if name not in _CODEC_REGISTRY:
        raise ValueError(f"Model {name} not registered")
    model = _CODEC_REGISTRY[name]()
    return model


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
