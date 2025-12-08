import importlib
import pkgutil
from typing import Callable

import fmri_codecs.codecs
from fmri_codecs.codecs.base import Codec


_CODEC_REGISTRY: dict[str, Callable[..., Codec]] = {}


def register_codec(name_or_func: str | Callable | None = None):
    def _decorator(func: Callable):
        name = name_or_func if isinstance(name_or_func, str) else func.__name__
        _CODEC_REGISTRY[name] = func
        return func

    if isinstance(name_or_func, Callable):
        return _decorator(name_or_func)
    return _decorator


def create_codec(name: str, **kwargs) -> Codec:
    if name not in _CODEC_REGISTRY:
        raise ValueError(f"Codec {name} not registered")
    codec = _CODEC_REGISTRY[name](**kwargs)
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
        if name.split(".")[-1] not in {"base", "common", "registry"}
    }
    return codec_modules
