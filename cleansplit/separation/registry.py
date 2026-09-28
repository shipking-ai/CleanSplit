"""Separator registry: add a backend by registering a factory, no pipeline changes needed."""

from __future__ import annotations

from typing import Callable

from .base import Separator

_FACTORIES: dict[str, Callable[..., Separator]] = {}


def register(name: str, factory: Callable[..., Separator]) -> None:
    _FACTORIES[name] = factory


def create(name: str, **kwargs) -> Separator:
    if name not in _FACTORIES:
        raise KeyError(f"unknown separator '{name}'. Available: {sorted(_FACTORIES)}")
    return _FACTORIES[name](**kwargs)


def available() -> list[str]:
    return sorted(_FACTORIES)


def _roformer(**kw):
    from .roformer import BSRoformerSeparator

    return BSRoformerSeparator(**kw)


def _folder(**kw):
    from .precomputed import StemFolderSeparator

    return StemFolderSeparator(**kw)


def _htdemucs(**kw):
    from .demucs_sep import HTDemucsSeparator

    return HTDemucsSeparator(**kw)


def _mdx23c(**kw):
    from .mdx23c_sep import MDX23CSeparator

    return MDX23CSeparator(**{k: v for k, v in kw.items() if k in ("checkpoint", "device", "num_overlap")})


register("bs_roformer_sw", _roformer)
register("bs_roformer_ep317", lambda **kw: _roformer(**{**kw, "model": "bs_roformer_ep317"}))
register("htdemucs_ft", _htdemucs)
def _scnet(**kw):
    from .scnet_sep import SCNetSeparator

    return SCNetSeparator(**{k: v for k, v in kw.items() if k in ("checkpoint", "device", "num_overlap")})


register("mdx23c_instvoc_hq", _mdx23c)
register("scnet_xl_ihf", _scnet)
register("stem_folder", _folder)


def _ensemble(use_demucs):
    def make(**kw):
        from .ensemble import EnsembleSeparator

        return EnsembleSeparator(use_demucs=use_demucs, **{k: v for k, v in kw.items() if k in ("device", "tta", "num_overlap")})

    return make


register("ensemble", _ensemble(False))
register("ensemble_demucs", _ensemble(True))
