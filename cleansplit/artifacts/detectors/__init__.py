"""Detector registry. Register a new (e.g. learned) detector with ``register(name, cls)``."""

from __future__ import annotations

from .base import Detector, EvidenceMap
from .cancellation import CancellationDetector
from .leakage import LeakageDetector
from .modulation import ModulationDetector
from .residual import ResidualDetector
from .spectral import HFNoiseDetector, MusicalNoiseDetector
from .transient import TransientDetector

_REGISTRY: dict[str, type[Detector]] = {}


def register(name: str, cls: type[Detector]) -> None:
    _REGISTRY[name] = cls


for _cls in (ResidualDetector, CancellationDetector, ModulationDetector, TransientDetector, MusicalNoiseDetector, HFNoiseDetector, LeakageDetector):
    register(_cls.name, _cls)


def create(names) -> list[Detector]:
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise KeyError(f"unknown detectors {unknown}; available {sorted(_REGISTRY)}")
    return [_REGISTRY[n]() for n in names]


def available() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["Detector", "EvidenceMap", "register", "create", "available"]
