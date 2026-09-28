"""Restoration interface.

    restore(stem, artifact_regions, context) -> RestorationProposal

A restorer proposes new audio for one or more stems around a region. It never writes anything itself:
``restoration.engine`` projects the proposal onto the region's time-frequency mask (so audio outside the region
support stays bit-identical no matter what the model did), then the mixture-consistency gate accepts or rejects it.
Generative models plug in here exactly like the deterministic baselines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..artifacts.region import ArtifactMap, ArtifactRegion
from ..audio.stft import STFTGrid


@dataclass
class RestorationContext:
    mixture: np.ndarray  # O_ref (C, N): model-matched mixture the stems should sum to
    stems: dict[str, np.ndarray]  # current stems (C, N), possibly already partially restored
    sample_rate: int
    grid: STFTGrid
    artifact_map: ArtifactMap

    @property
    def reconstructed(self) -> np.ndarray:
        R = np.zeros(self.mixture.shape, dtype=np.float64)
        for s in self.stems.values():
            R += s
        return R

    @property
    def residual(self) -> np.ndarray:
        return self.mixture.astype(np.float64) - self.reconstructed


@dataclass
class RestorationProposal:
    region: ArtifactRegion
    new_audio: dict[str, np.ndarray]  # stem -> (C, b - a) proposed audio for the excerpt [a, b)
    excerpt: tuple[int, int]
    restorer: str
    notes: list[str] = field(default_factory=list)


class Restorer(ABC):
    name = "abstract"
    generative = False

    @abstractmethod
    def restore(self, stem: str, regions: list[ArtifactRegion], context: RestorationContext) -> list[RestorationProposal]:
        """``stem`` is the region's stem ("mixture" for reconstruction-level regions)."""

    def describe(self) -> dict:
        return {"name": self.name, "generative": self.generative, "doc": (self.__doc__ or "").strip().splitlines()[0]}
