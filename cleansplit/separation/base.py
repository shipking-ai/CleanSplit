"""Separator interface. Implementations are swappable through ``separation.registry``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

CANONICAL_STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")


@dataclass
class SeparationResult:
    stems: dict[str, np.ndarray]  # name -> (C, N) float32, same rate/length as the input mixture
    sample_rate: int
    separator: str
    metadata: dict = field(default_factory=dict)
    # Sample indices where chunked inference windows start (for seam diagnostics); may be empty.
    chunk_starts: list[int] = field(default_factory=list)
    chunk_size: int | None = None


class Separator(ABC):
    name: str = "abstract"
    sample_rate: int = 44100
    channels: int = 2
    stems: tuple[str, ...] = CANONICAL_STEMS

    @abstractmethod
    def separate(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        """mixture: (C, N) float32 at ``self.sample_rate`` with ``self.channels`` channels."""

    def describe(self) -> dict:
        return {"name": self.name, "sample_rate": self.sample_rate, "channels": self.channels, "stems": list(self.stems)}

    def cache_key(self) -> dict:
        """Everything that changes the output; used to decide whether cached stems are reusable."""
        return self.describe()
