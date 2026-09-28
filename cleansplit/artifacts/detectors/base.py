"""Detector interface.

A detector turns an ``AnalysisContext`` into one or more ``EvidenceMap``s: a 2-D evidence array in [0, 1]
over explicit frequency rows and time columns, plus the underlying physical measure. Region extraction is
generic (``artifacts.extract``), so a learned detector only has to emit EvidenceMaps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ...analysis.context import AnalysisContext


@dataclass
class EvidenceMap:
    detector: str
    stem: str
    evidence: np.ndarray  # (R, K) float32 in [0, 1]
    measure: np.ndarray  # (R, K) float32, physical measure (units in measure_name)
    measure_name: str
    row_bounds_hz: np.ndarray  # (R, 2)
    col_bounds_s: np.ndarray  # (K, 2)
    artifact_types: list[str]
    # optional per-cell labels refining artifact_types, e.g. {"added_energy": bool mask}
    type_masks: dict[str, np.ndarray] = field(default_factory=dict)
    max_confidence: float = 1.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        R, K = self.evidence.shape
        if self.measure.shape != (R, K):
            raise ValueError("measure/evidence shape mismatch")
        if self.row_bounds_hz.shape != (R, 2) or self.col_bounds_s.shape != (K, 2):
            raise ValueError(f"{self.detector}: axis bounds do not match evidence shape {self.evidence.shape}")


def logistic(x: np.ndarray, center: float, width: float) -> np.ndarray:
    z = np.clip((np.asarray(x, dtype=np.float64) - center) / max(width, 1e-9), -60, 60)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


class Detector(ABC):
    name: str = "abstract"
    version: str = "0"
    requires_stems: bool = True

    @abstractmethod
    def detect(self, ctx: AnalysisContext) -> list[EvidenceMap]: ...

    def describe(self) -> dict:
        return {"name": self.name, "version": self.version, "doc": (self.__doc__ or "").strip().splitlines()[0]}
