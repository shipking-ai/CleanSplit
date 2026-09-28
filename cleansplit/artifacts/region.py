"""Artifact map: time-frequency regions, not whole-stem verdicts.

``confidence`` is a heuristic evidence score in [0, 1] produced by deterministic detectors. It is NOT a
calibrated probability. Calibration against synthetic ground truth is reported by ``cleansplit evaluate``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = "cleansplit.artifact_map/1"


@dataclass
class DetectorEvidence:
    confidence: float
    measure_name: str
    measure_peak: float
    measure_mean: float
    extra: dict = field(default_factory=dict)


@dataclass
class ArtifactRegion:
    stem: str  # canonical stem name, or "mixture" for reconstruction-level findings
    start_s: float
    end_s: float
    freq_low_hz: float
    freq_high_hz: float
    confidence: float
    artifact_types: list[str]
    detectors: dict[str, DetectorEvidence] = field(default_factory=dict)
    stem_level_db: float | None = None  # region's mean stem power relative to the stem's loudest analysis cell
    mixture_level_db: float | None = None  # same for the mixture
    id: str = ""
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not (self.end_s >= self.start_s):
            raise ValueError(f"region end {self.end_s} < start {self.start_s}")
        if not (self.freq_high_hz >= self.freq_low_hz):
            raise ValueError(f"region freq_high {self.freq_high_hz} < freq_low {self.freq_low_hz}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def overlaps(self, other: ArtifactRegion, time_pad_s: float = 0.0) -> bool:
        return (
            self.start_s - time_pad_s < other.end_s
            and other.start_s - time_pad_s < self.end_s
            and self.freq_low_hz < other.freq_high_hz
            and other.freq_low_hz < self.freq_high_hz
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["artifact_types"] = sorted(set(self.artifact_types))
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ArtifactRegion:
        d = dict(d)
        d["detectors"] = {k: DetectorEvidence(**v) for k, v in d.get("detectors", {}).items()}
        return cls(**d)


@dataclass
class ArtifactMap:
    regions: list[ArtifactRegion]
    source: dict = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)
    stem_summaries: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    schema: str = SCHEMA_VERSION

    def for_stem(self, stem: str) -> list[ArtifactRegion]:
        return [r for r in self.regions if r.stem == stem]

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "source": self.source,
            "analysis": self.analysis,
            "warnings": self.warnings,
            "stem_summaries": self.stem_summaries,
            "regions": [r.to_dict() for r in self.regions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ArtifactMap:
        if d.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"unsupported artifact map schema {d.get('schema')!r} (expected {SCHEMA_VERSION})")
        return cls(
            regions=[ArtifactRegion.from_dict(r) for r in d["regions"]],
            source=d.get("source", {}),
            analysis=d.get("analysis", {}),
            stem_summaries=d.get("stem_summaries", {}),
            warnings=d.get("warnings", []),
            schema=d["schema"],
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ArtifactMap:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
