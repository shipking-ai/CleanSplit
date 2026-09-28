"""Configuration dataclasses. Every threshold used by a detector lives here, with its unit.

Defaults were chosen from signal-processing reasoning (documented in docs/03_artifact_detectors.md) and then
checked against synthetic corruptions (cleansplit/tests/test_synthetic_detection.py). They are not tuned
on listening tests, and results must be read with that in mind.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class DetectorThresholds:
    center: float  # measure value where evidence = 0.5
    width: float  # logistic width (measure units); evidence 0.73 at center + width
    unit: str


@dataclass
class AnalysisConfig:
    # analysis grid (defaults = BS-RoFormer SW STFT)
    n_fft: int = 2048
    hop: int = 512
    smooth_bins: int = 5  # ~108 Hz
    smooth_frames: int = 5  # ~58 ms
    audibility_range_db: float = 60.0  # ignore cells more than this below the signal's loudest smoothed cell
    abs_floor_power: float = 1e-9  # absolute floor on smoothed per-bin power (sum over channels)

    # log-band pooling
    n_bands: int = 32
    band_fmin_hz: float = 40.0
    band_fmax_hz: float = 18000.0

    # region extraction
    hysteresis_high: float = 0.6
    hysteresis_low: float = 0.35
    min_region_cells: int = 4
    merge_gap_s: float = 0.05
    max_regions_per_stem: int = 2000

    # detectors: logistic evidence mapping
    residual: DetectorThresholds = field(default_factory=lambda: DetectorThresholds(-12.0, 1.5, "dB E/O_ref (smoothed)"))
    cancellation: DetectorThresholds = field(default_factory=lambda: DetectorThresholds(3.0, 1.0, "dB stem/|sum of stems|"))
    modulation: DetectorThresholds = field(default_factory=lambda: DetectorThresholds(0.6, 0.08, "1 - beta, beta = regression of mixture modulation on stem modulation"))
    modulation_fmin_hz: float = 3.0
    modulation_fmax_hz: float = 20.0
    modulation_window_s: float = 1.0
    modulation_hop_s: float = 0.25
    modulation_min_rel_db: float = -6.0  # stem modulation energy must be within this of the mixture's to be judged
    transient: DetectorThresholds = field(default_factory=lambda: DetectorThresholds(6.0, 1.5, "dB attack loss (mixture attack minus stem attack)"))
    transient_n_fft: int = 512
    transient_hop: int = 128
    transient_onset_rise_db: float = 9.0
    transient_ownership_db: float = -6.0  # stem must carry >= this share of the post-onset mixture energy
    musical_noise: DetectorThresholds = field(default_factory=lambda: DetectorThresholds(-15.0, 2.0, "dB fraction of stem patch energy in mask-spike cells"))
    musical_noise_spike_factor: float = 10.0  # cell must exceed this x its 5-cell median along time AND frequency
    musical_noise_fmin_hz: float = 1000.0
    musical_noise_patch_bins: int = 32
    musical_noise_patch_frames: int = 16
    hf_noise: DetectorThresholds = field(default_factory=lambda: DetectorThresholds(0.25, 0.06, "spectral flatness, stem minus mixture"))
    hf_noise_fmin_hz: float = 8000.0
    hf_min_share_db: float = -20.0
    leakage_level_range_db: tuple = (-35.0, -8.0)  # receiver/source level ratio considered leak-like
    leakage_min_env_std_db: float = 3.0  # source envelope must move at least this much (dB std) in the window
    leakage_max_confidence: float = 0.7

    # Experimental detectors are registered but NOT enabled by default because they failed the synthetic benchmark
    # (see docs/03_artifact_detectors.md):
    #   hf_noise      - flatness contrast cannot work when the mixture's HF is itself noise-like (cymbals).
    #   musical_noise - hit confidence on injected mask spikes was below its false-positive confidence on clean stems.
    enabled_detectors: tuple = ("residual", "cancellation", "modulation", "transient", "leakage")
    hf_label_min_hz: float = 6000.0  # regions entirely above this with noise-like stem content get a HF-noise label
    hf_label_min_flatness: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AnalysisConfig:
        kw = {}
        names = {f.name: f for f in fields(cls)}
        for k, v in d.items():
            if k not in names:
                raise KeyError(f"unknown analysis config key '{k}'")
            if isinstance(v, dict) and set(v) == {"center", "width", "unit"}:
                v = DetectorThresholds(**v)
            if isinstance(v, list):
                v = tuple(v)
            kw[k] = v
        return cls(**kw)
