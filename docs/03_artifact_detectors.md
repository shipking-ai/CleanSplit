# Artifact detectors (V0): design, evidence, and failures

Every detector compares stems against the **mixture** (model-matched: `O_ref`), the only reference available for a real song. Each one emits an `EvidenceMap` (evidence in [0,1] over explicit Hz rows × second columns). Region extraction (hysteresis 0.6/0.35 → connected components → boxes → conservative cross-detector merge) is shared. `confidence` is a heuristic evidence score, **not a probability**.

## Enabled by default

| Detector | Physical measure | Why it is valid | Known blind spots / confounds |
|---|---|---|---|
| `residual` (mixture) | local ‖O_ref − ΣS‖² / ‖O_ref‖² (dB), 5×5 TF smoothing; evidence 0.5 at −12 dB | For mask separators E = O(1 − ΣMᵢ): an exact map of mask-sum error. Cells labelled `missing_energy` / `added_energy` | Cannot see misassignment between stems |
| `cancellation` (per stem) | stem power / power of Σ stems (dB), 9×9 smoothing; 0.5 at +3 dB | With unrelated phases E\|ΣS\|² = ΣE\|S\|² ≥ \|Sᵢ\|². A stem clearly louder than the sum only exists because another stem cancels it (phasey when soloed) | A stationary anti-phase partial collision between distinct sources; mitigated by band-wide smoothing |
| `modulation` (per stem), **confidence capped at 0.7** | 1 − β, where β regresses the mixture's 3–20 Hz band-power modulation onto the stem's (complex modulation spectra, pooled ±1 band, 1 s windows); 0.5 at 0.6 | Band powers add, so a genuine stem fluctuation appears in the mixture *with the same waveform* (β ≈ 1) whatever other stems do. If its complement went to another stem, or the mix never had it, β ≈ 0 | Judged only where the stem's modulation energy is within 6 dB of the mixture's: under a much busier mix, β is dominated by unrelated fluctuation. Anti-correlated stems (ducking) look like artifacts. Needs ≥ 2 adjacent bands. **Weak**: see benchmark |
| `transient` (per stem) | attack(mixture) − attack(stem) at mixture onsets, 512/128 STFT, 4 bands; 0.5 at 6 dB | The mixture contains the stem, so a stem cannot legitimately have more pre-onset energy than the mixture at an onset it owns | Onsets owned by < −6 dB share are skipped |
| `leakage` (per stem) | std of the dB level difference between receiver and a louder stem (0.5 s windows), at −35..−8 dB level ratio; source must have detrended envelope std ≥ 3 dB; ≥ 2 bands; confidence capped at 0.7 | A leaked copy keeps a constant level ratio to its source; distinct co-timed parts do not | Leakage *under* the receiver's own content; exact musical doublings |

HF-noise **labelling**: regions above 6 kHz whose stem content is noise-like (energy-weighted flatness ≥ 0.3) get `high_frequency_noise`. This labels physical findings; it detects nothing by itself.

## Disabled (registered as experimental) and why

| Detector | Failure observed on the synthetic benchmark |
|---|---|
| `hf_noise` (flatness contrast, stem vs mixture) | Never fired on injected HF noise: when the mixture's HF is itself noise-like (hi-hats), the contrast is ~0. Those corruptions are caught by `residual` (lossy) or `cancellation` (conserving) instead |
| `musical_noise` (mask spikes: isolated on each side in time and frequency vs a 3–5-cell ring, on a stationary mixture, ≥ −6 dB share) | Hit confidence on injected spikes (0.60–0.72) stayed **below** its false-positive confidence on clean stems (up to 0.99, e.g. vibrato partials and fast-decaying plucks) |

## How the detectors were debugged (for the record)
Initial versions produced **309 false-positive regions** on exact ground-truth stems. Causes found and fixed:
1. Synthetic instruments had exactly equal-tempered, phase-stable shared partials (a test-data flaw, fixed with per-instrument detuning) and cancellation smoothing was too narrow.
2. Modulation on dB envelopes turned note on/offsets into 40–60 dB "modulation". Replaced by the mixture-consistency formulation on linear power.
3. Musical-noise spikes were judged against adjacent cells, but a single-cell mask spike necessarily spreads ±2 frames after ISTFT and re-analysis (STFT consistency). A symmetric ring then counted every plucked onset.
4. Leakage by envelope correlation fired on anything rhythmically aligned. Replaced by constant-ratio, detrended fine structure, and ≥ 2 bands.

5. The energy-ratio modulation detector (v4) passed the hand-picked warble test at confidence 1.0, but scored **0/10** on randomized warble boxes. In dense passages the mixture's own rhythmic modulation (drums, plucks) exceeds the warbled stem's. That was overfitting to one test box. It was replaced by the coherence (β) formulation (v5), which reached 5/10 at the cost of more clean false positives. No threshold improved both (sweep below), so its confidence is capped at 0.7.

| Modulation operating point (5 seeds, randomized boxes) | warble recall (conserve / lossy) | clean FP regions/min |
|---|---|---|
| v4 energy ratio, 0.5 at +6 dB | 0/5 / 0/5 | 6.0 |
| **v5 β, center 0.6, trust ≥ −6 dB (default)** | **2/5 / 3/5** | **19.5** |
| v5, center 0.75 | 0/5 / 1/5 | 9.0 |
| v5, trust ≥ −3 dB | 1/5 / 1/5 | 10.5 |
| v5, center 0.75, trust ≥ −10 dB | 0/5 / 1/5 | 22.5 |

## Synthetic benchmark
`cleansplit evaluate --synthetic` injects 10 randomized corruption scenarios per seed into caricature stems and reports recall per scenario and false positives per minute on clean stems (`outputs/_benchmarks/synthetic_detection.json`). Current numbers are in `docs/04_results.md`. Caveat: caricature instruments show that the detector mechanics work; they do not establish real-world accuracy.

## First real-song observation (one track, not a conclusion)
`Concrete Crown.wav` (hip-hop beat, 243 s) separated by SW:
- Mixture SNR 26.9 dB full-band; 30.2 dB model-matched. The DC-bin removal alone accounts for the 0–20 Hz residual dropping from −16 to −33 dB.
- Mask-sum error is largest at high frequencies (8–22 kHz ≈ −21 dB vs 60–250 Hz ≈ −32 dB): stems are missing HF energy. Least-squares gain 1.0186 (stems 0.16 dB quiet overall).
- Stem-over-sum 99.9th percentile ≈ +0.02…+0.09 dB: **no inter-stem cancellation** on this track.
- Modulation 99.9th percentile +3.7 dB and transient (other) 99th percentile +5.2 dB: below thresholds, so no per-stem regions.
- Crossfade (chunk seam) zones are indistinguishable from elsewhere (−34.5 vs −33.8 dB residual).
