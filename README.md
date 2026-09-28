# CleanSplit

Local-first six-stem separation with **artifact-aware analysis** and **mixture-consistent, region-restricted restoration**.

CleanSplit is built to answer one question experimentally, and to be able to answer "no":

> Can generative, artifact-aware restoration improve AI-separated six-stem audio while remaining faithful to the original mix?

It is not an "AI enhancer". Every change to a stem must stay inside a detected time-frequency region, leave all other samples bit-identical, and pass a mixture-consistency test against the original mixture. Otherwise it is rejected automatically.

## Status

| Phase | State |
|---|---|
| 1 Research & decisions | done: [`docs/01_research_decision_report.md`](docs/01_research_decision_report.md) |
| 2–3 Skeleton, audio pipeline | done, tested |
| 4 Separation (BS-RoFormer SW, local UVR checkpoint) | done, runs on RTX 2080 Super Max-Q (8 GB): [`docs/02_hardware_measurements.md`](docs/02_hardware_measurements.md). Also available: `--tta`, BS-RoFormer ep_317, HTDemucs_ft, and **`--separator ensemble`** (vocals = mean of SW+TTA and ep_317; exactly mixture-consistent). **Measured on 20 MUSDB18-HQ songs with real studio stems: +0.39 dB vocals median vs SW, better on 18/20.** The earlier one-song estimate (+1.8…+2.1 dB) was ~5× too optimistic; MDX23C and Demucs averaging did not survive real truth ([`docs/04_results.md` §11](docs/04_results.md)) |
| 5 Reconstruction / residual | done, tested |
| 6 Artifact detectors | V0 deterministic detectors plus a synthetic ground-truth benchmark: [`docs/03_artifact_detectors.md`](docs/03_artifact_detectors.md) |
| 7 Metrics / evaluation | done (mixture metrics, compare, reference-stem evaluation, detection benchmark) |
| 8 Restoration | interface, projection, acceptance gate, non-generative baselines, and a **ground-truth restoration experiment** done. A2SB integrated as `a2sb` (raw inpainting) and `a2sb_consistent` (generative allocation prior); fits 8 GB at fp16 (≈75 s per stereo 3 s window). **Measured on real music with known stems: no restorer beat doing nothing** (A2SB raw rejected 23/23 by the gate; the accepted restorers ended +1.2…+2.1 dB further from truth, and 61 of their 67 accepted changes were worse). Results: [`docs/04_results.md`](docs/04_results.md). **Apollo** (codec restoration, the one candidate aimed at genuinely lost information) added as `apollo`: 2.1 GB VRAM, 5x realtime, and **worse than doing nothing in 24 of 24 ground-truth measurements**; the gate rejected all 13 proposals on a real song ([`docs/04_results.md` §9](docs/04_results.md)) |

## Requirements

- Windows 10/11 (Linux should work, but is untested), Python 3.10+
- NVIDIA GPU recommended (CPU fallback works, but is slow); ffmpeg on PATH for non-WAV/FLAC inputs
- **BS-Rofo-SW-Fixed** checkpoint from an existing Ultimate Vocal Remover 5.6 install. CleanSplit never downloads or redistributes these weights: their license is unknown and the trainer is unidentified.

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -e ".[separation,dev]"
```

## Usage

```bash
cleansplit doctor
```
```bash
cleansplit analyze song.wav
```
```bash
cleansplit analyze song.wav --stems-dir path/to/uvr_exported_stems
```
```bash
cleansplit separate song.wav        # defaults are the best measured settings, not the fastest
```
```bash
cleansplit restore song.wav --restorer residual_reallocation
```
```bash
cleansplit evaluate --synthetic --seeds 0 1 2
```
```bash
cleansplit evaluate --restoration-experiment --seeds 0 1 2
```
```bash
cleansplit compare outputs/song/original.wav outputs/song/reconstruction/reconstructed.wav
```

Separated stems are cached (keyed by the input audio hash plus the separator configuration), so re-running `analyze` does not re-separate.

### Outputs

```
outputs/<song>/
  original.wav                      44.1 kHz stereo float mixture that everything is compared against
  stems/{vocals,drums,bass,guitar,piano,other}.wav + manifest.json
  reconstruction/reconstructed.wav  R = sum of stems (float64 sum, float32 WAV, never clipped)
  reconstruction/residual.wav       E = O - R
  reconstruction/residual_model_matched.wav   O_ref - R (O with the separator's DC-bin removal applied)
  analysis/artifact_map.json        time-frequency regions with per-detector evidence
  analysis/metrics.json             mixture fidelity, per-band residual, peaks, alignment, chunk-seam diagnostics
  analysis/report.json              summary, warnings, environment, limitations
  restoration/<restorer>/           restored stems, reconstructed, residual, restoration_report.json (every accept/reject with reasons)
```

## Architecture

```
cleansplit/
  audio/           decoding (soundfile -> ffmpeg), validation/conform, STFT (torch-compatible), region TF edits, synthetic songs
  separation/      Separator interface + registry; BS-RoFormer SW (vendored MIT code, own overlap-add); stem-folder and oracle separators
  reconstruction/  alignment (lag / polarity / gain / length, all reported), float64 sum, residual, model-matched reference
  analysis/        shared feature context, V0 pipeline orchestration
  artifacts/       ArtifactRegion / ArtifactMap schema, detectors (registry), evidence -> region extraction, corruption injectors
  restoration/     Restorer interface, baselines, projection + acceptance engine, restore pipeline
  metrics/         SNR / SI-SDR / Multi-Mel-SNR / LSD / band residual, compare, reference evaluation, detection benchmark
  models/          device probing, local checkpoint discovery + SHA-256 pinning
  config/          all thresholds, with units
  cli/             command-line entry point
  tests/           unit, synthetic ground-truth and restoration-gate tests
```

Swapping components: register a separator (`separation.registry.register`), a detector (`artifacts.detectors.register`, which only has to emit `EvidenceMap`s), or a restorer (`restoration.baselines.RESTORERS`). The pipeline, region extraction, projection and gate are shared.

## Honest limitations (V0)

- Detector `confidence` is a heuristic evidence score, **not** a calibrated probability.
- The mixture residual only measures mask-sum error. Energy placed in the wrong stem is invisible to it, which is why cancellation, modulation-consistency and leakage detectors exist.
- Without ground-truth stems, real-song findings are hypotheses. The synthetic benchmark checks the mechanics, not real-world accuracy.
- `hf_noise` and `musical_noise` detectors are registered but disabled: they failed the synthetic benchmark. `modulation` (warble) is weak: 5/10 recall at 19.5 false-positive regions/min, confidence capped at 0.7.
- **The mixture-consistency gate is necessary but not sufficient**: a restorer can pass it while making stems less faithful (measured, `docs/04_results.md`).
- **No generative restorer is enabled by default, because none has earned it.** A2SB and Apollo are both integrated and both measured worse than leaving the stems alone. Apollo restores a plausible top octave (right amount of energy, wrong content), which is a listening preference, not a fidelity gain.

## Licensing

CleanSplit code is MIT. `cleansplit/separation/bs_roformer/model.py` is adapted from ZFTurbo/Music-Source-Separation-Training (MIT, see `LICENSE_MSST.txt`) and lucidrains/BS-RoFormer (MIT). `third_party/apollo/` is vendored from [JusperLee/Apollo](https://github.com/JusperLee/Apollo) under **CC BY-SA 4.0** (attribution: Kai Li; pinned commit and the two deviations are documented in the file headers). `third_party/diffusion-audio-restoration/` is NVIDIA A2SB under the NVIDIA Source Code License-NC (non-commercial). Model weights are not included and carry their own terms: the Apollo checkpoints are CC BY-SA 4.0 and are read from an existing UVR install, never downloaded.
