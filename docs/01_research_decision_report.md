# CleanSplit — Phase 1 Research & Technical Decision Report

Date: 2026-09-16. Everything marked **[verified]** was checked against a primary source, local files, or code this session. Anything marked **[reported]** is a third-party claim we have not reproduced. Anything marked **[measured]** was run on the target machine (i9-10980HK / RTX 2080 Super Max-Q 8 GB / Windows 11).

---

## 1. Separator

### 1.1 BS-RoFormer SW ("BS-Rofo-SW-Fixed") — **SELECTED as primary separator**

| Property | Finding |
|---|---|
| Local availability | **[verified]** `%LOCALAPPDATA%\Programs\Ultimate Vocal Remover\models\MDX_Net_Models\BS-Rofo-SW-Fixed.ckpt` (699,412,152 bytes) plus `model_data\mdx_c_configs\BS-Rofo-SW-Fixed.yaml` |
| Integrity | **[verified]** SHA-256 `24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e` matches the LFS oid of `huggingface.co/enerjazzer/BS-ROFO-SW-Fixed` |
| Hosting | **[verified]** The original `jarredou/BS-ROFO-SW-Fixed` repo now returns HTTP 401 (the account was deleted, per the `openmirlab/bs-roformer-infer` README). The current rehost is `enerjazzer/BS-ROFO-SW-Fixed` (created 2026-04-23), with the README text "Edited version of that checkpoint to allow its use with UVR and MSST" |
| License | **[verified] Weights: none stated ("license: unknown").** No trainer or provenance is known. The rehoster says they did not train the model (quoted in `elicwhite/bs-roformer-sw-6stem-onnx`). Architecture code: lucidrains/BS-RoFormer (MIT). Inference framework: ZFTurbo/Music-Source-Separation-Training (MIT). |
| Stems / order | **[verified]** from the yaml: `[Bass, Drums, Other, Vocals, Guitar, Piano]`, 6 stems |
| Audio | **[verified]** 44.1 kHz, stereo, STFT n_fft 2048 / hop 512 / win 2048 (hann), 1025 bins, 62 bands, chunk_size 588,800 samples (13.35 s) |
| Architecture | **[verified]** from the code: dim 256, depth 12, 1+1 time/freq transformer per layer, 8 heads × 64. It produces a **complex mask per stem multiplied onto the mixture STFT**, with **no mixture-consistency projection**. It then applies `zero_dc=True`, which zeroes STFT bin 0 in every stem |
| Quality | **[reported]** MVSep Multisong SDR: vocals 11.30, bass 14.62, drums 14.11, guitar 9.05, piano 7.83, other 8.71 dB (mvsep.com/algorithms/77). Guitar and piano are clearly the weakest stems |
| Windows / 8 GB | UVR already runs it on this machine. The PyTorch path is **[measured]** in §6 |

**Consequences for CleanSplit design (derived from the code, not assumed):**

1. For each STFT bin, `R(f,t) = O(f,t)·ΣMᵢ(f,t)`, up to STFT consistency and chunk overlap-add. The mixture residual is therefore **`E(f,t) ≈ O(f,t)·(1 − ΣMᵢ(f,t))`**. That makes it an exact, cheap diagnostic of mask-sum error.
2. **The mixture residual cannot see leakage.** If energy is assigned to the wrong stem, the sum is unchanged and E stays zero. Leakage and inter-stem redistribution need separate *per-stem* detectors.
3. `zero_dc` removes content below about 21.5 Hz (plus window leakage) from **every** stem. A naive `O − R` always contains this sub-bass/DC component. CleanSplit therefore reports both the full-band residual and a **model-matched residual**, where the same bin-0 removal is applied to O. This keeps a known design property from being mistaken for an artifact.
4. Complex masks can make stems contain components that **cancel on summation**, so that Σ|Sᵢ|² > |ΣSᵢ|². This causes phasey or flanging sound when a stem is soloed, while the mix still nulls. It can be detected deterministically (see §4).
5. Chunked inference with overlap-add creates **known boundary positions**. A boundary-aware detector can test for chunk seams directly.

**Implementation choice:** vendor ZFTurbo's `bs_roformer.py` and `attend.py` (MIT), with attribution. Write CleanSplit's own chunked overlap-add loop so we control alignment, the windowing sum, and precision. We will **not** depend on `python-audio-separator` (it pulls the UVR stack) or on `bs-roformer-infer` (young package whose model registry changes; it can be evaluated later as a backend). **CleanSplit will never download or redistribute these weights.** It points at the user's existing file and pins the SHA-256.

### 1.2 HTDemucs 6s — secondary / cross-check separator (later)
- **[verified]** Demucs README: the piano source "is not working great at the moment", with a lot of bleeding and artifacts. The original author left Meta, and `facebookresearch/demucs` is effectively unmaintained. Code is MIT and the weights are public.
- UVR's local `v3_v4_repo` holds htdemucs / htdemucs_ft weights, **not** htdemucs_6s.
- Role: a second backend to prove the separator interface is swappable, and later a **cross-model disagreement** signal for the detector. It is not the primary separator.

### 1.3 MVSep
- MVSep hosts SW and newer ensembles, but it is a cloud service. It is excluded because CleanSplit must be local-first. Its SDR tables are used only as **[reported]** context.

---

## 2. Restoration candidates (Phase 8 — **not integrated in V0**)

| Model | Rate / channels | What it was trained for | Weights / license | 8 GB feasibility | Verdict |
|---|---|---|---|---|---|
| **NVIDIA A2SB** (`NVIDIA/diffusion-audio-restoration`) | 44.1 kHz, **mono** **[verified]** (model card) | Bandwidth extension (masks above a cutoff) and time-gap inpainting < 1 s; trained on ~10 k h of permissively licensed music/audio (FMA, Slakh, MusicNet, FreeSound…) **[verified]** | 3 × 2.26 GB ckpts (onesplit, twosplit 0–0.5, twosplit 0.5–1) **[verified]**; **NVIDIA OneWay Non-Commercial** model and **NVIDIA Source Code License-NC** code | ~2.26 GB fp32 weights per split; the 2-split ensemble swaps models by t. 200 sampling steps. Model card lists Linux only and Turing as supported hardware. Must be measured | **Primary Phase 8 candidate** |
| **AIDD** (`iftachShoham/AIDD`, ICLR 2026) | **24 kHz mono** via WavTokenizer `medium_speech_320_24k` **[verified]** | Gaps of 150–750 ms; ckpts trained on **MusicNet / MAESTRO** (classical / piano) **[verified]** | 2 × 1.58 GB, MIT **[verified]** | Probably fits | **Rejected as primary.** 24 kHz caps bandwidth at 12 kHz. A speech-codec tokenizer re-synthesizes the whole region, and the training domain does not match pop/rock stems. Possible piano-only baseline |
| **Apollo** (`JusperLee/Apollo`, also in UVR) | 44.1 kHz | Lossy-codec (MP3) restoration, not inpainting | 66 MB; **CC-BY-SA-4.0**; already on disk (`Vocal_Restore.ckpt`) | Easy | **Baseline only**: a non-inpainting "de-artifact" regressor, applied per region with crossfades |
| HiFi++ GAN restorers (MSR Challenge 2025 entries) | 44.1/48 kHz | Restore unprocessed stems from separated stems | Weights not confirmed released | — | Prior art, not usable yet |

### Why A2SB is the right first candidate (verified in its code)
- The sampler (`A2SB_lightning_module_api.py::ddpm_sample`) runs `pred_x0 = pred_x0·mask + (1−mask)·x₁` and `x_t = (1−mask)·x_t_true + mask·x_t` on a **full C×F×T mask**. So **time-frequency box masks are mechanically supported**, and unmasked bins are clamped to the input at every step. That matches CleanSplit's "touch only suspicious regions" rule exactly.
- Its STFT is **n_fft 2048 / hop 512 / win 2048 at 44.1 kHz, identical to BS-RoFormer SW**. Artifact regions map to A2SB mask indices 1:1.
- **Caveats to test, not assume:** (a) training masks were whole time-slices and high-band cutoffs, so TF-box masks are **out of distribution**; (b) it is mono, so stereo stems need per-channel or M/S processing, with stereo-image checks; (c) it is non-commercial; (d) Linux-only is stated; Windows means no DDP and needs code adaptation; (e) it generates *plausible* music, not *the* missing information. That last point is exactly why mixture-consistency rejection is mandatory.

### Mandatory non-generative baselines for the Phase 8 experiment
Without baselines, the experiment cannot answer "does *generative* restoration help?":
1. **Identity** (no change): the control.
2. **Residual reallocation / mixture-consistency projection** (Wisdom et al. 2019, arXiv:1811.08521). Inside flagged TF regions only, distribute E to stems in proportion to their local energy. This deterministically restores missing mixture energy.
3. **Region Wiener re-estimate** using the mixture and the other stems' magnitudes.

---

## 3. Evidence about the artifacts themselves
- **[reported]** Users describe separator artifacts as "warbling / going underwater", "phasey hi-hats that sounded … underwater", and "warbly and distorted" bass (Gearspace "Best Stem Separator 2025?" thread). UVR Roformer models are rated best in that thread, but not artifact-free.
- **[reported]** ISMIR 2025, "Perceptual Errors in Music Source Separation: looking beyond SDR averages": SDR generally correlates with perception, with significant deviations. The authors recommend error categorization and distributions over averages. This supports a region-level artifact map.
- **[reported]** "Musical Source Separation Bake-Off" (arXiv:2507.06917): SDR is best for vocals, while SI-SAR better predicts ratings for drums and bass. No single metric works across stems.
- **[reported]** Diffiner (arXiv:2305.05857): diffusion refiners **improve non-intrusive perceptual scores but lower SI-SDR/PESQ/STOI**. This is direct prior evidence that "sounds smoother" and "is more faithful" can diverge. CleanSplit's acceptance test must be fidelity-based.
- **[reported]** Musical-noise measures: log-kurtosis-ratio metrics correlate with perceived musical noise only "in a limited manner" unless perceptual masking is considered (arXiv:2105.13079). Our kurtosis detector is therefore one weak cue among several, not "the" detector.

---

## 4. V0 artifact analysis — deterministic, honest cues
All computed on the SW-native STFT grid (2048/512 @ 44.1 kHz), per stem, **relative to the mixture**:

| Detector | Cue | Targets |
|---|---|---|
| `residual` | Local `|E|²/|O|²` in dB (mask-sum error), TF-smoothed | reconstruction discrepancy, missing / added energy |
| `cancellation` | `Σ|Sᵢ|² / |ΣSᵢ|²` excess, attributed to stems with `|Sᵢ| > |O|` | phasing / flanging, energy that exists only because it cancels |
| `stem_exceeds_mix` | `|Sᵢ(f,t)| > |O(f,t)|·(1+margin)` | physically implausible stem energy |
| `modulation` | 2–20 Hz envelope-modulation energy of stem band envelopes vs mixture band envelopes | warble / watery / "underwater" |
| `transient` | onset sharpness and pre-onset energy of stem vs mixture at mixture onsets | transient smearing, pre-echo |
| `spectral_kurtosis` | local log-kurtosis ratio stem vs mixture | musical noise / "twinkling" |
| `hf_noise` | high-band (> 8 kHz) spectral flatness of stem vs mixture where the stem carries HF energy | high-frequency artifacts |
| `leakage` | local TF magnitude co-activation between stem pairs in bins where both are significant | possible leakage / shared content (ambiguous by nature, labeled as such) |
| `chunk_seam` | residual / discontinuity concentrated at the separator's known overlap boundaries | chunking artifacts |

Scores are robust z-scores per track (median/MAD), then hysteresis thresholding, TF connected components, and bounding boxes, giving `ArtifactRegion`s. **"confidence" in V0 is a heuristic score in [0,1], not a calibrated probability.** It is calibrated only against synthetic corruptions, and the report says so. Learned detectors plug into the same `Detector` interface later.

## 5. Metrics
- Mixture fidelity: SNR / SDR, SI-SDR (Le Roux et al. 2019, arXiv:1811.02508), residual energy (dB rel. mixture), band-wise residual, multi-resolution STFT distance, log-spectral distance, Multi-Mel-SNR (the MSR Challenge 2025 metric).
- Change control: max |Δ| and energy of Δ **outside** artifact regions (must be exactly 0, bit-identical), and inside regions.
- Ground-truth evaluation (when stems exist): per-stem SDR / SI-SDR / SI-SAR. MUSDB18-HQ has no guitar/piano stems. MoisesDB does, but is non-commercial and requires the user to download it. SW's training data may overlap either set (possible contamination, to be stated in results).
- Perceptual metrics (ViSQOL, PEAQ, FAD) are deferred: heavy, Windows-unfriendly, or embedding-dependent. The metric interface is designed to accept them.

## 6. Hardware measurements
See `docs/02_hardware_measurements.md` (filled from real runs).

## 7. Decisions summary
1. **Separator:** BS-Rofo-SW-Fixed from the user's local UVR folder, SHA-pinned, run through vendored MIT ZFTurbo code. Weights are never downloaded or redistributed by CleanSplit; their license is unknown, so use is personal/research only.
2. **Secondary separator:** HTDemucs 6s (later), for swappability and a cross-model disagreement signal.
3. **Restoration (Phase 8):** A2SB first, with TF-box masks on the shared STFT grid, **non-commercial**, feasibility to be measured on 8 GB. Mandatory baselines: identity, residual reallocation, region Wiener. AIDD is rejected as primary (24 kHz, speech tokenizer, classical-only training).
4. **Acceptance:** a restoration is accepted only if the mixture error does not increase beyond tolerance, audio outside regions is bit-identical, and no new unexplained energy appears. "Sounds smoother" is never an acceptance criterion.

## 8. What would actually make this better: research pass, 2026-09-28

Asked deliberately after the artifact work (docs/04 section 14), because that work narrowed the question: bleed is
solved, artifacts are what is left, and the only mechanism measured to remove artifacts here is **averaging
independent models**. So the research question is not "what is the best single model" but "what raises the ceiling".

### 8.1 The biggest structural gap in CleanSplit today: only vocals are ensembled

`vocals = mean(SW+TTA, ep317)` and that average is worth **+1.12 dB SAR** (section 14.1). Drums, bass, guitar and
piano come from **one model, SW**, with no averaging at all. So three quarters of the output gets none of the one thing
that works.

MVSep's Multisong leaderboard (100+ commercial tracks, a much larger and independent benchmark than the 20 MUSDB songs
used here) shows the community has already validated the fix. On the drums sort, the best entry on the whole board is
**"Drums Ensemble (MelBand + SCNet XL + BS Roformer SW)" at 14.3505 dB**, ahead of MVSep's own proprietary ensemble
(14.3339) and ahead of plain **"BS Roformer SW (6 stems)" at 14.1129** — the model CleanSplit already uses, alone.
That is **+0.24 dB SDR** from ensembling drums with two architecturally different models. On vocals, an SDR gain of
+0.39 dB corresponded to an SAR gain of +1.12 dB here, so the artifact gain may well be several times the SDR gain.

The three members are deliberately different architectures — convolutional (SCNet XL), mel-band transformer (MelBand
RoFormer), fixed-band transformer (BS-RoFormer SW). That is the same "uncorrelated errors cancel" mechanism, and it is
why diversity matters more than any single member's score.

### 8.2 External confirmation of the "comparable strength" rule

Three third-models were rejected here (MDX23C, HTDemucs_ft, MDX-Net Inst HQ 3), all of them materially weaker than the
pair, producing the rule: *a member more than ~1-2 dB below the pair drags the average down.* The MVSep vocals board
shows the same effect independently: **"BS RoFormer (11.89 + 12.33)" ensembles to 12.2597 — worse than its own better
member at 12.3339.** Averaging is not free; it is only worth it between near-equals.

### 8.3 Candidate models that are downloadable and strong enough to qualify

Top of the MVSep vocals board is MVSep-proprietary ("BS Roformer 124 bands", 12.33) and not obtainable. The strongest
**downloadable** entries sit just below: `bs_leap_xe_voc_unwa.ckpt` (11.7615), `unwa leap Xe` (11.7577),
`BS-Roformer_Leap by pcunwa` (11.7398), versus `mel_band_roformer_ep_125_sdr_11.2069` at 11.2468 (rank 100). All run in
the BS-RoFormer code path CleanSplit already has. Licence status of these community checkpoints is the usual problem:
distributed with UVR/MSST, no explicit licence (the same note already applies to SW and ep317).

### 8.4 The user's "AI that listens and fixes" — the form of it that works

Two post-hoc generative repairers were built and both lost (sections 4 and 9). **Iterative Audio Separation with
Mixture Consistency via MIMO Model Extension**, Ikemiya, Liao and Mitsufuji, Sony AI (arXiv:2609.07226, code MIT at
github.com/SonyResearch/mimo-audio-separation) does the same idea in a different place and it works. Their ablation on
MUSDB18-HQ, vocals:

| variant | params | SDR | SIR | SAR |
|---|---|---|---|---|
| (1) SISO BS-RoFormer (baseline) | 12.1M | 10.38 | 19.12 | 11.74 |
| (7) MIMO, 3 iterations | 16.3M | 10.72 | 20.84 | 11.89 |
| (8) (7) + stem discriminator | 16.3M | **11.05** | **21.29** | 12.14 |
| (10) (8) + generative | 20.2M | 11.00 | 21.35 | **12.16** |

**+0.67 dB SDR and +0.42 dB SAR over the baseline.** Three things matter for CleanSplit:
1. The gain comes from **iterating while holding mixture consistency** — the exact invariant this project already
   enforces. Their section 3.7 frames it as fixed-point iteration: the model is trained to converge to a
   self-consistent solution.
2. Most of the jump is the **discriminator** (row 7 to 8): a model whose whole job is to listen to a stem and judge
   whether it sounds real. That is the user's idea, and it is the best-performing part.
3. The **generative** variant gets the best SAR of all. So generative modelling does help artifacts — **inside the
   separator, trained jointly**, not as a repair pass afterwards. That is fully consistent with the local result that
   post-hoc repair lost 24/24: the same intuition, applied in the only place it can work.

**Cost: this requires training, not just inference.** They train BS-RoFormer from scratch for the comparison; MIMO is an
architectural change, not a post-processing wrapper. Released weights (GitHub releases, MIT code) include MIMO
BS-RoFormer large (71.4M, vocals SDR 11.62 vs 11.40 for their non-MIMO large) and a 4-source MIMO SCNet
(vocals 9.82 / drums 10.46 / bass 10.93 / other 7.69). Those absolute numbers are below what SW and ep317 achieve, so
the released weights are interesting for **architectural diversity in an ensemble**, not as a replacement.

### 8.5 The ceiling nobody can raise: source quality

MVSep's board includes the source-quality references, which put a hard number on it. Taking the *original studio stems*
and merely re-encoding them caps the achievable SDR:

| source | vocals SDR ceiling | drums | bass |
|---|---|---|---|
| WAV (lossless) | 113.8 | 115.0 | 115.5 |
| MP3 320 kbps CBR | 37.7 | 35.0 | 63.8 |
| M4A 320 kbps | 35.0 | 35.2 | 45.7 |
| **MP3 128 kbps CBR** | **20.1** | **19.8** | **25.8** |

Models are at ~12 dB, so 320 kbps is not the binding constraint. But it confirms the local finding that the user's own
`After 2.wav` is an MP3 decode with nothing above ~16 kHz (docs/04 section 9): for badly sourced material the encoder,
not the separator, sets the ceiling, and no amount of model quality recovers it.

### 8.6 Ranked conclusion

1. **Ensemble the other stems.** Biggest structural gap, externally validated recipe, needs SCNet XL + MelBand drums
   downloads. Expected: small SDR gain, possibly a multiple of it in SAR.
2. **Add or swap in a top-20 downloadable vocal model.** Qualifies under the comparable-strength rule; runs in
   existing code.
3. **MIMO / discriminator training** is the research-grade path and the vindication of the user's architecture idea,
   but it is a training project, not a configuration change.
4. Source quality gates everything above for lossy inputs.

## Sources
- BS-RoFormer paper: Lu et al., ICASSP 2024, arXiv:2309.02612 · https://github.com/lucidrains/BS-RoFormer
- https://github.com/ZFTurbo/Music-Source-Separation-Training (MIT; `models/bs_roformer/bs_roformer.py`, `utils/model_utils.py::demix`)
- https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed · https://huggingface.co/elicwhite/bs-roformer-sw-6stem-onnx · https://github.com/openmirlab/bs-roformer-infer
- https://mvsep.com/algorithms/77
- https://github.com/facebookresearch/demucs
- https://github.com/NVIDIA/diffusion-audio-restoration (README, modelcard.md, corruption/corruptions.py, A2SB_lightning_module_api.py, configs/ensemble_2split_sampling.yaml) · https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge · arXiv:2501.11311
- https://github.com/iftachShoham/AIDD · https://huggingface.co/TaliDror/AIDD · arXiv:2507.08333
- https://github.com/JusperLee/Apollo · arXiv:2409.08514
- MSR Challenge: arXiv:2601.04343 · arXiv:2603.04032 · https://msrchallenge.com/
- Diffiner: arXiv:2305.05857 · Musical noise: arXiv:2105.13079 · Bake-off: arXiv:2507.06917 · ISMIR 2025 poster 221
- Wisdom et al., mixture consistency, arXiv:1811.08521 · Le Roux et al., SI-SDR, arXiv:1811.02508
- https://gearspace.com/threads/best-stem-separator-2025.1443674/page-2
