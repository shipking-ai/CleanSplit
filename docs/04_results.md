# Results to date (2026-09-17)

Every number here was produced by the commands listed, on the target machine. Raw JSON is in `outputs/_benchmarks/` and `outputs/Concrete_Crown/`.

## 1. Synthetic detection benchmark
`cleansplit evaluate --synthetic --seeds 0 1 2 3 4`: 8 s caricature songs, one randomized corruption box per scenario per seed; a hit needs confidence ≥ 0.6 on the expected stem, overlapping the box.

| Scenario | Detected | Per expectation |
|---|---|---|
| smear (conserving) | 5/5 | transient 5/5 |
| smear (lossy) | 5/5 | transient 5/5, residual 5/5 |
| hf_noise (conserving) | 5/5 | cancellation 5/5 |
| hf_noise (lossy) | 5/5 | residual 5/5 |
| leakage drums → vocals | 5/5 | leakage 5/5 |
| cancellation guitar/piano | 4/5 | cancellation 4/5 on both stems |
| dropout (lossy) | 4/5 | residual 4/5 |
| warble (lossy) | 3/5 | modulation 3/5 (residual 5/5) |
| warble (conserving) | **2/5** | modulation 2/5 |
| musical noise | **0/5** | detector disabled by default (failed; see doc 03) |

Clean ground-truth stems: **19.5 regions/min**, all from the confidence-capped `modulation` detector; residual, cancellation, transient and leakage produced **none**.

Meaning: the physically grounded detectors (residual, cancellation, transient) and the leakage detector work mechanically. Warble and musical noise are **not solved**. Caricature instruments do not establish real-world accuracy.

## 2. Real song: `Concrete Crown.wav` (243 s hip-hop beat), BS-RoFormer SW
`cleansplit analyze`: 122.8 s separation (2.0× realtime), 1.83 GB peak VRAM, 223.5 s total.
- Mixture SNR 26.9 dB full-band, **30.2 dB model-matched**. SW's DC-bin zeroing accounts for the 0–20 Hz residual (−16 → −33 dB).
- Missing energy concentrates at high frequencies (8–22 kHz residual ≈ −21 dB vs 60–250 Hz ≈ −32 dB). The stems are 0.16 dB quiet overall.
- No inter-stem cancellation (99.9th percentile of stem/sum ≈ +0.05 dB). Transient cues stayed below threshold. No chunk-seam effect.
- *Correction:* the original run used warble detector v4, which flagged nothing. Re-analysed with the current detector (v5), the same SW stems get 66 warble-type regions (vocals 34, other 26, bass 4, drums 2). That is within v5's measured false-positive rate on clean stems (≈ 19.5 / min), so it is not evidence of artifacts.
- Guitar (−51 dB) and piano (−32 dB) are effectively empty, and they were correctly not flagged.
- 1870 regions, all `mixture` / `missing_energy`.

`cleansplit restore --max-regions 300` (residual_reallocation): **300/300 accepted**, median local mixture error −11.3 dB. Whole-song SNR 30.18 → 30.27 dB. About 90% of samples stay bit-identical; changed samples carry ≈ −46 dB of stem energy. With no ground truth for this song, this demonstrates consistency, **not** fidelity.

## 3. Ground-truth restoration experiment (real SW, known stems)
`cleansplit evaluate --restoration-experiment --seeds 0 1 2 --duration 20 --max-regions 100`

| Restorer | Accepted / proposals | Mean error change vs **true** stems, changed stems only | Moved closer in |
|---|---|---|---|
| identity (control) | 0 / 85 | n/a | n/a |
| residual_reallocation | 82 / 85 | −0.01 dB | 3/3 songs (negligible) |
| region_wiener | 84 / 85 | **+1.98 dB** (piano +8.7 … +11.0 dB, bass +1.6 … +2.6 dB) | **0/3** |

**Main finding.** The mixture-consistency gate accepted `region_wiener` almost every time, yet it moved stems *away* from the truth. Re-splitting the mixture consistently among the wrong stems is invisible to any test that only compares against the mixture. Consistency is **necessary but not sufficient**. Any restorer, and especially a generative one, must be judged on ground-truth data before its real-song outputs can be trusted.

**Caveat (serious).** SW is out of domain on the caricatures: separation SDR was vocals 0.0 dB and guitar 0.0 dB (estimates essentially silent), other −9.2 dB. The synthetic songs are structurally near-identical across seeds. So the three songs are not independent evidence, and the *magnitudes* say little about real music. The next experiment needs real multitrack stems (MoisesDB includes guitar and piano; MUSDB18-HQ has only 4 stems), downloaded by the user. The code already supports this through `run_experiment(songs=...)`, and possible train/test overlap with SW should be kept in mind.

## 4. Generative restoration (NVIDIA A2SB) on real music with known stems

### Setup
- **Material:** `Downloads\BUH - {vocals, drums, bass, instrumentals} - Treblo.wav` (a real hip-hop track separated by another service). Truth = {vocals, drums, bass, other = instrumentals − drums − bass}, which is exactly disjoint because instrumentals ≈ drums + bass + other (corr 0.995). **Mixture = exact sum of truth.** Two 30 s excerpts (60 s and 120 s).
- **Separator:** real BS-RoFormer SW. Groups: vocals, drums, bass one-to-one; other ← guitar + piano + other. Separation SDR vs this truth: vocals 15.8, bass 18.7, drums 4.2, other 3.5 dB. The drums/other "truth" is partly the other service's opinion, so vocals and bass are the trustworthy groups.
- **Restorers**, all on the *same* top-confidence regions (10 per excerpt, confidence ≥ 0.6):
  - `identity` (control)
  - `residual_reallocation` (non-generative)
  - `region_wiener` (non-generative)
  - `a2sb`: raw generative inpainting of each affected stem's TF box, 50 steps, fp16
  - `a2sb_consistent`: A2SB's inpainted magnitudes decide the mixture split, and stems are re-derived from O_ref; the matching control is `region_wiener`
- **Score:** stem error vs. truth inside each region's TF box, summed over groups, floored at −30 dB of truth energy, pooled as 10·log10(Σ after / Σ before). **Negative = closer to truth; 0 dB = identity.** Every proposal is scored, including ones the gate rejected (`tools/run_real_stem_experiment.py`, `tools/summarize_experiments.py`).

### Results: mixture-level (missing-energy) regions, 20 regions over two excerpts
| Restorer | Gate accepted | Pooled error vs truth | Regions closer / worse | vocals group | drums group |
|---|---|---|---|---|---|
| identity | 0/20 | 0 (control) | n/a | n/a | n/a |
| residual_reallocation | 20/20 | **+1.27 dB** | 3 / 16 | +5.5 dB | +3.0 dB |
| region_wiener | 20/20 | **+1.85 dB** | 1 / 19 | +5.4 dB | +4.7 dB |
| a2sb_consistent (generative) | 20/20 | **+2.07 dB** | 1 / 19 | +5.5 dB | +5.1 dB |
| a2sb raw (generative, scored as if applied) | **0/20** | +2.53 dB | 8 / 11 | +1.3 dB | +4.9 dB |

### Results: per-stem artifact regions
Real SW output produced **0** per-stem regions in excerpt 60 s and **3** (warble, confidence 0.61–0.68) in excerpt 120 s, which is within the detector's measured false-positive rate. On those 3: region_wiener +1.17 dB (3/3 worse), a2sb_consistent +1.77 dB (3/3 worse), raw a2sb **+20.4 dB** (rejected 3/3), residual_reallocation 1 accepted at +0.06 dB.

### Findings
1. **No restorer, generative or not, beat doing nothing.** Every accepted change moved stems away from the truth on average.
2. **The generative prior made allocation worse, not better.** `a2sb_consistent` is +0.2 dB worse than its non-generative twin `region_wiener` on mixture regions, and +0.6 dB worse on per-stem regions.
3. **The mixture-consistency gate behaved correctly for raw A2SB.** It rejected every raw A2SB proposal: local mixture error rose +10…+43 dB, because A2SB regenerates a stem from its own context with no knowledge of the mix. Those proposals were worse vs. truth on balance. It also rejected 8 region-proposals that would have helped. **As predicted, it did not protect against consistent-but-wrong reallocation:** it accepted 67 of 69 mixture-consistent proposals, and 61 of those 67 were worse (by more than 0.05 dB).
4. The dominant failure mode is **giving missing energy to the wrong stem**, above all to vocals where the true vocal is near-silent, which makes soloed vocals dirtier.
5. **Cost:** A2SB takes ≈ 75 s per stereo 3 s window at fp16 (3.7 GB peak); fp32 took 2,522 s per window on this laptop. fp16 and fp32 results differed by 0.57%.

### Limits of this evidence
One track, two 30 s excerpts, 23 regions, and a "truth" produced by another separator. A2SB was used as released (trained on time-gap and high-band masks, mono, 50 steps), not fine-tuned for stems or TF-box masks. This is evidence **against** off-the-shelf generative inpainting of SW stems on this material, not proof that no generative approach could work.

## 5. Why residual-based restoration cannot do much for SW: error decomposition
`tools/diagnose_missing_energy.py` (same BUH excerpts, whole 30 s, STFT domain, bin 0 excluded).
With truth groups Tᵢ and SW group estimates Sᵢ: Mᵢ = Tᵢ − Sᵢ, and the mixture residual E = Σ Mᵢ.

| | excerpt 60 s | excerpt 120 s |
|---|---|---|
| Residual E as a share of **total stem error energy** Σ‖Mᵢ‖² | **1.3%** | **3.4%** |
| Proportional reallocation (what `residual_reallocation` does), all stems | −0.00 dB | −0.04 dB |
| … vocals only | **+0.84 dB** | **+1.25 dB** |
| Oracle allocation of E, smoothed 5×9 cells (upper bound at restorer resolution) | −0.55 dB | −0.56 dB |
| Oracle allocation per cell (unreachable; exploits truth-only structure) | −2.80 dB | −2.64 dB |
| Proportional, only where one stem holds ≥ 95% of the weight, vocals | +0.45 dB | +0.56 dB |

**About 97% of SW's stem error is misallocation between stems, which cancels in the sum and is invisible to the mixture.** Any method that only redistributes the mixture residual addresses a few percent of the error. Even with perfect knowledge of the owner at the restorer's resolution, the gain would be about 0.55 dB. Restricting to "confident owner" cells does not remove the harm to vocals. This explains the negative results in section 4 at the root.

## 6. Test-time augmentation (improves allocation; not restoration)
`tools/tta_experiment.py`: SW on transformed inputs, outputs mapped back and averaged. SDR gain vs single pass:

| Averaged passes | vocals | drums | bass | other |
|---|---|---|---|---|
| base + channel-swap + polarity-invert, 60 s | +0.08 | −0.01 | −0.01 | +0.04 |
| base + channel-swap + polarity-invert, 120 s | +0.32 | +0.01 | +0.01 | +0.25 |
| + half-hop shift (4 passes), 60 s / 120 s | +0.24 / +0.41 | −0.01 / −0.12 | −0.00 / −0.12 | +0.08 / +0.44 |

- Single passes vary by up to ±0.7 dB per group with input alignment (the half-hop shift alone ranged −0.75 … +0.70 dB). Differences smaller than a few tenths of a dB between *single* runs are therefore not meaningful.
- **3-pass TTA is the only method in this project that moved stems closer to the truth without hurting any group** (worst case −0.01 dB), on both excerpts. The effect is modest, and it rests on 2 excerpts of one track. It is available as `--tta` (3× separation time). Averaging is linear, so mixture consistency is unchanged.

## 7. Per-stem ensembles of local models (the largest measured gain)

> **Largely superseded by section 11.** On 20 real multitracks the vocal gain is +0.39 dB median (not ~2 dB), and
> averaging drums and bass with Demucs is worse on a typical song. The BUH reference exaggerated both.
`tools/ensemble_experiment.py`: same two BUH excerpts and truth groups, equal weights only (nothing tuned).
Additional local models, all hash-verified against the public `Politrees/UVR_resources` mirror:
- **BS-RoFormer ep_317** (viperx, vocals/instrumental; same vendored architecture, STFT hop 441). 1.6 GB VRAM, 23 s per 30 s.
- **HTDemucs_ft** (Meta, 4 stems; `demucs` 4.0.1). Loaded with `weights_only` plus an explicit allowlist of exactly the four globals the pickles contain. 0.55 GB VRAM, 10 s per 30 s. Its stems do not sum to the mix (27.5 dB).

Single models vs truth (60 s / 120 s):

| Group | SW | ep_317 | HTDemucs_ft |
|---|---|---|---|
| vocals | 15.76 / 17.04 | 16.57 / 16.54 | 13.09 / 14.92 |
| drums | 4.19 / 5.68 | n/a | 7.31 / 6.66 |
| bass | 18.68 / 19.90 | n/a | 23.09 / 22.26 |
| other | 3.52 / 2.40 | n/a | 4.26 / 3.56 |

Full ensembles, gain vs single-pass SW (60 s / 120 s). "other" = mixture − all other stems, so every ensemble sums **exactly** to the mixture:

| Ensemble | vocals | drums | bass | other |
|---|---|---|---|---|
| SW + 3-pass TTA | +0.08 / +0.32 | −0.01 / +0.01 | −0.01 / +0.01 | +0.04 / +0.25 |
| **E1** `ensemble`: vocals = mean(SW+TTA, ep_317) | **+2.14 / +1.83** | −0.01 / +0.01 | −0.01 / +0.01 | **+0.69 / +1.11** |
| **E2** `ensemble_demucs`: E1 + drums, bass = mean(SW+TTA, HTDemucs_ft) | +2.14 / +1.83 | +2.10 / +1.18 | +2.74 / +2.02 | +2.37 / +2.36 |
| E3: E2 but vocals also averaged with HTDemucs_ft | +1.51 / +1.75 | +2.10 / +1.18 | +2.74 / +2.02 | +1.59 / +1.78 |

Findings:
1. **Vocals: averaging SW and ep_317 gains +1.8 to +2.1 dB on both excerpts, although neither model is consistently better alone** (ep_317 alone: +0.81 / −0.51). That is a real ensemble effect from complementary errors. Demucs is not involved, so a Demucs-like reference cannot explain it.
2. Adding a model that is weak on a stem hurts: E3 < E2 on vocals and other.
3. **Caveat on E2 drums and bass:** the reference stems come from an undisclosed commercial separator. If it is Demucs-like, HTDemucs is favoured, so the E2 drum and bass gains may be optimistic. E1's gains do not have this problem.
4. This targets exactly the error section 5 identified (misallocation between stems), using outside knowledge (other models) that the mixture cannot provide.
5. Guitar and piano are not re-estimated. Changes in the remainder go to `other` only, since no truth exists to validate reallocation inside that group.

Available as `--separator ensemble` and `--separator ensemble_demucs` (loads one model at a time on the GPU).

**Full song** (`cleansplit analyze "Concrete Crown.wav" --separator ensemble`, 243 s): 913 s total (SW with 3-pass TTA 530 s, ep_317 314 s), peak VRAM 1.83 GB. The stems sum to the mixture at 157 dB SNR (float precision). Per-stem detector flags are essentially unchanged vs plain SW analysed with the same detectors (vocals 34 vs 34, other 22 vs 26, bass 4 vs 4, drums 2 vs 2). In the 34 flagged vocal regions, SW and ep_317 agree closely (coherence median 0.99, level difference 0.4 dB, no partial cancellation from averaging), so the flags are not caused by the ensemble. On the BUH excerpts the ensemble produced exactly the same per-stem flags as SW.

**Truth-quality check:** the detectors applied to the BUH *reference* stems flag 458 / 178 cancellation regions on drums and other. The derived `other = instrumentals − drums − bass` contains inverted drum content wherever the service's instrumental is not exactly drums + bass + other. This confirms drums/other are unreliable as truth (and independently validates the cancellation detector). The vocals result above does not depend on them. Because `other` absorbs the exact remainder, the mixture residual of an ensemble is zero by construction. The residual detector therefore reports nothing for ensemble output, and per-stem detectors are the only analysis that applies.

## 8. What this means for Phase 8 (generative restoration)
1. **Current experimental answer: no.** Off-the-shelf A2SB, used raw or as an allocation prior, did not improve SW six-stem separation on real music with known stems; neither did the deterministic baselines, nor Apollo (section 9, 24/24 measurements worse). Doing nothing was best.
2. Passing the mixture gate is not evidence of improvement. Ground-truth scoring is mandatory, and the default CLI `restore` should be treated as experimental.
3. Section 5 changes the priorities. Residual-based restoration is capped at about 0.5 dB even with an oracle, and "confident owner" region selection was tested without success. The error that matters is **misallocation**, which only extra knowledge about the sources can reduce:
   - **Real multitrack truth** (MoisesDB / MUSDB18-HQ) to confirm sections 4–6 beyond one track and a separator-made reference.
   - **Ensembles / TTA: done** (section 7). Vocals +1.8 to +2.1 dB from a two-model average; TTA alone gives small, consistent gains. Next: more diverse local models per stem (MDX23C InstVoc HQ) and real multitrack truth to rule out reference bias.
   - **A generative refiner conditioned on the mixture *and* all stems jointly**, trained to correct allocation. That is a model to train, not one that exists off the shelf; single-stem inpainters like A2SB cannot see the allocation problem.
   - **Codec restoration: tested, section 9.** Apollo is the only candidate whose target (information the encoder deleted) is not misallocation, and it still lost every measurement. Its top-octave reconstruction is plausible in shape and wrong in content.
   - A2SB **bandwidth extension** is deprioritised: in the bands above 6 kHz the mixture residual is only 0.1–0.5% of the stem error in those same bands on these excerpts, so the high-band "missing energy" is not where SW's stems go wrong.

## 9. Apollo: generative codec restoration (measured 2026-09-17)

A2SB failed because it attacks missing information while ~97% of the error is misallocation (section 5). Apollo is
the opposite case: it is trained to restore what an MP3 encoder actually deleted. If any off-the-shelf generative
restorer belongs in this pipeline, it is this one, so it was measured the same way everything else was.

Setup: `tools/apollo_experiment.py`. The reference is the truth; damage is applied by me with LAME at 320 / 128 /
96 kbps, decoded back, delay-aligned by cross-correlation (measured lag 0 in every run, correlation 1.00000), then
scored. **96 and 128 kbps are the control**: they are the damage Apollo was trained on, so if the harness cannot
see an improvement there, a null result at 320 kbps would say nothing. Both released checkpoints, four references,
three bitrates = 24 measurements. Raw JSON in `outputs/_benchmarks/apollo_codec_*.json`.

### 9.1 The source material is already lossy

Spectral cutoff (Welch, 16384-point, knee at 40 dB below the 1-4 kHz level) of every file used in this project:

| File | Cutoff |
|---|---|
| `After 2.wav` (downloaded by the user) | ~16.0 kHz |
| `After 2.mp3` (320 kbps) | ~16.0 kHz |
| `Too Much On My Plate.wav` | ~16.2 kHz |
| `Concrete Crown.wav` | ~22.1 kHz (full band) |

LAME at 320 kbps lowpasses at ~20.5 kHz, so a 320 kbps file that stops at 16 kHz was encoded from an already-lossy
ancestor. Direct comparison of the two `After 2` files: **lag 0, correlation 1.00000, best gain 1.00000, SNR
74.5 dB**, and their energy above 16 kHz matches to 0.1 dB. The WAV is a decode of the MP3 rounded to 16 bits, not
an independent master. There is therefore no "lossless vs lossy" experiment to run on that song, and the UI's
`ensemble` vs `ensemble_demucs` A/B on it is a clean separator comparison, not a source-quality one.

### 9.2 Whole-signal enhancement vs exact truth

Waveform SNR against the truth, in dB (higher is better); Δ is against leaving the lossy file alone.

| Reference | Damage | Lossy | + mp3_enhancer | + vocal_restore |
|---|---|---|---|---|
| `Concrete Crown` mixture (full band) | 320k | 35.61 | 18.32 (−17.29) | 6.50 (−29.11) |
| | 128k | 20.35 | 15.68 (−4.66) | 6.32 (−14.03) |
| | 96k | 18.42 | 14.78 (−3.65) | 6.12 (−12.30) |
| `Concrete Crown` vocals stem | 320k | 35.33 | 18.26 (−17.07) | 22.15 (−13.18) |
| | 128k | 17.75 | 7.67 (−10.08) | 16.10 (−1.65) |
| | 96k | 15.30 | 7.19 (−8.11) | 14.31 (**−0.99**) |
| `After 2` vocals stem (16 kHz-limited) | 320k | 40.68 | 19.93 (−20.75) | 24.83 (−15.85) |
| | 128k | 17.74 | 15.03 (−2.72) | 16.80 (−0.94) |
| | 96k | 13.13 | 12.39 (−0.73) | 12.66 (**−0.47**) |
| `After 2` mixture (16 kHz-limited) | 320k | 42.99 | 19.57 (−23.42) | 7.16 (−35.84) |
| | 128k | 20.46 | 17.12 (−3.34) | 7.04 (−13.42) |
| | 96k | 16.21 | 14.96 (−1.24) | 6.69 (−9.52) |

**Apollo never beat doing nothing, in 24 of 24 measurements.** The closest it came is −0.47 dB (After 2 vocals,
96 kbps, `vocal_restore`), and the gap widens monotonically as the input gets cleaner: at 320 kbps it costs 13-36 dB.

Two secondary results are real and worth keeping:

1. **Matching the checkpoint to the material matters a lot.** `vocal_restore` on a vocal stem is 7-9 dB better than
   `mp3_enhancer` on the same stem; `mp3_enhancer` on a full mixture is 8-29 dB better than `vocal_restore` on one.
   `vocal_restore` applied to a mixture destroys everything below 250 Hz (band error −1.1 dB relative to signal at
   0-20 Hz, i.e. the error is nearly as large as the content).
2. **On full-band truth, Apollo restores spectral *shape* while moving away from the signal.** On the
   `Concrete Crown` mixture at 96/128 kbps, LSD improves by 2.46-2.86 and multi-mel-SNR is unchanged (−0.07, −0.11)
   while waveform SNR drops 3.6-4.7 dB. Energy above 16 kHz lands near truth (−50.7 vs −50.0 dB at 96 kbps, where
   the lossy input had −93.9 dB). It rebuilds a plausible top octave of roughly the right size — just not the one
   that was there.

The 16 kHz-limited references show the failure mode of that same behaviour: the truth has −94.6 dB above 16 kHz and
Apollo emits −52.0 dB, i.e. **42 dB of fabricated band** on material whose top octave was never recorded.

### 9.3 Region-restricted and gated, on a real song

`cleansplit restore "After 2.mp3" --restorer apollo:variant=vocal_restore --min-confidence 0.5` on the 13 flagged
regions of the `ensemble` split:

```
pass 0: 13 regions, 0 accepted
rejection_reasons: {"mixture error in region increased": 13, "adds energy not explained": 1}
stems_bit_identical: all six true
```

The mixture-consistency gate rejected every Apollo proposal, and the output is bit-identical to the input. Unlike
the A2SB case in section 4 — where the gate accepted 67 of 69 consistent-but-wrong proposals — here the gate and
the ground-truth scoring agree, because Apollo's error is added energy the mixture can see, not a reallocation
hidden inside it.

### 9.4 Conclusion

Apollo is **not** enabled anywhere by default. The honest summary is that it is a spectral-plausibility model, not
a restoration of the recording: it is useful if the goal is "make the top end sound present", and harmful by every
fidelity measure available here. That may still be what a listener wants, which is a different question from the
one this project asks, and it is not a question this apparatus can answer without a listening test.

The one situation where it might earn its place is the one the user's files are actually in — a top octave that was
destroyed upstream and has no truth anywhere. In that case nothing can be measured, only preferred. CleanSplit will
not silently apply it on that basis.


## 10. A third vocal model: MDX23C InstVoc HQ (measured 2026-09-21)

> **Superseded by section 11.** On 20 MUSDB18-HQ songs with real studio stems, adding MDX23C made the vocal
> average *worse* (two-model average better on 16/20 songs). MDX23C was removed from the default ensemble again.
> This section is kept as the record of why it was added.

Section 7 found that averaging two vocal models gains ~2 dB because their errors are partly independent. SW and
ep_317 are both BS-RoFormers, so the obvious question is whether a model with a different architecture adds more.
MDX23C InstVoc HQ is a convolutional TFC-TDF U-Net on an 8192-point STFT; its weights ship with UVR.

Setup: `tools/vocal_ensemble_experiment.py`. BUH vocal truth (the reliable stem, see section 7), **five** 30 s
excerpts at 30/60/90/120/150 s — more than section 7's two. Every equal-weight combination was scored; nothing was
tuned. **Decision rule, written into the script before it ran:** MDX23C joins the default ensemble only if the
three-model average beats the current two-model average on the pooled score *and* on a majority of excerpts.

Vocals SDR (dB):

| | 30 s | 60 s | 90 s | 120 s | 150 s | pooled |
|---|---|---|---|---|---|---|
| sw_tta | 17.46 | 15.84 | 15.39 | 17.37 | 18.80 | 16.89 |
| ep317 | 16.78 | 16.57 | 17.27 | 16.54 | 17.19 | 16.90 |
| mdx23c | 16.83 | 15.31 | 17.50 | 16.62 | 17.49 | 16.80 |
| avg(sw_tta, ep317) — previous default | 19.23 | 17.90 | 18.69 | 18.87 | 19.75 | 18.93 |
| avg(sw_tta, mdx23c) | 18.78 | 17.18 | 18.15 | 18.55 | 19.54 | 18.48 |
| avg(ep317, mdx23c) | 18.21 | 17.32 | 18.75 | 17.75 | 18.52 | 18.15 |
| **avg(sw_tta, ep317, mdx23c)** | **19.48** | **18.14** | **19.41** | **19.03** | **19.86** | **19.23** |

Result: **+0.30 dB pooled over the previous default, better on 5/5 excerpts** → adopted. `--separator ensemble`
and `ensemble_demucs` now use vocals = mean(SW+TTA, ep_317, MDX23C).

Observations:
1. The three single models are within 0.1 dB of each other pooled (16.80–16.90), yet no single model wins every
   excerpt: SW is best at 30/120/150 s, MDX23C at 90 s, ep_317 at 60 s. That is exactly the situation in which
   averaging helps.
2. Each pair gains 1.3–2.0 dB over its members, and the third model adds a further 0.30 dB. The gains shrink, as
   they should when the errors are only partly independent.
3. The architecture guess was not confirmed: the two-model pair with the most architectural diversity
   (sw_tta + mdx23c, 18.48) is *worse* than the two BS-RoFormers together (18.93). Diversity of architecture is
   not the same as diversity of error, and that is why this was measured instead of assumed.
4. Cost: MDX23C runs at 1.5× realtime with its native overlap of 8 and 1.48 GB peak VRAM (docs/02), so the
   ensemble takes about 2.5 minutes longer on a 4-minute song. It is loaded after the other two have been
   released, so the 8 GB budget is unchanged.
5. Same caveat as section 7: this is one song and a separator-made reference. Five excerpts are not five songs.

## 11. Real multitrack truth: MUSDB18-HQ, 20 songs (measured 2026-09-21)

Every ensemble number above came from **one** song (BUH) whose "truth" was produced by an undisclosed commercial
separator. MUSDB18-HQ (Zenodo 3338373, md5 verified, educational/non-commercial licence) has the real studio stems.

Protocol (`tools/musdb_eval.py`, written and smoke-tested before any MUSDB song was scored): the first 20 test songs
in alphabetical order (scope cut from 50 at the user's request; alphabetical order is content-blind), one 30 s
excerpt centred on each track's midpoint, SDR per song, equal-weight averages only, nothing tuned. A "yes" needs a
better median **and** wins on more than half the songs. Raw data: `outputs/_benchmarks/musdb18hq_test.json`.

SDR in dB (median / mean / pooled over 20 songs):

| Group | Candidate | Median | Mean | Pooled |
|---|---|---|---|---|
| vocals | SW | 12.77 | 12.98 | 11.86 |
| | SW + TTA | 12.83 | 13.03 | 11.90 |
| | ep317 | 13.48 | 13.40 | 11.80 |
| | MDX23C | 11.91 | 12.22 | 11.31 |
| | HTDemucs_ft | 9.70 | 9.74 | 9.27 |
| | **avg(SW+TTA, ep317)** | 13.16 | **13.67** | **12.25** |
| | avg(SW+TTA, ep317, MDX23C) | 13.08 | 13.55 | 12.20 |
| drums | SW + TTA | **12.38** | **12.76** | **10.95** |
| | HTDemucs_ft | 9.28 | 9.38 | 7.31 |
| | avg(SW+TTA, Demucs) | 11.83 | 11.58 | 10.00 |
| bass | SW + TTA | **10.76** | **9.49** | 4.91 |
| | HTDemucs_ft | 8.84 | 8.04 | 6.35 |
| | avg(SW+TTA, Demucs) | 9.75 | 9.34 | **6.73** |
| other | SW + TTA | **8.98** | **8.42** | 5.66 |
| | remainder of `ensemble` | 8.95 | 8.31 | 5.66 |
| | remainder of `ensemble_demucs` | 8.04 | 7.75 | **5.89** |

Pre-registered answers:

| Question | Δ median | Δ pooled | Wins | Answer |
|---|---|---|---|---|
| Q1 ensemble vocals (3-model, as shipped at the time) vs SW | +0.30 | +0.34 | 17/20 | yes |
| Q2 adding MDX23C to the vocal average | −0.09 | −0.05 | 4/20 | **no** |
| Q3a Demucs averaging, drums (vs SW+TTA) | −0.55 | −0.95 | **0/20** | **no** |
| Q3b Demucs averaging, bass (vs SW+TTA) | −1.01 | +1.81 | 7/20 | **no** |
| Q4a that average vs Demucs alone, drums | +2.54 | +2.68 | 20/20 | yes |
| Q4b that average vs Demucs alone, bass | +0.90 | +0.38 | 18/20 | yes |

Follow-up comparisons (not pre-registered, reported for the decision): two-model vocal average vs SW **+0.39 dB
median, 18/20**; vs the three-model average +0.09, 16/20. TTA alone is tiny but consistent: +0.02 to +0.05 dB, better
on 16–20/20 songs in every group.

What changed as a result:
1. **MDX23C removed from the default ensemble** (Q2). Five excerpts of one song said +0.30 dB, 5/5; twenty real
   songs say −0.09, 4/20. The pre-registered rule adopted it on BUH and removes it on MUSDB — the rule was right
   both times, the BUH evidence was not. MDX23C stays available as `--separator mdx23c_instvoc_hq`.
2. **The shipped vocal ensemble works, but the gain is +0.39 dB, not ~2 dB.** BUH overstated it about 5×.
3. **`ensemble_demucs` is worse on a typical song.** Drums lose on 20/20 songs. This matches the caveat recorded in
   section 7 that the BUH reference might be Demucs-like. Stated bias: the community models' training data may
   include these songs and Demucs' does not, so this "no" is not a verdict on Demucs in general.
4. **Its real use is insurance.** The bass pooled score is better (+1.81) only because of one song: in "Arise – Run
   Run Run" SW's bass scores 0.8 dB and Demucs' 14.5 dB — SW missed the instrument, averaging halves the damage.
   Without that song the averaged bass median drops from 11.24 to 10.11. That is the same failure the user heard on
   `After 2` (drums lane at −46.9 dB with `ensemble`, audible with `ensemble_demucs`). The UI now describes it that
   way: "try when a drum or bass lane comes out empty".

Open, and now measurable: a **disagreement detector**. When two models disagree about a stem by tens of dB (as SW
and Demucs do on that song), one of them has missed an instrument. That signal exists without any truth and would
tell the user *which* songs need `ensemble_demucs`, instead of making it a blanket choice.


## 12. A truth-free "a model missed something" flag (measured 2026-09-21)

Idea: two strong separators that disagree about a stem cannot both be right. agreement =
10·log10(|d|² / |s − d|²), with s = SW+TTA's estimate of the stem and d = HTDemucs_ft's. No truth needed.

Developed on MUSDB18-HQ test songs 1–20: Spearman(agreement, SW's true SDR) = **0.79** over 80 song-stems; all 8 SW
failures (SDR < 3 dB) rank among the 17 lowest. Threshold chosen there: **6 dB** (catches 7/8, flags 13/80).

Held-out test on songs 21–50 (`tools/disagreement_eval.py`), criteria written into the script before those songs
were separated: recall ≥ 0.5 **and** precision ≥ 0.4 **and** Spearman ≥ 0.5.

| | Result | Required |
|---|---|---|
| Spearman(agreement, SW SDR) | **0.827** (120 song-stems) | ≥ 0.5 |
| Recall (failures flagged) | **1.00** (4/4) | ≥ 0.5 |
| Precision (flags that are failures) | **0.21** (4/19) | ≥ 0.4 |
| Verdict | **FAIL** | |

What it means:
1. The *ranking* replicates and is strong: low agreement really does mean a worse stem (0.83 on unseen songs).
2. The *threshold* does not transfer: at 6 dB it raises about four false alarms per real failure on new songs, so it
   is **not shipped as a warning**. Per the rule, the threshold is not re-tuned on the held-out songs.
3. Two of the four held-out "failures" are degenerate: the true stem is silent (an instrumental's vocals, a bassless
   song), where SDR is meaningless (−75 dB, 0.00 dB). The failure definition was fixed in advance and is kept; it
   is noted because it inflates both the failure count and the difficulty of the precision target.
4. What could be shipped honestly later: the raw agreement number shown as information ("the two models agree on
   this stem to X dB; lower tends to mean worse, Spearman 0.8"), without a pass/fail threshold. Not done yet.

## 13. Audio → MIDI: stem-aware transcription vs the whole mix (measured 2026-09-21)

Question: CleanSplit already separates a song. Does transcribing each separated stem (with a hard instrument
constraint) beat what ordinary audio-to-MIDI tools do — transcribing the whole mix?

Models (researched 2026-09-21, see HANDOFF): **MuScriptor** medium (Kyutai + Mirelo, July 2026; 36 instrument groups
incl. drums and voice; code MIT, weights CC BY-NC 4.0, gated — the user accepted the licence with their own
account) and **Transkun** v2 (piano specialist with velocities; MIT; SHA-256 pinned, strict `weights_only` load).

Truth: **BabySlakh** (Zenodo 4603870, CC BY 4.0) — 20 tracks rendered from MIDI, so the exact notes are known.
One 30 s excerpt per track, centred on the midpoint. mir_eval, 50 ms onset tolerance, offsets ignored.
`tools/transcription_eval.py`; questions and pass rules written into it before any route was transcribed.

Pooled F1 over 20 tracks:

| Route | multi-F1 (pitch + onset + instrument) | onset-F1 (any instrument) | drum-F1 | piano-F1 |
|---|---|---|---|---|
| A full mix → MuScriptor (what other tools do) | 0.226 | 0.500 | 0.523 | 0.322 |
| C CleanSplit stems → MuScriptor, constrained | 0.274 | 0.488 | 0.651 | 0.421 |
| **C2 as C, piano stem → Transkun** | **0.332** | **0.548** | 0.651 | **0.606** |
| D true stems → MuScriptor (upper bound for C) | 0.334 | 0.475 | 0.714 | 0.467 |

Pre-registered answers:
- **Q1 stems beat the full mix: YES** — multi-F1 +0.048, better on 13/20 tracks.
- **Q2 Transkun for the piano stem: YES** — piano-F1 +0.186 (0.421 → 0.606).
Not pre-registered, reported for the default: the combined pipeline C2 vs A is better on **16/20** tracks, median
per-track multi-F1 0.183 → 0.316.

Reading:
1. The gain is mostly **instrument assignment and drums**, not raw pitch detection: onset-F1 barely moves from A to
   C (0.500 → 0.488) while multi-F1 (+0.048) and drum-F1 (+0.128) rise. Knowing *which stem* a note came from is the
   information a full-mix transcriber lacks.
2. Separation errors cost little for pitched instruments: C (0.274) is close to the oracle-stem D (0.334) given that
   the separators saw 16 kHz audio here; drums lose most (0.651 vs 0.714).
3. With Transkun on piano, C2 reaches the oracle-stem route's multi-F1 (0.332 vs 0.334): the specialist makes up for
   the separation loss.
4. Defaults set from this: `cleansplit midi` uses `--mode stems --piano transkun`.

Caveats, stated in the script before the run:
- BabySlakh is **16 kHz and synthetic** (sample-library renders). The separators were trained on 44.1 kHz real
  music and here see nothing above 8 kHz, which biases *against* the stem routes; MuScriptor may have seen Slakh
  in pre-training, which flatters every MuScriptor route alike.
- **Slakh has no singing**: the vocal route (voice constraint) is untested by this benchmark.
- Absolute F1 is lower than MuScriptor's paper reports on its own test set (multi-F1 0.48 for the 1.4B model);
  different data, a smaller model, 30 s excerpts cut mid-phrase. Only the within-benchmark comparison is claimed.
- MuScriptor writes no velocities (fixed 90); Transkun's piano notes carry real velocities.

Added after the first results (each written into the script before it was run):
- **Q3 normalise every stem to −20 dBFS RMS before MuScriptor: NO** — multi-F1 0.332 → 0.332 (+0.001), better on
  10/20 tracks, drum-F1 0.651 → 0.649. Prompted by After 2's Demucs drum stem at −48 dBFS; MuScriptor does not
  normalise its input. Caveat: BabySlakh stems sit at ordinary mix levels, so a stem as quiet as that one is barely
  represented here; the question is answered for typical stems, not for extreme ones. Not adopted.
- **Density guard** (not a pre-registered question; threshold taken from the truth, not from scores): a stem
  averaging over 40 notes/s is left out of the combined MIDI. BabySlakh's densest real stem averages 30.7 notes/s.
  Never fires on the shipped route (C2 unchanged at 0.332); on plain C it removes one 104 notes/s piano stem
  (0.274 → 0.298).

First real song (`After 2`, hip-hop, the user's own): from the `ensemble` split, vocals 675 / piano (Transkun) 494 /
guitar 4,932 notes looked plausible, but drums gave 0 and bass 51 (the split had missed the drums and 808s), and
`other` produced 28,871 notes (16k "French horn") because the missed low end landed in the `other` remainder.
From the `ensemble_demucs` split on a 60 s excerpt: drums 25, bass 117, other 564 (plausible piano, 9.4 notes/s).
Two bass stems differing by only −20 dB gave 5 vs 117 notes: **near-sine 808 bass is a real weakness of
MuScriptor**. The quality of the MIDI is bounded by the quality of the split.

---

## 14. Artifacts vs bleed: measuring the thing the project is for (2026-09-22)

Every number in sections 1–13 is SDR or plain SNR. Both lump together two faults that sound completely different
and have different cures. The BSS decomposition (`mir_eval.separation.bss_eval_sources`) splits the estimate into
target + interference + artifacts and scores them separately:

* **SIR — bleed.** Other instruments leaking into the stem. You hear a hi-hat in the vocal.
* **SAR — artifacts.** Energy that belongs to *no* source: warble, smearing, musical noise, chunk seams. You hear
  the separator itself.

"Split with no artifacts" is a statement about SAR. Nothing in this project had ever measured it.

**Protocol.** MUSDB18-HQ test songs 1–20 in alphabetical order, the same 30 s mid-song excerpts and the same cached
estimates as section 11 — no new separation, so this is a re-reading of results already on disk, not a new sample.
Downsampled to 16 kHz because bss_eval's filter projections are O(n²)-ish; this lowers absolute numbers slightly and
band-limits the comparison to ≤ 8 kHz (see §14.3). Median over songs. `tools/artifact_metrics.py`.

### 14.1 Baseline

| candidate | stem | SDR | SIR (bleed) | SAR (artifacts) |
|---|---|---|---|---|
| sw | vocals | 12.59 | 23.88 | 12.93 |
| sw_tta | vocals | 12.65 | 23.89 | 13.00 |
| sw_tta | drums | 13.52 | 24.04 | 14.06 |
| sw_tta | bass | 10.98 | 20.20 | 12.40 |
| sw_tta | other | 8.43 | 16.36 | 9.60 |
| demucs | vocals | 9.90 | 19.11 | 10.38 |
| **ensemble** (shipped) | vocals | **13.62** | 24.39 | **14.12** |
| ensemble | other | 8.36 | 15.76 | 9.67 |
| ensemble_demucs | drums | 12.56 | 23.11 | 12.98 |
| ensemble_demucs | bass | 10.48 | 18.35 | 12.36 |

**SIR sits 10–11 dB above SAR on every stem of every model, and SDR tracks SAR to within ~0.5 dB.** Bleed is not what
is left to fix; it was solved by the RoFormer generation of models. What remains audible is the separator's own
invented energy, and the headline SDR number has been an artifact score all along without saying so.

Two adoption decisions get independent confirmation on this axis:

* **Model averaging is an artifact remover.** The shipped vocal ensemble buys **+1.12 dB SAR** over SW+TTA alone
  (13.00 → 14.12) — the largest single quality gain measured anywhere in this project, larger than TTA (§6), larger
  than anything either generative restorer achieved (§4, §9, both negative). Averaging independent models cancels
  the part of each model's output that is uncorrelated between them, which is exactly the artifact term.
* **`ensemble_demucs` hurts here too** (drums −1.08 dB SAR, bass −0.04), matching its SDR verdict in §11. It stays
  opt-in insurance for songs where SW misses an instrument outright, not a default.

### 14.2 Can the artifacts be reduced? Four candidates, all rejected

`tools/artifact_reduction.py`, pre-registered in the script's docstring **before the run**: *a candidate replaces the
default only if median SAR improves on the vocals, drums AND bass stems, and median SDR drops by no more than
0.10 dB on any of them.* The SDR guard exists because §4 already measured this exact trap — a mixture-consistent
re-weighting that sounds smoother while moving further from the truth.

Change vs the shipped ensemble, median dB over the 20 songs:

| candidate | vocals ΔSAR | drums ΔSAR | bass ΔSAR | vocals ΔSDR | verdict |
|---|---|---|---|---|---|
| `+mdx23c` (3-model vocal average) | +0.22 | ±0.00 | ±0.00 | +0.13 | **no** (see §14.3) |
| `wiener1` (S′ = \|S\|¹/Σ\|S\|¹ · X) | −2.91 | −4.29 | −3.14 | −3.30 | **no** |
| `wiener2` (power 2) | −3.07 | −4.74 | −3.59 | −3.09 | **no** |
| `wiener2_smooth` (5-frame smoothing) | −3.60 | −5.53 | −4.29 | −3.75 | **no** |

**Wiener post-filtering is catastrophic here, and it fails on its own terms.** It is the textbook artifact reducer —
the output can only be a re-weighting of sound really present in the mixture, so invented energy cannot survive — and
it is what Open-Unmix/norbert and `demucs --wiener` do. It cost 3–5.5 dB of *SAR*, the very thing it exists to
protect, and 6–10 dB of SIR. The reason is specific to this system and worth stating: **the shipped stems are already
mixture-consistent by construction** (`other` is the exact remainder, so the four stems sum to the mixture sample for
sample). Wiener has no inconsistency left to remove; all it can do is throw away the models' phase-accurate output
and replace it with a coarse magnitude mask at 4096/1024 resolution, whose own mask error *is* a new artifact. The
smoothing variant, intended to stop mask flutter, is the worst of the three — it blurs the mask across transients.
This is a genuine negative result, not a tuning failure: no parameter of this family was adjusted after seeing it.

### 14.3 The `+mdx23c` disagreement, stated rather than resolved

`+mdx23c` fails the rule on a technicality — it only alters the vocal stem, so "SAR improves on drums and bass" is
vacuously false — and its vocal numbers (+0.22 dB SAR, +0.13 dB SDR) point the *opposite* way from §11, where adding
MDX23C was worse on 16 of 20 songs and it was removed from the ensemble.

The rule stands as written and the shipping decision does not change. The two measurements are not comparable:
§11 is full-band 44.1 kHz SNR with no filter allowance; §14 is band-limited to 8 kHz and bss_eval permits an optimal
distortion filter before scoring. The obvious hypothesis is that MDX23C helps below 8 kHz and hurts above it, which
would make both results correct. That is a hypothesis, not a finding — it is tested in §14.4, not assumed here.
Re-tuning the pre-registered rule to let this candidate through after seeing its numbers would be exactly the
failure mode this project is built to avoid.

### 14.4 The band hypothesis is falsified — and that changes how §14 must be read

`tools/dev/mdx23c_band_probe.py`, hypothesis and prediction written into the docstring before the run: *MDX23C helps
below 8 kHz and hurts above, so §11 and §14 are each right about the band they can see.* Falsification condition
stated up front: the delta has the same sign in both bands.

Vocals, 20 songs, full-rate 44.1 kHz, per-band SNR from the STFT error (Parseval; the window constant cancels).
Adding MDX23C to the 2-model vocal average:

| band | median Δ | mean Δ | songs improved |
|---|---|---|---|
| below 8 kHz | **−0.08 dB** | −0.12 | 4/20 |
| above 8 kHz | **−0.03 dB** | −0.12 | 8/20 |
| full band | −0.08 dB | −0.12 | 4/20 |

**Same sign in both bands. The prediction fails.** MDX23C does not help anywhere; it hurts slightly and fairly
evenly across the spectrum, on 16 of 20 songs, exactly as §11 found. The removal decision in §11 is confirmed by an
independent measurement and the §14.3 explanation I offered for the disagreement was wrong.

So the +0.22 dB SAR in §14.2 comes from somewhere else. The explanation offered here originally was that **`bss_eval`
fits an optimal distortion filter (a 512-tap projection onto filtered versions of the true sources) before scoring**,
absorbing EQ-shaped error into the "target" term instead of charging it as distortion. That mechanism is real and worth
knowing about.

> **Correction (§14.7).** It was not the cause here. The +0.22 dB was an artefact of the *unpaired* statistic, not of
> the filter: re-derived paired from the same `bss_eval` numbers, `+mdx23c` is **−0.04 dB on vocals, winning 6 of 20
> songs**. The filter allowance has not been shown to change any verdict in this project. See §14.7.

**How to read §14 from here on:**

* The **qualitative** conclusion stands: SIR and SAR are computed inside the same framework on the same audio, so
  "bleed is 10–11 dB cleaner than artifacts" is a valid comparison, and artifacts remain the right target.
* The **ensemble's +1.12 dB SAR** stands, because it is corroborated full-band and filter-free by §11
  (+0.39 dB median SNR vs SW, 18/20 songs). Two independent metrics, same direction.
* A **SAR-only gain with no full-band confirmation is not evidence of an audibly cleaner stem.** Any future candidate
  that passes the §14.2 rule must also clear a filter-free full-band check before it ships. That requirement is added
  now, after a falsified prediction — it makes adoption harder, not easier, so it is not a re-tuning of the bar in
  the candidate's favour. It is still worth keeping for that reason, even though §14.7 shows the specific anomaly that
  motivated it had a different cause.

### 14.5 Chunk overlap: the first artifact lever that works (2026-09-28)

The last untested lever, and the only one whose mechanism is the one already known to work here. Chunked inference
denoises each chunk independently, so the seams between chunks and each chunk's own invented energy are uncorrelated
between neighbours. Raising `num_overlap` makes every output sample the average of more independent passes — the same
cancellation that makes model ensembling remove artifacts (§14.1), applied within a single model.

Isolated deliberately: SW+TTA **alone**, not the shipped ensemble, so the overlap factor is the only difference
between arms. All four stems come from the one model. `tools/overlap_experiment.py`, cache per overlap, resumable.

**Both gates, 20 songs, `num_overlap` 4 vs the shipped 2:**

| | vocals | drums | bass | other |
|---|---|---|---|---|
| ΔSAR (artifacts, 16 kHz, bss_eval) | +0.03 | +0.08 | +0.03 | +0.09 |
| ΔSDR (16 kHz, bss_eval) | +0.03 | +0.07 | +0.08 | +0.09 |
| **ΔSNR (full band 44.1 kHz, filter-free)** | **+0.09** | **+0.05** | **+0.05** | +0.07 |
| songs improved, full band | **18/20** | **19/20** | 14/20 | 16/20 |

* Pre-registered artifact rule (§14.2): SAR up on vocals, drums and bass, SDR down no more than 0.10 dB on each →
  **PASS** (SAR up on all four stems; SDR up on all four, so the guard never binds).
* Filter-free full-band gate (`tools/fullband_check.py`, required by §14.4 and written *before* these SAR numbers were
  computed) → **PASS**, positive on every stem, better on 18/20 and 19/20 songs for vocals and drums.

**This is the first candidate in the entire artifact and restoration line of work to clear a gate** — after two
generative restorers (§4, §9), a third vocal model (§10, §14.3), and four post-filters (§14.2) all failed. It is
also the only one whose two metrics agree in sign and in direction on a large majority of songs.

**Effect size, stated plainly: +0.03 to +0.09 dB is not audible on its own.** What makes it worth recording is not
the magnitude but the consistency — 18/20 and 19/20 songs is not sampling noise, and unlike every rejected candidate
it costs nothing in truth-distance. The cost is time: **about 2× the GPU time** (docs/02 measured 3.6× realtime at
overlap 2, 3.1× at 4 for a single pass; with TTA the arm ran at roughly 0.26× realtime on a loaded machine).

**Decision (revised 2026-09-28 by the user): every default is now the best measured setting, and speed is the
opt-out.** My first call was to expose `--overlap` and leave the default at 2, on the grounds that doubling every
render for a sub-0.1 dB gain is a poor trade. The user overruled it — "whatever split mode brings the BEST quality
should be the default" — which is exactly the trade the pre-registered rule said was theirs to make, so the rule now
applies as written: overlap 4 replaces 2.

Three defaults changed, and it is worth being blunt that two of them were much bigger than the overlap finding that
prompted this:

| default | was | now | measured gain | cost |
|---|---|---|---|---|
| `--separator` | `bs_roformer_sw` | **`ensemble`** | **+1.12 dB vocal SAR**, +0.39 dB SNR, 18/20 songs (§11, §14.1) | ~4 model passes |
| `--overlap` | 2 | **4** | +0.03…+0.09 dB SAR, 14–19/20 songs (§14.5) | ~2× |
| `--tta` | off | **on** (`--no-tta` to disable) | better on all four stems (§6, §14.1) | 3× on the SW pass |

**The separator default was the real bug.** Every measurement in §11 and §14 says `ensemble` is the best thing this
project can produce, and the CLI had been defaulting to single-pass SW with no TTA — the *fastest* option — since
before any of it was measured. Two other places silently pinned the fast path: `cleansplit midi` built its own
argument namespace with `overlap 2, tta False`, and `EnsembleSeparator` had no `num_overlap` parameter at all, so
`--overlap` was dropped for the one recipe that ships. All three are fixed and covered by
`test_defaults_are_the_best_measured_setting_not_the_fastest`, which asserts the defaults *and* that opting out still
works, so this cannot silently regress to the fast path again.

`evaluate` is deliberately exempt and stays on single-pass SW: the experiment protocols in §4, §5 and §12 were run
with it, and moving that default would silently change what an already-published number means.

What did **not** change, because the measurements do not support it: `ensemble_demucs` stays opt-in (worse on both
axes, §11 and §14.1), MDX23C stays out (§14.3, §14.4), `--fp16` stays off (it adds numerical noise to exactly the
residuals this project measures), and TTA is **not** switched on for the ep317 member inside the ensemble — that
would be an unmeasured change to a measured recipe. It is a cheap experiment and is listed as open work rather than
guessed at.

### 14.6 Overlap 8: real, consistent, and still not worth it — plus a correction to this section's first version

`num_overlap` 8 passed the section 14.2 rule too, and on the bss_eval table it looked considerably *better* than 4:

| arm | vocals dSAR | drums dSAR | bass dSAR | other dSAR |
|---|---|---|---|---|
| overlap 4 | +0.03 | +0.08 | +0.03 | +0.09 |
| overlap 8 | **+0.32** | +0.11 | +0.02 | +0.10 |

A +0.32 dB vocal gain would have been ten times what overlap 4 delivered, and under "best quality is the default" that
would have made 8 the new default at another 2x in GPU time. It is not real: on the filter-free full-band metric the
vocal stem does not move at all. That is the second time bss_eval's optimal distortion filter has manufactured a
result here, and it is why the section 14.4 gate is mandatory rather than advisory.

**This section's first version then got its own verdict right for the wrong reason, and the correction is instructive.**
It reported the two overlaps as "identical to two decimal places", concluding the benefit *saturates* at 4. Two defects
produced that: `tools/fullband_check.py` computed its deltas as median-of-arm minus median-of-baseline (unpaired) while
only its win counts were paired, and it matched songs **by list position**, so the overlap-8 arm — which had 9 of 20
songs cached at the time — was lined up against whichever songs happened to occupy the same slots. Both are fixed; the
statistic is now `paired()` in that file, matched by song name, covered by `cleansplit/tests/test_paired_statistics.py`.

Rescored properly, all three arms complete at 20 songs, paired medians against single-pass `sw_tta`:

| arm | vocals | drums | bass | other |
|---|---|---|---|---|
| overlap 4 | +0.08 (18/20) | +0.09 (19/20) | +0.07 (14/20) | +0.05 (16/20) |
| overlap 8 | +0.07 (19/20) | +0.08 (18/20) | +0.08 (15/20) | +0.05 (17/20) |

And head to head, overlap 8 against overlap 4 as the baseline:

| | vocals | drums | bass | other |
|---|---|---|---|---|
| overlap 8 vs 4 | +0.01 (16/20) | +0.01 (15/20) | +0.01 (12/20) | +0.00 (11/20) |

So overlap 8 is **not** indistinguishable from overlap 4. It is genuinely, consistently better — by +0.01 dB, winning
12 to 16 songs out of 20. Under the gate as originally written (any paired gain > 0, winning more than half) **it
passes**, and would have doubled every render.

That is a flaw in the gate, not in overlap 8: with enough songs, any arbitrarily small consistent gain clears a
"greater than zero" bar, and cost is unbounded. The gate now carries an effect-size floor of **+0.02 dB**, taken from
the rule already pre-registered in `tools/cache_arm.py` before this run rather than invented for this verdict. Stated
plainly, because the order matters: the floor was added *after* seeing overlap 8 pass, and it is the reason the verdict
stands.

**Decision: overlap 4 stays the default; overlap 8 is rejected on effect size against cost** — +0.01 dB is inaudible,
and it costs roughly half the throughput (0.13x realtime versus 0.26x on the same machine). This is still the first
time in this project that "best quality" and "faster" point at nearly the same setting.

Three lessons worth keeping:
1. The unpaired statistic cuts **both** ways. In section 14.5's follow-up it invented a gain that was not there; here
   it *hid* a real one, by reporting two arms as equal when one wins on 12-16 of 20 songs. Neither error is safe.
2. Pairing by position is not pairing. It is indistinguishable from correct code whenever every arm happens to be
   complete, which is exactly when nobody checks — and these caches fill incrementally, so incomplete is the norm.
3. A decision rule needs a minimum effect size, not just a sign. "Better than zero" plus "costs whatever it costs" is
   not a quality policy; it is a licence to spend unlimited time on inaudible gains.

### 14.7 One root cause, blamed on the wrong thing twice: it was the unpaired statistic, not bss_eval's filter

Two anomalies in §14 were reported as evidence that `bss_eval`'s optimal distortion filter manufactures results:

* `+mdx23c` gaining **+0.22 dB** vocal SAR while being worse full-band (§14.2, §14.4);
* overlap 8 gaining **+0.32 dB** vocal SAR while the full-band metric did not move (§14.6, first version).

Both attributions were wrong, and the real cause is the same in both cases: the **unpaired** statistic. Every row in
`tools/artifact_reduction.py` and `tools/overlap_experiment.py` carries its song name, so the pairing can be recovered
from the JSON they already wrote, with no GPU work — `tools/paired_rescore.py` does exactly that. Re-derived from the
**same `bss_eval` SAR numbers**, matched per song:

| candidate | stem | unpaired ΔSAR | paired ΔSAR | won |
|---|---|---|---|---|
| `+mdx23c` | vocals | **+0.22** | **−0.04** | 6/20 |
| `+mdx23c` | vocals (SDR) | +0.13 | −0.07 | 4/20 |
| `+mdx23c` | other | +0.01 | −0.03 | 3/20 |
| overlap 4 | vocals | +0.03 | +0.09 | 18/20 |
| overlap 8 | vocals | **+0.32** | **+0.10** | 19/20 |
| overlap 8 | drums | +0.11 | +0.13 | 20/20 |
| overlap 8 | bass | +0.02 | +0.12 | 17/20 |

Three sign flips, all of them `+mdx23c`, all in the direction that had made a bad candidate look good. And overlap 8's
notorious +0.32 dB is **+0.10 dB** once paired — indistinguishable from overlap 4's +0.09, which is what the full-band
gate was saying all along.

**The consequence is a simplification.** Once every comparison is paired, `bss_eval`'s SAR and the filter-free
full-band SNR **agree on every case tested here**:

| | paired ΔSAR (vocals) | paired Δfull-band SNR (vocals) |
|---|---|---|
| overlap 4 vs single-pass | +0.09 | +0.08 |
| overlap 8 vs single-pass | +0.10 | +0.07 |
| `+mdx23c` | −0.04 | worse, 16/20 |

The two metrics never actually disagreed. An unpaired median was making one of them look untrustworthy.

**What this does and does not change.**

* **All four §14.2 rejections stand.** Paired, every candidate is negative on every stem — the Wiener variants by
  2.4–4.6 dB. No verdict in §14 flips.
* **The §14.4 filter explanation is withdrawn as the cause.** The filter allowance is a real property of `bss_eval` and
  a documented hazard, but it has not been shown to change a single verdict in this project. Claiming it did, twice, was
  a misattribution on my part: I reached for the sophisticated explanation and did not check the boring one first.
* **The filter-free full-band gate stays.** It was right in both disputed cases, it is the metric that tracks what a
  listener hears, and an independent check is worth keeping regardless of which mechanism motivated it.
* **Paired statistics are now enforced in code, not policy** — `paired()` in `tools/fullband_check.py`, matched by song
  name, with `cleansplit/tests/test_paired_statistics.py` failing on both the unpaired and the by-position forms.

The general lesson, and the reason this section exists rather than a quiet edit: an error in the *statistic* looks
exactly like an error in the *measurement*, and it is much cheaper to check. Three different explanations were offered
for these anomalies across §14.3, §14.4 and §14.6 before the actual cause was found, and the actual cause was one line
of arithmetic in the comparison, sitting in the tool that was supposed to be the safeguard.

### 14.8 Ranking every quality lever by what it costs: ensembling is ~40x more compute-efficient than TTA

§14.5–14.7 measured levers one at a time against a fixed baseline, which answers "does it help?" but not "is it the
best use of the next doubling of GPU time?" — the actual question behind a `--quality` switch. This section normalises
every measured lever by its cost.

**Cost is countable exactly, not timed.** `roformer.py` sets `step = chunk // num_overlap`, so the number of forward
passes is linear in `num_overlap`; TTA is exactly 3 passes; each ensemble member is its own pass. So relative compute is
`models x num_overlap x (3 if TTA else 1)` in units of one chunk-forward. The shipped default is SW+TTA at overlap 4
(3x4 = 12) plus ep317 at overlap 4 (1x4 = 4) — **16 units**. Only overlap 8's wall-clock was ever measured
(0.129x realtime), so wall-clock is deliberately not used here; pass counts are exact and machine-independent.

All gains below are **paired medians on vocals** with win counts, re-derived in §14.7 from the committed benchmark JSON:

| lever | compute | gain (vocals, paired) | won | dB per doubling of compute |
|---|---|---|---|---|
| **add ep317 (the ensemble)** | 12 → 16 (**1.33x**) | **+0.42** vs SW+TTA | 18/20 | **+1.01** |
| overlap 2 → 4 | 6 → 12 (2x) | +0.08 | 18/20 | +0.08 |
| TTA, at overlap 2 | 2 → 6 (3x) | +0.04 | 20/20 | +0.025 |
| overlap 4 → 8 | 12 → 24 (2x) | +0.01 | 16/20 | +0.01 |

**Averaging a second, architecturally different model is roughly 40x more compute-efficient than test-time augmentation
and 12x more than doubling the overlap.** It is not close. This is the same conclusion as §14.1's +1.12 dB SAR and
docs/01 §8.2's comparable-strength rule, but stated in the unit that matters when choosing where to spend time.

Two honest qualifications:

1. **ep317's gain is vocals-only.** It is a vocal model; it contributes nothing to drums, bass or `other`. Overlap and
   TTA help every stem. So the ranking above is the vocal-stem ranking, and it is the flagship stem — but the reason
   drums and bass have no equivalent lever is precisely that they have no ensemble partner, which is what the SCNet
   experiment (§14.9, pending) is testing.
2. **The gains are measured against different baselines and are not additive.** TTA's +0.04 dB was measured at overlap
   2; overlap and TTA plausibly cancel the same uncorrelated error, which is the pending question below.

**A falsifiable prediction, recorded before the running experiment finishes.** If overlap and TTA remove overlapping
error, then TTA's +0.04 dB at overlap 2 must **shrink** at overlap 4, because overlap 4 has already removed part of
what TTA was removing. Prediction: **at overlap 4, TTA's paired median gain on vocals is below +0.04 dB, and below the
+0.02 dB adoption floor.** Falsification: TTA gains ≥ +0.04 dB at overlap 4, which would mean the two mechanisms are
independent and TTA is simply underpriced at 3x. Either way the cost ranking above already says TTA is the worst-value
component of the default recipe apart from overlap 8, which is not in it.

**This is the table the `--quality` tiers will be built from**, once the TTA and SCNet verdicts land — so that each tier
carries a real measured cost in passes and a real measured dB, rather than an adjective.
