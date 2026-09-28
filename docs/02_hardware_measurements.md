# Hardware measurements (target machine)

Machine: Intel i9-10980HK, 32 GB RAM, NVIDIA RTX 2080 Super Max-Q 8 GB (compute capability 7.5, driver 610.88), Windows 11 Pro 26200.
Software: Python 3.11.15, PyTorch 2.11.0+cu128, CUDA 12.8.

All numbers below were measured on this machine on 2026-09-16. Nothing is extrapolated.

## BS-RoFormer SW (BS-Rofo-SW-Fixed), vendored MSST code

| Item | Result |
|---|---|
| Checkpoint | 699,412,152 bytes, SHA-256 `24e7d35e…916e` (matches the `enerjazzer` rehost) |
| Strict `load_state_dict` into vendored model | all 1939 keys matched, 174.7 M parameters, float32 weights |
| Weights on GPU | 674 MB |
| Attention kernel | SDPA memory-efficient/math. Flash attention is unavailable on compute capability 7.5, and upstream also disables it on Windows |

Single-chunk forward pass (random stereo input, batch 1):

| Chunk | Precision | Peak VRAM | Time |
|---|---|---|---|
| 4 s | fp32 | 1,027 MB | 1.77 s (includes warm-up) |
| 8 s | fp32 | 1,378 MB | 1.18 s |
| 13.35 s (native 588,800 samples) | fp32 | 1,845 MB | 2.57 s |
| 8 s | fp16 autocast | 1,207 MB | 0.91 s |
| 13.35 s | fp16 autocast | 1,525 MB | 1.09 s |

Full song, `cleansplit analyze` (native chunk, overlap 2, fp32):

| Song | Duration | Separation | Peak VRAM | Analysis (CPU) | Total |
|---|---|---|---|---|---|
| Concrete Crown.wav (44.1 kHz stereo WAV) | 243.4 s | 122.8 s (≈2.0× realtime) | 1,828 MB | ≈95 s | 223.5 s |

**Conclusion:** SW runs comfortably on 8 GB, using less than a quarter of VRAM at its native context. fp32 is the default because fp16 adds rounding noise to exactly the residuals CleanSplit measures. `--fp16` exists for smaller GPUs. CPU fallback is implemented but was not timed; expect it to be far slower than realtime.

## A2SB (NVIDIA Audio-to-Audio Schrödinger Bridge): measured 2026-09-16
Code: `third_party/diffusion-audio-restoration` @ `02ddff01` (NVIDIA Source Code License-NC). Weights: `models/a2sb/` (NVIDIA OneWay Noncommercial License). Probe: `tools/a2sb_probe.py`; raw results in `outputs/_benchmarks/a2sb_vram_probe.json`.

| Item | Result |
|---|---|
| Checkpoints | onesplit 0–1, twosplit 0–0.5, twosplit 0.5–1; each 2,262,312,618 bytes, **SHA-256 matches the HF LFS oid** for all three |
| Loading | `torch.load(weights_only=True)` works (no arbitrary unpickling); `vf_model.*` loads **strictly** into `AttnUNetF` |
| Parameters | 565.5 M per network (float32 in checkpoint) |
| VRAM free at start | 7,115 MB of 8,192 MB |

One denoiser forward pass at NVIDIA's inference window: 3 channels (mag^0.25 plus instantaneous phase) × 1024 bins × 256 frames (≈3 s of mono audio):

| Configuration | Weights | Batch | Peak VRAM | Time per pass |
|---|---|---|---|---|
| onesplit fp32 | 2,172 MB | 1 | 3,367 MB | **0.55 s** |
| onesplit fp32 | 2,172 MB | 2 | 4,555 MB | 7.1–10.0 s ⚠ pathological |
| onesplit fp16 weights + autocast | 1,101 MB | 1 | 1,956 MB | **0.42 s** |
| onesplit fp16 weights + autocast | 1,101 MB | 2 | 2,577 MB | 0.84 s |
| twosplit, both networks resident, fp32 | 4,354 MB | 1 | 5,541 MB | 2.10 s ⚠ (4× slower than onesplit fp32) |
| twosplit, both networks resident, fp16 + autocast | 2,195 MB | 1 | 3,049 MB | **0.39 s** |

Precision (same input, one pass, real weights): autocast with fp32 weights differs from fp32 by **0.12%** (relative L2); fp16 weights + autocast differ by **0.28%**. This is for a *single* pass. Error accumulation over 50–200 sampling steps is **not yet measured** and must be checked before trusting fp16 restorations.

Findings:
1. **A2SB fits on this GPU.** The recommended configuration is the NVIDIA 2-split ensemble with both networks resident in fp16 plus autocast: 3.05 GB peak, 0.39 s per pass at batch 1.
2. **NVIDIA's defaults do not suit this laptop.** Their API uses batch 16 and fp32. fp32 at batch 2 and the fp32 two-network setup ran 4–18× slower than linear, even though the reported peak allocation stayed below free VRAM. The cause is **not verified** (candidates: WDDM paging to system RAM, allocator reserve above allocated, kernel selection).
3. Bare `.half()` inference fails: the network creates float32 tensors internally (embeddings, `GroupNorm32` casts to float32). fp16 must run under `torch.autocast`.
4. Cost estimate (not measured end to end): 50 steps × 0.39 s ≈ **20 s per 3 s window per channel**, so ≈ 40 s per stereo region window. The 300 real-song regions from `docs/04_results.md` would take ≈ 3.3 h. Whole-stem restoration of a 243 s song: 163 windows (hop 128 frames) × 50 steps × 0.39 s ≈ 53 min per channel. NVIDIA's inpainting script uses 200 steps, which is 4× longer.

## Apollo (JusperLee band-sequence restoration): measured 2026-09-17

16.54 M parameters, 66.5 MB fp32 weights, both released checkpoints identical in shape (`sr 44100, win 20 ms,
feature_dim 256, layer 6`). Loaded from UVR's `models/Apollo_Models`, SHA-256 pinned, `weights_only=True`.

Single forward pass, stereo, fp32, `tools/apollo_probe.py` (RTX 2080 Super Max-Q, 8 GB):

| Chunk | Peak allocated | Peak reserved | Seconds | Speed |
|---|---|---|---|---|
| 5 s | 2100 MB | 2938 MB | 1.01 | 5.0x realtime |
| 10 s | 4136 MB | 5734 MB | 1.98 | 5.0x realtime |
| 20 s | 8201 MB | 11386 MB | 33.04 | 0.6x realtime |
| 30 s | 12266 MB | 17066 MB | 87.31 | 0.3x realtime |
| 60 s | OOM | | | |

VRAM grows linearly at about **410 MB per second of stereo audio** — the band-split representation keeps 80 bands
x feature_dim x frames resident. Past ~16 GB reserved the driver falls back to shared system memory, which is why
20 s and 30 s are 50-300x slower rather than failing: those rows are thrashing, not compute.

**Operating point: 5 s chunks, 1 s overlap, 0.5 s context pad** (the default in `restoration/apollo.py`).
A 4-minute song takes about 60 s at 2.1 GB peak, so Apollo can run while nothing else is resident on the card.


## MDX23C InstVoc HQ (TFC-TDF-net v3), vendored MSST code: measured 2026-09-21

448 MB fp32 checkpoint, config `model_2_stem_full_band_8k.yaml` (n_fft 8192, hop 1024, dim_f 4096, native chunk
261,120 samples = 5.92 s). Loaded strict, SHA-256 verified. 30 s stereo excerpt of `Concrete Crown.wav`, fp32:

| num_overlap | Seconds for 30 s | Speed | Peak allocated |
|---|---|---|---|
| 2 | 8.3 | 3.6x realtime | 1482 MB |
| 4 | 9.6 | 3.1x realtime | 1482 MB |
| 8 (model default) | 19.8 | 1.5x realtime | 1482 MB |

Peak VRAM does not depend on overlap (one chunk is resident at a time). Vocals + instrumental reproduce the
mixture to -64.4 / -65.3 / -65.8 dB, so the two outputs are close to complementary without being forced to be.
