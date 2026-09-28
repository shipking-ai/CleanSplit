<div align="center">

# CleanSplit

**Local-first stem separation that measures itself honestly — and is built to prove its own ideas wrong.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-93%20(83%20CPU%20%2B%2010%20GPU)-brightgreen.svg)](cleansplit/tests)
[![Benchmark](https://img.shields.io/badge/benchmark-MUSDB18--HQ%2020%20songs-orange.svg)](docs/04_results.md)
[![VRAM](https://img.shields.io/badge/VRAM-8%20GB-lightgrey.svg)](docs/02_hardware_measurements.md)
[![Offline](https://img.shields.io/badge/inference-100%25%20local-success.svg)](#requirements)

[Quick start](#quick-start) · [Quality tiers](#quality-tiers) · [Results](#results) · [What didn't work](#what-didnt-work) · [Docs](#documentation)

</div>

---

## What this is

A six-stem separator (vocals · drums · bass · guitar · piano · other) with **artifact-aware analysis** and
**mixture-consistent, region-restricted restoration**. Everything runs on your machine. No uploads, no API calls.

It exists to answer one question experimentally — and to be able to answer **"no"**:

> Can generative, artifact-aware restoration improve AI-separated audio while staying faithful to the original mix?

So far the measured answer is **no**, and that answer is the deliverable. Seven restoration and ensembling ideas have
been tested and rejected on real studio truth. [What didn't work](#what-didnt-work) is the most useful section here.

## Why it's built this way

This is not an "AI enhancer" with a confident README. Four rules are enforced in code, not in prose:

| Rule | How it's enforced |
|---|---|
| **Clean audio stays untouched** | A restoration may only alter samples inside a detected time-frequency region. Everything else is bit-identical, and that is asserted. |
| **The mix must still add up** | Stems sum to the mixture sample-for-sample: `other` is the exact remainder. A restoration that breaks mixture consistency is **auto-rejected**. |
| **Decision rules are written before the run** | Every experiment's adoption rule lives in its script's docstring, pre-registered, with a falsifiable prediction and an explicit falsification condition. |
| **Comparisons are paired, with win counts** | Median-of-A minus median-of-B is banned. It manufactured two false results here before being caught ([§14.7](docs/04_results.md)) and is now blocked by [tests](cleansplit/tests/test_paired_statistics.py). |

Gains below **+0.02 dB** are not adopted no matter how consistent they are, because a bar of "better than zero" buys
inaudible gains at unbounded cost. That floor cost overlap-8 its place in the default recipe.

## Quick start

```bash
uv venv .venv --python 3.11
```
```bash
uv pip install --python .venv/Scripts/python.exe torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```
```bash
uv pip install --python .venv/Scripts/python.exe -e ".[separation,dev]"
```
```bash
cleansplit doctor
```

Separate a song at the best measured quality (this is the default — you don't pass anything):

```bash
cleansplit separate "song.wav"
```

Four times faster, keeping 95% of the vocal gain:

```bash
cleansplit separate "song.wav" --quality balanced
```

Inspect it in the desktop UI — lanes, solo/mute, A/B compare, flagged artifact regions:

```bash
cleansplit ui
```

## Quality tiers

Compute is counted in **forward passes**, not timed, so the ratios hold on any machine
(`step = chunk // num_overlap`, so cost is linear in overlap; TTA is exactly 3 passes).

| `--quality` | Recipe | Compute | Vocals vs `fast` | Notes |
|---|---|---|---|---|
| `fast` | SW alone, overlap 2 | **2 units** | — | 8× cheaper than `best` |
| `balanced` | SW + ep317, overlap 2, no TTA | **4 units** | **+0.442 dB** (18/20) | **95% of best's gain.** Drums/bass identical to `fast` |
| `best` *(default)* | SW+TTA + ep317, overlap 4 | **16 units** | **+0.464 dB** (18/20) | Also +0.11 dB drums, +0.10 dB bass |

No tier advertises a single dB figure, because **per-song it ranges from −0.09 dB to +5.12 dB**
([§14.11](docs/04_results.md)). Usually `fast` is close; occasionally it collapses. Any explicit
`--separator` / `--overlap` / `--tta` overrides the tier.

> **Why `balanced` works:** ranked by dB per doubling of compute, averaging a second model is **~40× better value than
> TTA** and ~12× better than doubling the overlap ([§14.8](docs/04_results.md)). `balanced` keeps the lever that pays and
> drops the two that barely do.

## Results

MUSDB18-HQ test set, 20 songs, real studio stems, 30 s excerpt centred on each track's midpoint — no content selection.
All deltas are **paired medians** with win counts. Full protocol and numbers: [`docs/04_results.md`](docs/04_results.md).

| Measurement | Result |
|---|---|
| Shipped ensemble vocals vs single-pass SW | **+0.45 dB**, 18/20 songs |
| Same, on artifact-only SAR | **+1.12 dB** — the largest quality gain in the project |
| Chunk overlap 2 → 4 | +0.08 dB vocals, +0.09 drums, +0.07 bass (14–19/20) |
| Artifacts vs bleed | SIR sits **10–11 dB above SAR** on every stem: bleed is solved, **artifacts are the remaining error** |
| Misallocation vs lost information | **~97%** of the error is audio filed under the wrong stem, not missing audio |

That last line is why single-stem generative repair cannot work here: the information is present in the mix, just in the
wrong place, so it cancels in the sum and no amount of "imagining" the missing part helps.

### Ceilings, measured before building anything

Both are computed by **reading the ground truth** — they cannot ship, they exist to decide whether a feature is worth
writing at all.

| Idea | Oracle ceiling | Verdict |
|---|---|---|
| Per-song / adaptive ensemble **weighting** | **+0.013 dB** | Axis closed permanently. Equal weighting is *exactly* optimal. |
| Per-TF-tile **selector** ("listen and pick the better model") | **+0.38 dB** | Real budget — but unreachable without an external prior. |

## What didn't work

Published because a negative result measured properly is worth more than a positive result measured loosely.

| Candidate | Outcome |
|---|---|
| **A2SB** generative inpainting | Rejected 23/23 by the mixture-consistency gate |
| **Apollo** codec restoration | Worse than doing nothing in **24 of 24** ground-truth measurements |
| **MDX23C** in the vocal ensemble | Worse on 16/20 songs once real truth was used |
| **HTDemucs_ft** averaged into drums/bass | −0.55 dB drums on **20/20** songs |
| **Wiener** post-filtering (3 variants) | −2.4 to −4.6 dB. Fails *by construction*: stems already sum to the mixture |
| **Overlap 8** | Real but +0.01 dB, for 2× the render time — below the adoption floor |
| **4 truth-free tile combiners** | All rejected. Disagreement says *how much* error, never *which model* has it |

Two of these were nearly adopted on bad statistics, and both near-misses are written up rather than quietly deleted:
[§14.7](docs/04_results.md) traces a "+0.32 dB" gain that was an artefact of an unpaired median, and explains why the
sophisticated explanation I reached for twice was wrong.

## How it works

```
input ─▶ separate ─▶ reconstruct ─▶ detect artifacts ─▶ propose restoration ─▶ GATE ─▶ accept / reject
             │            │                │                                     │
      ensemble of     residual vs      deterministic                    mixture consistency
      independent     the mixture      TF detectors +                   + region containment
      models                           synthetic benchmark             + bit-identical elsewhere
```

The gate is the point. It is designed to say no, and it says no to almost everything.

## Requirements

- Windows 10/11 (Linux likely fine, untested), Python 3.10+
- NVIDIA GPU recommended; measured on an **RTX 2080 Super Max-Q, 8 GB** ([`docs/02`](docs/02_hardware_measurements.md)).
  CPU works but is slow. `ffmpeg` on PATH for non-WAV/FLAC input.
- **Model weights are not included or downloaded.** `BS-Rofo-SW-Fixed` comes from an existing Ultimate Vocal Remover 5.6
  install. Their licence is unknown and the trainer unidentified, so CleanSplit never redistributes them.

## Documentation

| Document | Contents |
|---|---|
| [`docs/01_research_decision_report.md`](docs/01_research_decision_report.md) | Architecture research, model survey, primary sources |
| [`docs/02_hardware_measurements.md`](docs/02_hardware_measurements.md) | Real throughput and VRAM on 8 GB |
| [`docs/03_artifact_detectors.md`](docs/03_artifact_detectors.md) | Detectors + synthetic ground-truth benchmark |
| [`docs/04_results.md`](docs/04_results.md) | **Every measurement, including the failures.** Start at §14 |
| [`HANDOFF.md`](HANDOFF.md) | Running engineering log, newest first |
| [`QUICKSTART.md`](QUICKSTART.md) | Command reference |

## Licence

MIT for CleanSplit's own code — see [`LICENSE`](LICENSE), which also lists the vendored third-party components
(`bs_roformer/`, `mdx23c/`, `scnet/` MIT; `ui/static/vendor/` BSD-3; `third_party/apollo/` CC BY-SA 4.0) and states
plainly that model weights are not redistributed.
