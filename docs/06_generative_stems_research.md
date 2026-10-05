# CleanSplit — Generative Stems: Research & Decision Report

Date: 2026-10-05. Same convention as [docs/01](01_research_decision_report.md): **[verified]** was checked against a
primary source this session, **[reported]** is a third-party claim we have not reproduced, **[measured]** was run on the
target machine (i9-10980HK / RTX 2080 Super Max-Q **8 GB** / Windows 11).

**Question asked:** can CleanSplit have its own generative model that writes a song stem by stem and labels each stem by
instrument?

**Short answer:** the instrument-labelling half is free, the generative half splits into one tractable tier and two that
are not, and the single biggest finding is a **licensing problem at the symbolic layer that the field's own authors
flag**. Details below; the verdict is in §6.

---

## 1. The labelling requirement dissolves

If stems are generated one instrument at a time, the label *is* the generator that was called. No classifier, no error
rate, no confusion matrix to report. A classifier is only needed on the strictly worse pipeline — generate a mixture,
then separate it, then guess what came out. This is recorded because it was half the original question and it costs
nothing to satisfy.

---

## 2. What the project already has and already knows

| | Finding |
|---|---|
| `cleansplit/audio/synthetic.py` | **[verified]** `make_song()` already generates all six stems independently (`vocals`, `drums`, `bass`, `guitar`, `piano`, `other`) and sums them. Its own docstring: "crude instrument caricatures, not realistic music". Used by the test suite. This is tier 0 and it exists. |
| BabySlakh | **[verified]** `docs/04_results.md:401` uses it as transcription ground truth (Zenodo 4603870, CC BY 4.0, 20 tracks). |
| Recorded limits of it | **[verified]** docs/04 §13 already states BabySlakh "is **16 kHz and synthetic**", that the separators "see nothing above 8 kHz, which biases *against* the stem routes", and that "**Slakh has no singing**". |

### 2.1 One of those three recorded limits is an artifact of the subset, not the dataset

| Property | BabySlakh | Slakh2100 |
|---|---|---|
| Sample rate | **[verified]** 16 kHz — "All of the audio is in the wav format and has a sample rate of 16 kHz" (Zenodo 4603870) | **[verified]** **44.1 kHz**, 16-bit, **mono**, FLAC (Zenodo 4599666) |
| Tracks | **[verified]** 20 | **[verified]** 2,100 (1500/375/225 train/valid/test), 145 h of mixtures, 104.3 GB |
| Licence | **[verified]** CC BY 4.0 | **[verified]** "Creative Commons Attribution 4.0 International" |

**This matters more than anything else in the near term.** The 16 kHz ceiling that docs/04 records as biasing the
benchmark against stem routes is a property of the 20-track *debug* subset, not of Slakh. Moving to full Slakh2100
removes two of the three recorded limitations (sample rate, n=20) for the price of a 104 GB download and no new code.

What it does **not** fix: Slakh2100 is **mono**, and the separators are stereo 44.1 kHz models — so a mono benchmark
still mis-states their operating conditions. And "no singing" remains true, so the vocal route stays untested by it.

### 2.2 Slakh cannot be re-rendered without paid, platform-locked software

**[verified]** from `ethman/slakh-generation` (the authors' own generation code, MIT):
- synthesis via **RenderMan**, a JUCE-based VST host with Python bindings (public domain / unlicense);
- instruments are **Native Instruments Kontakt** + Kontakt libraries — **commercial**;
- the README states **"this project is Mac only!!!!"**.

So "render more Slakh-style data ourselves" is not a free action, and not one this Windows machine can take with that
toolchain. A generator of our own would need a different renderer and different samples (§4).

---

## 3. The decisive finding: the symbolic layer is copyright-encumbered, per its own authors

The obvious way to get novel multitrack data is to generate MIDI per instrument and render it. The best open-weights
candidate is the **Anticipatory Music Transformer** (Stanford CRFM, Thickstun et al.).

**[verified]** from its model card (`johnthickstun.com/assets/pdf/music-modelcard.pdf`, read directly):
- **Licence: "Apache License, Version 2.0"** — permissive, as hoped.
- Trained on the **Lakh MIDI dataset**.
- But under its own *Copyright* heading: **"The Lakh MIDI dataset contains large quantities of copyrighted music. The
  copyright status of models trained on this data—and music sampled from these models—is an open legal question."**

That is the model's authors flagging it, not a third-party objection. And the encumbrance is **upstream of Slakh too**:
Slakh2100's audio is CC BY 4.0, but it is rendered **from Lakh MIDI** — so the clean licence on the audio sits on top of
transcriptions of copyrighted recordings.

**Consequence for this project specifically.** CleanSplit is public, published under the maintainer's own name, and
already refuses to redistribute weights with unclear licences (docs/01 §1.1: "Weights: none stated"; the MuScriptor
caveat: "CC BY-NC 4.0: this MIDI is fine for your own music, not for a paid product"). An Apache-2.0 model whose own card
calls its output's copyright status "an open legal question" is the same class of risk the project already declines
elsewhere. Using it to *generate redistributable benchmark audio* would be inconsistent with that standard.

---

## 4. A clean-room tier-1 toolchain does exist

If the symbolic source is ours (procedural, or our own rules, or public-domain scores) rather than a Lakh-trained model,
the render side is fully permissive and runs on this machine:

| Component | Finding |
|---|---|
| `sfizz` | **[verified]** SFZ sampler engine, **BSD-2-Clause**. `sfztools/sfizz-render` is a CLI that renders a MIDI/SMF file + an SFZ to a WAV — also **BSD-2-Clause**. Render each instrument's MIDI separately ⇒ stems that are exactly separated and exactly labelled by construction. |
| FluidSynth | **[verified]** LGPL-2.1, renders SF2/SF3 to disk. Workable but copyleft-adjacent; sfizz's BSD is the cleaner choice. |
| **VSCO 2 Community Edition** | **[verified]** **CC0** (public domain), ~3,000 samples / ~70 packs, original WAVs **44.1 kHz** 16/24-bit, with a "Vanilla SFZ" build for sfizz. Chamber orchestra. |
| **VCSL** (Versilian Community Sample Library) | **[verified]** **CC0**, general-purpose, 44.1 or 48 kHz, 16/24-bit WAV, SFZ version in releases. Explicitly "even make commercial software, no royalties, no credit". |
| Karoryfer Growlybass / Emilyguitar | **[reported]** free, open-source sample libraries, 310 × 24-bit 44.1 kHz samples for a Squier Jazz bass, seven SFZ mappings. **Licence not verified this session** — Karoryfer uses its own terms; must be checked before any redistribution. |

**Gap to be honest about:** VSCO 2 CE and VCSL are strongest on orchestral/acoustic material. A convincing pop/rock kit,
electric bass and electric guitar under CC0 is the weak spot, and **singing is not available at all** — which is
precisely the gap docs/04 already records for Slakh. A generator built this way would reproduce Slakh's own blind spot.

---

## 5. Tiers 2 and 3: the neural options, and why they are not the recommendation

| Approach | Finding |
|---|---|
| **MSDM** — Multi-Source Diffusion Models | **[verified]** Mariani, Tallini, Postolache, Mancusi, Cosmo, Rodolà; **ICLR 2024 oral**; arXiv 2302.02257. Learns the joint density of sources sharing a context, so one model does generation, separation, **and source imputation** ("playing a piano track that goes well with the drums"). **Trained on Slakh2100** — so it inherits §3's encumbrance and Slakh's mono/no-vocals character. |
| **StemGen** | **[verified]** Parker, Spijkervet, Kosta, Yesiler, Kuznetsov, Wang, Avent, Chen, Le (Stability AI + TikTok); **ICASSP 2024**; arXiv 2312.08723. Non-autoregressive transformer that **generates a stem conditioned on the other stems** — i.e. "listens" to context. Closest published match to the user's request. |
| **MusicGen-Stem** | **Not verified.** Referred to from recollection in conversation; no primary source located this session. **Treat as nonexistent until checked.** |
| Training our own from scratch | **No.** Models in this class are trained on tens of thousands of hours across datacenter clusters. **8 GB VRAM is not in the conversation**; exact GPU-hour figures were *not* verified this session and should not be quoted. |

Note that MSDM and StemGen are *research results*, not drop-in weights with a licence audit done. Neither was checked
this session for released weights or their terms.

---

## 6. Verdict

1. **Do not build a generative model.** Not now, and tier 3 not at all on this hardware.
2. **Do the free win first: move the transcription benchmark from BabySlakh to full Slakh2100.** It is the same licence,
   already cited in docs/04, and it retires two of the three limitations §13 records. No new code, no new dependency, no
   new licence question. Remaining caveats (mono, no singing) must stay written down.
3. **If novel multitrack ground truth is still wanted after that**, build tier 1 — our own symbolic layer (procedural,
   *not* Lakh-trained) rendered per-instrument through `sfizz` + CC0 libraries. Justify it as a **measurement tool that
   emits data with known ground truth**, never as music generation.
4. **Do not use a Lakh-trained symbolic model to produce redistributable audio**, on the authors' own copyright note
   (§3). This is the same standard the project already applies to model weights.
5. **Mixing/mastering (the previous question) stays the better investment**, because mastering has measurable targets
   and generation has none.

### 6.1 Pre-registered success criterion, if tier 1 is ever built

Written before any code, in the style of the experiment scripts:

> **RULE.** A tier-1 generator earns a place in this repository only if, on ≥ 20 generated songs, the separators'
> measured SDR/SNR per stem is **rank-correlated** with their MUSDB18-HQ results (Spearman ρ ≥ 0.6 across the six
> stems). If generated material ranks the separators differently from real music, it is not a benchmark — it is a
> synthetic artifact with its own biases, and any number measured on it is uninterpretable.
>
> **PREDICTION.** It fails on `guitar` and `piano` first, because those are the two stems docs/01 §1.1 already records
> as SW's weakest ([reported] MVSep SDR 9.05 and 7.83 dB) and the two most dependent on sample-library realism.
>
> **FALSIFIED IF** ρ ≥ 0.6 holds *and* the generated set reproduces the known `best`-vs-`fast` ordering from docs/04
> §14.11. Then it is a usable benchmark and the vocal gap (§4) becomes the only blocker.

---

## 7. What was not checked this session

Stated so nobody reads this as more complete than it is:
- Karoryfer library licences (§4).
- Whether MSDM or StemGen released weights, and under what terms (§5).
- Whether `MusicGen-Stem` exists at all (§5).
- Slakh2100's actual download integrity / per-stem layout — only its Zenodo metadata was read, not the archive.
- MoisesDB's licence is **contradictory across sources** and is recorded separately in §8.

## 8. MoisesDB — a real-multitrack alternative, with a licence conflict

**[verified]** 240 previously unreleased tracks, 45 artists, 12 genres, 3–10 stems each, two-level instrument taxonomy;
vocals/drums/bass present in nearly all songs (ISMIR 2023, Pereira et al., arXiv 2307.15913). This would answer the
"no singing" and "mono" gaps that Slakh cannot.

**But the licence is reported two ways and the difference is material:**
- Zenodo record 10265363 states **"Creative Commons Attribution 4.0 International"**.
- The maintainers' own repository `moises-ai/moises-db` states **"MoisesDB is distributed under the Creative Commons
  Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)"**.

The maintainers' own repo is the stronger authority, and **NC + ShareAlike** is the same constraint class the project
already flags for MuScriptor. **Resolve this against music.ai's own download terms before using MoisesDB for anything
published.** Until then, treat it as non-commercial research only.
