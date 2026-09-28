# CleanSplit UI — design direction

Researched 2026-09-17 from real products, not from what a model reaches for by default.

## What we are NOT building (the "AI slop" tells, documented)
Indigo/purple hero gradient · glassmorphism cards with neon glow · Inter in every weight · three feature cards in a row with an icon and two lines of text · hover bounce · emoji as icons · a landing page pretending to be an app. These are the listed fingerprints of AI-generated UI (SmoothUI, 925studios, dev.to "purple gradient problem"), and the fix is to pick a real direction and lock its tokens before writing any markup.

## What good products actually do (observed)
- **Linear** — near-black page floor, a four-step surface ladder, hairline borders instead of shadows, **one** chromatic accent used only on the brand mark, focus rings and one primary button per view. Hierarchy comes from luminance, not weight.
- **Raycast** — almost-black canvas, single warm accent, one typeface throughout, cards defined by hairlines and inset highlight strokes ("pressed key" feel), slight positive letter-spacing on body text.
- **Vercel / Geist** — the ink *is* the brand: a long neutral ramp, accent as punctuation, 4 px spacing scale, mono paired with sans for anything technical.
- **LALAL.AI** — near-black, one yellow accent, one card, one dropdown, one button. The whole product is a single decision plus a file.
- **RipX** — the differentiator is a *visual editor*: you see the sound and edit what you see.

## CleanSplit's direction: **measurement instrument**, not SaaS landing page
The product is a local lab bench for audio: it separates, then it *measures* and shows where the separation is unreliable. Nothing else on the market shows the residual and an artifact map. That is the thing the UI should make visible, the way a scope or an RX spectrogram does.

### Tokens (lock these; no ad-hoc values)
```
--bg          #0B0C0D   page floor (not pure black: waveform blacks must read against it)
--surface-1   #131517   panels
--surface-2   #1A1D1F   raised rows, inputs
--surface-3   #232729   hover / selected
--hairline    #2A2F33   1px borders carry all separation; no drop shadows
--text        #E8EAEC
--text-dim    #9AA1A6
--text-faint  #646B70
--accent      #E8A33D   amber: the VU-meter/tape needle colour. Brand mark, focus ring, ONE primary action per view.
--ok          #4FB477   measurement states only, never decoration
--warn        #E8A33D
--bad         #D6544A
--stem-vocals #E4B363   stem identity colours: warm→cool by frequency register, muted (never saturated)
--stem-drums  #C76F54
--stem-bass   #8C6FC7
--stem-guitar #5E9C8F
--stem-piano  #7C8CA8
--stem-other  #6B7280
radius: 4px (6px on panels). spacing: 4px scale. motion: 120ms ease-out, no bounce, no scale-up hovers.
```
Type: **IBM Plex Sans** for UI, **IBM Plex Mono** for every number (dB, timecode, Hz). Industrial, engineering-drawn, and deliberately not the default. Base 13px, tracking +0.1px on body. All measurements are mono and right-aligned so columns line up.

### Layout: a deck, not a dashboard
```
┌ rail ─┬ header: song title · 3:51 · 44.1 kHz · [Split ▾] [Analyze] ────────────┐
│ songs │ ── stem lanes (the main object) ─────────────────────────────────────── │
│       │  ▎vocals   [S][M] ▁▃▅▇▅▃▁ waveform, artifact marks under it     −7.2 dB │
│       │  ▎drums    [S][M] ▁▁▂▁▁▂▁                                     −34.9 dB │
│       │  … bass / guitar / piano / other                                        │
│       │ ── inspector (right, collapsible) ────────────────────────────────────── │
│       │   reconstruction 157 dB · residual −157 dB · 13 flagged regions          │
│       │   flagged list: 0:32.7 vocals 6.9–10.2 kHz  warble  0.70                 │
└───────┴ transport: ◀▶ play · 1:12 / 3:51 · [solo] [mute] · zoom ───────────────┘
```
Rules that follow from this:
1. **The stem lanes are the product.** They fill the window; everything else is chrome. No hero, no cards.
2. **Flagged regions are drawn on the audio**, as thin amber marks under the waveform at their exact time, and as a spectrogram overlay box when a lane is expanded. Clicking one seeks there and solos that stem. That is the feature no competitor has, so it gets the pixels.
3. **Numbers are first-class**: every lane shows its level; the inspector shows the measurements. Mono font, aligned, unit always present.
4. **One primary action per view** (amber). Everything else is a hairline ghost button.
5. **Keyboard first**: `space` play/pause · `1–6` solo stem · `alt+1–6` mute · `[` `]` prev/next flagged region · `S` split · `A` analyze.
6. **Progress is honest**: separation takes minutes, so the header shows the real stage ("SW pass 2 of 3 · 4:10 left"), never a fake spinner.
7. **Empty state** = a drop target and a file picker. One sentence. No marketing copy.

### Implementation plan (proposed)
- **Backend**: FastAPI wrapping the existing CLI functions; a job queue (one GPU job at a time), server-sent events for progress. Python is already the project's language, so no second runtime.
- **Frontend**: no build step, no node toolchain. Plain ES modules + vendored `wavesurfer.js` v7 (BSD-3) for waveforms/multitrack. Peaks are precomputed server-side so a 4-minute stereo float WAV never has to be shipped to the browser whole.
- **Serving**: `cleansplit ui` opens `http://127.0.0.1:8770` bound to localhost only. Local-first, no telemetry, no CDN.
- **Assets**: fonts vendored locally (IBM Plex, SIL OFL).
