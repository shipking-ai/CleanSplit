# CleanSplit: simple guide

CleanSplit splits a song into 6 parts (vocals, drums, bass, guitar, piano, other) on your own PC, then checks how good the split is.

## Before every session
Open **PowerShell** and go to the project folder:
```
cd C:\AI\CleanSplit
```

## 1. Check that everything works (once)
```
.venv\Scripts\cleansplit.exe doctor
```
You should see your graphics card, ffmpeg, and `sha256_verified` for the model. If so, you're ready.

## 2. Split a song
```
.venv\Scripts\cleansplit.exe separate "C:\Users\<user>\Music\My Song.wav" --out outputs\best
```
That is the best quality this project can produce — you do not have to ask for it. The defaults are the best
*measured* settings (`--separator ensemble`, `--overlap 4`, TTA on), not the fastest ones.

**If you are in a hurry:**
```
.venv\Scripts\cleansplit.exe separate "C:\Users\<user>\Music\My Song.wav" --separator bs_roformer_sw --no-tta --overlap 2 --out outputs\fast
```
- Put the song path in quotes. WAV, FLAC and MP3 all work.
- The 6 files land in `outputs\...\My_Song\stems\` (spaces in the name become `_`).
- Time for a 4-minute song: roughly **30 minutes** on the default best-quality settings, or about **2 minutes** on the
  fast ones. The gap in time is large and the gap in quality is small — the table says how small.
- Files are 32-bit float WAV. They never clip, and any DAW opens them.

| Option | What it does | Measured difference |
|---|---|---|
| *(defaults)* | `ensemble` + overlap 4 + TTA | the best of everything measured on 20 real multitracks |
| `--separator bs_roformer_sw` | one model instead of two | **-0.39 dB** vocals, worse on 18/20 songs; ~4x faster |
| `--no-tta` | one pass instead of 3 averaged | slightly worse on all four stems; 3x faster |
| `--overlap 2` | half the chunk overlap | -0.05..-0.09 dB, worse on 14-19/20 songs; 2x faster |
| `--separator ensemble_demucs` | also averages drums/bass with HTDemucs_ft | **worse** (-0.55 dB drums on 20/20 songs). Only worth trying if a drum or bass lane comes out empty |
| `--fp16` | half precision | saves VRAM, adds numerical noise to the residuals this tool measures |

Use a different `--out` folder per option so results don't overwrite each other.

## 3. Check the quality of a split
```
.venv\Scripts\cleansplit.exe analyze "C:\Users\<user>\Music\My Song.wav" --separator ensemble --out outputs\ensemble
```
It reuses stems it already made, so it won't split again. It writes:
- `reconstruction\residual.wav`: what's missing when the stems are added back together. The quieter, the better.
- `analysis\report.json`: a summary. `reconstruction_headline` gives the quality in dB (higher is better).
- `analysis\artifact_map.json`: moments that *might* sound warbly or leaky. These are hints only; listen before trusting them.

Already have stems from UVR? Analyze those instead:
```
.venv\Scripts\cleansplit.exe analyze "C:\Users\<user>\Music\My Song.wav" --stems-dir "C:\path\to\uvr\stems" --out outputs\uvr
```

## 4. Compare two audio files
```
.venv\Scripts\cleansplit.exe compare "outputs\sw\My_Song\stems\vocals.wav" "outputs\ensemble\My_Song\stems\vocals.wav"
```
Shows how different they are (dB numbers). Your ears are still the best judge.

## 5. "Restore": experimental, not recommended
```
.venv\Scripts\cleansplit.exe restore "C:\Users\<user>\Music\My Song.wav" --out outputs\sw
```
It runs, but tests showed it makes stems **less** accurate. Kept only for experiments.

## 6. Tests (optional)
```
.venv\Scripts\python.exe -m pytest -q cleansplit\tests
```
Everything should say `passed` (takes about 3–5 minutes).

## Good to know
- Everything runs offline on your PC. Nothing is uploaded.
- Quality results so far come from one test track: trust your ears over the numbers.
- Model licenses: the SW weights have no license (personal use only), and the A2SB files in `models\a2sb` are non-commercial.
- Full details: `README.md` and `docs\04_results.md`.
