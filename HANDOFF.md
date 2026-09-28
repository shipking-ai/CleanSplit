# CleanSplit — HANDOFF

**Updated after every code change.** Newest entry on top. If you are picking this up cold, read
[README.md](README.md) for what the project is, [QUICKSTART.md](QUICKSTART.md) for how to run it, and
[docs/04_results.md](docs/04_results.md) for what has actually been measured.

---

## 2026-09-28 (latest) — CodeQL suite narrowed, 18 alerts dismissed with reasons, and a real cache race fixed

**Security tab is now empty, and that is a claim about 18 written justifications, not about silence.**

`queries: security-and-quality` → **`security-extended`** in `.github/workflows/codeql.yml` (a3204ca). The broad
suite reported decisions this project has already recorded: `py/file-not-closed` on the probes is the exact pattern
`pyproject.toml` deliberately ignores `SIM115` for. Style is ruff's job and it runs on every push. That change alone
closed roughly ten quality alerts, leaving 18 `py/path-injection`.

**All 18 dismissed individually**, each with its own comment (`tools/`-free, done over the API; the script is not
tracked). Two different claims, so two different reasons:
- **15 × "false positive"** — `service.py` (12), `server.py` (3). CodeQL does not model `_stem_name()` or
  `is_relative_to()` as sanitisers, so it keeps tracing a flow that is guarded. Nine of the twelve sit **on the guard
  lines themselves** (`resolve()`, `is_relative_to()`, `is_file()`), which is the clearest evidence of the limitation.
- **3 × "won't fix"** — `server.py:41` (`create_job` takes the path of the file to separate: that IS the feature, on a
  127.0.0.1-bound local tool), `io.py:87` (`load_audio(path)`) and `region.py:113` (`ArtifactMap.load(path)`). A
  function whose sole parameter is the file to open cannot be fixed by validating it; the trust boundary is the
  caller's, and the UI caller does enforce one.

If a genuine alert ever appears it will now be the only thing in that tab. Re-open any of these with the API if the
reasoning stops holding.

**A flaky CI failure turned out to be a real bug in the shipped UI** (f9b307a). The ubuntu/py3.10 leg of run
36480845648 failed with `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)` from
`peaks()` and passed on rerun. The empty string it decoded was the point: `peaks` wrote ~58 kB of JSON with
`write_text`, which creates the file and then fills it, while the read side trusted `is_file()`. FastAPI serves those
handlers from a threadpool, so two requests for one song do overlap — this was never only a test problem. Writes now
go via a temp file plus `os.replace`; reads treat undecodable JSON as a cache miss and recompute.

Two things worth keeping from fixing it:
- **`os.replace` is atomic on Windows but fails outright if any other handle has the destination open** (WinError 5) —
  a concurrent reader, antivirus, or the search indexer. The first version of the fix broke on Windows, and
  `test_concurrent_peaks_requests_never_observe_a_partial_cache` (8 threads, cold cache) is what caught it. Suppressed
  now, which is safe because these peaks are a pure function of (audio, buckets), so losing the race costs nothing.
- **`test_a_half_written_peaks_cache_is_recomputed_rather_than_raising` reproduces the CI message verbatim** against
  the old read path. Verified by temporarily removing the fix, not assumed.

Also made the backslash traversal string in `test_ui.py` raw — it built the intended text only by accident, through
invalid escape sequences Python already warns about.

**Still outstanding:** the social preview card (`docs/assets/social-preview.png`) has to be uploaded by hand at
Settings → General → Social preview; GitHub exposes no API for it. Unreleased song titles remain in 9 tracked files
and in history (renaming is offered but needs a history rewrite on a now-public repo). 30 MUSDB test songs unrendered.

---

## 2026-09-28 (later) — public, and code scanning immediately found a real path-traversal bug

Repo is **public**: https://github.com/shipking-ai/CleanSplit. Pre-publication audit found no audio, weights or
dataset tracked, nothing over 2 MB anywhere in history, no credentials, and no vendored NVIDIA code. The maintainer's
Windows account name was scrubbed from 17 path strings first.

**CodeQL + zizmor SARIF now upload to the Security tab** (both needed a public repo). And CodeQL earned its place
immediately, contrary to what the workflow comment predicted:

**`cleansplit/ui/service.py` had a genuine path traversal.** `song_dir()` confines variant and slug with
`is_relative_to`, but `stem` went straight into `d / "stems" / f"{stem}.wav"` and the only check was `is_file()` --
which asks whether a file is there, never where. A stem of `../../../../secret` served a file **outside the output
root entirely**, and `peaks()` was worse: it WRITES a cache file at an attacker-chosen path. Reaching it needs a
literal slash to survive Starlette's routing, so it was latent rather than live -- but `service.py` is a plain Python
API and the fix does not belong in the router.

Fixed with a `_stem_name()` validator plus a resolved-containment check, and **demonstrated** rather than assumed: with
the guards removed, the original code serves `C:\...\Temp\<tmp>\secret.wav`; with them, it refuses.

**The existing traversal test was passing for the wrong reason**, which is the part worth remembering. It asserted
`FileNotFoundError` for `audio_path(..., "../../original")` and got one -- because no such file existed, not because
traversal was blocked. The new test plants a real file first. It also had to be corrected mid-write: `../../` from
`stems/` only reaches the variant folder, so **four** levels are needed to leave the output root, and the shallower
version passed against the vulnerable code too. Counted, not guessed.

**zizmor caught my own bad pin.** I pinned `github/codeql-action@7999b86 # v4` by reading `.object.sha` from the tag
ref -- but `v4` is an **annotated tag**, so that is the tag object, not the commit. Actions resolved it anyway, which
is why it ran green and I would not have noticed. Now `2892aa5`, and all six pins re-verified by dereferencing.

Two CodeQL `error` findings are false positives, checked individually: `py/non-iterable-in-for-loop` on
`conform.py:74` (iterating a 2-D NumPy boolean array, which CodeQL reads as a Python `bool`) and
`py/uninitialized-local-variable` in `test_separator_gpu.py` (CodeQL does not know `pytest.skip` never returns).

**Social preview card** at `docs/assets/social-preview.png`, generated by `make_readme_assets.py social` from the same
measured numbers as everything else. GitHub has no API for social previews, so it must be uploaded by hand at
Settings -> General -> Social preview. Rendered at 100 dpi deliberately: font sizes are points, so at 200 dpi a
46-point heading is ~160 px on a 640 px card and the first attempt collided with itself.

**Verified:** ruff clean, **91 passed, 10 deselected**; CI, CodeQL and zizmor all green.

---

## 2026-09-28 — TTA removed from the default; SCNet rejected; CI was testing less than it claimed

**Two queued verdicts landed, and both changed something.**

**1. SCNet XL rejected on all four stems (docs/04 §14.14).** Adding it to each stem of the ensemble at overlap 4 gave
-0.07 drums / -0.15 bass / -0.13 other / -0.06 vocals. The interesting part is that it **falsified docs/01 §8.2's
"comparable strength helps" rule on 3 of 4 stems**: drums, other and vocals all had gaps near 1 dB, which the rule
predicted would help, and all three hurt. Only bass (4 dB gap) behaved as written. Averaging pays for decorrelated
error, not comparable strength, and the rule never measured independence -- so it cannot be used to pick the next model
to try. Also the third confirmation that the full-band gate is mandatory: SCNet *improved* SAR on drums, bass and other
while making all three worse.

**2. TTA is out of the default recipe (docs/04 §14.15). `best` went from 16 forward passes to 8.**
`tools/eval/synth_arm.py` (new) prices an ensemble recipe from already-cached member arms with **no GPU** -- the
ensemble averages only vocals and `other` is the exact remainder, so `other = other_sw + (vocals_sw - vocals_ep317)/2`
and the mixture cancels. Validated, not assumed: the synthesised arm reproduces §14.13's independently rendered
`shipped` vocals median of 13.61 dB exactly.

TTA inside the shipped ensemble buys **+0.008 vocals / +0.015 drums / +0.033 bass** for 2x compute. Two of three gated
stems are under the +0.02 dB floor, so the gate fails. Mechanism confirmed as predicted: drums is +0.015 in both the
SW-solo and ensemble measurements (identical, because ensemble drums *is* the SW member), and vocals halves from +0.027
to +0.008 (averaging with a non-TTA member dilutes it).

**TTA never made a single song worse -- 0 regressions in 160 song-stem comparisons -- and was dropped anyway.** That is
the sharpest case for the floor the project has: reliability and magnitude are different questions. Third decision the
floor has changed (§14.6, §14.13, §14.15), all in the same direction.

The two pre-registered rules disagreed, which is reported rather than smoothed over: `cache_arm.py` asked for two of
three stems (SW solo passes), `synth_arm.py` asked for all three (the ensemble fails). Applied to the ensemble,
cache_arm's own two-of-three rule also fails. Neither was edited after the fact.

**New tier ladder, all re-measured from cache, no GPU:**

| tier | recipe | units | vocals vs fast | drums | bass |
|---|---|:--:|:--:|:--:|:--:|
| fast | SW, ov2 | 2 | — | — | — |
| balanced | mean(SW, ep317), ov2 | 4 | +0.442 · 18/20 | +0.000 · 0/20 | +0.000 · 0/20 |
| **best** | mean(SW, ep317), ov4 | **8** | **+0.460 · 18/20** | +0.095 · 18/20 | +0.061 · 16/20 |
| `--tta` | + 3-pass TTA on SW | 16 | +0.464 · 18/20 | +0.106 | +0.099 |

**CI was green-washing three separate things, all now fixed.** It had never actually passed; the first real look at the
logs showed:
- **The Python matrix was a lie.** No `python-version` anywhere, so `setup-uv` provisioned nothing and both `py3.10`
  legs ran 3.12 (3.12.10 Windows, 3.12.3 Ubuntu). The 3.10 floor had never been tested.
- **Both Ubuntu legs ran zero tests.** `uv pip install --system` hits PEP 668 on the image Python: *"The interpreter at
  /usr is externally managed"*. Now `uv venv` + `uv run`.
- **26 ruff violations**, because the ruff config was added without ever running it. Fixed 14 properly (including a dead
  `total = None` in `ui/service.py`); `RUF001` and two `E70x` are now ignored **with reasons** -- the UI deliberately
  writes real typography, and the asset generator is dense plotting code.
- **Two tests needing checkpoint FILES were unmarked**, so they failed on any machine without weights. They need no GPU,
  so a `gpu` mark would be wrong; they now skip themselves when the weights are absent, verified in both directions.

**`cleansplit/tests/test_docs_match_measurements.py` (new, 7 tests): the prose must match the committed JSON.** Cheapest
guard in the repo against its worst failure mode -- a stale number reads exactly like a fabricated one. It caught a real
problem on its first run: **`fullband_check.py` keys a verdict by arm but not by baseline and writes to one shared
path**, so a later ladder run against `fast` silently overwrote the TTA verdict against `ens_ov4`. Since
`ens_tta_ov4` *passes* against `fast`, the surviving file said PASS where §14.15 says FAIL. Nothing had been published
from it. The verdict now lives in its own `musdb18hq_tta_in_ensemble.json`, the test asserts both files, and `--out`'s
help records the hazard.

**Also:** `.github/workflows/zizmor.yml` (new) audits the workflows themselves -- pinned to a SHA,
`advanced-security: false` while the repo is private. CI now uses `permissions: {}` with per-job grants,
`persist-credentials: false`, pinned runner images (`ubuntu-24.04`, `windows-2025`) and `timeout-minutes`.

**zizmor's first run found 7 issues and all 7 were real:** five unpinned action references and two Dependabot blocks
with no cooldown. Every action is now pinned to a release SHA with the version in a trailing comment (checkout v7.0.1,
ruff-action v4.1.0, setup-uv v10.2.0, zizmor-action v0.6.4), and `cooldown` + `groups` are set. Pinning also cleared a
Node 20 deprecation warning -- worth recording that those majors were being **force-run on Node 24**, not failing, so
the claim that Node-20 actions are dead as of 2026-09-23 is wrong; they warn.

**Then the Ubuntu legs failed on their first genuinely-executing run**, with `A virtual environment already exists at:
.venv`: `setup-uv` creates `.venv` itself when it provisions an interpreter, so a bare `uv venv` aborts. Windows did not
hit it. Now `uv venv --clear`, and the step prints the interpreter so the matrix is checkable from the log rather than
trusted -- which is what hid the fake matrix in the first place.

**Verified:** ruff clean; **90 passed, 10 deselected** locally. The user-song render completed all three songs (the
earlier exit-code-1 was a trailing `tail` on a wrong log path, not the render).

**Open:** confirm CI green on this commit; two Dependabot PRs (`checkout@v7`, `setup-uv@v7`) were failing only because
of the ruff/test problems above and should be mergeable now; remaining 30 MUSDB test songs; repo still private.

---

## Current state (2026-09-28) — git: pushed to github.com/shipking-ai/CleanSplit (private)

| Area | State |
|---|---|
| Six-stem separation (BS-RoFormer SW, ep317, HTDemucs_ft, ensembles) | works, measured |
| Artifact analysis + artifact map | works, measured (docs/03) |
| Restoration + mixture-consistency gate | works; **every restorer tested is worse than doing nothing** (docs/04 §4, §5, §9) |
| Desktop UI (`cleansplit ui`) | works: lanes, solo/mute, A/B compare, inspector, flagged spots |
| Apollo codec restoration | **done and written up** (docs/04 §9). Integrated, measured, not enabled by default |
| MDX23C InstVoc HQ | **removed from the ensemble again** after MUSDB (worse on 16/20 real songs). Still available as `--separator mdx23c_instvoc_hq` |
| Real multitrack truth | **MUSDB18-HQ, 20 songs, done** (docs/04 §11). `ensemble` vocals **+0.45 dB paired** median vs SW, 18/20 (§14.8; the +0.39 dB in §11 was the unpaired form, i.e. understated). `ensemble_demucs` worse on a typical song, useful when SW misses an instrument |
| Artifacts vs bleed (SAR/SIR) | **measured, 20 songs** (docs/04 §14). Artifacts dominate: SIR is 10–11 dB cleaner than SAR everywhere. 4 reduction candidates **all rejected**; Wiener post-filtering fails on its own terms. **`--overlap 4` passes both gates** (§14.5) — first thing in this project to clear one. **Now the default, along with `ensemble` and TTA** (user: best quality wins over speed). **ov8 rejected on effect size, +0.01 dB for 2x the time** (§14.6) |
| Tests | **89** (79 non-GPU + 10 gpu-marked; +4 locking the paired statistic). 2026-09-28: 75 non-GPU passed in 82 s AND **all 10 GPU tests passed in 47 min** on the new best-quality defaults. `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest -q` |

Python: `C:\AI\CleanSplit\.venv\Scripts\python.exe` (uv-managed, **no pip** — use
`uv pip install --python .venv/Scripts/python.exe <pkg>`). Set `PYTHONIOENCODING=utf-8` for anything that prints Δ or dB.

### Verified after the last code change
`cleansplit restore "After 2.mp3" --restorer apollo:variant=vocal_restore` was re-run against the current code
(after `ApolloRestorer` learned to handle mixture-level regions) and produced **the same result**: 13 eligible,
0 accepted, all six stems bit-identical, rejection reasons `{"mixture error in region increased": 13,
"adds energy not explained": 1}`. docs/04 §9.3 is correct as written.

---

## The headline answer (for "where's the AI that fixes the stems?")

Two generative restorers are integrated and both lost:

| Restorer | Target | Result |
|---|---|---|
| A2SB (NVIDIA, non-commercial) | inpaint a flagged TF box | gate rejected all raw proposals; the consistent variant was accepted 67/69 and **61 of those were worse** vs truth |
| Apollo (CC BY-SA) | repair codec damage | **worse than doing nothing in 24/24 ground-truth measurements**; gate rejected 13/13 on a real song |

Root cause, measured in docs/04 §5: **~97% of the separator's error is misallocation** — energy that is present in
the mix but filed under the wrong stem. It cancels in the sum, so the mixture cannot see it, and no single-stem
generative prior can fix it. The thing that actually worked is ensembling — but on 20 real songs (docs/04 §11) the vocal gain is **+0.39 dB
median**, not the +1.8…+2.1 dB that one song suggested.

---

## Recently changed

### 2026-09-28 (13) — Repo reorganised, CI added, and CI immediately found a broken package

**CI's first run found that `cleansplit/models/` was never in the repository.** `.gitignore` line 8 was an unanchored
`models/`, written to exclude weight directories; it also matched `cleansplit/models/`, which is **source**. So
`__init__.py`, `checkpoints.py` and `device.py` had never been committed and **every clone produced a package that
could not import**. Invisible locally, because the files exist on disk. Pattern is now anchored (`/models/`,
`/data/models/`) with the reason recorded inline. Ten minutes of CI paid for itself.

**tools/ reorganised** from 26 flat scripts plus a `dev/` grab-bag into: `eval/` (reusable harnesses), `experiments/`
(pre-registered, decision rule in the docstring), `render/` (audio and figures), `probes/` (one-off), `queues/` (GPU
serialisation). `tools/README.md` explains the eval-vs-experiments split. All 18 moved scripts had `parents[1]` →
`parents[2]` and their `sys.path` insert repointed to `tools/eval`; 43 files had path references rewritten; one
relocated tool was run end to end to prove it.

**Two model directories merged.** 6.4 GB of A2SB weights sat in a root-level `models/` while everything else was in
`data/models/`. Now `data/models/a2sb`, with `_a2sb_ckpt_dir()` still honouring the legacy path so existing installs
keep working. Stray run logs → `outputs/_logs/`.

**Lint:** explicit ruff config (there was none). **B905 deliberately NOT ignored** — a silently truncating `zip` is the
same failure as the by-position pairing bug (§14.7), so every first-party `zip` now passes `strict=True`. One sed
flipped a docstring describing *upstream's* behaviour; caught and reverted.

**`.github/`:** CI (ruff + the non-GPU suite on Linux and Windows, py3.10/3.12, no torch and no weights — verified in a
torch-free venv first: 75 passed, 8 skipped), dependabot, issue templates including **"Challenge a measurement"**, a PR
template carrying the measurement bar, CODEOWNERS, SECURITY.md, CONTRIBUTING.md.

### 2026-09-28 (12) — Vocal recipe verdict, GitHub repo live, README rebuilt with generated assets

**VERDICT: `both_tta` REJECTED** (docs/04 §14.13). Giving ep317 its own TTA gains **+0.01 dB winning 16/20** — real and
consistent — for **+50% GPU time** on every render (the ensemble goes 4 passes to 6). Below the +0.02 dB floor, so it
does not ship. **Second time the floor has changed a decision, both times the same way: a consistent gain not worth its
cost.** Win counts measure reliability, not size. The control `ep317_tta_alone` is −0.28 dB (3/20), confirming the
averaging does the work, not the augmentation. TTA now has three independent confirmations as the worst-value lever.

**Repo is live: https://github.com/shipking-ai/CleanSplit — PRIVATE**, per the user's answer. NOTE: `gh` has two
accounts authenticated (`shipking-ai` active, `mattycigemp-crypto`); the active one was used. 189 files, 557 KB packed,
no weights and no audio (both gitignored).

**README rebuilt** after researching what well-regarded repos actually do (awesome-readme, demucs, uv). Devices adopted:
centred HTML header, `<picture>` with `prefers-color-scheme` so nothing looks broken in dark mode, one tight badge row,
quick-start above the fold, `<details>` collapsible for the cost ranking, GitHub `> [!IMPORTANT]` callout.

`tools/render/make_readme_assets.py` (new) generates every figure from measured data — nothing illustrative:
* `banner-{light,dark}.svg` — hand-written SVG, one mixture waveform fanning into four stem lanes.
* `tiers-{light,dark}.png` — from `musdb18hq_quality_tiers.json`.
* `spectrograms-{light,dark}.png` — from the rendered audio that was actually scored: mixture, vocals, drums, bass, and
  the **true-scale** error. The error panel is labelled "14 dB down" because on a 100 dB colour scale a 14 dB gap still
  looks bright, and an unlabelled panel would read as "the error is as big as the signal".

**Commits no longer carry Claude attribution footers** (user instruction, 2026-09-28; saved to memory).

### 2026-09-28 (11) — SCNet was registered but unreachable from the CLI, and the cost claim in --separator help was wrong

Two things found while preparing to re-render the user's own songs at the current best settings.

1. **`scnet_xl_ihf` was hardcoded out of the CLI.** It was vendored, registered, tested and measured, but
   `sep_opts` carried a literal `choices=[...]` list that never got it, so no user could name it. `--separator` choices
   now come from `separator_choices()`, derived from the registry, excluding only `stem_folder` (an evaluation helper
   that reads pre-separated stems from a directory). New test asserts registry and CLI choices agree, so registering a
   separator is now sufficient to expose it.
2. **The help text's cost claim was wrong.** It said the ensemble is "about 4 model passes, so roughly 4x slower than
   bs_roformer_sw". Those are not the same comparison: 4 passes per chunk is correct, but `bs_roformer_sw` also defaults
   to TTA, so at equal flags the ensemble is ~1.3x, and 8x only against the fully fast setting
   (`--separator bs_roformer_sw --no-tta --overlap 2`). Rewritten with §14.8's pass counts, and it now states §14.11's
   spread rather than implying a fixed cost.

**Source audio for the user's own songs is not in the repo**, but every old run saved its input: `outputs/*/original.wav`
— Concrete Crown 4.06 min, After 2 3.86 min, Too Much On My Plate 3.16 min, all 44.1 kHz float. ~11 min of audio,
roughly an hour of GPU at the best settings. Deliberately queued to run AFTER the vocal-recipe verdict (3 songs away)
so it is not rendered twice if `both_tta` is adopted. SCNet could still change drums/bass/other later; that re-render
cost is known and accepted.

Tests: CLI suite 5 passed.

### 2026-09-28 (10) — There is listenable audio again, and the medians were hiding a 5 dB spread

**User asked what they could hear. The honest answer was: nothing current.** The newest audio in `outputs/` was from
2026-09-17 — eleven days old, rendered with the OLD defaults. Every §14 result lived in `data/musdb_cache/*.npz` and was
never written out.

`tools/render/render_audio.py` (new) fixes that **with no GPU time**, because the caches already hold the separated audio. It
writes the *same arrays that were scored* — not a re-render — per song: `mixture`, `truth/`, `best/` (current default:
ensemble at overlap 4), `fast/` (single-pass SW at overlap 2, no TTA), `error_best/`, `error_fast/`, plus `_loud` copies
of each error normalised to −3 dBFS. 30 s excerpts, the fixed protocol midpoint. Default picks the worst, median and
best song by vocal SNR so both ends are audible. Written to `outputs/listen/` (gitignored, 879 MB for 3 songs).

**The finding this exposed — and it changes how the `--quality` switch must be described.** Per song, best vs fast on
vocals: Detsky Sad **+5.12 dB**, Ben Carrigan +0.69, Girls Under Glass +0.44 (and on that last one drums/bass/other are
*slightly negative*, −0.01/−0.07/−0.09). The headline median of +0.45 dB is honest but the spread runs roughly **−0.09 to
+5.12 dB**. A tier label cannot promise "0.45 dB worse"; the true claim is "the same on most material, up to several dB
worse on some". Written up as **docs/04 §14.11**.

### 2026-09-28 (10b) — The oracle selector's +0.38 dB is real but unreachable from the models' outputs

`tools/experiments/tile_combine.py` (new) tested whether any TRUTH-FREE per-TF-bin combiner captures part of the +0.38 dB ceiling
`oracle_headroom.py` measured. All parameter-free formulas, so nothing is fitted; all free at inference.

| combiner | ΔSNR | won | verdict |
|---|---|---|---|
| `min_mag` | **−0.376** | 1/20 | no |
| `mag_mean_phase` | +0.002 | 14/20 | no |
| `geo_mag` | −0.023 | 4/20 | no |
| `consensus_gate` | −0.007 | 2/20 | no |

**Prediction CONFIRMED** — all four rejected, `min_mag` worst as predicted. `mag_mean_phase` is the instructive
near-miss: wins 14/20 but by +0.002 dB, a hundredth of the floor. Consistent and worthless, which is what the floor is
for.

The sharp conclusion: all four use the only truth-free signal two models offer — their **disagreement** — which says
*how much* error a tile holds but not *which* model holds it. The average is already minimum-variance for two
uncorrelated equal-strength errors (§14.9), so disagreement-driven reweighting must lose. **The user's listening-AI idea
is not dead — its ceiling is +0.38 dB, a third of the project's biggest win — but every cheap route to it is now
measured and closed.** What remains is a trained selector (Sony MIMO's in-separator discriminator, docs/01 §8.4).
Written up as **docs/04 §14.10**.

### 2026-09-28 (9) — The free quality lever does not exist: equal weighting is exactly optimal (axis worth ≤0.013 dB)

`tools/experiments/ensemble_weight.py` (new). Every lever in §14.8 buys quality with compute; the vocal blend **weight** would buy
it with nothing, and 0.5 was picked because it is obvious, never because it measured best. Pre-registered with a
**held-out split** — fit on the first 10 songs, choose the weight there only, check on the untouched last 10 — because
tuning a weight on the same 20 songs this project reports against is fitting the test set.

**w\* = 0.50 exactly.** Every other weight on a 0.05 grid is worse; the curve is symmetric and flat (±0.05 costs
0.003–0.004 dB, splits songs 5/10); w=0 is −0.344 dB and w=1 is −0.417 dB. Prediction in the docstring (w\* in
[0.4, 0.6], no held-out gain) **CONFIRMED**.

**The number that closes the axis: the oracle per-song weight, reading the truth, is worth +0.013 dB.** So any
weighting scheme at all — fixed, per-song, adaptive — is capped at 0.013 dB, a seventh of doubling the overlap and a
thirtieth of adding ep317. Written up as **docs/04 §14.9**. No future weighting idea needs measuring.

Two free findings from the curve: the two models are of near-equal strength (which is *why* 0.5 is optimal — docs/01
§8.2's comparable-strength rule confirmed from a new direction), and **ep317 alone beats SW+TTA alone on vocals** by
~0.07 dB while both lose heavily to their average, so the averaging does the work, not either member.

Also corrected the understated headline number in the places that state it as a live claim (CLI `--separator` help,
`ensemble.py` docstring, §14.4's corroboration line, both defaults tables, `oracle_headroom.py`): **+0.45 dB paired**,
not +0.39 dB unpaired. §11's own narrative is left as written, with §14.8 explaining the difference.

### 2026-09-28 (8) — §11's headline number was UNDERSTATED, and ensembling is ~40x better value than TTA

Re-derived the four pre-registered MUSDB answers and the shipped recipe's headline claim paired, from
`outputs/_benchmarks/musdb18hq_test.json`, no GPU. `musdb_eval.py` already paired its *win counts* by song name (unlike
`artifact_reduction.py`), but its reported delta was still a difference of medians.

* **All four pre-registered answers (Q1–Q4) hold paired. No sign flips.**
* **The flagship claim was understated.** Shipped `ensemble` vocals vs single-pass SW: docs said +0.39 dB (unpaired);
  paired it is **+0.45 dB, 18/20**. Against SW+TTA it is **+0.42 dB, 18/20**.
* **Free partial answer on the pending TTA question:** `sw_tta` vs `sw` on vocals is **+0.04 dB, winning 20/20** — at
  overlap 2. Perfectly consistent, and tiny for 3x the compute.

**docs/04 §14.8 (new) ranks every lever by cost.** Compute is countable exactly rather than timed: `roformer.py` sets
`step = chunk // num_overlap`, so passes are linear in overlap; TTA is exactly 3 passes; each ensemble member is one.
The default is 16 units (SW+TTA@ov4 = 12, ep317@ov4 = 4).

| lever | compute | vocals (paired) | won | dB per doubling |
|---|---|---|---|---|
| **add ep317** | 1.33x | **+0.42** | 18/20 | **+1.01** |
| overlap 2 → 4 | 2x | +0.08 | 18/20 | +0.08 |
| TTA @ov2 | 3x | +0.04 | 20/20 | +0.025 |
| overlap 4 → 8 | 2x | +0.01 | 16/20 | +0.01 |

**Averaging a second architecturally-different model is ~40x more compute-efficient than TTA and ~12x more than
doubling overlap.** Qualified honestly in §14.8: ep317 helps vocals only, and the gains are not additive.

**Pre-registered prediction, written before the running experiment finishes:** if overlap and TTA cancel the same
uncorrelated error, TTA's +0.04 dB at overlap 2 must shrink at overlap 4 — predicted **below the +0.02 dB floor**.
Falsified if TTA gains ≥ +0.04 dB at overlap 4.

`tools/eval/paired_rescore.py` is the tool for all of this; it re-derives any benchmark JSON with a `song` field per row.

### 2026-09-28 (7) — It was never bss_eval's filter. It was the unpaired statistic, and I blamed the wrong thing twice

`tools/eval/paired_rescore.py` (new) re-derives any existing benchmark JSON as a paired statistic **with no GPU work** —
`artifact_reduction.py` and `overlap_experiment.py` both recorded a `song` field on every row, so the per-song pairing
was recoverable from files already on disk. Both of those tools decided their verdicts on `median(cand) - median(base)`
with **no win counts at all**, and between them they produced §14.2's four rejections and §14.5's overlap adoption.

Re-derived from the **same bss_eval SAR numbers**:

| candidate | stem | unpaired ΔSAR | paired ΔSAR | won |
|---|---|---|---|---|
| `+mdx23c` | vocals | **+0.22** | **−0.04** | 6/20 |
| overlap 8 | vocals | **+0.32** | **+0.10** | 19/20 |
| overlap 4 | vocals | +0.03 | +0.09 | 18/20 |

Both of the anomalies I attributed to bss_eval's 512-tap distortion filter — in §14.4 and again in §14.6, and twice to
the user in conversation — were the unpaired median instead. The +0.22 flips sign; the +0.32 becomes +0.10, which is
overlap 4's +0.09. **Once paired, bss_eval SAR and the filter-free full-band SNR agree on every case tested.** The two
metrics never disagreed; a bad statistic was making one look untrustworthy.

Written up as **docs/04 §14.7**, with §14.4's explanation explicitly withdrawn as the cause and §14.6 rewritten.

What holds: **all four §14.2 rejections stand** (paired, every candidate is negative on every stem; the Wiener variants
by 2.4–4.6 dB). No verdict in §14 flips. The full-band gate stays — it was right in both disputed cases — but its
justification is now "an independent filter-free check is worth having", not "bss_eval manufactured a result".

The honest lesson, recorded because it will recur: I offered three explanations across §14.3, §14.4 and §14.6 for
anomalies whose cause was one line of arithmetic in the comparison, inside the tool meant to be the safeguard. An error
in the statistic looks exactly like an error in the measurement and is far cheaper to check. Check it first.

### 2026-09-28 (6) — The mandatory gate itself was computing the wrong statistic. Fixed, and (5)'s verdict re-derived

**`tools/eval/fullband_check.py` — the filter-free gate every quality decision has had to clear since §14.4 — had two bugs,
and one of them had already changed a reported result.**

1. **Deltas were unpaired.** It printed `median(arm) - median(baseline)` while only its win counts were paired. This is
   the exact statistic banned in §14.5 after it invented a +0.32 dB gain for ep317 — and it was sitting in the gate.
2. **Songs were matched by list position**, not by name, so arms with different coverage were compared song-by-slot
   instead of song-by-song. The overlap-8 arm had 9 of 20 songs cached when it was first scored, so this was live, not
   hypothetical. These caches fill incrementally, so mismatched coverage is the normal case, not an edge case.

Both fixed: the statistic is now a pure `paired()` function matched by song name, with 4 new tests in
`cleansplit/tests/test_paired_statistics.py` — including one that fails on by-position pairing, and one where two arms
share a median while the candidate wins 2 of 3 songs. (Writing those killed a wrong premise of mine: a candidate cannot
win *every* song while the medians stay equal, because elementwise domination raises every order statistic.)

**Entry (5) below reached the right verdict for the wrong reason.** Rescored with all three arms complete at 20 songs,
overlap 8 is **not** "indistinguishable from overlap 4". It is consistently better — paired **+0.01 dB** on
vocals/drums/bass, winning 12–16 of 20 — and it **passed** the gate as written. So the gate was the problem: "any
paired gain > 0" adopts arbitrarily small gains at unbounded cost. It now carries a **+0.02 dB effect-size floor**,
taken from the rule already pre-registered in `tools/render/cache_arm.py` before this run rather than invented for this
verdict. The floor was added *after* seeing overlap 8 pass; that order is stated in the docstring and in §14.6, because
it is the reason the verdict stands. **Overlap 8 is rejected on effect size against cost, not for being identical.**

Same floor and paired form now in `tools/experiments/vocal_best_recipe.py` (its deltas were unpaired too — it is mid-run, so its
scoring pass gets re-run from cache) and `tools/experiments/stem_ensemble_experiment.py` (queued, so it picks this up before it runs).

Tests: **79 non-GPU passed.** GPU chain still serialized: vocal recipe → SCNet → TTA redundancy. The `--quality` switch
is still deliberately unwritten; two of its three numbers are still being measured.

### 2026-09-28 (5) — Overlap 8 REJECTED (gain saturates at 4), licensing resolved, TTA redundancy queued
User: *"do what's best for quality (and speed)"* — so these calls are mine, stated with the evidence.

**Overlap 8: not adopted.** *(Superseded by entry (6): the verdict holds, but "the gain saturates at 4" is wrong — the gate was computing an unpaired statistic. Overlap 8 is really +0.01 dB better and is rejected on effect size against cost.)*
On bss_eval it looked much better than 4 (vocals **+0.32** dB SAR vs +0.03). On the paired, filter-free, full-band gate
that §14.4 makes mandatory, the two are **indistinguishable**:

| arm | vocals | drums | bass | other |
|---|---|---|---|---|
| overlap 4 | +0.09 (18/20) | +0.05 (19/20) | +0.05 (14/20) | +0.07 (16/20) |
| overlap 8 | +0.09 (19/20) | +0.05 (18/20) | +0.05 (15/20) | +0.06 (17/20) |

Identical to two decimals, win counts differing by one song — noise. Overlap 8 costs ~half the throughput (0.13x vs
0.26x realtime) for nothing. **First time in this project that "best quality" and "faster" agreed.** docs/04 §14.6.
Second time bss_eval's filter allowance has manufactured a result, and the second time the unpaired median has; both
are now banned in favour of paired medians with win counts.

**Licensing resolved for a public release — Apollo KEPT, not deleted.** New top-level `LICENSE`: MIT for our code with
a plain statement of the non-MIT directories. I said last entry that deleting `third_party/apollo` was the clean fix;
on reflection that was wrong. A repo may carry third-party directories under their own licences when it is explicit and
the licence text travels with them, and CleanSplit *imports* Apollo rather than deriving from it. Deleting it would
remove the code behind a published negative result (§9, worse than doing nothing 24/24) and the two documented edits
mean a re-clone would not reproduce it. **Reproducibility of a negative result beats licence tidiness**; anyone who
cannot accept share-alike can delete the directory, and the LICENSE says so. Also fixed: `wavesurfer.esm.js` was
vendored with **no copyright notice**, which BSD-3 requires — its LICENSE now sits beside it. `pyproject.toml` updated.

**New question queued, and it is the biggest SPEED lever available: is TTA redundant with overlap 4?**
Both do the same thing — TTA averages 3 passes over transformed copies, overlap-add averages every sample over
`num_overlap` independently denoised chunks. Both cancel error that is uncorrelated between passes. Sizes are
suspiciously similar (TTA at overlap 2: +0.06/+0.02..0.05 dB; overlap 4 alone: +0.09/+0.05/+0.05) and overlap 8 added
nothing on top of 4, which is what saturation looks like. If they cancel the *same* error we are paying 3x for a
benefit already bought at 2x.
- `tools/render/cache_arm.py` — new, generic: fills one named cache arm with one named separator configuration, so any two
  configurations can be compared paired without a hand-rolled script that accidentally differs in more than one
  variable. Carries the pre-registered rule.
- **RULE: TTA stays only if, at overlap 4, it gives a paired median gain ≥ +0.02 dB AND wins on more than half of the
  20 songs, on at least two of vocals/drums/bass.** Otherwise it is dropped as redundant and the default separation
  gets **~3x faster at no measured cost.**
- Queued last (`tools/queues/tta_queue.sh`) behind the vocal-recipe and SCNet runs; one GPU job at a time.

GPU chain now: vocal recipe (ep317 TTA asymmetry) -> SCNet across 20 songs -> TTA redundancy. The `--quality` switch is
deliberately still unwritten: its tiers must carry real dB costs, and three of the numbers are still being measured.


### 2026-09-28 (4) — Answers acted on: committed, SCNet XL IHF integrated, licence audit
User answered the popup: download **SCNet XL + MelBand drums**; **"I do want to make this project public"**; time budget
**"1 and 3"** (whatever it takes AND give me a quality switch); **commit everything now**.

**1. Committed. `d1dfc6c`, 164 files, 36,964 lines, clean tree.** First commit in the repo's life; two weeks of work had
been untracked. `.gitignore` excludes weights (`*.ckpt/*.pth/*.onnx`), `data/` (42 GB) and `outputs/` (4.6 GB) but
**force-includes `outputs/_benchmarks/*.json`** (4.9 MB, 31 files) — those are the measured record that makes docs/04
checkable. Two things fixed while staging: a stale `main.py.tmp.8656.*` from 12 Sept (deleted), and
`third_party/diffusion-audio-restoration` was an **embedded git clone** that would have committed as a broken gitlink
(now ignored, with the pinned commit + clone command recorded in the new `third_party/README.md`).

**2. Licence audit for going public** (`third_party/README.md`, pyproject already declares MIT for our code):

| component | licence | public-release status |
|---|---|---|
| CleanSplit code | MIT | fine |
| `bs_roformer/model.py`, `mdx23c/model.py`, `scnet/` | MIT (ZFTurbo/MSST) | fine, MIT-compatible |
| **`third_party/apollo/`** | **CC BY-SA 4.0** | **share-alike clash with an MIT repo — needs a decision** |
| `ui/static/vendor/wavesurfer.esm.js` | BSD-3 per docs/05 | **no copyright header in the vendored file — must add one** |
| all model weights | various / none | not redistributed, correct |
Apollo lost every measurement (§9) and is off by default, so deleting it in favour of a fetch script (as A2SB already
is) is the clean fix. **Not done unilaterally — it removes working code behind a published result.**

**3. SCNet XL IHF integrated** — the four-stem CONVOLUTIONAL partner for drums/bass/other, which have never had one.
- Downloaded from MSST release v1.0.15 (214 MB, sha256 `ac25975f…b74f`), config + weights in `data/models/`.
  **MIT-licensed weights — the only separator here whose weights carry an explicit permissive licence.**
- `cleansplit/separation/scnet/` vendored byte-for-byte from MSST @ `a8a86223…` (MIT). Upstream's SCNet passes no
  window to `torch.stft/istft`; that raises a UserWarning and is **left alone on purpose** — "fixing" it would deviate
  from the numerics behind the author's reported SDR.
- `scnet_sep.py` mirrors `roformer.py`/`mdx23c_sep.py` overlap-add exactly, so the separators differ only in the model.
- `checkpoints.py` gained `SCNET_XL_IHF` and `cleansplit_models_dir()` (`CLEANSPLIT_MODELS` or `data/models`), searched
  before the UVR trees — our downloads stay out of someone else's install.
- Smoke test on 15 s: **2.1 GB peak VRAM** (comfortable on 8 GB), ~0.5x realtime, loads `strict=True`.
- `tools/experiments/stem_ensemble_experiment.py` — **pre-registered** per stem: adopt `mean(SW+TTA, SCNet)` only if paired median
  full-band SNR improves AND it wins >half the songs AND median SAR does not drop. It also records a **falsifiable
  prediction**: SCNet's drums are ~0.6 dB below SW's, inside the 1-2 dB band, so the comparable-strength rule predicts
  it HELPS on drums — and the script prints `** RULE CONTRADICTED **` if a within-2 dB member hurts instead.
- Queued behind the overlap/vocal-recipe work (`tools/queues/scnet_queue.sh`); one GPU job at a time.
- **85 tests**, 75 non-GPU passing. New `test_scnet_config_and_checkpoint_are_the_ones_measured` pins the sha256 and the
  four source names, because a silently swapped checkpoint would look like a working ensemble while actually averaging
  two band-split transformers — the thing already measured not to work.

Still owed from the answers: the `--quality` switch (tiers can only be written once the SCNet and overlap-8 verdicts
are in, so the printed dB costs are real), and MelBand drums (MSST lists **no** drums-dedicated mel_band model; the
MVSep ensemble's MelBand member is a 4-stem model that still needs identifying).


### 2026-09-28 (3) — Research pass: what would actually raise the ceiling (docs/01 section 8)
Sources: MVSep Multisong leaderboard (100+ commercial tracks, independent of our 20 MUSDB songs) and
arXiv:2609.07226 (Sony AI, MIT code + released weights). Full write-up in docs/01 section 8. Headlines:

1. **Biggest structural gap: only VOCALS are ensembled.** Drums, bass, guitar and piano come from SW alone, so three
   quarters of the output gets none of the +1.12 dB SAR that averaging buys. MVSep's drums board has
   **"Drums Ensemble (MelBand + SCNet XL + BS Roformer SW)" as the best entry on the whole board (14.3505)**, above
   MVSep's own proprietary ensemble and above plain "BS Roformer SW (6 stems)" (14.1129) — **+0.24 dB SDR** from
   exactly the recipe we are missing. Needs SCNet XL + MelBand drums downloads.
2. **Our "comparable strength" rule is confirmed externally.** MVSep vocals: "BS RoFormer (11.89 + 12.33)" ensembles
   to 12.2597, i.e. **worse than its own better member (12.3339)**. Same effect as our three rejected third models.
3. **The user's listen-and-fix idea works — inside the separator, not after it.** Sony's MIMO framework iterates while
   holding mixture consistency (our own invariant; they frame it as fixed-point iteration) and adds a **stem
   discriminator** whose only job is to judge whether a stem sounds real. Vocals: SDR 10.38 -> **11.05**, SAR 11.74 ->
   12.14, and the **generative** variant gets the best SAR (12.16). Most of the gain is the discriminator. Consistent
   with our 24/24 failure of post-hoc repair: right idea, wrong place. **Requires training, not a config change.**
4. **Lossy sources put a hard ceiling on everything:** re-encoding the true stems caps vocals SDR at 37.7 dB (MP3 320)
   and **20.1 dB (MP3 128)**. Models sit at ~12 dB so 320 is not binding, but it confirms docs/04 section 9 on the
   user's own MP3-sourced file.

Ranked: (1) ensemble the other stems, (2) add/swap a top-20 downloadable vocal model, (3) MIMO+discriminator is a
training project, (4) source quality gates all of it. Items 1 and 2 need downloads -> asked the user.


### 2026-09-28 (2) — Inventory of every model already on disk; only one was still untested
User wants the absolute best split, so: what else is already installed that could join the vocal average (the one
mechanism in this project that measurably works, +1.12 dB SAR)?

`C:\Users\<user>\AppData\Local\Programs\Ultimate Vocal Remover\models` holds:

| file | size | status |
|---|---|---|
| `BS-Rofo-SW-Fixed.ckpt` | 667 MB | in use (SW) |
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | 610 MB | in use (2nd vocal model) |
| `MDX23C-8KFFT-InstVoc_HQ.ckpt` | 427 MB | tested, **rejected** (docs/04 sections 11, 14.3, 14.4) |
| `Demucs_Models/v3_v4_repo/*.th` (5) | — | tested as htdemucs_ft, **worse** (sections 11, 14.1) |
| `Apollo_Models/Vocal_Restore.ckpt` | — | tested, **worse in 24/24** (section 9) |
| `UVR-MDX-NET-Inst_HQ_3.onnx` | 64 MB | **the only untested one** |
| `VR_Models/1_HP-UVR.pth`, `UVR-DeNoise-Lite.pth` | 121 / 17 MB | old VR architecture, not wired up |

**Getting MDX-Net Inst HQ 3 runnable took real source-hunting, and that matters for trusting the result.** UVR keys a
model's inference parameters by the md5 of its last 10 kB (`ad1501a5b998eb4b37c8ab81b1403305`). That key is in
**neither** the local `model_data.json` (86 entries) **nor** TRvlvr/application_data's `mdx_model_data`. Guessing the
parameters would have produced garbage that looked exactly like "this model does not help" — the single failure mode
this project exists to avoid. Resolved from primary sources instead:
- `dim_f=3072, dim_t=256` — **read out of the ONNX file itself** (graph input shape `[batch, 4, 3072, 256]`, via the
  `onnx` package; `onnxruntime` installed to run it).
- `hop=1024`, `chunk = hop*(dim_t-1)`, `trim = n_fft//2` — **UVR's own `separate.py`** on GitHub, the code that ships
  these models. (UVR is installed here as a frozen exe, so the local copy has no readable source.)
- `n_fft=6144` — **inferred**: the only value where `dim_f == n_fft/2`, the convention for this family.
- `compensate=1.0` — **inferred**, and principled rather than arbitrary: it is the value that keeps
  `vocals = mixture - instrumental` exactly mixture-consistent, which every CleanSplit recipe requires anyway.

Because two parameters are inferred, `tools/probes/mdxnet_onnx_probe.py` **self-checks before reporting anything**: it
scores the model's own instrumental against the true instrumental and refuses to say anything about the model's quality
if that is below 6 dB. **It came out at 15.27 dB, so the plumbing is right.**

**ANSWER, 20 songs: it does NOT help, and the on-disk avenue is now CLOSED.** Self-check 16.30 dB, so the
implementation is sound and the result is about the model, not the plumbing.

| vocal estimate | median SNR |
|---|---|
| MDX-Net Inst HQ 3 alone | 10.42 dB |
| shipped pair `mean(sw_tta, ep317)` | **13.16 dB** |
| pair + MDX-Net averaged in | 12.82 dB (**-0.34 dB, better on only 1/20 songs**) |

Same shape as the MDX23C result and for the same reason: a model **2.7 dB weaker** than the pair drags the average
down. Averaging only cancels artifacts when the members are of comparable strength — that is the real lesson from
three rejected third models (MDX23C, Demucs, MDX-Net), and it is a *prediction* for any future candidate: a model more
than ~1-2 dB below the pair will hurt, so check its solo score before wiring it into the average. A stronger third
vocal model would have to be downloaded, which needs the user's go-ahead. `VR_Models/1_HP-UVR.pth` is an older, weaker
architecture still and was not wired up on the same reasoning.

New deps: `onnx`, `onnxruntime` (CPU; the probe runs off the GPU on purpose, so it does not contend with the overlap
and vocal-recipe runs).

**Operational note worth keeping: do not run two GPU jobs at once on this machine.** The overlap-8 arm and the GPU test
suite together pinned VRAM at 7.8 of 8 GB and 100% utilisation, and the overlap arm fell from 0.13x to **0.06x**
realtime — more than half the throughput lost to thrashing. `tools/queues/gpu_queue.sh` now serialises the remaining work
(GPU tests -> overlap 8 -> vocal recipe) and echoes each stage's result. Resume commands are inside it.


### 2026-09-28 — Defaults are now the BEST MEASURED setting, not the fastest (user decision)
User: *"Whatever split mode brings the BEST quality, should be the default."* That overrules my §14.5 call to leave
overlap at 2, and it exposed something bigger than the overlap question.

| default | was | now | measured gain | cost |
|---|---|---|---|---|
| `--separator` | `bs_roformer_sw` | **`ensemble`** | **+1.12 dB vocal SAR**, +0.45 dB paired SNR, 18/20 songs | ~4 model passes |
| `--overlap` | 2 | **4** | +0.03…+0.09 dB SAR, 14–19/20 songs | ~2× |
| `--tta` | off | **on**, `--no-tta` to disable | better on all four stems | 3× on the SW pass |

**The separator default was the real find.** Every number in docs/04 §11 and §14 says `ensemble` is the best this
project can make, and the CLI had defaulted to single-pass SW with no TTA — the *fastest* option — since before any of
it was measured. Two other places silently pinned the fast path as well:
- `cleansplit midi` built its own `argparse.Namespace` with `"overlap": 2, "fp16": False, "tta": False`, so the MIDI
  command always separated at the old fast settings no matter what the user asked for. Now `overlap 4, tta True`.
- `EnsembleSeparator` had **no `num_overlap` parameter at all**, so `--overlap` was silently dropped for the default
  recipe (found in the previous entry, fixed there).

Library defaults moved too, so the UI and any direct caller get the same: `BSRoformerSeparator.num_overlap` 2 → 4,
`EnsembleSeparator.num_overlap` default 4. UI mode list rewritten (`cleansplit/ui/service.py`): Ensemble is first and
labelled "best measured quality", the quoted speeds doubled to match overlap 4 (~7× song length), `ensemble_demucs`
now says "measurably worse on drums and bass", SW says "lowest quality".

**Deliberately NOT changed, with reasons:**
- `evaluate` stays pinned to `bs_roformer_sw` — the experiment protocols behind docs/04 §4, §5 and §12 were run with
  it, and moving that default would silently change what an already-published number means.
- `ensemble_demucs` stays opt-in (measurably worse on both axes), MDX23C stays out (§14.3/§14.4), `--fp16` stays off.
- **TTA is NOT switched on for the ep317 member inside the ensemble.** It would probably help and it is cheap to
  test, but it is an unmeasured change to a measured recipe. Listed as open work, not guessed at.

Tests: **84**. New `test_defaults_are_the_best_measured_setting_not_the_fastest` asserts the defaults for
`separate`/`analyze`/`midi`, that `evaluate` stays pinned, **and** that `--no-tta --overlap 2 --separator
bs_roformer_sw` still works — so this cannot quietly regress to the fast path. 73 non-GPU passed after the change.

Docs: docs/04 §14.5 decision paragraph rewritten (marked as a user revision of my call), QUICKSTART §2 rewritten so
the plain command is the best one and the fast flags are the opt-out, README example de-flagged.


### 2026-09-22 — Measuring the thing the project is actually for: ARTIFACTS
The user's reminder: "the whole point of this is to split with no artifacts". Every number until now was SDR, which
mixes two different faults. The BSS decomposition separates them: **SIR = bleed**, **SAR = artifacts** (energy in the
estimate that belongs to no source — warble, smearing, musical noise).

**Baseline, locked (`tools/eval/artifact_metrics.py --limit 20`, MUSDB18-HQ 20 songs x 30 s, 16 kHz, median dB):**

| candidate | stem | SDR | SIR (bleed) | SAR (artifacts) |
|---|---|---|---|---|
| sw_tta | vocals | 12.65 | 23.89 | 13.00 |
| sw_tta | drums | 13.52 | 24.04 | 14.06 |
| sw_tta | bass | 10.98 | 20.20 | 12.40 |
| sw_tta | other | 8.43 | 16.36 | 9.60 |
| **ensemble** (shipped) | vocals | **13.62** | 24.39 | **14.12** |
| ensemble | other | 8.36 | 15.76 | 9.67 |
| ensemble_demucs | drums | 12.56 | 23.11 | 12.98 |
| ensemble_demucs | bass | 10.48 | 18.35 | 12.36 |
| demucs | vocals | 9.90 | 19.11 | 10.38 |

Three things this settles:
1. **Bleed is not the problem. Artifacts are.** SIR sits 10–11 dB above SAR on every stem of every model, and
   SDR tracks SAR to within ~0.5 dB. What is left to hear is damage the model does *inside* the stem.
2. **Averaging models is an artifact remover, not just an SDR trick.** The shipped vocal ensemble buys
   **+1.12 dB SAR** over sw_tta alone (13.00 → 14.12) — the largest single gain measured in this project.
3. **`ensemble_demucs` is confirmed harmful on the artifact axis too** (drums −1.08 dB SAR, bass −0.04), independently
   of the SDR verdict in docs/04 §11. It stays opt-in insurance for songs where SW misses an instrument outright.
**Artifact reduction: four candidates, all REJECTED** (`tools/experiments/artifact_reduction.py --limit 20`, pre-registered bar:
median SAR up on vocals AND drums AND bass, median SDR down <= 0.10 dB on each). Median dSAR vs the shipped ensemble:

| candidate | vocals | drums | bass | verdict |
|---|---|---|---|---|
| `+mdx23c` | +0.22 | 0.00 | 0.00 | no — only touches vocals, so the rule is vacuously false; see below |
| `wiener1` | −2.91 | −4.29 | −3.14 | no |
| `wiener2` | −3.07 | −4.74 | −3.59 | no |
| `wiener2_smooth` | −3.60 | −5.53 | −4.29 | no |

Wiener post-filtering — the textbook artifact reducer (Open-Unmix/norbert, `demucs --wiener`) — **fails on its own
terms**, costing 3–5.5 dB of the SAR it exists to protect. The reason is specific to CleanSplit: the shipped stems are
**already mixture-consistent by construction** (`other` is the exact remainder), so Wiener has no inconsistency to
remove and can only swap the models' phase-accurate output for a coarse 4096/1024 magnitude mask whose own error is
a fresh artifact. Genuine negative result; nothing in that family was tuned after seeing it.

**`+mdx23c` looked like it disagreed with docs/04 §11 (which removed MDX23C). It doesn't — and my explanation for the
disagreement was wrong.** `tools/probes/mdx23c_band_probe.py` (new) pre-registered the hypothesis *"MDX23C helps below
8 kHz and hurts above"* with an explicit falsification condition. Full-rate 44.1 kHz per-band SNR, 20 songs:
**−0.08 dB below, −0.03 dB above, 4/20 songs improved. Same sign in both bands → FALSIFIED.** MDX23C hurts
everywhere; §11's removal decision is confirmed by an independent measurement. The §14 gain came from `bss_eval`
fitting an optimal 512-tap distortion filter before scoring, which absorbs EQ-shaped error into the target term.
**Consequence, added after the falsification (it tightens the bar, it does not loosen it): a SAR-only gain is not
evidence of a cleaner stem — any future candidate must also pass a filter-free full-band check before it ships.**
Written up as docs/04 §14.4.
- `tools/experiments/overlap_experiment.py` — new, **pre-registered, same bar**. SW+TTA alone (so the overlap factor is the only
  thing that changes between arms) at `num_overlap` 2 (shipped) vs 4 vs 8. Rationale: chunked inference denoises each
  chunk independently, so the seams and each chunk's own invented energy are uncorrelated — more overlap averages more
  independent passes, the same mechanism that makes model ensembling cancel artifacts. Cache is per overlap
  (`data/musdb_cache/sw_tta_ov<N>/`), so it is resumable. Later change: `--out` plus a per-arm "N songs scored" line,
  so a complete arm can be scored in one process while another is still filling its cache — an arm with a partial
  cache is scored over the songs it has and the count says so.
- `tools/eval/fullband_check.py` — new. The filter-free full-band gate §14.4 demands of every SAR candidate: median
  `snr_db` at the native 44.1 kHz, all four stems, no downsampling and no filter allowance (the §3/§6/§7/§11 metric).
  Gate: median must improve on vocals, drums AND bass; `other` is reported, not gated (it is the remainder, so it
  absorbs the other three's mistakes). **Written and run before the overlap SAR numbers were visible**, so it cannot
  have been shaped by them.

**OVERLAP 4 PASSES BOTH GATES — the first candidate in the whole artifact/restoration line of work to do so.**
docs/04 §14.5. 20 songs, SW+TTA alone, vs the shipped `num_overlap=2`:

| | vocals | drums | bass | other |
|---|---|---|---|---|
| ΔSAR (artifacts) | +0.03 | +0.08 | +0.03 | +0.09 |
| ΔSDR | +0.03 | +0.07 | +0.08 | +0.09 |
| **ΔSNR full-band, filter-free** | **+0.09** | **+0.05** | **+0.05** | +0.07 |
| songs improved, full band | **18/20** | **19/20** | 14/20 | 16/20 |

Effect size is **inaudible on its own** (sub-0.1 dB). What makes it real is consistency — 18/20 and 19/20 is not
noise — and that both metrics agree in sign, unlike every rejected candidate. Cost: **~2x GPU time**.

**Decision: `--overlap` exposed and documented, default stays 2.** The pre-registered rule said a passing overlap
"replaces the shipped 2"; it earned that on quality. The same rule said the speed/quality trade is the user's, so the
deviation is recorded in docs/04 §14.5 rather than buried: doubling every render for an inaudible gain is a poor
default, and withholding a change is the conservative direction, not the flattering one.

**Bug found and fixed while wiring the lever up:** `--overlap` was **silently ignored for `--separator ensemble`**,
the default recipe — `_make_separator` built the ensembles with `device` only, and `EnsembleSeparator` had no
`num_overlap` parameter at all. So the one lever that passed both gates could not reach the code path that ships.
`EnsembleSeparator(num_overlap=...)` now forwards to both RoFormer members (HTDemucs_ft is unaffected, it has its own
float overlap), the registry factory and CLI pass it through, `--overlap`'s help text carries the measured numbers,
and `test_overlap_reaches_both_roformer_members_of_the_ensemble_and_the_cache_key` (no GPU, constructors only) covers
the plumbing **and** the cache key, so a re-run at another overlap cannot silently reuse cached stems. **83 tests.**

`num_overlap=8` still measuring (`outputs/overlap_experiment.log`, ~50 min of separation left, then ~25 min scoring).
docs/02 has it ~2.4x slower again than 4, so it needs a much larger gain to change the recommendation.

### 2026-09-21 (7) — First real song through the MIDI converter: works, but `other` is garbage
`cleansplit midi "After 2.mp3"` (reused the `ensemble` split), 1943 s total, MuScriptor peak 1713 MB:
vocals 675 notes, piano (Transkun) 494, guitar 4932, **bass 51** (808s mostly missed), **drums 0** (the split's
drum lane is nearly empty, the known SW failure on this song), **other 28,871 — 16k "French horn"**, 1434 s of the
run. Tempo: beat_this refused ("no fixed tempo, 196 ms RMS from 80.1 BPM") → wrote the 120 BPM default.
Changes so far:
- `_muscriptor_worker.py` prints `[progress] item/items chunk/chunks` every 5 chunks; `muscriptor_backend.py` now
  streams the worker's stderr (Popen + reader thread) into a `progress(i, n, c, t, label)` callback, keeping the
  last 15 lines for errors; `pipeline.py` logs "MIDI: other (4/5) 40%". Previously a 30-minute run was silent.
- Tempo fallback in the worker: when beat_this finds no constant tempo, use the **median beat interval**
  (40-240 BPM range) and mark `grid.fallback`; the report caveat says so. 120 BPM was certainly wrong.
- Probe (`tools/probes/midi_after2_probe.py`, 60 s, `outputs/midi_probe.log`): `ensemble` split → drums 0, bass 5,
  other 4175 notes (70/s); `ensemble_demucs` split → drums 25, bass 117, other 564 (all piano, 9.4/s), 4× faster.
  Constraining `other` away from drums made it worse (117/s): the garbage is the *content* — SW missed the drums
  and 808s on this song, so their energy landed in the `other` remainder. The two bass stems differ by only −20 dB
  yet gave 5 vs 117 notes → near-sine 808 bass is a weak spot of MuScriptor (now a report caveat).
- **Density guard** (`pipeline.py`, `SUSPECT_NOTES_PER_S = 40`): threshold from BabySlakh's exact MIDI (densest
  real stem averaged 30.7 notes/s). A stem above it keeps its own `.mid` but is left out of the combined file and
  named in the report. Checked against the benchmark: never fires on the shipped route C2 (0.332 → 0.332), fires once
  on plain C (a 104 notes/s piano) and lifts it 0.274 → 0.298. +1 test; transcription tests 7/7.
- **Q3 (pre-registered in `transcription_eval.py` before running)**: scale every stem sent to MuScriptor to
  −20 dBFS RMS (MuScriptor does not normalise input; After 2's Demucs drum stem sits at −48 dBFS). Adopt if pooled
  multi-F1 improves AND > 10/20 tracks. **Result: NO** (+0.001, 10/20). Not adopted. Caveat: BabySlakh stems are at
  normal levels, so very quiet stems are barely tested. Written into docs/04 §13 with the After 2 findings.
- `outputs/ensemble/After_2/midi/After_2_clean.mid` — the combined file rebuilt without the garbage `other` track
  (made by hand from the existing file; the guard now does this automatically). Sent to the user with vocals/piano/guitar.
- **After 2 v2** (`--separator ensemble_demucs`, `outputs/ensemble_demucs/After_2/midi/`): 788 s (was 1943 s),
  **78.9 BPM** from the median-beat fallback (was a 120 placeholder), drums **175** (was 0), bass **454** (was 51),
  other 5,708 (24.7/s, under the guard; was 28,871), guitar 4,532, vocals 699, piano 498. Guard did not fire.
  Drums still look under-counted for hip-hop (0.76 hits/s). Sent to the user.

### 2026-09-21 (6) — MIDI eval done; defaults set from it
- **UI MIDI button**: `service.submit_midi(variant, slug)` + worker branch for `kind == "midi"`;
  `POST /api/midi`; songs carry a `midi` flag; `/api/reveal` accepts `sub: "midi"`. Header button reads
  "MIDI" → "MIDI…" while running → "Open MIDI" (reveals `midi/`); key **M**. `test_ui.py` +1 test (404 for unknown
  song, job runs the pipeline, flag false when no .mid). 8/8 UI tests pass.
- **Result** (docs/04 §13, `outputs/_benchmarks/transcription_babyslakh.json`): pooled multi-F1 full mix 0.226 →
  stems 0.274 → **stems + Transkun piano 0.332** (oracle stems 0.334). Q1 stems beat mix **YES** (+0.048, 13/20);
  Q2 Transkun on piano **YES** (+0.186 piano-F1). Combined vs full mix: better on 16/20, median 0.183 → 0.316.
- `cleansplit/cli/main.py`, `cleansplit/transcription/pipeline.py` — defaults now `--mode stems --piano transkun`.
- `cleansplit/tests/test_transcription.py` — existing pipeline test pins `piano="muscriptor"`; new test checks the
  default sends piano to Transkun (mocked), not MuScriptor, and that Transkun velocities reach the MIDI. 6/6 pass.
- Caveats in §13: 16 kHz synthetic audio (biased against stems), no singing in Slakh (vocal route untested),
  MuScriptor may have seen Slakh.

### 2026-09-21 (5) — Disagreement flag (held-out test running) + MIDI converter research
**Disagreement flag.** agreement = 10·log10(|d|²/|s−d|²) between SW+TTA's and HTDemucs_ft's estimate of a stem —
needs no truth. On MUSDB songs 1–20: Spearman with SW's true SDR **0.79** (80 song-stems); all 8 SW failures
(SDR < 3 dB) rank in the 17 lowest. Threshold **6 dB** chosen on those 20 (catches 7/8, flags 13/80).
- `tools/eval/disagreement_eval.py` — new. **Pre-registered** pass criteria in the docstring (recall ≥ 0.5, precision
  ≥ 0.4, Spearman ≥ 0.5) for the **30 held-out songs 21–50**, written before they were separated. Running in the
  background → `outputs/disagreement_heldout.log`.
- **Held-out result: FAIL** (docs/04 §12). Spearman 0.827 and recall 4/4 pass, **precision 0.21 < 0.4** fails. Not
  shipped as a warning; threshold deliberately not re-tuned on the held-out songs. The ranking is real; showing the
  raw agreement number without a threshold is a possible later option.

**MIDI converter — built so far (2026-09-21):**
- User accepted the MuScriptor licence and logged in to HuggingFace themselves (`hf auth whoami` → MindCoDev);
  medium and large both return 200/206. Token lives in `~/.cache/huggingface/token`; never print it.
- `.venv-transcribe/` — separate uv env: `muscriptor==0.3.0` + torch 2.11 cu128 (reused from uv cache). In
  `.gitignore`. Kept apart because MuScriptor pins fastapi/numpy differently and its weights are non-commercial.
- `cleansplit/transcription/` — new package:
  - `notes.py` — `TranscribedNote` (onset, offset, pitch, instrument group, is_drum, velocity|None, source) and
    `write_midi` (one track per stem+instrument; GM program map; fixed velocity 90 where the backend has none).
  - `_muscriptor_worker.py` — runs *inside* `.venv-transcribe`, JSON in/out, imports nothing from CleanSplit.
  - `muscriptor_backend.py` — `transcribe_many(items, model, dtype)`; per-stem hard instrument constraints
    (`STEM_INSTRUMENTS`: vocals→voice, drums→drums, bass→basses, piano→pianos, guitar→guitars, other→free).
  - `transkun_backend.py` — `TranskunPiano`: SHA-256 pinned (`50a80010…d44c`), `weights_only=True`,
    **strict** load (upstream CLI uses unsafe load + `strict=False`).
- Main venv: `transkun==2.0.1`, `pretty_midi`, `mir_eval` installed.
- Smoke tests: MuScriptor medium on 15 s of After 2 vocals → 55 voice notes, 12.4 s, **1688 MB** peak (first load
  92 s incl. 1.2 GB weight download). Transkun on 30 s of After 2 piano stem → 41 notes with velocities, 22 s,
  **855 MB** peak.
- BabySlakh downloaded + extracted: `data/babyslakh/babyslakh_16k/Track00001..20` (16 kHz **mono**, 226 stems
  listed, 209 with audio, 213 with MIDI; the `audio_rendered` flag in metadata.yaml is False for all of them and
  is ignored; 4 stems have audio but no MIDI).
- `tools/eval/transcription_eval.py` — new. Routes A full mix / C CleanSplit stems (constrained) / C2 stems with piano →
  Transkun / D oracle true stems. mir_eval, 50 ms onsets, offsets ignored; multi-F1 (instrument-aware), onset-F1,
  drum-F1 (GM keys → kick/snare/hihat/tom/cymbal), piano-F1. **Pre-registered:** Q1 C beats A on pooled multi-F1 AND
  >10/20 tracks → per-stem becomes the default; Q2 C2 beats C on piano-F1 → piano stems go to Transkun. Biases
  written in up front (16 kHz input hurts C; Slakh may be in MuScriptor's pre-training; no singing in Slakh).
  Caches in `data/transcription_cache/` (per route JSON + separated stems). Running → `outputs/transcription_eval.log`.
- `_muscriptor_worker.py` / `muscriptor_backend.py` — optional `grid_from`: tempo, bar lines and MuScriptor's own
  onset-lag estimate (beat_this via MuScriptor) from the full mix. Returned as `info["grid"]`.
- `cleansplit/transcription/pipeline.py` — new `transcribe_split(song_dir, mode, model, piano)`: skips stems below
  −60 dBFS, subtracts the measured onset lag, writes `midi/<song>.mid` (all tracks, detected BPM), `midi/<stem>.mid`,
  `midi/report.json` (timings, VRAM, tempo, note counts, caveats: no velocities from MuScriptor, constant tempo,
  non-commercial weights).
- `cleansplit/cli/main.py` — new `cleansplit midi SONG [--mode stems|mix] [--model small|medium|large]
  [--piano muscriptor|transkun] [--separator ...]`; reuses a cached split or separates first. **Defaults
  (stems, muscriptor for piano) are provisional until the eval answers Q1/Q2.** Not yet run end to end.
- `cleansplit/transcription/notes.py` — MIDI written at **960 ticks/beat**; pretty_midi's default 220 snapped notes
  to a ~3 ms grid at ~100 BPM (caught by the new test).
- `cleansplit/tests/test_transcription.py` — new, 5 non-GPU tests: MIDI round-trip (times, pitches, tracks, drums,
  velocity default + clamping, zero-length notes), stem constraints are valid MuScriptor names, pipeline skips
  silent stems / constrains each stem / subtracts the onset lag, mix mode, the eval's note matcher. **5/5 pass.**

**MIDI converter — research notes:**
- **MuScriptor** (Kyutai + Mirelo, arXiv 2607.08168, July 2026): open multi-instrument AMT, 36 instrument groups
  incl. drums and voice, small 103M / medium 307M / large 1.4B, 16 kHz mono, code MIT, weights **CC BY-NC 4.0 and
  gated on HuggingFace** (the user must accept the licence with their own account and run `hf auth login`
  themselves — I may not create accounts or enter tokens). Self-reported Multi-F1 48.2 vs YourMT3+ 21.9 on their
  own test set. **No velocity output.** Works best at steady tempo.
- **Transkun** (piano, MIT, pip `transkun`, velocity + pedal) — piano specialist.
- **basic-pitch** (Spotify, Apache-2.0, ungated) — lightweight fallback, pitch bends.
- **ADTOF** (drums; weights non-commercial), **ROSVOT** (singing voice, ACL'24) — specialists, not yet checked.
- YourMT3+ is GPL-3.0 (would constrain CleanSplit's licence if vendored).

### 2026-09-21 (4) — MUSDB18-HQ results; MDX23C reverted
Real studio truth, 20 songs, pre-registered questions (full tables in docs/04 §11):
- Q1 vocal ensemble beats SW: **yes**, but small — shipped 3-model +0.30 median 17/20; the 2-model average
  +0.39 median, **18/20**.
- Q2 MDX23C helps: **no** — −0.09 median, 4/20. The BUH result (+0.30, 5/5) did not replicate.
- Q3 Demucs averaging helps drums/bass: **no** — drums 0/20, bass 7/20.
- Q4 that average beats Demucs alone: yes (20/20, 18/20). The BUH smoke-test hint did not hold either.
Changes made because of it:
- `cleansplit/separation/ensemble.py` — **vocals back to mean(SW+TTA, ep317)**; MDX23C removed; docstring carries the
  MUSDB evidence and the history. Recipe string reverted, so 3-model cache entries are not reused.
- `cleansplit/tests/test_separator_gpu.py` — recipe test back to the 2-model average (MDX23C's own test stays).
- `cleansplit/cli/main.py`, `cleansplit/ui/service.py` — honest labels: `ensemble` "best measured";
  `ensemble_demucs` "try when a drum or bass lane comes out empty" (its one measured win is the song where SW
  missed the bass entirely: SW 0.8 dB vs Demucs 14.5 dB).
- `README.md` status row, `docs/04_results.md` — new §11; "superseded" notes on §7 and §10.
- Lesson recorded: one song with a separator-made reference overstated the ensemble gain ~5× and produced two
  wrong adoption decisions. Do not adopt anything on BUH evidence alone again.


### 2026-09-21 (3) — MUSDB18-HQ: real multitrack truth (download in progress)
The user approved downloading MUSDB18-HQ (Zenodo record 3338373, `musdb18hq.zip`, 22,656,664,047 bytes, md5
`12d4f2ecd55245a4688754dd76363103`, open access, **educational / non-commercial licence**). MoisesDB needs an
account, which I may not create; the user can sign up if they want it.
- **Download**: `data/musdb18hq/musdb18hq.zip`, resumable `curl -C -` in a 10-attempt loop. Zenodo throttles to
  ~2.2 MB/s → **~3 hours**. If the session dies, resume with:
  `cd data/musdb18hq && curl -L -C - --retry 5 -o musdb18hq.zip "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"`
  then check the md5 before using it.
- `.gitignore` — `data/` added (the dataset must never be committed: licence and size).
- `tools/eval/musdb_eval.py` — new. `extract` (test/ only), `run` (per-model, per-song `.npz` cache in
  `data/musdb_cache/`, so it resumes after interruption), `score`. **Protocol and three questions fixed in the
  docstring before any result:** one 30 s excerpt per song centred on the midpoint; median/mean/pooled SDR; a "yes"
  needs a better median AND wins on >25/50. Q1 ensemble vocals vs SW; Q2 does MDX23C still help; Q3 Demucs
  averaging for drums/bass. Leakage bias stated up front (community models may have trained on the test songs;
  Demucs did not → bias is against Demucs). Env overrides `CLEANSPLIT_MUSDB_{TEST,CACHE,OUT}` exist for the smoke test.
- Smoke test on a 2-song fake MUSDB built from the BUH stems: **all 5 models ran, cached, scored, JSON written.**
  It surfaced a question the protocol was missing: on those BUH excerpts Demucs *alone* beat the SW+Demucs average
  for drums (4.86 vs 4.13 dB) and bass (20.63 vs 19.21). Added **Q4 (average vs Demucs alone)** to the pre-registered
  list, with a note that it was added after the BUH smoke test and before any MUSDB song was scored. If Q4 comes
  back "no" on MUSDB, `ensemble_demucs` should take drums/bass from Demucs alone rather than the average.
- Download at 1.13 / 22.66 GB when this was written.
- **Scope cut to 20 songs** at the user's request (first 20 test songs alphabetically, content-blind). Stated in the
  `musdb_eval.py` docstring; a "yes" now needs wins on >10/20.
- Zenodo speed check: 6 parallel range requests got 3.3 MB/s combined, same as one connection → the cap is per
  client, not per connection. No faster route from Zenodo.
- `tools/queues/musdb_chain.sh N` — new. Waits for the zip to reach full size, **checks the md5 and stops on a
  mismatch**, extracts test/, runs `musdb_eval.py run --limit N`. Running in the background with N=20, log in
  `outputs/musdb_run.log`. If it dies: re-run the same command; the per-song cache means finished models are skipped.

### 2026-09-21 (2) — MDX23C adopted into the ensemble
- **Result** (`outputs/vocal_ensemble.log`): pooled vocals SDR 18.93 → **19.23 dB (+0.30)**, better on **5/5**
  excerpts. Pre-registered rule satisfied → adopted. Full table in docs/04 §10.
- `cleansplit/separation/ensemble.py` — vocals = mean(SW+TTA, ep317, MDX23C); MDX23C loaded after the other two are
  released (8 GB budget unchanged); recipe string and cache key updated, so old cached ensemble splits are not reused.
- `cleansplit/cli/main.py` — `mdx23c_instvoc_hq` is a `--separator` choice with its own branch (SW's chunk/fp16/tta
  options do not apply); ensemble help text updated.
- `cleansplit/ui/service.py` — separator labels/timings updated ("SW×3 + ep317 + MDX23C", ~4× song length).
- `README.md`, `docs/04_results.md` §10, `docs/02_hardware_measurements.md` (MDX23C table) — written up.
- `tools/experiments/vocal_ensemble_experiment.py` — JSON write crashed on a numpy int64 after printing the result; cast to
  Python types. Re-run end to end: **identical to the first run to 0.01 dB** in every cell (deterministic), JSON at
  `outputs/_benchmarks/vocal_ensemble_BUH.json`.
- Tests: `test_separator_gpu.py` — new `test_mdx23c_separates_aligned_vocals_and_instrumental`; the ensemble
  recipe test now checks the 3-model average. `test_cli_e2e.py` accepts the new choice.
- Surprise worth remembering: the *most* architecturally diverse pair (SW + MDX23C, 18.48) is worse than the two
  BS-RoFormers together (18.93). Architecture diversity ≠ error diversity; measure, do not assume.

### 2026-09-21 (1) — MDX23C InstVoc HQ integration
Why: ensembling is the only thing that has measurably improved the stems (docs/04 §7), and ensembles gain from
models whose errors differ. SW and ep317 are both BS-RoFormers; MDX23C is a convolutional TFC-TDF U-Net.
- `cleansplit/separation/mdx23c/{__init__,model}.py` — vendored from MSST @ `050cae73…` (the same pinned commit
  as the BS-RoFormer code, MIT). Two deviations in the header: `prefer_target_instrument` inlined, config passed
  as an attribute dict. Numerics untouched.
- `cleansplit/separation/mdx23c_sep.py` — new `MDX23CSeparator` (vocals + instrumental), MSST-identical chunked
  overlap-add (native chunk 261,120, overlap 8, reflect pad, chunk//10 fade), reuses `fade_window` /
  `load_msst_config` from `roformer.py`.
- `cleansplit/models/checkpoints.py` — `MDX23C_INSTVOC_HQ` pinned: sha256 `49d51472…d816`, 448,101,203 bytes,
  config `model_2_stem_full_band_8k.yaml` (confirmed via UVR's `model_data.json` hash map, not guessed).
- `cleansplit/separation/registry.py` — registered `mdx23c_instvoc_hq`.
- Smoke test, 30 s of Concrete Crown, RTX 2080 Super Max-Q: strict state-dict load, hash verified, **1482 MB peak**;
  overlap 2 / 4 / 8 = 8.3 / 9.6 / 19.8 s; vocals + instrumental reproduce the mixture to −64…−66 dB.
- `tools/experiments/vocal_ensemble_experiment.py` — new. Scores every equal-weight combination of {SW+TTA, ep317, MDX23C}
  against BUH vocal truth on **5** excerpts (30/60/90/120/150 s, twice as many as §7). **Decision rule fixed before
  running:** MDX23C joins the default ensemble only if the 3-model average beats the current 2-model average on the
  pooled score *and* on a majority of excerpts. Running now → `outputs/vocal_ensemble.log`.

### 2026-09-17 (3) — Apollo measured and written up
- `docs/04_results.md` — new **§9** (source material already lossy; whole-signal results table; region-restricted
  and gated run; conclusion). §8 cross-linked. Title date bumped.
- `docs/02_hardware_measurements.md` — Apollo VRAM/speed table: ~410 MB per second of stereo audio, 5 s chunk =
  2100 MB at 5× realtime, 20 s = 8.2 GB and thrashes, 60 s OOM. Operating point 5 s / 1 s overlap / 0.5 s pad.
- `README.md` — status row for Apollo, a limitation line saying no generative restorer is enabled by default, and a
  licensing paragraph for the two vendored third-party trees.
- `cleansplit/restoration/apollo.py` — `ApolloRestorer` now handles **mixture-level** regions by picking the stems
  that own the box (`_targets`, same energy-share rule as A2SB: top 2 stems within 20 dB). Before this it returned
  no proposals for those regions, so they silently produced no decision at all.
- `cleansplit/restoration/pipeline.py` — `_folder_name()`: restorer names carry options
  (`apollo:variant=vocal_restore`) and `:`/`=` are illegal in Windows paths, which crashed the run after the work
  was already done.
- `cleansplit/tests/test_apollo.py` — new, 5 tests (registration + option parsing, folder-name sanitising,
  shape/level preservation, **chunk seams < −25 dB vs a single pass**, region locality or bit-identity on reject).
- `cleansplit/restoration/apollo.py` (2nd pass) — the per-song enhanced-stem cache now keys on a blake2b hash of
  the **whole** array, not a 4096-sample prefix: with `--max-iterations > 1` a later pass can change a stem
  anywhere, and the prefix key would have returned a stale enhancement and scored the wrong audio. Added
  `describe()` (variant, target rules, licence) and dropped an unused import.

### 2026-09-17 (2) — Apollo integration
- `third_party/apollo/{__init__,apollo,base_model}.py`, `LICENSE` — vendored from
  github.com/JusperLee/Apollo @ `e84bcacc59d5455f05d86a5c97dd4aeb3c14dbb6`, CC BY-SA 4.0. Two deviations in the
  file header: no huggingface_hub mixin, no debug print. Numerics untouched.
- `cleansplit/restoration/apollo.py` — new. `ApolloModel` (whole-signal, chunked overlap-add, `weights_only` load
  with an explicit allowlist) and `ApolloRestorer` (region-restricted proposal for the gate).
- `cleansplit/restoration/baselines.py` — registered `apollo` in `RESTORERS`.
- `cleansplit/models/checkpoints.py` — `APOLLO` registry (both UVR checkpoints, SHA-256 + size pinned, provenance
  and licence recorded) and `find_apollo()`. Nothing is downloaded.
- `tools/probes/apollo_probe.py`, `tools/experiments/apollo_experiment.py` — new.

### 2026-09-17 (1) — Desktop UI finished
- `cleansplit/ui/static/app.js` — A/B now switches **everything** from the active side (waveforms, level readouts,
  flagged marks, inspector, header), not just the waveform; `activeSrc()` is the single source of truth.
  Inspector visibility is deliberately not touched on Tab (it would change deck width and make the two
  waveforms incomparable). Fixed the flagged-spot click index (it indexed the unfiltered region list).
- `cleansplit/ui/static/app.css` — narrow-window rules: <1180 px the inspector floats over the deck, <980 px the
  rail and lanes shrink. `pywebview` min size is 980×620 to match.
- `cleansplit/ui/{server,service,desktop}.py`, `cleansplit/cli/main.py` — the app now reads/writes `outputs/`
  (one sub-folder per separator) instead of `outputs/ui`, so existing CLI splits show up.
- `cleansplit/tests/test_ui.py` — new, 7 tests.
- `pyproject.toml` — declared the `ui` extra (fastapi, uvicorn, pywebview) and added `httpx` to `dev`.

---

## Traps for whoever is next

1. **"Stems rebuild the song +157 dB"** on ensemble splits is exact *by construction* (`other` is the remainder),
   not a quality score. The UI says so inline. Never quote it as evidence.
2. **The user's own files are already lossy.** `After 2.wav` is a decode of `After 2.mp3` (lag 0, correlation
   1.00000, gain 1.00000, SNR 74.5 dB) and everything except `Concrete Crown.wav` stops at ~16 kHz. Any experiment
   that treats one of those WAVs as a lossless reference is measuring nothing.
3. **Passing the mixture-consistency gate is not evidence of improvement** (docs/04 §4). Ground-truth scoring is
   mandatory before believing any restorer.
4. `vocal_restore` (Apollo) applied to a full mixture destroys everything below 250 Hz. It is a vocals-only model.

## Open work, in priority order

1. ~~Disagreement detector~~ — tested, failed its precision bar on held-out songs (docs/04 §12).
1b. The remaining 30 MUSDB test songs: `tools/eval/musdb_eval.py run` (no `--limit`); the cache makes the first 20 free.
2. ~~MDX23C InstVoc HQ as a third vocal model~~ — in progress, see 2026-09-21 above.
3. A **mixture-conditioned generative refiner trained to fix allocation** — the error that actually dominates.
   That is a model to train, not one that exists off the shelf.
4. UI: a listening-test mode would be the honest way to answer the one question the measurements cannot
   (is Apollo's fabricated top octave *preferred*, even though it is wrong?).
