# Contributing

The unusual rule in this project: **quality claims are measured, not argued.** Everything else is ordinary.

## Setup

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -e ".[dev,ui]"
```

Torch is optional — the test suite does not need it, which is why CI installs neither torch nor any model weights.
Add `.[separation]` if you want to actually separate audio.

## Before opening a PR

```bash
ruff check .
```
```bash
pytest cleansplit/tests -q -m "not gpu"
```

The 10 GPU-marked tests need an 8 GB card and local checkpoints, and are run by hand.

## Changing separation quality

A change to the default recipe has to clear the same bar as everything already in `docs/04_results.md`:

1. **Write the decision rule first**, in the script's docstring, before the run — with a falsifiable prediction and an
   explicit falsification condition. See `tools/experiments/` for a dozen examples.
2. **Compare paired**: the median of per-song differences, plus a win count. Never median-of-A minus median-of-B —
   that statistic manufactured two false results here (docs/04 §14.7).
3. **Clear the filter-free full-band gate**: `tools/eval/fullband_check.py`.
4. **Clear +0.02 dB.** A consistent gain below the floor is rejected on cost. That has happened twice (§14.6, §14.13),
   both times to a change that won on most songs.

A negative result is a welcome contribution. Seven are published here.

## Style

`ruff check .` is the whole style guide. Two conventions it does not enforce:

- **Comments say why, not what.** The interesting comments here record why an alternative was rejected.
- **Vendored code is never edited.** `third_party/` and `cleansplit/separation/{bs_roformer,mdx23c,scnet}` stay
  byte-identical to upstream so they can be diffed against it; they are excluded from lint for that reason.

Update `HANDOFF.md` with anything a person picking this up cold would need.
