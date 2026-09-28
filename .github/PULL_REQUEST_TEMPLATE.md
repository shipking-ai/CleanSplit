## What this changes

<!-- One or two sentences. -->

## If it changes separation quality

Quality claims in this repo are measured, not argued. A change to the default recipe needs:

- [ ] **Paired** medians with win counts — never median-of-A minus median-of-B (docs/04 §14.7)
- [ ] The filter-free full-band gate: `tools/eval/fullband_check.py`
- [ ] A gain of at least **+0.02 dB** — the effect-size floor (docs/04 §14.6, §14.13)
- [ ] The decision rule written in the script's docstring **before** the run

Not applicable to docs, tooling or refactors — say so and skip it.

## Checks

- [ ] `ruff check .` clean
- [ ] `pytest cleansplit/tests -q -m "not gpu"` passes
- [ ] `HANDOFF.md` updated
