# tools/

Everything here is a command-line script, run from the repository root with the project's venv. None of it is imported
by the `cleansplit` package — the package never depends on this directory.

| Folder | What lives here | Rule of thumb |
|---|---|---|
| [`eval/`](eval) | Reusable measurement harnesses | Produces a number that `docs/04_results.md` cites |
| [`experiments/`](experiments) | Pre-registered experiments | Its **docstring carries the decision rule, written before the run** |
| [`render/`](render) | Produces audio and figures | Writes files a human looks at or listens to |
| [`probes/`](probes) | One-off investigations | Answered a question once; kept for provenance, not maintained |
| [`queues/`](queues) | Shell scripts that serialise GPU work | 8 GB of VRAM holds one job; two at once halves throughput |

## The distinction that matters

`eval/` and `experiments/` are not the same thing, and the split is deliberate.

**`eval/`** measures. `musdb_eval.py` is the shared harness every other script imports; `fullband_check.py` is the
filter-free gate a candidate must clear; `paired_rescore.py` re-derives an old result as a paired statistic.

**`experiments/`** decides. Each script states, in its docstring and **before it is run**, the question, the arms, the
adoption rule, and a falsifiable prediction with its falsification condition. That is why the rejections in
`docs/04_results.md` are trustworthy: the bar could not be moved after the numbers were seen.

```python
# tools/experiments/quality_tiers.py, verbatim from the docstring
# PRE-REGISTERED: `balanced` is worth shipping as a tier only if it recovers at
# least 70% of best's gain over fast, on vocals, measured as paired medians.
```

It recovered 95%. Had it recovered 60%, the rule would have rejected it.

## Running anything here

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/eval/fullband_check.py sw_tta_ov4 sw_tta_ov8 --baseline sw_tta_ov4
```

`PYTHONIOENCODING=utf-8` is needed on Windows for any script that prints `Δ` or `±`. Most scripts read cached estimates
from `data/musdb_cache/` and need no GPU; the ones that separate audio do, and should be run one at a time.
