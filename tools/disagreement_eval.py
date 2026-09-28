"""Held-out test of the model-disagreement flag on MUSDB18-HQ test songs 21-50.

    python tools/disagreement_eval.py        # separates songs 21-50 with SW+TTA and HTDemucs_ft (cached), then scores

The flag needs no truth: agreement = 10 log10(|d|^2 / |s - d|^2), where s is SW+TTA's estimate of a stem and d is
HTDemucs_ft's. Low agreement means two strong models disagree about what is in that stem, so at least one of them has
missed or misfiled something.

DEVELOPED on songs 1-20 (alphabetical): Spearman(agreement, SW SDR) = 0.79 over 80 song-stems; the threshold
T = 6 dB was chosen there (catches 7/8 SW failures, i.e. SDR < 3 dB, flagging 13/80 stems, precision 0.54).

PRE-REGISTERED for songs 21-50, fixed before they were separated. Nothing below may be changed after the run:
  rule      flag a stem when agreement < 6 dB
  failure   SW+TTA SDR < 3 dB against the real studio stem
  pass if   recall >= 0.5  AND  precision >= 0.4  AND  Spearman(agreement, SW SDR) >= 0.5
Output: outputs/_benchmarks/disagreement_heldout.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import musdb_eval as M  # noqa: E402

THRESHOLD_DB = 6.0
FAIL_SDR_DB = 3.0
OUT = M.ROOT / "outputs" / "_benchmarks" / "disagreement_heldout.json"


def agreement_db(s: np.ndarray, d: np.ndarray) -> float:
    s, d = s.astype(np.float64), d.astype(np.float64)
    return float(10 * np.log10(np.sum(d**2) / max(np.sum((s - d) ** 2), 1e-20)))


def main() -> None:
    import torch
    from cleansplit.metrics.signal import snr_db

    items = list(M.songs())[20:]
    print(f"held-out songs: {len(items)}", flush=True)
    seps = M._separators()
    for key in ("sw_tta", "demucs"):
        todo = [(n, m) for n, m, _ in items if not (M.CACHE / key / f"{n}.npz").is_file()]
        if not todo:
            continue
        sep = seps[key]()
        for i, (name, mix) in enumerate(todo, 1):
            g = M._grouped(key, {k: v.astype(np.float32) for k, v in sep.separate(mix, M.SR).stems.items()})
            (M.CACHE / key).mkdir(parents=True, exist_ok=True)
            np.savez(M.CACHE / key / f"{name}.npz", **g)
            print(f"{key}: {i}/{len(todo)} {name}", flush=True)
        sep._model = None
        torch.cuda.empty_cache()

    rows = []
    for name, _, truth in items:
        s = np.load(M.CACHE / "sw_tta" / f"{name}.npz")
        d = np.load(M.CACHE / "demucs" / f"{name}.npz")
        for g in M.GROUPS:
            rows.append({"song": name, "stem": g, "agreement_db": agreement_db(s[g], d[g]),
                         "sw_sdr_db": float(snr_db(truth[g].astype(np.float64), s[g].astype(np.float64)))})
    ag = np.array([r["agreement_db"] for r in rows])
    sdr = np.array([r["sw_sdr_db"] for r in rows])
    flag, fail = ag < THRESHOLD_DB, sdr < FAIL_SDR_DB
    tp = int((flag & fail).sum())
    recall = tp / max(int(fail.sum()), 1)
    precision = tp / max(int(flag.sum()), 1)
    rho = float(spearmanr(ag, sdr).correlation)
    passed = bool(recall >= 0.5 and precision >= 0.4 and rho >= 0.5)
    print(f"\nheld-out: {len(rows)} song-stems, {int(fail.sum())} SW failures, {int(flag.sum())} flagged")
    print(f"recall {recall:.2f} ({tp}/{int(fail.sum())}), precision {precision:.2f}, Spearman {rho:.3f} -> "
          f"{'PASS' if passed else 'FAIL'}")
    for r in sorted(rows, key=lambda r: r["agreement_db"])[:12]:
        print(f"  {r['song'][:38]:38s} {r['stem']:7s} agree {r['agreement_db']:6.1f}  SW SDR {r['sw_sdr_db']:6.2f}")
    OUT.write_text(json.dumps({"threshold_db": THRESHOLD_DB, "fail_sdr_db": FAIL_SDR_DB, "recall": recall,
                               "precision": precision, "spearman": rho, "pass": passed, "rows": rows}, indent=1))
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
