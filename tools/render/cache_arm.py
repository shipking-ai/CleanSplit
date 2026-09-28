"""Fill one named cache arm with a named separator configuration, so any two configurations can be compared paired.

    python tools/render/cache_arm.py sw_ov4 --separator bs_roformer_sw --overlap 4 --no-tta [--limit 20]

Estimates land in data/musdb_cache/<arm>/ in tools/eval/musdb_eval.py's four-group convention, which is what
tools/eval/fullband_check.py and tools/eval/artifact_metrics.py read. Resumable: a song already cached is skipped.

Exists because the interesting comparisons keep needing a cell nobody separated yet, and hand-rolling a one-off
script each time is how arms end up differing in more than the one variable under test.

THE QUESTION THIS WAS WRITTEN FOR, pre-registered before the run:

  Is test-time augmentation REDUNDANT with chunk overlap?

  Both do the same thing. TTA averages 3 passes over transformed copies of the input (original, channel-swapped,
  polarity-inverted); overlap-add averages every output sample over `num_overlap` independently denoised chunks. Both
  cancel whatever part of the error is uncorrelated between passes. If they cancel the *same* error, running both is
  paying twice for one benefit -- and TTA is the expensive one, at 3x the GPU time versus overlap 4's 2x.

  Measured evidence that they might overlap: at overlap 2, TTA was worth only +0.06 dB on vocals and +0.02..+0.05 dB
  elsewhere (docs/04 section 14.1); overlap 4 alone was worth +0.09/+0.05/+0.05 dB (section 14.5). Similar sizes, and
  overlap 8 added nothing on top of 4 (section 14.6), which is exactly what saturation looks like.

  DECISION RULE: TTA stays in the default recipe only if, at overlap 4, it gives a paired median gain of at least
  +0.02 dB AND wins on more than half of the 20 songs, on at least two of vocals/drums/bass. Otherwise it is dropped
  as redundant, which makes the default separation about 3x faster at no measured cost.

  Compare with:  python tools/eval/fullband_check.py sw_ov4 sw_tta_ov4 --baseline sw_ov4 --limit 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "eval"))
import musdb_eval as M


def main(arm: str, separator: str, overlap: int, tta: bool, limit: int | None) -> None:
    import torch

    from cleansplit.separation import registry

    out = M.CACHE / arm
    items = [(n, m) for n, m, _ in M.songs(limit)]
    todo = [(n, m) for n, m in items if not (out / f"{n}.npz").is_file()]
    if not todo:
        print(f"{arm}: already complete ({len(items)} songs)")
        return
    kw = {"num_overlap": overlap}
    if separator.startswith("bs_roformer") or separator.startswith("ensemble"):
        kw["tta"] = tta
    sep = registry.create(separator, **kw)
    print(f"{arm}: {separator} overlap={overlap} tta={tta}, {len(todo)} songs to go", flush=True)
    for i, (name, mix) in enumerate(todo, 1):
        stems = {k: v.astype(np.float32) for k, v in sep.separate(mix, M.SR).stems.items()}
        grouped = M._grouped("sw" if "roformer_sw" in separator else separator, stems)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"{name}.npz", **grouped)
        print(f"{arm}: {i}/{len(todo)} {name}", flush=True)
    sep._model = None
    torch.cuda.empty_cache()
    print(f"{arm}: done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", help="cache directory name under data/musdb_cache/")
    p.add_argument("--separator", default="bs_roformer_sw")
    p.add_argument("--overlap", type=int, default=4)
    p.add_argument("--tta", dest="tta", action="store_true", default=True)
    p.add_argument("--no-tta", dest="tta", action="store_false")
    p.add_argument("--limit", type=int, default=20)
    a = p.parse_args()
    main(a.arm, a.separator, a.overlap, a.tta, a.limit)
