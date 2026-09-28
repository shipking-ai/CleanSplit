"""Build a cache arm for the per-stem ensemble out of two already-cached member arms, with no GPU.

    python tools/eval/synth_arm.py ens_ov4      --sw sw_ov4     --voc ep317_ov4
    python tools/eval/synth_arm.py ens_tta_ov4  --sw sw_tta_ov4 --voc ep317_ov4

Why this is exact and not an approximation. EnsembleSeparator averages ONE stem: vocals = mean(sw, ep317). drums,
bass, guitar and piano are the SW member untouched, and `other` is the remainder of the mixture. So given the SW
member's four cached groups and the ep317 member's vocals, the ensemble's groups are fully determined:

    vocals = (v_sw + v_voc) / 2
    drums  = d_sw          bass = b_sw
    other  = mix - vocals - drums - bass
           = (mix - v_sw - d_sw - b_sw) + (v_sw - vocals)
           = o_sw + (v_sw - v_voc) / 2

The second line is the point: every cached arm is mixture-consistent by construction (its `other` IS the remainder),
so the mixture cancels and `other` follows from the cached arrays alone. No audio is read and no model is run, which
is what makes it possible to price a recipe nobody has rendered yet.

THE QUESTION THIS WAS WRITTEN FOR, pre-registered before the arms were scored:

  Does the 3-pass TTA inside the shipped default earn its keep at overlap 4?

  The default `best` recipe is mean(SW+TTA, ep317) at overlap 4 = 12 + 4 = 16 forward passes. Dropping TTA from the
  SW member gives mean(SW, ep317) at overlap 4 = 4 + 4 = 8 passes -- half the compute for the same two models and the
  same overlap. tools/render/cache_arm.py already asked this of SW ALONE and measured +0.027 vocals / +0.015 drums /
  +0.033 bass. This asks it of the thing that actually ships, where TTA's contribution to vocals is halved by the
  averaging with ep317 but its contribution to drums and bass is not diluted at all.

  DECISION RULE: TTA stays in the default only if the expensive arm clears tools/eval/fullband_check.py's gate
  against the cheap one -- paired median >= +0.02 dB and wins on more than half of the 20 songs, on vocals, drums AND
  bass. Otherwise the default becomes the 8-pass recipe and `best` is redefined, because a 2x compute cost has to buy
  something measurable.

  PREDICTION: it fails on vocals (TTA's +0.027 dB halves to about +0.013 dB when averaged with a non-TTA member) and
  passes on bass. Drums decides it, and drums already missed the floor at +0.015 dB on SW alone.
  FALSIFIED IF: the ensemble arm gains MORE on vocals than SW alone did, which would mean the averaging amplifies
  TTA's contribution rather than diluting it, and the dilution argument above is wrong.

Verify with:  python tools/eval/fullband_check.py ens_tta_ov4 --baseline ens_ov4 --limit 20
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


def build(arm: str, sw: str, voc: str, force: bool) -> None:
    src_sw, src_voc, out = M.CACHE / sw, M.CACHE / voc, M.CACHE / arm
    for p in (src_sw, src_voc):
        if not p.is_dir():
            raise SystemExit(f"missing member arm: {p}")
    shared = sorted({p.stem for p in src_sw.glob("*.npz")} & {p.stem for p in src_voc.glob("*.npz")})
    if not shared:
        raise SystemExit(f"{sw} and {voc} have no songs in common")
    out.mkdir(parents=True, exist_ok=True)
    made = 0
    for name in shared:
        dst = out / f"{name}.npz"
        if dst.is_file() and not force:
            continue
        a = {k: v.astype(np.float64) for k, v in np.load(src_sw / f"{name}.npz").items()}
        b = np.load(src_voc / f"{name}.npz")["vocals"].astype(np.float64)
        if a["vocals"].shape != b.shape:
            raise SystemExit(f"{name}: member shapes differ, {a['vocals'].shape} vs {b.shape}")
        half = 0.5 * (a["vocals"] - b)           # what the averaging removes from vocals
        g = {"vocals": a["vocals"] - half, "drums": a["drums"], "bass": a["bass"], "other": a["other"] + half}
        # The recipe's whole point is that stems still sum to the mixture. The mixture is not loaded here, so assert
        # the equivalent invariant that does not need it: the synthesised groups sum to the member's groups.
        lhs = sum(g[k] for k in M.GROUPS)
        rhs = sum(a[k] for k in M.GROUPS)
        if not np.allclose(lhs, rhs, atol=1e-9):
            raise SystemExit(f"{name}: mixture consistency broken, max |diff| = {np.abs(lhs - rhs).max():.3e}")
        np.savez(dst, **{k: v.astype(np.float32) for k, v in g.items()})
        made += 1
    print(f"{arm}: {made} written, {len(shared) - made} already present -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesise an ensemble cache arm from cached member arms (no GPU).")
    ap.add_argument("arm", help="name of the arm to create under data/musdb_cache/")
    ap.add_argument("--sw", required=True, help="cached arm supplying drums, bass, other and half of vocals")
    ap.add_argument("--voc", required=True, help="cached arm supplying the other half of vocals (ep317)")
    ap.add_argument("--force", action="store_true", help="rewrite songs that already exist")
    a = ap.parse_args()
    build(a.arm, a.sw, a.voc, a.force)


if __name__ == "__main__":
    main()
