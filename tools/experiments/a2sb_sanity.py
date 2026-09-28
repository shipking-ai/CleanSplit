"""A2SB integration sanity checks on one window (GPU):
 1. representation round trip with an empty mask reproduces the input,
 2. inpainting a TF box of a known synthetic stem: error vs ground truth inside the box, compared with
    zeroing the box (lower bound "do nothing sensible") and with the untouched truth,
 3. fp16+autocast vs fp32 over the full sampling run (same seed): does precision change the result?,
 4. time per stereo window.
"""
import json
import time

import numpy as np

from cleansplit.audio.stft import STFTGrid, stft
from cleansplit.audio.synthetic import make_song
from cleansplit.audio.tf_edit import TFBox, box_mask
from cleansplit.restoration.a2sb import HOP, SEGMENT, WINDOW_FRAMES, A2SBModel

g = STFTGrid()
mix, stems = make_song(8.0, seed=0)
a = 2 * 44100 // HOP * HOP
seg = stems["piano"][:, a : a + SEGMENT].astype(np.float64)
box = TFBox(a / 44100 + 1.2, a / 44100 + 1.6, 400.0, 2400.0)
mask = box_mask(g, WINDOW_FRAMES, box, frame_offset=a // HOP, taper_bins=0, taper_frames=0)[1:] > 0
res = {"mask_cells": int(mask.sum()), "segment_samples": SEGMENT}


def box_err(y):
    Y = stft(y, g, dtype=np.complex128)[:, 1:]
    T = stft(seg, g, dtype=np.complex128)[:, 1:]
    m = mask[None]
    return float(np.sum(np.abs((Y - T) * m) ** 2) / np.sum(np.abs(T * m) ** 2))


for precision in ("fp32", "fp16"):
    model = A2SBModel(precision=precision)
    empty = np.zeros_like(mask)
    rt = model.inpaint(seg, empty, n_steps=3, seed=0)
    inner = slice(4096, -4096)
    res[f"{precision}_roundtrip_rel_err"] = float(np.linalg.norm(rt[:, inner] - seg[:, inner]) / np.linalg.norm(seg[:, inner]))
    model.torch.cuda.synchronize()
    t0 = time.time()
    out = model.inpaint(seg, mask, n_steps=50, seed=0)
    model.torch.cuda.synchronize()
    res[f"{precision}_seconds_stereo_window_50_steps"] = round(time.time() - t0, 1)
    res[f"{precision}_box_rel_err_vs_truth"] = box_err(out)
    res[f"{precision}_peak_vram_mb"] = round(model.torch.cuda.max_memory_allocated() / 2**20)
    np.save(f"outputs/_benchmarks/a2sb_sanity_{precision}.npy", out)
    print(precision, {k: v for k, v in res.items() if k.startswith(precision)}, flush=True)
    del model
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

zeroed = seg.copy()
Z = stft(seg, g, dtype=np.complex128)
res["zeroed_box_rel_err_vs_truth"] = 1.0  # by definition: error energy == truth energy in the box
o32 = np.load("outputs/_benchmarks/a2sb_sanity_fp32.npy")
o16 = np.load("outputs/_benchmarks/a2sb_sanity_fp16.npy")
res["fp16_vs_fp32_rel_diff_full_sampling"] = float(np.linalg.norm(o16 - o32) / np.linalg.norm(o32 - seg + 1e-12))
print(json.dumps(res, indent=1))
open("outputs/_benchmarks/a2sb_sanity.json", "w").write(json.dumps(res, indent=1))
