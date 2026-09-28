"""Measure Apollo's VRAM and speed on this machine (RTX 2080 Super Max-Q, 8 GB).

    python tools/probes/apollo_probe.py [variant]

Prints one row per chunk length: peak allocated/reserved VRAM and seconds per chunk, then the projected
wall time for a 4-minute song. Numbers go into docs/02_hardware_measurements.md as [measured].
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from cleansplit.restoration.apollo import ApolloModel

SR = 44100


def main(variant: str = "mp3_enhancer") -> None:
    model = ApolloModel(variant=variant, device="cuda")
    print(json.dumps(model.describe(), indent=2))
    rng = np.random.default_rng(0)
    rows = []
    for chunk_s in (5.0, 10.0, 20.0, 30.0, 60.0):
        x = (rng.standard_normal((2, int(chunk_s * SR))) * 0.05).astype(np.float32)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model.enhance(x, chunk_s=chunk_s + 1, overlap_s=0.0, pad_s=0.0)  # one forward, no chunking
            torch.cuda.synchronize()
            t0 = time.time()
            model.enhance(x, chunk_s=chunk_s + 1, overlap_s=0.0, pad_s=0.0)
            torch.cuda.synchronize()
            dt = time.time() - t0
            rows.append({
                "chunk_s": chunk_s,
                "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
                "seconds": round(dt, 3),
                "realtime_x": round(chunk_s / dt, 1),
            })
        except torch.cuda.OutOfMemoryError:
            rows.append({"chunk_s": chunk_s, "error": "OOM"})
            torch.cuda.empty_cache()
        print(rows[-1], flush=True)
    ok = [r for r in rows if "seconds" in r]
    if ok:
        best = max(ok, key=lambda r: r["realtime_x"])
        print(f"\n4-minute song at chunk {best['chunk_s']} s: ~{240 / best['realtime_x']:.0f} s "
              f"({best['peak_alloc_mb']} MB peak allocated)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mp3_enhancer")
