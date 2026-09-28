"""Dev: VRAM/time of the A2SB denoiser architecture with random weights (memory does not depend on weight values)."""
import sys
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2] / "third_party" / "diffusion-audio-restoration"
sys.path.insert(0, str(REPO))
from networks import AttnUNetF, SinusoidalTemporalEmbedding

init = yaml.safe_load(open(REPO / "configs" / "ensemble_2split_sampling.yaml"))["model"]["vf_model"]["init_args"]
m = AttnUNetF(**init).eval().cuda()
params = sum(p.numel() for p in m.parameters())
print(f"params {params / 1e6:.1f} M, weights {torch.cuda.memory_allocated() / 2**20:.0f} MB")
emb = SinusoidalTemporalEmbedding(64, 0.5).cuda()
for bs in (1, 2, 4):
    x = torch.randn(bs, 3, 1024, 256, device="cuda")
    e = emb(torch.full((bs,), 0.5, device="cuda"))
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            m(x, e)
            torch.cuda.synchronize()
            t = time.time()
            m(x, e)
            torch.cuda.synchronize()
        print(f"fp32 batch {bs}: peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB, {time.time() - t:.3f} s/forward")
    except torch.OutOfMemoryError:
        print(f"fp32 batch {bs}: OOM")
        break
    del x
    torch.cuda.empty_cache()
