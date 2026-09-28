"""Measure A2SB feasibility on this GPU: checkpoint integrity, strict weight loading, weight VRAM, and peak VRAM /
time of one denoiser forward pass at NVIDIA's inference window (3 x 1024 x 256 frames = ~3 s at 44.1 kHz).

Usage: python -u tools/a2sb_probe.py
A2SB code and weights: NVIDIA Source Code License-NC / NVIDIA OneWay Noncommercial License (not part of CleanSplit).
Note: on Windows (WDDM), exceeding *free* VRAM does not OOM; it silently pages to system RAM and runs 10-100x slower.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "diffusion-audio-restoration"
CKPT_DIR = ROOT / "models" / "a2sb"
sys.path.insert(0, str(REPO))

from networks import AttnUNetF, SinusoidalTemporalEmbedding  # noqa: E402

SLOW_S = 5.0


def to_device(model, dtype):
    """fp16 = half weights, always run under torch.autocast: the network creates float32 tensors internally
    (embeddings, GroupNorm32 casts to float32), so bare .half() inference fails with dtype mismatches."""
    return model.to("cuda", dtype=dtype)


def run(model, x, emb, dtype):
    with torch.autocast("cuda", dtype=torch.float16, enabled=dtype == torch.float16):
        return model(x, emb)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_vf(p: Path, init: dict):
    ck = torch.load(p, map_location="cpu", weights_only=True)  # no arbitrary unpickling
    sd = ck["state_dict"] if "state_dict" in ck else ck
    vf = {k[len("vf_model."):]: v for k, v in sd.items() if k.startswith("vf_model.")}
    m = AttnUNetF(**init)
    m.load_state_dict(vf, strict=True)
    return m.eval(), sorted({str(t.dtype) for t in vf.values()})


def forward_stats(model, dtype, bs, t_to_emb):
    x = torch.randn(bs, 3, 1024, 256, device="cuda", dtype=dtype)
    emb = t_to_emb(torch.full((bs,), 0.5, device="cuda")).to(dtype)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        t0 = time.time()
        run(model, x, emb, dtype)  # warm-up (kernel selection)
        torch.cuda.synchronize()
        warm = time.time() - t0
        if warm > SLOW_S * 3:
            return {"batch": bs, "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20), "sec_warmup": round(warm, 2), "pathological": True}
        t0 = time.time()
        y = run(model, x, emb, dtype)
        torch.cuda.synchronize()
        dt = time.time() - t0
    return {
        "batch": bs,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "sec_per_forward": round(dt, 3),
        "finite": bool(torch.isfinite(y).all()),
        "pathological": dt > SLOW_S,
    }


def main():
    init = yaml.safe_load(open(REPO / "configs" / "ensemble_2split_sampling.yaml"))["model"]["vf_model"]["init_args"]
    expected = {Path(f["path"]).name: (f["lfs"]["oid"], f["size"]) for f in json.loads((CKPT_DIR / "lfs_tree.json").read_text()) if f.get("lfs")}
    free, total = torch.cuda.mem_get_info()
    report = {"device": torch.cuda.get_device_name(0), "vram_total_mb": round(total / 2**20), "vram_free_at_start_mb": round(free / 2**20), "checkpoints": {}, "runs": []}
    print(json.dumps({k: v for k, v in report.items() if k != "checkpoints"}), flush=True)

    models = {}
    for name in sorted(expected):
        p = CKPT_DIR / name
        info = {"present": p.is_file()}
        if p.is_file():
            info["sha256_ok"] = sha256(p) == expected[name][0] and p.stat().st_size == expected[name][1]
            m, dtypes = load_vf(p, init)
            info.update(strict_load=True, params_M=round(sum(t.numel() for t in m.parameters()) / 1e6, 1), dtypes=dtypes)
            models[name] = m
        report["checkpoints"][name] = info
        print(name, info, flush=True)

    t_to_emb = SinusoidalTemporalEmbedding(n_bands=64, min_freq=0.5).cuda()
    one = models["A2SB_onesplit_0.0_1.0_release.ckpt"]

    for label, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        if label == "fp32" and "--skip-fp32" in sys.argv:
            continue
        m = to_device(one, dtype)
        torch.cuda.empty_cache()
        weights_mb = round(torch.cuda.memory_allocated() / 2**20)
        for bs in (1, 2):
            row = {"config": f"onesplit {label}", "weights_mb": weights_mb, **forward_stats(m, dtype, bs, t_to_emb)}
            report["runs"].append(row)
            print(row, flush=True)
            if row["pathological"]:
                break
        one = m.to("cpu", dtype=torch.float32)
        del m
        torch.cuda.empty_cache()

    # 2-split ensemble with both networks resident (no CPU<->GPU swapping between t < 0.5 and t >= 0.5)
    for label, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        a = to_device(models["A2SB_twosplit_0.0_0.5_release.ckpt"], dtype)
        b = to_device(models["A2SB_twosplit_0.5_1.0_release.ckpt"], dtype)
        torch.cuda.empty_cache()
        weights_mb = round(torch.cuda.memory_allocated() / 2**20)
        row = {"config": f"twosplit both resident {label}", "weights_mb": weights_mb, **forward_stats(b, dtype, 1, t_to_emb)}
        report["runs"].append(row)
        print(row, flush=True)
        models["A2SB_twosplit_0.0_0.5_release.ckpt"] = a.to("cpu", dtype=torch.float32)
        models["A2SB_twosplit_0.5_1.0_release.ckpt"] = b.to("cpu", dtype=torch.float32)
        del a, b
        torch.cuda.empty_cache()

    # fp16 fidelity vs fp32 on the same input (real weights)
    x = torch.randn(1, 3, 1024, 256)
    emb = t_to_emb(torch.full((1,), 0.5, device="cuda")).cpu()
    with torch.inference_mode():
        m32 = one.to("cuda")
        y32 = run(m32, x.cuda(), emb.cuda(), torch.float32).float().cpu()
        yamp = run(m32, x.cuda(), emb.cuda(), torch.float16).float().cpu()  # fp32 weights + autocast
        m16 = to_device(m32, torch.float16)
        y16 = run(m16, x.cuda().half(), emb.cuda().half(), torch.float16).float().cpu()
    report["autocast_fp32_weights_vs_fp32_relative_output_error"] = float((yamp - y32).norm() / y32.norm())
    print("autocast (fp32 weights) vs fp32 relative output error", report["autocast_fp32_weights_vs_fp32_relative_output_error"], flush=True)
    rel = float((y16 - y32).norm() / y32.norm())
    report["fp16_vs_fp32_relative_output_error"] = rel
    print("fp16 vs fp32 relative output error", rel, flush=True)

    out = ROOT / "outputs" / "_benchmarks" / "a2sb_vram_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
