"""Is UVR-MDX-NET-Inst_HQ_3 (the one untested model already on disk) worth adding to the vocal average?

    python tools/dev/mdxnet_onnx_probe.py [--limit 3] [--nfft 6144] [--compensate 1.0]

Kept in tools/dev/ deliberately: two inference parameters are INFERRED, not sourced, so nothing here is fit to be a
shipped separator or a published number until the self-check below passes.

Where each parameter comes from:
  dim_f = 3072, dim_t = 256   READ FROM THE ONNX FILE ITSELF (graph input shape [batch, 4, 3072, 256]).
  hop = 1024                  UVR's own separate.py (`self.hop = 1024`), the code that ships these models.
  chunk = hop * (dim_t - 1)   UVR's separate.py (`self.chunk_size = self.hop * (self.mdx_segment_size-1)`).
  trim = n_fft // 2           UVR's separate.py.
  n_fft = 6144                INFERRED. UVR reads it from `mdx_n_fft_scale_set`, and this model's hash
                              (md5 of the last 10 kB = ad1501a5b998eb4b37c8ab81b1403305) is in NEITHER the local
                              model_data.json (86 entries) NOR TRvlvr/application_data's mdx_model_data. 6144 is the
                              only value for which dim_f == n_fft/2, which is the convention for this family.
  compensate = 1.0            INFERRED. UVR multiplies the output by a per-model `compensate` (~1.02 for this family)
                              that is likewise unavailable for this hash. 1.0 is not an arbitrary pick: it is the
                              value that keeps vocals = mixture - instrumental exactly mixture-consistent, which every
                              CleanSplit recipe requires anyway.

SELF-CHECK FIRST, and this is the point of the probe: if either inferred parameter is wrong the output is garbage, and
a garbage run would look exactly like "this model does not help" -- the failure mode docs/04 exists to prevent. So the
probe reports the model's own INSTRUMENTAL SNR against the true instrumental before anything else. UVR reports ~11 dB
SDR for this model, so:
    > 6 dB   the implementation is sane; the averaging numbers below mean something.
    < 6 dB   the parameters are wrong. Report NOTHING about the model's quality. Fix or abandon.

Only then: does averaging its vocals into the shipped pair help? Reported, not decided -- a real adoption test belongs
in tools/vocal_best_recipe.py with a pre-registered rule, and only if this probe says the plumbing is right.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import musdb_eval as M

ONNX = Path(r"C:\Users\wegot\AppData\Local\Programs\Ultimate Vocal Remover\models\MDX_Net_Models\UVR-MDX-NET-Inst_HQ_3.onnx")
SHA256 = "317554b07fe1ea5279a77f2b1520a41ea4b93432560c4ffd08792c30fddf9adc"
DIM_F, DIM_T, HOP = 3072, 256, 1024


class MDXNetOnnx:
    """UVR's MDX-Net ONNX path: STFT -> [B, 4, dim_f, dim_t] (real/imag x 2 channels) -> mask -> iSTFT, overlap-free
    chunks with n_fft//2 trimmed from each end, exactly as separate.py does it."""

    def __init__(self, n_fft: int, compensate: float):
        import hashlib

        import onnxruntime as ort

        b = ONNX.read_bytes()
        got = hashlib.sha256(b).hexdigest()
        if got != SHA256:
            raise RuntimeError(f"checkpoint changed: {got}")
        self.n_fft, self.compensate = n_fft, compensate
        self.trim = n_fft // 2
        self.chunk = HOP * (DIM_T - 1)
        self.gen = self.chunk - 2 * self.trim
        if self.gen <= 0:
            raise ValueError(f"n_fft {n_fft} too large for chunk {self.chunk}")
        self.sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
        self.win = np.hanning(n_fft).astype(np.float32)

    def _stft(self, x: np.ndarray) -> np.ndarray:
        import torch

        t = torch.from_numpy(x)
        S = torch.stft(t, self.n_fft, HOP, window=torch.from_numpy(self.win), center=True,
                       return_complex=True, pad_mode="constant")[..., :DIM_F, :DIM_T]
        return torch.stack([S.real, S.imag], 1).reshape(1, 4, DIM_F, DIM_T).numpy().astype(np.float32)

    def _istft(self, spec: np.ndarray, n: int) -> np.ndarray:
        import torch

        s = torch.from_numpy(spec).reshape(2, 2, DIM_F, DIM_T)
        full = torch.zeros(2, self.n_fft // 2 + 1, DIM_T, dtype=torch.complex64)
        full[:, :DIM_F] = torch.complex(s[:, 0], s[:, 1])
        y = torch.istft(full, self.n_fft, HOP, window=torch.from_numpy(self.win), center=True, length=n)
        return y.numpy()

    def instrumental(self, mix: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(mix, dtype=np.float32)
        n = x.shape[-1]
        pad = np.zeros((2, self.trim), np.float32)
        buf = np.concatenate([pad, x, pad, np.zeros((2, self.gen), np.float32)], -1)
        out = []
        i = 0
        while i < buf.shape[-1] - self.chunk:
            seg = buf[:, i:i + self.chunk]
            y = self._istft(self.sess.run(None, {"input": self._stft(seg)})[0], self.chunk)
            out.append(y[:, self.trim:-self.trim])
            i += self.gen
        return (np.concatenate(out, -1)[:, :n] * self.compensate) if out else np.zeros_like(x)


def main(limit: int, n_fft: int, compensate: float) -> None:
    from cleansplit.metrics.signal import snr_db

    sep = MDXNetOnnx(n_fft, compensate)
    print(f"n_fft={n_fft} (inferred) compensate={compensate} chunk={sep.chunk} trim={sep.trim}", flush=True)
    inst_snr, voc = [], []
    for name, mix, truth in M.songs(limit):
        mix64 = mix.astype(np.float64)
        inst_true = sum(truth[g].astype(np.float64) for g in ("drums", "bass", "other"))
        inst = sep.instrumental(mix).astype(np.float64)
        v = mix64 - inst
        t = truth["vocals"].astype(np.float64)
        e = {k: np.load(M.CACHE / k / f"{name}.npz")["vocals"].astype(np.float64) for k in ("sw_tta", "ep317")}
        pair = (e["sw_tta"] + e["ep317"]) / 2
        trio = (e["sw_tta"] + e["ep317"] + v) / 3
        row = dict(inst=float(snr_db(inst_true, inst)), mdx=float(snr_db(t, v)),
                   pair=float(snr_db(t, pair)), trio=float(snr_db(t, trio)))
        inst_snr.append(row["inst"])
        voc.append(row)
        print(f"  {name[:40]:40s} instrumental {row['inst']:6.2f} | vocals mdx {row['mdx']:6.2f} "
              f"pair {row['pair']:6.2f} trio {row['trio']:6.2f}", flush=True)

    m = float(np.median(inst_snr))
    print(f"\nSELF-CHECK  median instrumental SNR {m:.2f} dB -> "
          f"{'SANE, the numbers below mean something' if m > 6 else 'FAILED: parameters are wrong, report nothing about this model'}")
    if m <= 6:
        return
    for k in ("mdx", "pair", "trio"):
        print(f"  vocals {k:5s} median {np.median([r[k] for r in voc]):6.2f} dB")
    d = np.median([r["trio"] for r in voc]) - np.median([r["pair"] for r in voc])
    print(f"\n  adding it to the average: {d:+.2f} dB median, better on "
          f"{sum(r['trio'] > r['pair'] for r in voc)}/{len(voc)} songs (indicative only, n={len(voc)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--nfft", type=int, default=6144)
    p.add_argument("--compensate", type=float, default=1.0)
    a = p.parse_args()
    main(a.limit, a.nfft, a.compensate)
