"""Does Apollo repair codec damage? Controlled experiment with exact ground truth.

    python tools/experiments/apollo_experiment.py REFERENCE.wav [--bitrates 320k 128k 96k] [--variants mp3_enhancer vocal_restore]
                                      [--seconds 60] [--offset 30] [--out outputs/_benchmarks/apollo_codec.json]

Method. The reference is the truth. Each bitrate is a *known* damage: encode the reference with LAME, decode it
back, undo the encoder's delay by cross-correlation, and trim both to the same length. Then score
    lossy   vs reference   (how much damage the codec did)
    apollo  vs reference   (what is left after Apollo)
on the same metrics. Apollo helps only if its distance to the truth is smaller than the lossy file's.

The low bitrates are the control. If Apollo cannot improve 96 kbps audio - the damage it was trained on - then a
null result at 320 kbps says nothing about Apollo and everything about the harness. Reporting 320 kbps alone
would be an unfalsifiable experiment.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from cleansplit.analysis.pipeline import write_json
from cleansplit.audio.io import load_audio, save_audio
from cleansplit.metrics.signal import (
    band_residual_db,
    log_spectral_distance,
    multi_mel_snr_db,
    multires_stft_distance,
    si_sdr_db,
    snr_db,
)
from cleansplit.reconstruction.core import _shift, estimate_lag

SR = 44100


def mp3_roundtrip(x: np.ndarray, bitrate: str, tmp: Path) -> np.ndarray:
    """Encode (C, N) float to MP3 with LAME and decode it back, as a real lossy file would be."""
    src, mp3, back = tmp / "src.wav", tmp / f"{bitrate}.mp3", tmp / f"{bitrate}.wav"
    save_audio(src, x, SR)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-codec:a", "libmp3lame",
                    "-b:a", bitrate, str(mp3)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp3), "-f", "wav", "-c:a",
                    "pcm_f32le", str(back)], check=True)
    return load_audio(back).audio.astype(np.float64)


def align_to(reference: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int, float]:
    """MP3 encoding adds encoder delay; without undoing it every metric measures the delay, not the damage."""
    n = min(reference.shape[-1], y.shape[-1])
    lag, corr = estimate_lag(reference[..., :n], y[..., :n], max_lag=4096)
    return _shift(y[..., :n], lag), lag, corr


def _lowpass(x: np.ndarray, hi_hz: float) -> np.ndarray:
    """Brickwall in the FFT domain: crude for listening, exact for measurement."""
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(x.shape[-1], 1.0 / SR)
    X[..., f > hi_hz] = 0
    return np.fft.irfft(X, n=x.shape[-1], axis=-1)


def score(reference: np.ndarray, estimate: np.ndarray) -> dict:
    n = min(reference.shape[-1], estimate.shape[-1])
    r, e = reference[..., :n], estimate[..., :n]
    return {
        "snr_db": snr_db(r, e),
        "si_sdr_db": si_sdr_db(r, e),
        "multi_mel_snr_db": multi_mel_snr_db(r, e, SR),
        "lsd": log_spectral_distance(r, e),
        "multires_stft": multires_stft_distance(r, e, SR),
        # Split the two things a super-resolution model does: repair damage where the truth has content,
        # and invent a top octave where it has none. One number cannot tell those apart.
        "snr_db_below_16k": snr_db(_lowpass(r, 16000.0), _lowpass(e, 16000.0)),
        "energy_above_16k_db": float(10 * np.log10(np.mean((e - _lowpass(e, 16000.0)) ** 2) + 1e-30)),
        "ref_energy_above_16k_db": float(10 * np.log10(np.mean((r - _lowpass(r, 16000.0)) ** 2) + 1e-30)),
        "error_by_band_db": band_residual_db(r, r - e),
    }


def run(reference_path: str, bitrates, variants, seconds: float | None, offset: float, out: Path) -> dict:
    d = load_audio(reference_path)
    if d.sample_rate != SR:
        raise SystemExit(f"reference must be 44.1 kHz, got {d.sample_rate}")
    ref = d.audio.astype(np.float64)
    if seconds:
        a = int(offset * SR)
        ref = ref[..., a:a + int(seconds * SR)]
    print(f"reference {reference_path}: {ref.shape[-1] / SR:.1f} s", flush=True)

    from cleansplit.restoration.apollo import ApolloModel

    result = {"reference": str(Path(reference_path).resolve()), "seconds": ref.shape[-1] / SR,
              "offset_s": offset, "bitrates": {}}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        lossy = {}
        for br in bitrates:
            y, lag, corr = align_to(ref, mp3_roundtrip(ref, br, tmp))
            lossy[br] = y
            result["bitrates"][br] = {"encoder_delay_samples": lag, "alignment_corr": round(corr, 5),
                                      "lossy": score(ref, y), "apollo": {}}
            print(f"  {br}: lag {lag}, lossy SNR {result['bitrates'][br]['lossy']['snr_db']:.2f} dB", flush=True)

        for variant in variants:
            model = ApolloModel(variant=variant, device="cuda")
            result.setdefault("models", {})[variant] = model.describe()
            for br in bitrates:
                t0 = time.time()
                y = model.enhance(lossy[br]).astype(np.float64)
                y, lag, corr = align_to(ref, y)
                s = score(ref, y)
                s["seconds"] = round(time.time() - t0, 1)
                s["post_lag"] = lag
                base = result["bitrates"][br]["lossy"]
                s["delta_snr_db"] = s["snr_db"] - base["snr_db"]
                s["delta_multi_mel_snr_db"] = s["multi_mel_snr_db"] - base["multi_mel_snr_db"]
                s["delta_lsd"] = s["lsd"] - base["lsd"]
                s["delta_snr_db_below_16k"] = s["snr_db_below_16k"] - base["snr_db_below_16k"]
                s["helped"] = bool(s["delta_snr_db"] > 0 and s["delta_multi_mel_snr_db"] > 0)
                result["bitrates"][br]["apollo"][variant] = s
                print(f"  {br} + {variant}: SNR {s['snr_db']:.2f} ({s['delta_snr_db']:+.2f}) | "
                      f"<16k {s['snr_db_below_16k']:.2f} ({s['delta_snr_db_below_16k']:+.2f}) | "
                      f"mel {s['delta_multi_mel_snr_db']:+.2f} | LSD {s['delta_lsd']:+.3f} | "
                      f">16k energy {s['energy_above_16k_db']:.1f} vs truth {s['ref_energy_above_16k_db']:.1f} dB", flush=True)
            del model
            import torch

            torch.cuda.empty_cache()
    write_json(out, result)
    print(f"\nwritten {out}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("reference")
    p.add_argument("--bitrates", nargs="+", default=["320k", "128k", "96k"])
    p.add_argument("--variants", nargs="+", default=["mp3_enhancer", "vocal_restore"])
    p.add_argument("--seconds", type=float, default=60.0, help="0 = whole file")
    p.add_argument("--offset", type=float, default=30.0)
    p.add_argument("--out", default="outputs/_benchmarks/apollo_codec.json")
    a = p.parse_args()
    run(a.reference, a.bitrates, a.variants, a.seconds or None, a.offset, Path(a.out))
