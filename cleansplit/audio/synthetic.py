"""Deterministic synthetic six-stem songs with known ground truth.

These are crude instrument caricatures, not realistic music. Their purpose is to give the analysis pipeline
signals with the right *structure* (harmonic + vibrato voice, broadband transients, low sustained bass,
inharmonic decaying piano partials, plucked guitar, slow pad) so injected corruptions can be scored exactly.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

SR = 44100
STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")


def _env(n, sr, attack, release):
    e = np.ones(n)
    a = min(int(attack * sr), n)
    r = min(int(release * sr), n - a)
    if a:
        e[:a] = np.linspace(0, 1, a)
    if r:
        e[n - r :] = np.linspace(1, 0, r)
    return e


def _pan(x, p):
    """p in [-1, 1]; constant-power."""
    th = (p + 1) * np.pi / 4
    return np.stack([np.cos(th) * x, np.sin(th) * x])


def _place(buf, start, sig):
    s = int(start)
    if s >= buf.shape[-1]:
        return
    e = min(buf.shape[-1], s + sig.shape[-1])
    buf[..., s:e] += sig[..., : e - s]


def _cents(f, c):
    return f * 2 ** (c / 1200)


def vocals(n, sr, rng):
    out = np.zeros(n)
    notes = [_cents(f, 7) for f in (220.0, 246.9, 261.6, 293.7, 261.6, 246.9, 220.0, 196.0)]
    dur, gap = 0.6, 0.12
    t_pos = 0.25
    formants = [(700, 130), (1220, 160), (2600, 250)]
    for i, f0 in enumerate(notes * 4):
        m = int(dur * sr)
        t = np.arange(m) / sr
        vib = 1 + (2 ** (30 / 1200) - 1) * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.15) / 0.1, 0, 1)
        phase = 2 * np.pi * np.cumsum(f0 * vib) / sr
        sig = np.zeros(m)
        for k in range(1, 40):
            fk = k * f0
            if fk > 12000:
                break
            amp = sum(np.exp(-0.5 * ((fk - fc) / bw) ** 2) for fc, bw in formants) + 0.05 / k
            sig += amp * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
        breath = rng.standard_normal(m) * 0.02
        sig = sig + sosfilt(butter(4, 3000, "highpass", fs=sr, output="sos"), breath)
        _place(out, t_pos * sr, sig * _env(m, sr, 0.04, 0.08))
        t_pos += dur + gap
        if t_pos * sr >= n:
            break
    return _pan(out, 0.0)


def drums(n, sr, rng):
    L = np.zeros((2, n))
    beat = 0.5
    hp = butter(4, 7000, "highpass", fs=sr, output="sos")
    bp = butter(2, [1000, 8000], "bandpass", fs=sr, output="sos")
    k = 0
    while k * beat / 2 * sr < n:
        t0 = k * beat / 2 * sr
        if k % 4 == 0:
            m = int(0.35 * sr)
            t = np.arange(m) / sr
            f = 50 + 100 * np.exp(-t / 0.04)
            kick = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.12)
            _place(L, t0, _pan(kick, 0.0))
        if k % 4 == 2:
            m = int(0.25 * sr)
            t = np.arange(m) / sr
            snare = sosfilt(bp, rng.standard_normal(m)) * np.exp(-t / 0.07) * 0.8 + np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05) * 0.4
            _place(L, t0, _pan(snare, 0.05))
        m = int(0.08 * sr)
        t = np.arange(m) / sr
        hat = sosfilt(hp, rng.standard_normal(m)) * np.exp(-t / 0.02) * 0.5
        _place(L, t0, _pan(hat, 0.3))
        k += 1
    return L


def bass(n, sr, rng):
    out = np.zeros(n)
    notes = [_cents(f, -4) for f in (55.0, 73.42, 82.41, 65.41)]
    for i in range(int(np.ceil(n / sr))):
        f0 = notes[i % len(notes)]
        m = int(0.95 * sr)
        t = np.arange(m) / sr
        sig = sum(np.sin(2 * np.pi * k * f0 * t) / k ** 1.3 for k in range(1, int(2500 / f0)))
        _place(out, i * sr, sig * np.exp(-t / 0.8) * _env(m, sr, 0.005, 0.05))
    return _pan(out, 0.0)


def piano(n, sr, rng):
    out = np.zeros(n)
    chords = [tuple(_cents(f, 3) for f in c) for c in ((261.6, 329.6, 392.0), (220.0, 261.6, 329.6), (174.6, 220.0, 261.6), (196.0, 246.9, 293.7))]
    B = 0.0004
    for i in range(int(np.ceil(n / (2 * sr)))):
        m = int(1.9 * sr)
        t = np.arange(m) / sr
        sig = np.zeros(m)
        for f0 in chords[i % len(chords)]:
            for k in range(1, 16):
                fk = k * f0 * np.sqrt(1 + B * k * k)
                if fk > 16000:
                    break
                sig += np.sin(2 * np.pi * fk * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t * (1.2 + 0.35 * k)) / k**1.2
        _place(out, (2 * i + 0.1) * sr, sig * _env(m, sr, 0.003, 0.05))
    return _pan(out, -0.35)


def guitar(n, sr, rng):
    out = np.zeros(n)
    arp = [_cents(f, -9) for f in (164.8, 196.0, 246.9, 329.6, 246.9, 196.0)]
    step = 0.25
    j = 0
    while (1.0 + j * step) * sr < n:
        f0 = arp[j % len(arp)]
        m = int(0.6 * sr)
        t = np.arange(m) / sr
        pos = 0.18
        sig = sum(
            np.sin(np.pi * k * pos) / k * np.sin(2 * np.pi * k * f0 * t) * np.exp(-t * (2.5 + 0.6 * k))
            for k in range(1, int(9000 / f0))
        )
        _place(out, (1.0 + j * step) * sr, sig * _env(m, sr, 0.002, 0.03))
        j += 1
    return _pan(out, 0.4)


def other(n, sr, rng):
    t = np.arange(n) / sr
    sig = np.zeros(n)
    for f0 in (_cents(130.8, 13), _cents(164.8, 13), _cents(196.0, 13)):
        for det in (-0.004, 0.004):
            ph = rng.uniform(0, 2 * np.pi)
            sig += sum(np.sin(2 * np.pi * k * f0 * (1 + det) * t + ph * k) / k for k in range(1, int(3000 / f0)))
    sig = sosfilt(butter(2, 2500, "lowpass", fs=sr, output="sos"), sig)
    sig *= np.clip(t / 0.8, 0, 1) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.15 * t))
    side = sosfilt(butter(2, 2500, "lowpass", fs=sr, output="sos"), rng.standard_normal(n)) * 0.02
    return np.stack([sig + side, sig - side])


_GEN = {"vocals": vocals, "drums": drums, "bass": bass, "guitar": guitar, "piano": piano, "other": other}
_TARGET_RMS_DB = {"vocals": -18.0, "drums": -19.0, "bass": -20.0, "guitar": -24.0, "piano": -24.0, "other": -26.0}


def make_song(duration_s: float = 8.0, seed: int = 0, sr: int = SR) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Returns (mixture (2, N) float32, stems dict (2, N) float32) with mixture == sum(stems) exactly in float64
    and peak below -1 dBFS."""
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    stems = {}
    for name in STEMS:
        x = _GEN[name](n, sr, rng)
        rms = np.sqrt(np.mean(x**2)) + 1e-12
        stems[name] = x * (10 ** (_TARGET_RMS_DB[name] / 20) / rms)
    mix = sum(stems.values())
    peak = np.max(np.abs(mix))
    g = min(1.0, 10 ** (-1 / 20) / peak)
    stems = {k: (v * g).astype(np.float32) for k, v in stems.items()}
    mix = np.zeros((2, n), dtype=np.float64)
    for v in stems.values():
        mix += v.astype(np.float64)
    return mix.astype(np.float32), stems
