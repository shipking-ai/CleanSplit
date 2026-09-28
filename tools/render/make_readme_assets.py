"""Generate the README's banner and figures. Run from the repo root; writes to docs/assets/.

Everything drawn here comes from measured data in outputs/_benchmarks/ or from the rendered audio in outputs/listen/.
Nothing is illustrative: if a number appears in a figure it appears in docs/04_results.md too.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

INK = {"light": "#0f1116", "dark": "#e6edf3"}
MUTE = {"light": "#5b6472", "dark": "#8b949e"}
ACCENT = "#3fb950"     # green, the "kept" colour
WARN = "#f85149"       # red, the "rejected" colour
COOL = "#58a6ff"       # blue


def banner(theme: str) -> None:
    """Wordmark on the left, one mixture waveform fanning out into four stem lanes on the right.

    Hand-written SVG so it stays crisp at any width and adds no dependency. Two themes, selected in the README by a
    <picture> element, because a fixed-colour banner looks broken for half of GitHub's readers.
    """
    ink, mute = INK[theme], MUTE[theme]
    rng = np.random.default_rng(7)
    W, H = 1280, 300
    SPLIT = 812          # where the mixture ends and the fan begins
    LANE_X0, LANE_X1 = 880, 1232
    LANE_Y = (86, 142, 196, 246)

    def wave(y: float, amp: float, colour: str, x0: float, x1: float, n: int, width: float = 1.6) -> str:
        xs = np.linspace(x0, x1, n)
        env = np.sin(np.linspace(0, np.pi, n)) ** 0.5
        v = np.convolve(rng.standard_normal(n) * amp * env, np.ones(5) / 5, mode="same")
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y + q:.2f}" for i, (x, q) in enumerate(zip(xs, v, strict=True)))
        return f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}" stroke-linecap="round"/>'

    mix = wave(166, 40, ink, 660, SPLIT, 170, 1.8)
    cols = (COOL, ACCENT, "#d29922", mute)
    amps = (22, 16, 12, 8)
    lanes = "".join(wave(y, a, c, LANE_X0, LANE_X1, 210) for y, a, c in zip(LANE_Y, amps, cols, strict=True))
    fan = "".join(
        f'<path d="M{SPLIT + 4},166 C{SPLIT + 40},166 {LANE_X0 - 40},{y} {LANE_X0 - 6},{y}" '
        f'fill="none" stroke="{c}" stroke-width="1.2" opacity="0.55"/>' for y, c in zip(LANE_Y, cols, strict=True))
    names = ("vocals", "drums", "bass", "other")
    tags = "".join(
        f'<text x="{LANE_X1 + 10}" y="{y + 4}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" '
        f'font-size="11" fill="{c}" opacity="0.9">{n}</text>' for y, n, c in zip(LANE_Y, names, cols, strict=True))

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="CleanSplit">
<text x="60" y="150" font-family="ui-sans-serif,-apple-system,Segoe UI,Helvetica,Arial" font-size="78" font-weight="700" fill="{ink}" letter-spacing="-2.5">CleanSplit</text>
<text x="64" y="188" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="15" fill="{mute}">local-first stem separation, measured honestly</text>
<text x="64" y="216" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="13" fill="{ACCENT}">+0.45 dB vocals &#183; 18/20 songs &#183; 7 ideas rejected on the evidence</text>
{mix}{fan}{lanes}{tags}
</svg>'''
    (OUT / f"banner-{theme}.svg").write_text(svg, encoding="utf-8")
    print("wrote", OUT / f"banner-{theme}.svg")


def tiers(theme: str) -> None:
    """Compute cost vs measured vocal gain, from musdb18hq_quality_tiers.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = ROOT / "outputs" / "_benchmarks" / "musdb18hq_quality_tiers.json"
    if not f.is_file():
        print("skip tiers: no benchmark json"); return
    d = json.loads(f.read_text())
    v = d["per_stem"]["vocals"]
    names = ["fast", "balanced", "best"]
    units = [d["units"][n] for n in names]
    gain = [0.0, v["balanced_minus_fast_db"], v["best_minus_fast_db"]]
    ink, mute = INK[theme], MUTE[theme]

    fig, ax = plt.subplots(figsize=(8.6, 3.2), dpi=200)
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    cols = [mute, ACCENT, COOL]
    labels = [n + chr(10) + f"{u} units" for n, u in zip(names, units, strict=True)]
    bars = ax.barh(labels, gain, color=cols, height=0.5, zorder=3)
    for b, g, n in zip(bars, gain, names, strict=True):
        # the baseline bar has zero length, so its annotation goes outside the axes origin rather than inside the bar
        txt = "baseline" if n == "fast" else f"+{g:.3f} dB"
        ax.text(g + 0.012, b.get_y() + b.get_height() / 2, txt, va="center",
                color=mute if n == "fast" else ink, fontsize=11,
                fontweight="normal" if n == "fast" else "bold")
    ax.set_xlabel("vocal SNR gain over `fast`  (paired median, 20 MUSDB18-HQ songs)", color=mute, fontsize=10)
    ax.set_xlim(0, max(gain) * 1.28)
    ax.tick_params(colors=ink, labelsize=11)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(mute)
    ax.grid(axis="x", color=mute, alpha=0.18, zorder=0)
    ax.set_title("balanced costs 2x less than best and keeps 96% of the gain",
                 color=ink, fontsize=12, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / f"tiers-{theme}.png", transparent=True)
    plt.close(fig)
    print("wrote", OUT / f"tiers-{theme}.png")


def spectrograms(theme: str) -> None:
    """Mixture, three separated stems, and the residual error -- from the audio that was actually scored.

    Two details that matter for honesty: every panel shares one colour scale, so brightness is comparable across them,
    and the error panel is the TRUE-SCALE difference from the studio stem, not an amplified copy. It is visibly dimmer
    because it is about 14 dB down; if it looked as bright as the vocal panel, the separation would be worthless.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf

    base = ROOT / "outputs" / "listen" / "Detsky Sad - Walkie Talkie"
    want = [("mixture.wav", "input mixture"), ("best/vocals.wav", "vocals"), ("best/drums.wav", "drums"),
            ("best/bass.wav", "bass"), ("error_best/vocals.wav", "vocal error — 14 dB down")]
    if not all((base / w).is_file() for w, _ in want):
        print("skip spectrograms: rendered audio not found"); return
    ink, mute = INK[theme], MUTE[theme]
    cmap = plt.get_cmap("magma")
    fig, axes = plt.subplots(1, len(want), figsize=(15.0, 2.9), dpi=200)
    fig.patch.set_alpha(0)
    for ax, (rel, title) in zip(axes, want, strict=True):
        x, sr = sf.read(str(base / rel), dtype="float64", always_2d=True)
        # +1e-10 keeps digital silence out of log10(0): without it those bins become -inf, render as NaN, and show
        # through the transparent background as white rectangles.
        m = x.mean(1)[: sr * 10] + 1e-10
        ax.set_facecolor(cmap(0.0))
        ax.specgram(m, NFFT=2048, Fs=sr, noverlap=1536, cmap=cmap,
                    vmin=-135, vmax=-35, mode="magnitude", scale="dB")
        ax.set_title(title, color=ink, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 16000)
        ax.set_yticks([0, 4000, 8000, 12000, 16000])
        ax.set_yticklabels(["0", "4k", "8k", "12k", "16k"] if ax is axes[0] else [""] * 5)
        ax.set_xticks([])
        ax.tick_params(colors=mute, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(mute); sp.set_alpha(0.35)
    axes[0].set_ylabel("Hz", color=mute, fontsize=9)
    fig.text(0.5, 0.015, "same 10 s, one shared colour scale — the last panel is everything the separator got wrong",
             color=mute, fontsize=9.5, ha="center")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(OUT / f"spectrograms-{theme}.png", transparent=True)
    plt.close(fig)
    print("wrote", OUT / f"spectrograms-{theme}.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for theme in ("light", "dark"):
        if which in ("all", "banner"): banner(theme)
        if which in ("all", "tiers"): tiers(theme)
        if which in ("all", "spectrograms"): spectrograms(theme)
