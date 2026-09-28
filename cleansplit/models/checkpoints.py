"""Local checkpoint discovery and integrity checks.

CleanSplit never downloads or redistributes separator weights. It locates files the user
already has (e.g. inside Ultimate Vocal Remover) and verifies them against pinned hashes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KnownCheckpoint:
    key: str
    filename: str
    config_filename: str
    sha256: str
    size_bytes: int
    provenance: str
    license_note: str


BS_ROFO_SW_FIXED = KnownCheckpoint(
    key="bs_roformer_sw",
    filename="BS-Rofo-SW-Fixed.ckpt",
    config_filename="BS-Rofo-SW-Fixed.yaml",
    sha256="24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e",
    size_bytes=699_412_152,
    provenance=(
        "Community 6-stem BS-RoFormer ('SW'); original host huggingface.co/jarredou/BS-ROFO-SW-Fixed "
        "(account deleted); hash-identical rehost huggingface.co/enerjazzer/BS-ROFO-SW-Fixed"
    ),
    license_note="Weights: no license stated, trainer unknown. Personal/research use; do not redistribute.",
)

BS_ROFORMER_EP317 = KnownCheckpoint(
    key="bs_roformer_ep317",
    filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    config_filename="model_bs_roformer_ep_317_sdr_12.9755.yaml",
    sha256="5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa",
    size_bytes=639_331_213,
    provenance="viperx BS-RoFormer vocals/instrumental (UVR/MSST model zoo); hash-identical at huggingface.co/Politrees/UVR_resources",
    license_note="Community model weights distributed with UVR; no explicit license. Personal/research use.",
)

MDX23C_INSTVOC_HQ = KnownCheckpoint(
    key="mdx23c_instvoc_hq",
    filename="MDX23C-8KFFT-InstVoc_HQ.ckpt",
    config_filename="model_2_stem_full_band_8k.yaml",  # UVR model_data.json maps this checkpoint's hash to it
    sha256="49d51472769e34a2501cd1da782346a3212555c3a5619fc2c53507445528d816",
    size_bytes=448_101_203,
    provenance="MDX23C InstVoc HQ (TFC-TDF-net v3, vocals/instrumental, 8k FFT) from the UVR model zoo",
    license_note="Community model weights distributed with UVR; no explicit license. Personal/research use.",
)

# SCNet XL IHF: four-stem convolutional model, downloaded by the user (not bundled with UVR), so it lives in
# CleanSplit's own models directory rather than a UVR tree. MIT-licensed code AND weights, published as a GitHub
# release asset -- the only separator here whose weights carry an explicit, permissive licence.
SCNET_XL_IHF = KnownCheckpoint(
    key="scnet_xl_ihf",
    filename="model_scnet_ep_36_sdr_10.0891.ckpt",
    config_filename="config_scnet_xl_ihf.yaml",
    sha256="ac25975f0f5704f3d1a3c3c251505b7a0f417a22eafe82773440ee4f7e14b74f",
    size_bytes=214_063_778,
    provenance=("SCNet XL IHF from ZFTurbo/Music-Source-Separation-Training release v1.0.15 "
                "(config_musdb18_scnet_xl_more_wide_v5.yaml + model_scnet_ep_36_sdr_10.0891.ckpt). "
                "Author-reported MUSDB test SDR: drums 11.81, vocals 11.42, bass 9.23, other 7.88."),
    license_note="MIT (MSST repository), weights published as a release asset of that MIT repository.",
)

KNOWN = {BS_ROFO_SW_FIXED.key: BS_ROFO_SW_FIXED, BS_ROFORMER_EP317.key: BS_ROFORMER_EP317}

# Meta HTDemucs fine-tuned bag (MIT code; weights distributed by Meta, bundled with UVR). File name = signature-sha256[:8].
HTDEMUCS_FT_SHA256 = {
    "f7e0c4bc-ba3fe64a.th": "ba3fe64ae8ef66ac9a4857222ce48efbdc5eb3ad375cb79dd13debee5aaa4066",
    "d12395a8-e57c48e6.th": "e57c48e6b0e38af4f7118d7bd08c49f0a0c0edf7d09143bdd902ea0d237303e6",
    "92cfc3b6-ef3bcb9c.th": "ef3bcb9c8b40d14ae5d51b6db2587339cc12c6b77c0be151ce6d69002e087bf2",
    "04573f0d-f3cf25b2.th": "f3cf25b222c4eed7cd49dd8b2c9597d50c18bd154090f7b919cfa5f93cf22c49",
}

# NVIDIA A2SB (huggingface.co/nvidia/audio_to_audio_schrodinger_bridge, ckpt/). NVIDIA OneWay Noncommercial License.
A2SB_SHA256 = {
    "A2SB_onesplit_0.0_1.0_release.ckpt": "fc3dac27c63ce736f5f9e72e12420afb65d63e7899947e6280d132a8f833b8ad",
    "A2SB_twosplit_0.0_0.5_release.ckpt": "07ec74c7c2455383b39cdc3a214faf0857261b8d70cf58b1d351b1363a82b74b",
    "A2SB_twosplit_0.5_1.0_release.ckpt": "92950134c19ea9b74c637aa414d9e192de2913adbf39fefe35a661dbb2357907",
}
A2SB_SIZE_BYTES = 2_262_312_618

# Apollo (github.com/JusperLee/Apollo, CC BY-SA 4.0). Weights ship with UVR 5.6.x under models/Apollo_Models/.
APOLLO = {
    "mp3_enhancer": KnownCheckpoint(
        key="apollo_mp3_enhancer",
        filename="Default_Model.bin",
        config_filename="apollo.yaml",
        sha256="99d9af7f1ff20e63c393035513a655392818d66b4d7fc23d658175c1f15e8d76",
        size_bytes=66_541_845,
        provenance="JusperLee's Apollo MP3 enhancer, trained on 32-128 kbps MP3 damage (paper arXiv:2409.08514); bundled with UVR",
        license_note="CC BY-SA 4.0. Attribution required; derivatives must share alike.",
    ),
    "vocal_restore": KnownCheckpoint(
        key="apollo_vocal_restore",
        filename="Vocal_Restore.ckpt",
        config_filename="apollo.yaml",
        sha256="61b93332da9cde13925b2726b0b31167b05bc20030271b19a583ccf76d69354b",
        size_bytes=66_532_080,
        provenance="Lew's vocal super-resolution fine-tune of Apollo (github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee); bundled with UVR",
        license_note="Derivative of Apollo: CC BY-SA 4.0.",
    ),
}


def find_apollo(key: str = "mp3_enhancer", explicit: str | Path | None = None) -> tuple[Path, KnownCheckpoint]:
    """Locate one of the Apollo checkpoints UVR installs. CleanSplit never downloads it."""
    ck = APOLLO[key]
    searched = []
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p, ck
        searched.append(str(p))
    for root in uvr_models_dirs():
        p = root / "Apollo_Models" / ck.filename
        searched.append(str(p))
        if p.is_file():
            return p, ck
    raise FileNotFoundError(f"{ck.filename} not found. Searched: {searched}. Set CLEANSPLIT_UVR_MODELS or pass a path.")


def uvr_models_dirs() -> list[Path]:
    """Candidate UVR 5.x model directories on this machine (Windows installer default first)."""
    cands = []
    if env := os.environ.get("CLEANSPLIT_UVR_MODELS"):
        cands.append(Path(env))
    if la := os.environ.get("LOCALAPPDATA"):
        cands.append(Path(la) / "Programs" / "Ultimate Vocal Remover" / "models")
    cands.append(Path.home() / "Ultimate Vocal Remover" / "models")
    return [c for c in cands if c.is_dir()]


def cleansplit_models_dir() -> Path:
    """Where CleanSplit keeps weights it downloaded itself (flat: checkpoint and .yaml side by side).

    Separate from the UVR tree on purpose: these are ours to manage, and mixing them into someone else's install
    would make it unclear what may be deleted. Never committed to git (see .gitignore).
    """
    if env := os.environ.get("CLEANSPLIT_MODELS"):
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / "models"


def find_checkpoint(ck: KnownCheckpoint, explicit: str | Path | None = None) -> tuple[Path, Path]:
    """Return (checkpoint, yaml config) paths or raise FileNotFoundError with the places searched."""
    searched = []
    own = cleansplit_models_dir()
    p, cfg = own / ck.filename, own / ck.config_filename
    searched.append(str(p))
    if p.is_file() and cfg.is_file():
        return p, cfg
    if explicit:
        p = Path(explicit)
        cfg = p.with_suffix(".yaml")
        if not cfg.is_file():
            alt = p.parent / "model_data" / "mdx_c_configs" / ck.config_filename
            cfg = alt if alt.is_file() else cfg
        if p.is_file() and cfg.is_file():
            return p, cfg
        searched.append(str(p))
    for root in uvr_models_dirs():
        p = root / "MDX_Net_Models" / ck.filename
        cfg = root / "MDX_Net_Models" / "model_data" / "mdx_c_configs" / ck.config_filename
        searched.append(str(p))
        if p.is_file() and cfg.is_file():
            return p, cfg
    raise FileNotFoundError(
        f"{ck.filename} (+ {ck.config_filename}) not found. Searched: {searched}. "
        "Pass --checkpoint or set CLEANSPLIT_UVR_MODELS."
    )


def _hash_cache_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "CleanSplit"
    base.mkdir(parents=True, exist_ok=True)
    return base / "sha256_cache.json"


def sha256_file(path: Path, use_cache: bool = True) -> str:
    """SHA-256 with a (path, size, mtime) keyed cache so a 700 MB file is hashed once."""
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    cache_file = _hash_cache_path()
    cache = {}
    if use_cache and cache_file.is_file():
        try:
            cache = json.loads(cache_file.read_text())
        except (OSError, json.JSONDecodeError):
            cache = {}
        if key in cache:
            return cache[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    digest = h.hexdigest()
    if use_cache:
        cache[key] = digest
        with contextlib.suppress(OSError):
            cache_file.write_text(json.dumps(cache, indent=1))
    return digest


def verify(ck: KnownCheckpoint, path: Path) -> str:
    size = path.stat().st_size
    if size != ck.size_bytes:
        raise ValueError(f"{path}: size {size} != expected {ck.size_bytes}")
    digest = sha256_file(path)
    if digest != ck.sha256:
        raise ValueError(f"{path}: sha256 {digest} != pinned {ck.sha256}")
    return digest
