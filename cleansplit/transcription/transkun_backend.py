"""Transkun v2: piano transcription with velocities (Yan & Duan; MIT; pip ``transkun``, weights ship in the wheel).

Loaded the CleanSplit way rather than through ``transkun.transcribe``: the upstream CLI calls ``torch.load`` without
``weights_only`` and loads the state dict with ``strict=False``, which would silently accept a mismatched
checkpoint. Here the checkpoint is SHA-256 pinned, loaded with ``weights_only=True`` and strictly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .notes import TranscribedNote

WEIGHTS_SHA256 = "50a80010effc2a59ffcd068a95cd2b29bd7f23a27a3515bc3ccd209c89a3d44c"  # transkun 2.0.1 pretrained/2.0.pt


class TranskunPiano:
    def __init__(self, device: str = "cuda"):
        import torch
        import transkun
        from transkun.ModelTransformer import TransKun

        pre = Path(transkun.__file__).parent / "pretrained"
        weights, conf_path = pre / "2.0.pt", pre / "2.0.conf"
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest != WEIGHTS_SHA256:
            raise ValueError(f"{weights}: sha256 {digest} != pinned {WEIGHTS_SHA256}")
        conf = TransKun.Config()
        for k, v in json.loads(conf_path.read_text())["Model"]["config"].items():
            setattr(conf, k, v)
        self.torch = torch
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        model = TransKun(conf=conf)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("best_state_dict", state.get("state_dict")), strict=True)
        self.model = model.to(self.device).eval()
        self.fs = int(model.fs)

    def transcribe(self, audio: np.ndarray, sample_rate: int, source: str = "piano") -> list[TranscribedNote]:
        """audio (C, N) at any rate; Transkun wants (N, C) at 44.1 kHz."""
        x = np.asarray(audio, dtype=np.float32)
        if sample_rate != self.fs:
            import soxr

            x = soxr.resample(x.T, sample_rate, self.fs).T
        with self.torch.inference_mode():
            est = self.model.transcribe(self.torch.from_numpy(np.ascontiguousarray(x.T)).to(self.device),
                                        discardSecondHalf=False)
        out = []
        for n in est:
            if n.pitch > 0:  # negative "pitches" are Transkun's pedal events (CC numbers), not notes
                out.append(TranscribedNote(float(n.start), float(n.end), int(n.pitch), "acoustic_piano", False,
                                           int(n.velocity), f"{source}:transkun"))
        return out
