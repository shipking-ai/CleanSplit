"""Per-stem ensemble of local separators, six stems, exactly mixture-consistent.

  vocals          = mean(SW with 3-pass TTA, BS-RoFormer ep317)
  drums, bass     = SW+TTA                                   (default)
                  = mean(SW+TTA, HTDemucs_ft)                (use_demucs=True)
  guitar, piano   = SW+TTA
  other           = mixture - (all other stems)              -> sum of stems == mixture exactly

Evidence (docs/04_results.md sections 7 and 10; real music with known stems, equal weights, nothing tuned):
  default:     vocals +2.14 / +1.83 dB, other group +0.69 / +1.11 dB, drums and bass within +-0.01 dB vs single-pass SW
               on one song (BUH, separator-made reference).
  MUSDB18-HQ, 20 real multitracks (docs/04 sections 11, 14.8): vocals +0.45 dB PAIRED median vs single-pass SW, 18/20
               songs. Much smaller than the BUH number, and the one that counts.
  MDX23C was added as a third vocal model on the strength of five BUH excerpts (section 10) and REMOVED again after
               MUSDB: the two-model average beat the three-model one on 16/20 songs (median +0.09 dB). It remains
               available on its own as `mdx23c_instvoc_hq`.
  use_demucs on MUSDB: drums worse on 20/20 songs (median -0.55 dB), bass worse on 13/20 (median -1.01 dB). Its
               one real win is insurance: on a song where SW misses an instrument entirely (MUSDB "Arise - Run Run
               Run": SW bass 0.8 dB, Demucs 14.5 dB) the average halves the damage. See section 11.
  use_demucs:  additionally drums +2.10 / +1.18 dB, bass +2.74 / +2.02 dB; the reference stems come from an undisclosed
               commercial separator that may be Demucs-like, so these two numbers may be optimistic.
Guitar and piano are not re-estimated: there was no truth to validate a reallocation inside the 'other' group, so any
change in the remainder goes to 'other' only.
"""

from __future__ import annotations

import time

import numpy as np

from .base import SeparationResult, Separator


class EnsembleSeparator(Separator):
    name = "ensemble"
    sample_rate = 44100
    channels = 2
    stems = ("bass", "drums", "other", "vocals", "guitar", "piano")

    def __init__(self, use_demucs: bool = False, device: str = "auto", tta: bool = True, num_overlap: int = 4):
        """num_overlap applies to both RoFormer members; HTDemucs_ft is unaffected (it has its own float overlap).
        4 is the default because it is the best measured: cleaner than 2 on every stem, on 14-19 of 20 MUSDB songs,
        for about 2x the GPU time (docs/04 section 14.5). Pass 2 to halve the time for a sub-0.1 dB loss."""
        from .demucs_sep import HTDemucsSeparator
        from .roformer import BSRoformerSeparator

        self.use_demucs = use_demucs
        self.name = "ensemble_demucs" if use_demucs else "ensemble"
        self.sw = BSRoformerSeparator(model="bs_roformer_sw", device=device, tta=tta, num_overlap=num_overlap)
        self.voc = BSRoformerSeparator(model="bs_roformer_ep317", device=device, num_overlap=num_overlap)
        self.dm = HTDemucsSeparator(device=device) if use_demucs else None

    def cache_key(self) -> dict:
        return {
            **self.describe(),
            "sw": self.sw.cache_key(),
            "vocals": self.voc.cache_key(),
            "demucs": self.dm.cache_key() if self.dm else None,
            "recipe": "vocals=mean(sw,ep317); drums,bass=" + ("mean(sw,demucs)" if self.dm else "sw") + "; other=remainder",
        }

    def _release(self, sep):
        sep._model = None
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    def separate(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        t0 = time.time()
        sw = self.sw.separate(mixture, sample_rate)
        self._release(self.sw)  # one model on the GPU at a time (8 GB budget)
        stems = {k: v.astype(np.float64) for k, v in sw.stems.items()}
        voc = self.voc.separate(mixture, sample_rate)
        self._release(self.voc)
        stems["vocals"] = 0.5 * (stems["vocals"] + voc.stems["vocals"].astype(np.float64))
        meta = {"members": {"sw": sw.metadata, "vocals_model": voc.metadata}}
        if self.dm is not None:
            dm = self.dm.separate(mixture, sample_rate)
            self._release(self.dm)
            for k in ("drums", "bass"):
                stems[k] = 0.5 * (stems[k] + dm.stems[k].astype(np.float64))
            meta["members"]["demucs"] = dm.metadata
        rest = sum(v for k, v in stems.items() if k != "other")
        stems["other"] = mixture.astype(np.float64) - rest
        meta.update(
            recipe=self.cache_key()["recipe"],
            zero_dc=False,  # 'other' absorbs the exact remainder, so the sum equals the unprocessed mixture
            seconds=round(time.time() - t0, 2),
        )
        return SeparationResult({k: v.astype(np.float32) for k, v in stems.items()}, sample_rate, self.name, meta, sw.chunk_starts, sw.chunk_size)
