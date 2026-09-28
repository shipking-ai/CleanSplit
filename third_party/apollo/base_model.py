# Vendored from https://github.com/JusperLee/Apollo @ e84bcacc59d5455f05d86a5c97dd4aeb3c14dbb6 (2026-08-24)
# License: CC BY-SA 4.0 (see LICENSE in this folder). Author: Kai Li (JusperLee).
# CleanSplit deviations, deliberately minimal so the numerics stay upstream-exact:
#   base_model.py: the huggingface_hub PyTorchModelHubMixin base and from_pretrain/serialize helpers are removed.
#     CleanSplit never downloads weights; it loads the checkpoint the user already has and builds the module itself.
#   apollo.py: the constructor's debug print of the band layout is removed.
# Nothing else is changed: the layers, the STFT, the band split and the forward pass are upstream code.

import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, sample_rate, in_chan=1):
        super().__init__()
        self._sample_rate = sample_rate
        self._in_chan = in_chan

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def sample_rate(self):
        return self._sample_rate
