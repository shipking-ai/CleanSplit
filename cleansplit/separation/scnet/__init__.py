"""SCNet (Sparse Compression Network), vendored for the SCNet XL IHF four-stem model.

Upstream: ZFTurbo/Music-Source-Separation-Training @ a8a862231de29e6832a26c590e5c29f82f736f66,
models/scnet/ (MIT, see ../bs_roformer/LICENSE_MSST.txt). Architecture by the SCNet authors
(arXiv:2401.13276), MSST implementation by ZFTurbo.

Vendored UNCHANGED, byte for byte, so the numerics stay upstream-exact -- CleanSplit's wrapper in ../scnet_sep.py
supplies the config and the overlap-add, exactly as it does for BS-RoFormer and MDX23C. Why SCNet is here: it is
CONVOLUTIONAL, not a band-split transformer, and it is the only four-stem model available whose errors should be
uncorrelated with SW's for that reason. Averaging only cancels artifacts between architecturally independent members
(docs/01 section 8.1, docs/04 section 14.1).
"""
from .scnet import SCNet
