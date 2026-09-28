"""Non-generative restorers. They are the controls the generative experiment must beat."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from ..audio.stft import istft, stft
from ..audio.tf_edit import TFBox, box_mask, excerpt_bounds
from .base import RestorationContext, RestorationProposal, Restorer


def _excerpt_specs(ctx: RestorationContext, box: TFBox, margin_s: float = 0.5):
    a, b = excerpt_bounds(ctx.mixture.shape[-1], ctx.grid, box, margin_s)
    O = stft(ctx.mixture[..., a:b].astype(np.float64), ctx.grid, dtype=np.complex128)
    S = {k: stft(v[..., a:b].astype(np.float64), ctx.grid, dtype=np.complex128) for k, v in ctx.stems.items()}
    mask = box_mask(ctx.grid, O.shape[-1], box, frame_offset=a // ctx.grid.hop)
    return a, b, O, S, mask


def _box(r):
    return TFBox(r.start_s, r.end_s, r.freq_low_hz, r.freq_high_hz)


class IdentityRestorer(Restorer):
    """Control: proposes no change."""

    name = "identity"

    def restore(self, stem, regions, context):
        out = []
        for r in regions:
            a, b = excerpt_bounds(context.mixture.shape[-1], context.grid, _box(r))
            out.append(RestorationProposal(r, {}, (a, b), self.name, ["no-op control"]))
        return out


def _weights(S: dict, keys, smooth=(5, 9)):
    P = {k: uniform_filter((np.abs(S[k]) ** 2).sum(axis=0), size=smooth, mode="nearest") for k in keys}
    tot = sum(P.values()) + 1e-20
    return {k: P[k] / tot for k in keys}


class ResidualReallocationRestorer(Restorer):
    """Inside the region only, give each stem its energy-proportional share of the mixture residual E = O_ref - R
    (mixture-consistency projection, Wisdom et al. 2019, restricted to the region). Recovers missing energy; cannot
    fix misassignment between stems."""

    name = "residual_reallocation"

    def restore(self, stem, regions, context):
        out = []
        for r in regions:
            a, b, O, S, mask = _excerpt_specs(context, _box(r))
            E = O - sum(S.values())
            keys = list(S) if stem == "mixture" else [stem]
            W = _weights(S, list(S))
            new = {}
            for k in keys:
                new_spec = S[k] + W[k] * E
                new[k] = istft(new_spec, context.grid, b - a)
            out.append(RestorationProposal(r, new, (a, b), self.name))
        return out


class RegionWienerRestorer(Restorer):
    """Inside the region, re-estimate ALL stems as real, non-negative, temporally smoothed Wiener shares of O_ref:
    S_i' = w_i * O_ref with w_i = P_i / sum_j P_j (smoothed ~100 ms). Removes inter-stem cancellation, missing
    energy and fast mask flutter by construction; also blurs genuine fast detail. Cannot fix leakage."""

    name = "region_wiener"

    def restore(self, stem, regions, context):
        out = []
        for r in regions:
            a, b, O, S, mask = _excerpt_specs(context, _box(r))
            W = _weights(S, list(S), smooth=(5, 9))
            new = {k: istft(W[k] * O, context.grid, b - a) for k in S}
            out.append(RestorationProposal(r, new, (a, b), self.name))
        return out


RESTORERS = {c.name: c for c in (IdentityRestorer, ResidualReallocationRestorer, RegionWienerRestorer)}


def _a2sb(**kw):
    from .a2sb import A2SBRestorer  # lazy: needs torch, NVIDIA code and weights

    return A2SBRestorer(**kw)


def _a2sb_consistent(**kw):
    from .a2sb import A2SBConsistentRestorer

    return A2SBConsistentRestorer(**kw)


def _apollo(**kw):
    from .apollo import ApolloRestorer  # lazy: needs torch and the Apollo weights UVR installs

    return ApolloRestorer(**kw)


RESTORERS["a2sb"] = _a2sb
RESTORERS["a2sb_consistent"] = _a2sb_consistent
RESTORERS["apollo"] = _apollo


def create(name: str, **kwargs) -> Restorer:
    """Restorer names may carry options, e.g. 'a2sb:n_steps=25,precision=fp32'."""
    if ":" in name:
        name, opts = name.split(":", 1)
        for kv in filter(None, opts.split(",")):
            k, v = kv.split("=")
            kwargs[k] = int(v) if v.lstrip("-").isdigit() else v
    if name not in RESTORERS:
        raise KeyError(f"unknown restorer '{name}'; available {sorted(RESTORERS)}")
    return RESTORERS[name](**kwargs)
