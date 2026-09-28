"""BS-RoFormer network (inference-only vendored copy).

Upstream: ZFTurbo/Music-Source-Separation-Training @ 050cae7345f4ac1e1e27e066c2c5cdc0a2cdb679
          models/bs_roformer/bs_roformer.py + attend.py  (MIT, see LICENSE_MSST.txt)
Architecture originally by Phil Wang (lucidrains/BS-RoFormer, MIT) after
Lu et al., "Music Source Separation with Band-Split RoPE Transformer", ICASSP 2024.

Changes vs upstream (module/parameter names are unchanged so checkpoints load strictly):
  * beartype decorators/imports removed (typing only).
  * Attention uses torch.nn.attention.sdpa_kernel when available; flash kernels are never
    requested on compute capability < 8.0 or on Windows (same policy as upstream).
  * PoPE positional embedding support removed (not used by any checkpoint CleanSplit loads).
  * Training loss branch removed; ``forward`` returns separated audio only.
  * ``forward_masks`` added: returns the complex masks, needed for exact mask-sum diagnostics.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from torch.nn import Module, ModuleList

try:  # torch >= 2.3
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _HAS_SDPA_KERNEL = True
except Exception:  # pragma: no cover - old torch
    _HAS_SDPA_KERNEL = False


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


def _cuda_backends() -> list:
    """SDPA backends allowed on the current CUDA device (upstream policy)."""
    if not torch.cuda.is_available():
        return []
    major, _minor = torch.cuda.get_device_capability()
    if major >= 8 and os.name != "nt":
        return [SDPBackend.FLASH_ATTENTION]
    return [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class Attend(nn.Module):
    def __init__(self, dropout=0.0, flash=False, scale=None):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.flash = flash

    def flash_attn(self, q, k, v):
        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)
        dropout_p = self.dropout if self.training else 0.0
        if _HAS_SDPA_KERNEL and q.is_cuda:
            with sdpa_kernel(_cuda_backends()):
                return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

    def forward(self, q, k, v):
        scale = default(self.scale, q.shape[-1] ** -0.5)
        if self.flash:
            return self.flash_attn(q, k, v)
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
        attn = self.attn_dropout(sim.softmax(dim=-1))
        return torch.einsum("b h i j, b h j d -> b h i d", attn, v)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, rotary_embed=None, flash=True):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head
        self.rotary_embed = rotary_embed
        self.attend = Attend(flash=flash, dropout=dropout)
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(dim_inner, dim, bias=False), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        q, k, v = rearrange(self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)
        out = self.attend(q, k, v)
        gates = self.to_gates(x)
        out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LinearAttention(Module):
    """Linear attention (El-Nouby et al., arXiv:2106.09681)."""

    def __init__(self, *, dim, dim_head=32, heads=8, scale=8, flash=False, dropout=0.0):
        super().__init__()
        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h d n", qkv=3, h=heads),
        )
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.attend = Attend(scale=scale, dropout=dropout, flash=flash)
        self.to_out = nn.Sequential(Rearrange("b h d n -> b n (h d)"), nn.Linear(dim_inner, dim, bias=False))

    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.to_qkv(x)
        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()
        return self.to_out(self.attend(q, k, v))


class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        flash_attn=True,
        linear_attn=False,
    ):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            if linear_attn:
                attn = LinearAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=flash_attn)
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    rotary_embed=rotary_embed,
                    flash=flash_attn,
                )
            self.layers.append(ModuleList([attn, FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)]))
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class BandSplit(Module):
    def __init__(self, dim, dim_inputs: Tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([nn.Sequential(RMSNorm(d), nn.Linear(d, dim)) for d in dim_inputs])

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)
        return torch.stack([f(s) for s, f in zip(x, self.to_features)], dim=-2)


def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh):
    dim_hidden = default(dim_hidden, dim_in)
    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)
    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        net.append(nn.Linear(layer_dim_in, layer_dim_out))
        if ind == len(dims) - 2:
            continue
        net.append(activation())
    return nn.Sequential(*net)


class MaskEstimator(Module):
    def __init__(self, dim, dim_inputs: Tuple[int, ...], depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs
        dim_hidden = dim * mlp_expansion_factor
        self.to_freqs = ModuleList(
            [
                nn.Sequential(MLP(dim, d * 2, dim_hidden=dim_hidden, depth=depth), nn.GLU(dim=-1))
                for d in dim_inputs
            ]
        )

    def forward(self, x):
        x = x.unbind(dim=-2)
        return torch.cat([mlp(b) for b, mlp in zip(x, self.to_freqs)], dim=-1)


DEFAULT_FREQS_PER_BANDS = (
    (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)
)


class BSRoformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        linear_transformer_depth=0,
        freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        flash_attn=True,
        dim_freqs_in=1025,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: Optional[Callable] = None,
        zero_dc=True,
        mask_estimator_depth=2,
        mlp_expansion_factor=4,
        use_torch_checkpoint=False,
        skip_connection=False,
        **_training_only_kwargs,
    ):
        super().__init__()
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.skip_connection = skip_connection
        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False,
        )
        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            tran_modules = []
            if linear_transformer_depth > 0:
                tran_modules.append(Transformer(depth=linear_transformer_depth, linear_attn=True, **transformer_kwargs))
            tran_modules.append(Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs))
            tran_modules.append(Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs))
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = RMSNorm(dim)
        self.stft_kwargs = dict(
            n_fft=stft_n_fft, hop_length=stft_hop_length, win_length=stft_win_length, normalized=stft_normalized
        )
        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        freqs = stft_n_fft // 2 + 1
        assert len(freqs_per_bands) > 1
        assert sum(freqs_per_bands) == freqs, f"bands must sum to {freqs}, got {sum(freqs_per_bands)}"

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)
        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)
        self.mask_estimators = nn.ModuleList(
            [
                MaskEstimator(
                    dim=dim,
                    dim_inputs=freqs_per_bands_with_complex,
                    depth=mask_estimator_depth,
                    mlp_expansion_factor=mlp_expansion_factor,
                )
                for _ in range(num_stems)
            ]
        )
        self.zero_dc = zero_dc

    def _stft(self, raw_audio):
        """(b, s, t) -> complex (b, f*s, t) with frequency leading, channels interleaved per bin."""
        device = raw_audio.device
        raw_audio, ps = pack_one(raw_audio, "* t")
        window = self.stft_window_fn(device=device)
        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)
        stft_repr = unpack_one(stft_repr, ps, "* f t c")
        return rearrange(stft_repr, "b s f t c -> b (f s) t c")

    def forward_masks(self, raw_audio):
        """Return (mixture_stft_real [b, f*s, t, 2], complex masks [b, n, f*s, t])."""
        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")
        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), "channel mismatch with model"

        stft_repr = self._stft(raw_audio)
        x = rearrange(stft_repr, "b f t c -> b t (f c)")
        x = self.band_split(x)

        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):
            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block
                x, ft_ps = pack([x], "b * d")
                x = linear_transformer(x)
                (x,) = unpack(x, ft_ps, "b * d")
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_transformer(x)
            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_transformer(x)
            (x,) = unpack(x, ps, "* f d")

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)
        mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, "b n t (f c) -> b n f t c", c=2)
        return stft_repr, torch.view_as_complex(mask.float().contiguous())

    def forward(self, raw_audio):
        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")
        length = raw_audio.shape[-1]
        stft_repr, mask = self.forward_masks(raw_audio)

        stft_repr = torch.view_as_complex(stft_repr.float().contiguous())
        stft_repr = rearrange(stft_repr, "b f t -> b 1 f t") * mask
        stft_repr = rearrange(stft_repr, "b n (f s) t -> (b n s) f t", s=self.audio_channels)
        if self.zero_dc:
            stft_repr[:, 0] = 0.0

        window = self.stft_window_fn(device=raw_audio.device)
        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=window, return_complex=False, length=length)
        return rearrange(recon_audio, "(b n s) t -> b n s t", s=self.audio_channels, n=self.num_stems)
