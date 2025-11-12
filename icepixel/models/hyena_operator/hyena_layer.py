"""Standalone Hyena operator primitives used by the Hyena2DForecaster."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class OptimModule(nn.Module):
    """Minimal interface mirroring safari's OptimModule register helper."""

    def register(self, name: str, tensor: torch.Tensor, lr: Optional[float] = None, wd: float = 0.0) -> None:
        """Register a tensor with optional optimizer hyper-parameters."""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            param = nn.Parameter(tensor)
            if lr is not None:
                setattr(param, "_optim", {"lr": lr, "weight_decay": wd})
            self.register_parameter(name, param)


class Sin(nn.Module):
    def __init__(self, dim: int, w: float = 10.0, train_freq: bool = True) -> None:
        super().__init__()
        if train_freq:
            self.freq = nn.Parameter(w * torch.ones(1, dim))
        else:
            self.register_buffer("freq", w * torch.ones(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5, **_: float) -> None:
        super().__init__()
        self.seq_len = seq_len

        t = torch.linspace(0, 1, self.seq_len)[None, :, None]
        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)

        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    def __init__(
        self,
        d_model: int,
        fast_decay_pct: float = 0.3,
        slow_decay_pct: float = 1.5,
        target: float = 1e-2,
        modulation_lr: float = 0.0,
        modulate: bool = True,
        shift: float = 0.0,
        **_: float,
    ) -> None:
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


def fftconv(u: torch.Tensor, k: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)

    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


@torch.jit.script
def mul_sum(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (q * y).sum(dim=1)


class HyenaFilter(OptimModule):
    def __init__(
        self,
        d_model: int,
        emb_dim: int = 3,
        order: int = 16,
        fused_fft_conv: bool = False,
        seq_len: int = 1024,
        lr: float = 1e-3,
        lr_pos_emb: float = 1e-5,
        dropout: float = 0.0,
        w: float = 1.0,
        wd: float = 0.0,
        bias: bool = True,
        num_inner_mlps: int = 2,
        normalized: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if emb_dim % 2 == 0 or emb_dim < 3:
            raise ValueError("emb_dim must be odd and >= 3")

        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.normalized = normalized

        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        act = Sin(dim=order, w=w)
        layers = [nn.Linear(emb_dim, order), act]
        for _ in range(num_inner_mlps):
            layers.append(nn.Linear(order, order))
            layers.append(act)
        layers.append(nn.Linear(order, d_model, bias=False))
        self.implicit_filter = nn.Sequential(*layers)

        self.modulation = ExponentialModulation(d_model, **kwargs)

        for layer in self.implicit_filter:
            if isinstance(layer, nn.Linear):
                setattr(layer.weight, "_optim", {"lr": lr, "weight_decay": wd})
                if layer.bias is not None:
                    setattr(layer.bias, "_optim", {"lr": lr, "weight_decay": wd})

    def filter(self, L: int, *args, **kwargs) -> torch.Tensor:
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        L: int,
        k: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if k is None:
            k = self.filter(L)

        k = k[0] if isinstance(k, tuple) else k
        bias_term = bias if bias is not None else self.bias
        y = fftconv(x, k, bias_term)
        return y


class HyenaOperator(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int,
        order: int = 2,
        filter_order: int = 64,
        dropout: float = 0.0,
        filter_dropout: float = 0.0,
        **filter_args,
    ) -> None:
        super().__init__()
        if order < 2:
            raise ValueError("Hyena order must be at least 2")
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)

        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)

        self.short_filter = nn.Conv1d(
            inner_width,
            inner_width,
            kernel_size=3,
            padding=2,
            groups=inner_width,
        )

        self.filter_fn = HyenaFilter(
            d_model * (order - 1),
            order=filter_order,
            seq_len=l_max,
            channels=1,
            dropout=filter_dropout,
            **filter_args,
        )

    def forward(self, u: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u_proj = self.in_proj(u)
        u_proj = rearrange(u_proj, "b l d -> b d l")

        uc = self.short_filter(u_proj)[..., :l_filter]
        *x, v = uc.split(self.d_model, dim=1)

        k = self.filter_fn.filter(l_filter)[0]
        k = rearrange(k, "l (o d) -> o d l", o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, "(o d) -> o d", o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], "b d l -> b l d")
        y = self.out_proj(y)
        return y
