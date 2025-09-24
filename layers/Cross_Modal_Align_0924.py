# Cross_Modal_Align_0923.py
from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np

from layers.TS_Pos_Enc_0923 import Transpose, get_activation_fn, positional_encoding_like


# ------------------------------------------------------------
# Cross-modal encoder block (TimeCMA 스타일 개선판)
# ------------------------------------------------------------
class CrossModal(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: Optional[int] = None,
        norm: str = 'LayerNorm',
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = 'gelu',
        res_attention: bool = False,
        n_layers: int = 1,
        pre_norm: bool = False,
        store_attn: bool = False,
        # --- 추가 ---
        use_pe: bool = False,
        pe_type: str = 'sincos',
        pe_drop: float = 0.0,
        pe_targets: str = 'qk',
        use_gating: bool = True,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TSTEncoderLayer(
                d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                activation=activation, res_attention=res_attention,
                pre_norm=pre_norm, store_attn=store_attn,
                use_gating=use_gating
            )
            for _ in range(n_layers)
        ])
        self.res_attention = res_attention

        # PE
        self.use_pe = use_pe
        self.pe_targets = pe_targets.lower()
        self.pe_dropout = nn.Dropout(pe_drop) if pe_drop > 0 else nn.Identity()

        if self.use_pe:
            # Buffer 등록: forward마다 재생성 방지
            self.register_buffer("pe_cache", None, persistent=False)
            self.pe_type = pe_type

    def _maybe_add_pe(self, x: Tensor) -> Tensor:
        if self.pe_cache is None or self.pe_cache.size(0) < x.size(1):
            self.pe_cache = positional_encoding_like(
                x, pe=self.pe_type, learn_pe=False
            )  # [S,D]
        pe = self.pe_cache[:x.size(1), :].to(x.device, x.dtype)
        return x + self.pe_dropout(pe)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        if self.use_pe:
            if 'q' in self.pe_targets: q = self._maybe_add_pe(q)
            if 'k' in self.pe_targets: k = self._maybe_add_pe(k)
            if 'v' in self.pe_targets: v = self._maybe_add_pe(v)

        scores = None
        if self.res_attention:
            for mod in self.layers:
                q, scores = mod(q, k, v, prev=scores,
                                key_padding_mask=key_padding_mask,
                                attn_mask=attn_mask)
            return q
        else:
            for mod in self.layers:
                q = mod(q, k, v, key_padding_mask=key_padding_mask,
                        attn_mask=attn_mask)
            return q


# ------------------------------------------------------------
# Transformer Encoder Layer with Gating & Realformer
# ------------------------------------------------------------
class TSTEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_k: Optional[int] = None,
                 d_v: Optional[int] = None,
                 d_ff: Optional[int] = 256,
                 store_attn: bool = False,
                 norm: str = 'LayerNorm',
                 attn_dropout: float = 0.0,
                 dropout: float = 0.0,
                 bias: bool = True,
                 activation: str = "gelu",
                 res_attention: bool = False,
                 pre_norm: bool = False,
                 use_gating: bool = True):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        if d_k is None: d_k = d_model // n_heads
        if d_v is None: d_v = d_model // n_heads
        if d_ff is None: d_ff = 4 * d_model

        self.res_attention = res_attention
        self.pre_norm = pre_norm
        self.store_attn = store_attn

        # Attention
        self.self_attn = _MultiheadAttention(
            d_model, n_heads, d_k, d_v,
            attn_dropout=attn_dropout, proj_dropout=dropout,
            res_attention=res_attention
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)

        # FFN
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)

        # 게이팅 (초기값 0.1 → 안정적 시작)
        self.use_gating = use_gating
        if self.use_gating:
            self.gate_attn = nn.Parameter(torch.tensor(0.1))
            self.gate_ffn = nn.Parameter(torch.tensor(0.1))

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        if self.pre_norm:
            q = self.norm_attn(q); k = self.norm_attn(k); v = self.norm_attn(v)

        if self.res_attention:
            q2, attn, scores = self.self_attn(q, k, v, prev,
                                              key_padding_mask=key_padding_mask,
                                              attn_mask=attn_mask)
        else:
            q2, attn = self.self_attn(q, k, v,
                                      key_padding_mask=key_padding_mask,
                                      attn_mask=attn_mask)

        if self.store_attn: self.attn = attn

        # Residual + gating
        if self.use_gating:
            g = F.softplus(self.gate_attn)
            q = q + self.dropout_attn(g * q2)
        else:
            q = q + self.dropout_attn(q2)

        if not self.pre_norm:
            q = self.norm_attn(q)

        # FFN
        if self.pre_norm: q = self.norm_ffn(q)
        q2 = self.ff(q)
        if self.use_gating:
            g2 = F.softplus(self.gate_ffn)
            q = q + self.dropout_ffn(g2 * q2)
        else:
            q = q + self.dropout_ffn(q2)
        if not self.pre_norm: q = self.norm_ffn(q)

        if self.res_attention:
            return q, scores
        else:
            return q


# ------------------------------------------------------------
# Multihead Attention + Realformer
# ------------------------------------------------------------
class _MultiheadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_k: Optional[int] = None,
                 d_v: Optional[int] = None,
                 res_attention: bool = False,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0,
                 qkv_bias: bool = True,
                 lsa: bool = False):
        super().__init__()
        if d_k is None: d_k = d_model // n_heads
        if d_v is None: d_v = d_model // n_heads
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            d_model, n_heads, attn_dropout=attn_dropout,
            res_attention=res_attention, lsa=lsa
        )
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model),
            nn.Dropout(proj_dropout)
        )

    def forward(self, Q: Tensor, K: Optional[Tensor] = None,
                V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s, k_s, v_s, prev, key_padding_mask, attn_mask
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s, key_padding_mask, attn_mask
            )

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        return (output, attn_weights, attn_scores) if self.res_attention else (output, attn_weights)


# ------------------------------------------------------------
# Scaled Dot-Product Attention
# ------------------------------------------------------------
class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 attn_dropout: float = 0.0,
                 res_attention: bool = False,
                 lsa: bool = False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.extra_scale = nn.Parameter(torch.tensor(0.5), requires_grad=lsa)  # 초기값 0.5 안정화

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        eff_scale = F.softplus(self.scale * self.extra_scale)
        attn_scores = torch.matmul(q, k) * eff_scale
        if prev is not None: attn_scores = attn_scores + prev

        if attn_mask is not None:
            attn_scores += attn_mask if attn_mask.dtype != torch.bool else attn_scores.masked_fill(attn_mask, -np.inf)

        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return (output, attn_weights, attn_scores) if self.res_attention else (output, attn_weights)
