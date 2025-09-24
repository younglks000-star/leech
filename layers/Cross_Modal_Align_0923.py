# Cross_Modal_Align.py
from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np

# 새 PE 유틸을 쓰기 위함 (네가 저장한 TS_Pos_Enc_0923.py가 layers/TS_Pos_Enc.py로 import 되도록 경로만 맞추면 됨)
from layers.TS_Pos_Enc_0923 import Transpose, get_activation_fn, positional_encoding_like


# ------------------------------------------------------------
# Cross-modal encoder block (TimeCMA 스타일) - 개선판
#   - Positional Encoding 주입 옵션(use_pe, pe_type, pe_drop, pe_targets)
#   - Residual attention(Realformer) 유지
#   - 게이팅(learnable gate)으로 cross 영향량 제어
#   - d_ff 기본값 개선(4*d_model)
#   - LayerNorm 경로 일원화
#   - LSA(learnable scaling of attention temperature) 옵션 유지
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
        # --- 추가: 포지셔널 인코딩/게이팅 ---
        use_pe: bool = False,
        pe_type: str = 'sincos',
        pe_drop: float = 0.0,
        pe_targets: str = 'qk',  # 'q', 'k', 'v' 포함 조합(ex: 'qk', 'qkv', 'k')
        # 게이팅(어텐션/FFN 잔차에 대한 스칼라 게이트)
        use_gating: bool = True
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

        # PE 주입 옵션
        self.use_pe = use_pe
        self.pe_type = pe_type
        self.pe_targets = pe_targets.lower()
        self.pe_dropout = nn.Dropout(pe_drop) if pe_drop > 0 else nn.Identity()

    def _maybe_add_pe(self, x: Tensor) -> Tensor:
        # x: [B, S, D]
        pe = positional_encoding_like(x, pe=self.pe_type, learn_pe=False)  # [S,D] param
        # param이지만 그래디언트 꺼짐 → constant 취급 / device/dtype 자동 일치
        return x + self.pe_dropout(pe)

    def forward(
        self,
        q: Tensor, k: Tensor, v: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None
    ):
        """
        q: [B, S_q, D], k: [B, S_k, D], v: [B, S_k, D]
        """
        scores = None

        if self.use_pe:
            # 선택된 대상에만 포지셔널 인코딩 더함
            if 'q' in self.pe_targets:
                q = self._maybe_add_pe(q)
            if 'k' in self.pe_targets:
                k = self._maybe_add_pe(k)
            if 'v' in self.pe_targets:
                v = self._maybe_add_pe(v)

        if self.res_attention:
            for mod in self.layers:
                q, scores = mod(q, k, v, prev=scores,
                                key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return q
        else:
            for mod in self.layers:
                q = mod(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return q


class TSTEncoderLayer(nn.Module):
    def __init__(
        self,
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
        use_gating: bool = True
    ):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        if d_k is None: d_k = d_model // n_heads
        if d_v is None: d_v = d_model // n_heads
        if d_ff is None: d_ff = 4 * d_model  # 기본값 강화

        self.res_attention = res_attention
        self.pre_norm = pre_norm
        self.store_attn = store_attn

        # Multi-Head attention
        self.self_attn = _MultiheadAttention(
            d_model, n_heads, d_k, d_v,
            attn_dropout=attn_dropout, proj_dropout=dropout,
            res_attention=res_attention
        )

        # Add & Norm (Attention)
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.LayerNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        # Add & Norm (FFN)
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.LayerNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        # 게이팅: cross 영향량을 안정적으로 조절 (스칼라 파라미터, 초기 1.0)
        self.use_gating = use_gating
        if self.use_gating:
            self.gate_attn = nn.Parameter(torch.tensor(1.0))
            self.gate_ffn = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        q: Tensor, k: Tensor, v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None
    ) -> Tensor:

        # ---------- Attention sublayer ----------
        if self.pre_norm:
            q = self.norm_attn(q)
            k = self.norm_attn(k)
            v = self.norm_attn(v)

        if self.res_attention:
            q2, attn, scores = self.self_attn(
                q, k, v, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        else:
            q2, attn = self.self_attn(
                q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

        if self.store_attn:
            self.attn = attn

        # Residual + (optional) gating
        if self.use_gating:
            # gate는 양수로 제한하기 위해 softplus 사용 (초기값 1 → 거의 통과)
            g = F.softplus(self.gate_attn)
            q = q + self.dropout_attn(g * q2)
        else:
            q = q + self.dropout_attn(q2)

        if not self.pre_norm:
            q = self.norm_attn(q)

        # ---------- FFN sublayer ----------
        if self.pre_norm:
            q = self.norm_ffn(q)

        q2 = self.ff(q)

        if self.use_gating:
            g2 = F.softplus(self.gate_ffn)
            q = q + self.dropout_ffn(g2 * q2)
        else:
            q = q + self.dropout_ffn(q2)

        if not self.pre_norm:
            q = self.norm_ffn(q)

        if self.res_attention:
            return q, scores
        else:
            return q


class _MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        res_attention: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qkv_bias: bool = True,
        lsa: bool = False
    ):
        super().__init__()
        if d_k is None: d_k = d_model // n_heads
        if d_v is None: d_v = d_model // n_heads

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            d_model, n_heads, attn_dropout=attn_dropout,
            res_attention=self.res_attention, lsa=lsa
        )

        # Project output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(
        self,
        Q: Tensor,
        K: Optional[Tensor] = None,
        V: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None
    ):
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear → split heads
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)  # [bs, n_heads, S_q, d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)  # [bs, n_heads, d_k, S_k]
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # [bs, n_heads, S_k, d_v]

        # SDPA
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

        # merge heads
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    - res_attention: Realformer residual attention 지원
    - lsa: learnable scale (temperature) 옵션
    """
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0,
                 res_attention: bool = False, lsa: bool = False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        # 기본 scale은 고정, lsa=True면 학습 가능(temperature)
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        # 추가로 안전하게 스케일을 제한하기 위한 learnable factor
        self.extra_scale = nn.Parameter(torch.tensor(1.0), requires_grad=lsa)

    def forward(
        self,
        q: Tensor,  # [bs, n_heads, S_q, d_k]
        k: Tensor,  # [bs, n_heads, d_k, S_k]
        v: Tensor,  # [bs, n_heads, S_k, d_v]
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None
    ):
        # scores: [bs, n_heads, S_q, S_k]
        eff_scale = self.scale * torch.clamp(self.extra_scale, 1e-2, 50.0)
        attn_scores = torch.matmul(q, k) * eff_scale

        # Realformer residual attention
        if prev is not None:
            attn_scores = attn_scores + prev

        # Mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: [bs, S_k] → broadcast to [bs, 1, 1, S_k]
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # value combine
        output = torch.matmul(attn_weights, v)  # [bs, n_heads, S_q, d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights
