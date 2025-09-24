# TS_Pos_Enc.py
__all__ = [
    'Transpose', 'get_activation_fn',
    'moving_avg', 'series_decomp',
    'PositionalEncoding', 'SinCosPosEncoding',
    'Coord2dPosEncoding', 'Coord1dPosEncoding',
    'positional_encoding',
    # 추가 유틸
    'positional_encoding_like', 'sincos_periodic', 'build_time_fourier',
    'make_node_posenc', 'PositionalEncoding1D'
]

import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor


# ---------------------------
# helpers
# ---------------------------
def _as_like_device_dtype(ref: Tensor, x: Tensor) -> Tensor:
    """x를 ref의 device/dtype으로 맞춰 반환."""
    return x.to(device=ref.device, dtype=ref.dtype)


def _maybe_to(x: Tensor, device=None, dtype=None) -> Tensor:
    if device is None and dtype is None:
        return x
    return x.to(device=device, dtype=dtype)


def pv(msg: str, verbose: bool = False):
    if verbose:
        print(msg)


# ---------------------------
# basic modules
# ---------------------------
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous: bool = False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x: Tensor) -> Tensor:
        out = x.transpose(*self.dims)
        return out.contiguous() if self.contiguous else out


def get_activation_fn(activation):
    if callable(activation): 
        return activation()
    a = activation.lower()
    if a == "relu": return nn.ReLU()
    if a == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. Use "relu", "gelu", or a callable.')


class moving_avg(nn.Module):
    """
    1D 이동평균(AvgPool1d)로 추세를 추출.
    입력 x: [B, L, C]
    """
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        assert kernel_size > 0 and stride > 0
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        B, L, C = x.shape
        # 양끝 패딩(복제)
        pad = (self.kernel_size - 1) // 2
        if pad > 0:
            front = x[:, 0:1, :].repeat(1, pad, 1)
            end = x[:, -1:, :].repeat(1, pad, 1)
            x = torch.cat([front, x, end], dim=1)
        # AvgPool1d는 [B, C, L] 형식을 기대 → 변환 후 되돌리기
        x = self.avg(x.permute(0, 2, 1))   # [B, C, L+pad*2] -> [B, C, L]
        x = x.permute(0, 2, 1)             # [B, L, C]
        return x


class series_decomp(nn.Module):
    """
    시계열 분해: x = residual + trend
    입력/출력: [B, L, C]
    """
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ---------------------------
# positional encodings
# ---------------------------
def PositionalEncoding(q_len: int, d_model: int, normalize: bool = True,
                       device=None, dtype=None) -> Tensor:
    """
    표준 Sin/Cos 1D 위치 인코딩. 반환: [q_len, d_model]
    """
    pe = torch.zeros(q_len, d_model, device=device, dtype=dtype)
    position = torch.arange(0, q_len, device=device, dtype=dtype).unsqueeze(1)  # [q_len, 1]
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-(math.log(10000.0) / d_model))
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10 + 1e-6)
    return pe


SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(q_len: int, d_model: int, exponential: bool = False,
                       normalize: bool = True, eps: float = 1e-3, verbose: bool = False,
                       device=None, dtype=None) -> Tensor:
    """
    2D 격자형 좌표 기반 위치 인코딩(실험용). 반환: [q_len, d_model]
    """
    x = .5 if exponential else 1.0
    cpe = None
    for i in range(100):
        v1 = torch.linspace(0, 1, q_len, device=device, dtype=dtype).reshape(-1, 1) ** x
        v2 = torch.linspace(0, 1, d_model, device=device, dtype=dtype).reshape(1, -1) ** x
        cpe = 2 * (v1 * v2) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        m = cpe.mean().item()
        if abs(m) <= eps: 
            break
        x += .001 if m > eps else -.001
    if cpe is None:
        cpe = torch.zeros(q_len, d_model, device=device, dtype=dtype)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10 + 1e-6)
    return cpe


def Coord1dPosEncoding(q_len: int, exponential: bool = False, normalize: bool = True,
                       device=None, dtype=None) -> Tensor:
    """
    1D 좌표 기반 위치 인코딩. 반환: [q_len, 1]
    """
    power = 0.5 if exponential else 1.0
    cpe = 2 * (torch.linspace(0, 1, q_len, device=device, dtype=dtype).reshape(-1, 1) ** power) - 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10 + 1e-6)
    return cpe


def positional_encoding(pe, learn_pe: bool, q_len: int, d_model: int,
                        device=None, dtype=None) -> nn.Parameter:
    """
    다양한 방식의 위치 인코딩을 생성해 nn.Parameter로 반환.
    """
    if pe is None:
        W_pos = torch.empty((q_len, d_model), device=device, dtype=dtype)
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1), device=device, dtype=dtype)
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model), device=device, dtype=dtype)
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe in ('normal', 'gauss'):
        W_pos = torch.zeros((q_len, 1), device=device, dtype=dtype)
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1), device=device, dtype=dtype)
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True, device=device, dtype=dtype)
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True, device=device, dtype=dtype)
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, device=device, dtype=dtype)
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True,  normalize=True, device=device, dtype=dtype)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True, device=device, dtype=dtype)
    else:
        raise ValueError(
            f"{pe} is not a valid pe. "
            "Choose from: 'gauss'/'normal', 'zeros', 'zero', 'uniform', "
            "'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', or None."
        )
    return nn.Parameter(W_pos, requires_grad=learn_pe)


# ---------------------------
# extra utilities (멀티모달용)
# ---------------------------
def positional_encoding_like(x: Tensor, pe: str = 'sincos', learn_pe: bool = False) -> nn.Parameter:
    """
    입력 x: [B, S, D] 또는 [S, D]를 받아 같은 S,D의 위치 인코딩 파라미터를 반환.
    배치에 더해서 쓰기 쉬움.
    """
    if x.dim() == 3:
        _, S, D = x.shape
    elif x.dim() == 2:
        S, D = x.shape
    else:
        raise ValueError(f"Expected 2D/3D tensor, got {x.shape}")
    return positional_encoding(pe, learn_pe, q_len=S, d_model=D, device=x.device, dtype=x.dtype)


def sincos_periodic(t: Tensor, periods: Tensor) -> Tensor:
    """
    t: [..., 1] 시간스칼라(예: 일 단위 float), periods: [K] 주기(예: 7, 30.4, 91.3, 365.25)
    반환: [..., 2K]  (sin, cos concat)
    """
    # 방송가능한 형태로 변환
    t = t.unsqueeze(-1)  # [..., 1] -> [..., 1]
    periods = periods.to(device=t.device, dtype=t.dtype).view(*([1] * (t.dim() - 1)), -1)  # [..., K]
    ang = 2 * math.pi * t / (periods + 1e-6)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [..., 2K]


def build_time_fourier(L: int, device=None, dtype=None,
                       periods=(7.0, 30.4375, 91.3125, 365.25)) -> Tensor:
    """
    길이 L에 대해 여러 주기의 사인/코사인 포지션 피처 반환: [L, 2K]
    """
    t = torch.arange(L, device=device, dtype=dtype).unsqueeze(-1)  # [L,1]
    P = torch.tensor(periods, device=device, dtype=dtype)          # [K]
    return sincos_periodic(t, P)                                   # [L,2K]


def make_node_posenc(latlon: Tensor, d_model: int, method: str = 'linear') -> Tensor:
    """
    노드(클러스터) 좌표(lat, lon 정규화 [-1,1] 또는 [0,1])로부터 d_model 차원 포지션 반환.
    latlon: [N, 2]  (lat, lon)
    method: 'linear' | 'sincos'
    반환: [N, d_model]
    """
    N, two = latlon.shape
    assert two == 2, "latlon must be [N,2]"
    latlon = latlon.to(dtype=torch.float32)  # 내부 계산은 float32

    if method == 'sincos':
        # 2차원 사인코사인 확장 후 선형사상
        # [N, 4] -> Linear -> [N, d_model]
        phi = torch.cat([torch.sin(latlon), torch.cos(latlon)], dim=-1)  # [N,4]
        proj = nn.Linear(4, d_model, bias=False)
        with torch.no_grad():
            nn.init.xavier_uniform_(proj.weight)
        return proj(phi)
    else:
        # 단순 선형사상: [N,2] -> [N,d_model]
        proj = nn.Linear(2, d_model, bias=False)
        with torch.no_grad():
            nn.init.xavier_uniform_(proj.weight)
        return proj(latlon)


class PositionalEncoding1D(nn.Module):
    """
    배치 입력에 쉽게 더할 수 있는 1D learnable PE.
    입력 x: [B, S, D] → x + PE(S,D)
    """
    def __init__(self, max_len: int, d_model: int, init: str = 'sincos', learn_pe: bool = True):
        super().__init__()
        if init == 'sincos':
            pe = PositionalEncoding(max_len, d_model, normalize=True)
        elif init == 'zeros':
            pe = torch.zeros(max_len, d_model)
        else:
            pe = torch.empty(max_len, d_model)
            nn.init.uniform_(pe, -0.02, 0.02)
        self.pe = nn.Parameter(pe, requires_grad=learn_pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B,S,D]
        B, S, D = x.shape
        return x + self.pe[:S, :].to(device=x.device, dtype=x.dtype)
