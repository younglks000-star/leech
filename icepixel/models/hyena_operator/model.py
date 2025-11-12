"""2D sea-ice forecaster that swaps the temporal Conv1D block with a Hyena operator."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyena_layer import HyenaOperator


class Hyena2DForecaster(nn.Module):
    """CNN encoder/decoder with Hyena operator for temporal modelling."""

    def __init__(
        self,
        input_size: tuple[int, int] = (448, 304),
        in_time_points: int = 30,
        out_time_points: int = 7,
        n_layers: int = 5,
        hidden_dim: int = 64,
        temporal_dim: int = 512,
        hyena_order: int = 2,
        hyena_filter_order: int = 64,
        hyena_dropout: float = 0.0,
        hyena_filter_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.in_time_points = in_time_points
        self.out_time_points = out_time_points
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.temporal_dim = temporal_dim

        H, W = input_size

        # Spatial encoder
        encoder_layers = []
        in_ch = 1
        out_ch = 32
        for i in range(n_layers):
            encoder_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if i < n_layers - 1:
                encoder_layers.append(nn.MaxPool2d(2))
                H, W = H // 2, W // 2
            in_ch = out_ch
            out_ch = min(out_ch * 2, hidden_dim)
        self.spatial_encoder = nn.Sequential(*encoder_layers)

        self.encoded_h = H
        self.encoded_w = W
        self.encoded_channels = in_ch
        self.spatial_feat_size = self.encoded_channels * self.encoded_h * self.encoded_w

        # Temporal module built around the Hyena operator
        self.temporal_in = nn.Linear(self.spatial_feat_size, temporal_dim)
        self.temporal_norm = nn.LayerNorm(temporal_dim)
        self.hyena = HyenaOperator(
            d_model=temporal_dim,
            l_max=in_time_points,
            order=hyena_order,
            filter_order=hyena_filter_order,
            dropout=hyena_dropout,
            filter_dropout=hyena_filter_dropout,
        )
        self.temporal_dropout = nn.Dropout(hyena_dropout)
        self.temporal_out = nn.Linear(temporal_dim, self.spatial_feat_size)
        self.post_norm = nn.LayerNorm(temporal_dim)

        # Spatial decoder
        decoder_layers = []
        dec_in_ch = self.encoded_channels
        dec_out_ch = max(dec_in_ch // 2, 16)
        for _ in range(n_layers - 1):
            decoder_layers += [
                nn.ConvTranspose2d(dec_in_ch, dec_out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(dec_out_ch),
                nn.ReLU(inplace=True),
            ]
            dec_in_ch = dec_out_ch
            dec_out_ch = max(dec_out_ch // 2, 16)
        decoder_layers += [nn.Conv2d(dec_in_ch, 1, kernel_size=3, padding=1), nn.Sigmoid()]
        self.spatial_decoder = nn.Sequential(*decoder_layers)

        print("Hyena2D Forecaster initialized:")
        print(f"  Encoded spatial: (C={self.encoded_channels}, H={self.encoded_h}, W={self.encoded_w})")
        print(f"  Spatial feature size: {self.spatial_feat_size}")
        print(f"  Temporal dim: {self.temporal_dim}")
        print(f"  Input: ({in_time_points} days) → Output: ({out_time_points} days)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in, C, H, W = x.shape
        assert C == 1, "Input channel must be 1 (SIC)."

        x_flat = x.view(B * T_in, C, H, W)
        enc = self.spatial_encoder(x_flat)
        enc = enc.view(B, T_in, self.encoded_channels, self.encoded_h, self.encoded_w)

        enc_flat = enc.view(B, T_in, -1)
        temporal = self.temporal_in(enc_flat)
        hyena_in = self.temporal_norm(temporal)
        hyena_out = self.hyena(hyena_in)
        hyena_out = self.temporal_dropout(hyena_out)
        temporal = temporal + hyena_out
        temporal = self.post_norm(temporal)

        temporal = temporal.transpose(1, 2)
        if self.in_time_points != self.out_time_points:
            temporal = F.interpolate(
                temporal,
                size=self.out_time_points,
                mode="linear",
                align_corners=False,
            )
        temporal = temporal.transpose(1, 2)

        spatial_feat = self.temporal_out(temporal)
        spatial_feat = spatial_feat.view(
            B * self.out_time_points, self.encoded_channels, self.encoded_h, self.encoded_w
        )

        dec = self.spatial_decoder(spatial_feat)
        out = dec.view(B, self.out_time_points, 1, self.input_size[0], self.input_size[1])
        return out

    def get_model_info(self) -> dict[str, int | tuple[int, ...]]:
        return {
            "model_name": "Hyena2DForecaster",
            "input_size": self.input_size,
            "in_time_points": self.in_time_points,
            "out_time_points": self.out_time_points,
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "temporal_dim": self.temporal_dim,
            "encoded_shape": (self.encoded_channels, self.encoded_h, self.encoded_w),
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def create_hyena2d_model(
    input_size: tuple[int, int] = (448, 304),
    seq_input: int = 30,
    seq_output: int = 7,
    n_layers: int = 5,
    hidden_dim: int = 64,
    temporal_dim: int = 512,
    hyena_order: int = 2,
    hyena_filter_order: int = 64,
    hyena_dropout: float = 0.0,
    hyena_filter_dropout: float = 0.0,
    device: str = "cuda",
) -> Hyena2DForecaster:
    print("\n" + "=" * 60)
    print("Creating Hyena 2D Forecaster Model")
    print("=" * 60)

    model = Hyena2DForecaster(
        input_size=input_size,
        in_time_points=seq_input,
        out_time_points=seq_output,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        temporal_dim=temporal_dim,
        hyena_order=hyena_order,
        hyena_filter_order=hyena_filter_order,
        hyena_dropout=hyena_dropout,
        hyena_filter_dropout=hyena_filter_dropout,
    ).to(torch.device(device))

    info = model.get_model_info()
    print("\nModel Information:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")
    return model
