import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN2DForecaster(nn.Module):
    """
    Pure CNN 2D Baseline for Sea-Ice Prediction
    Input : (B, T_in, 1, H, W)
    Output: (B, T_out, 1, H, W)
    """

    def __init__(
        self,
        input_size=(448, 304),
        in_time_points=30,
        out_time_points=7,
        n_layers=5,
        hidden_dim=64,
    ):
        super().__init__()

        self.input_size = input_size
        self.in_time_points = in_time_points
        self.out_time_points = out_time_points
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        H, W = input_size

        # ---------------------------
        # Spatial Encoder (per frame)
        # ---------------------------
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

        # Encoded shape
        self.encoded_h = H
        self.encoded_w = W
        self.encoded_channels = in_ch
        self.spatial_feat_size = self.encoded_channels * self.encoded_h * self.encoded_w

        # ---------------------------
        # Temporal processing (Conv1D on time)
        # ---------------------------
        temporal_channels = hidden_dim * 8
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.spatial_feat_size, temporal_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(temporal_channels, temporal_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Project back to spatial feature size for each output step
        self.spatial_proj = nn.Conv1d(temporal_channels, self.spatial_feat_size, kernel_size=1)

        # ---------------------------
        # Spatial Decoder
        # ---------------------------
        decoder_layers = []
        in_ch = self.encoded_channels
        out_ch = max(in_ch // 2, 16)
        for _ in range(n_layers - 1):
            decoder_layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
            out_ch = max(out_ch // 2, 16)
        decoder_layers += [nn.Conv2d(in_ch, 1, kernel_size=3, padding=1), nn.Sigmoid()]
        self.spatial_decoder = nn.Sequential(*decoder_layers)

        print("Pure CNN2D Forecaster initialized (NO LSTM):")
        print(f"  Encoded spatial: (C={self.encoded_channels}, H={self.encoded_h}, W={self.encoded_w})")
        print(f"  Spatial feature size: {self.spatial_feat_size}")
        print(f"  Input: ({in_time_points} days) → Output: ({out_time_points} days)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T_in, 1, H, W)
        """
        B, T_in, C, H, W = x.shape
        assert C == 1, "Input channel must be 1 (SIC)."

        # 1) Encode all frames at once (vectorized)
        x_flat = x.view(B * T_in, C, H, W)
        enc = self.spatial_encoder(x_flat)  # (B*T_in, C_enc, H_enc, W_enc)
        enc = enc.view(B, T_in, self.encoded_channels, self.encoded_h, self.encoded_w)

        # 2) Temporal Conv1D on flattened spatial features
        enc_flat = enc.view(B, T_in, -1).transpose(1, 2)  # (B, spatial_feat_size, T_in)
        t_feat = self.temporal_conv(enc_flat)             # (B, temporal_channels, T_in)

        # 3) Resize time dim: T_in → T_out
        if self.in_time_points != self.out_time_points:
            t_feat = F.interpolate(
                t_feat, size=self.out_time_points, mode="linear", align_corners=False
            )  # (B, temporal_channels, T_out)

        # 4) Project to spatial feature vectors for each output step
        s_feat = self.spatial_proj(t_feat)                # (B, spatial_feat_size, T_out)
        s_feat = s_feat.transpose(1, 2).contiguous()      # (B, T_out, spatial_feat_size)
        s_feat = s_feat.view(
            B * self.out_time_points, self.encoded_channels, self.encoded_h, self.encoded_w
        )

        # 5) Decode to full resolution frames
        dec = self.spatial_decoder(s_feat)                # (B*T_out, 1, H, W)
        out = dec.view(B, self.out_time_points, 1, self.input_size[0], self.input_size[1])
        return out

    def get_model_info(self):
        return {
            "model_name": "CNN2DForecaster",
            "input_size": self.input_size,
            "in_time_points": self.in_time_points,
            "out_time_points": self.out_time_points,
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "encoded_shape": (self.encoded_channels, self.encoded_h, self.encoded_w),
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def create_cnn2d_model(
    input_size=(448, 304),
    seq_input=30,
    seq_output=7,
    n_layers=5,
    hidden_dim=64,
    device="cuda",
):
    print("\n" + "=" * 60)
    print("Creating CNN 2D Forecaster Model")
    print("=" * 60)

    model = CNN2DForecaster(
        input_size=input_size,
        in_time_points=seq_input,
        out_time_points=seq_output,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
    ).to(torch.device(device))

    info = model.get_model_info()
    print("\nModel Information:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")
    return model
