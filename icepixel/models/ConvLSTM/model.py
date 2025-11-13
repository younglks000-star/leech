# -*- coding: utf-8 -*-
"""
ConvLSTM 기반 Sea-Ice Forecaster

입출력 형식:
    Input : (B, T_in, 1, H, W)
    Output: (B, T_out, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # 입력(채널 + hidden)을 합쳐서 한 번에 4개 게이트 계산
        self.gates = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, input_tensor, hidden_state):
        """
        input_tensor: (B, 1, H, W)
        hidden_state: (h_cur, c_cur), 각각 (B, hidden_dim, H, W)
        """
        h_cur, c_cur = hidden_state

        # 입력 채널 차원 보장
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(1)  # (B, 1, H, W)

        combined = torch.cat([input_tensor, h_cur], dim=1)  # (B, C+hidden, H, W)
        gates = self.gates(combined)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        c_next = forgetgate * c_cur + ingate * cellgate
        h_next = outgate * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMForecaster(nn.Module):
    """
    ConvLSTM + 간단한 CNN Head 기반 Baseline

    Input : (B, T_in, 1, H, W)
    Output: (B, T_out, 1, H, W)
    """

    def __init__(
        self,
        input_size=(448, 304),
        in_time_points=30,
        out_time_points=7,
        hidden_dims=(32, 64, 128),
        kernel_size=3,
    ):
        super().__init__()

        self.input_size = input_size
        self.in_time_points = in_time_points
        self.out_time_points = out_time_points
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size

        H, W = input_size
        assert len(hidden_dims) == 3, "hidden_dims는 3개 값 (h1, h2, h3)이 필요합니다."

        # ---------------------------
        # ConvLSTM (시간축 인코딩)
        # ---------------------------
        self.conv_lstm = ConvLSTMCell(
            input_dim=1,
            hidden_dim=hidden_dims[0],
            kernel_size=kernel_size,
        )

        # ---------------------------
        # Spatial CNN Head
        # ---------------------------
        # 첫 번째 stage
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(
            hidden_dims[0],
            hidden_dims[1],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        # 두 번째 stage
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            hidden_dims[1],
            hidden_dims[2],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        # 풀링 이후 해상도 계산
        H_enc = H // 2 // 2
        W_enc = W // 2 // 2
        self.encoded_h = H_enc
        self.encoded_w = W_enc

        flatten_dim = hidden_dims[2] * H_enc * W_enc

        # ---------------------------
        # Fully Connected Head
        # ---------------------------
        self.fc1 = nn.Linear(flatten_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, out_time_points * H * W)

        self.sigmoid = nn.Sigmoid()

        print("ConvLSTM Forecaster initialized:")
        print(f"  Input  time steps: {in_time_points}")
        print(f"  Output time steps: {out_time_points}")
        print(f"  Input size: (H={H}, W={W})")
        print(f"  Encoded feature map: (C={hidden_dims[2]}, H={H_enc}, W={W_enc})")
        print(f"  Flatten dim: {flatten_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T_in, 1, H, W)
        """
        B, T_in, C, H, W = x.shape
        assert C == 1, "ConvLSTMForecaster는 입력 채널 1을 가정합니다."
        assert (
            T_in == self.in_time_points
        ), f"T_in({T_in}) != in_time_points({self.in_time_points})"

        # 초기 hidden / cell 상태
        h = torch.zeros(
            B, self.hidden_dims[0], H, W, device=x.device, dtype=x.dtype
        )
        c = torch.zeros(
            B, self.hidden_dims[0], H, W, device=x.device, dtype=x.dtype
        )

        # 시간축으로 ConvLSTM 수행 (마지막 hidden만 사용)
        for t in range(T_in):
            x_t = x[:, t, 0, :, :].unsqueeze(1)  # (B, 1, H, W)
            h, c = self.conv_lstm(x_t, (h, c))

        # Spatial CNN Head
        x_feat = self.maxpool1(h)
        x_feat = F.relu(self.conv1(x_feat))
        x_feat = self.maxpool2(x_feat)
        x_feat = F.relu(self.conv2(x_feat))

        # FC Head
        x_flat = x_feat.view(B, -1)
        x_flat = F.relu(self.fc1(x_flat))
        x_flat = F.relu(self.fc2(x_flat))
        x_flat = self.fc3(x_flat)  # (B, T_out * H * W)

        # 최종 출력 reshape
        out = x_flat.view(
            B, self.out_time_points, self.input_size[0], self.input_size[1]
        )  # (B, T_out, H, W)
        out = self.sigmoid(out).unsqueeze(2)  # (B, T_out, 1, H, W)

        return out

    def get_model_info(self):
        return {
            "model_name": "ConvLSTMForecaster",
            "input_size": self.input_size,
            "in_time_points": self.in_time_points,
            "out_time_points": self.out_time_points,
            "hidden_dims": self.hidden_dims,
            "kernel_size": self.kernel_size,
            "encoded_shape": (
                self.hidden_dims[2],
                self.encoded_h,
                self.encoded_w,
            ),
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


def create_convlstm_model(
    input_size=(448, 304),
    seq_input=30,
    seq_output=7,
    hidden_dims=(32, 64, 128),
    kernel_size=3,
    device="cuda",
):
    print("\n" + "=" * 60)
    print("Creating ConvLSTM Forecaster Model")
    print("=" * 60)

    model = ConvLSTMForecaster(
        input_size=input_size,
        in_time_points=seq_input,
        out_time_points=seq_output,
        hidden_dims=hidden_dims,
        kernel_size=kernel_size,
    ).to(torch.device(device))

    info = model.get_model_info()
    print("\nModel Information:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")

    return model
