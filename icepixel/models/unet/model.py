# -*- coding: utf-8 -*-
"""
UNet 기반 Sea-Ice Forecaster (시간축을 채널로 사용하는 버전)

입력/출력 형식:
    Input : (B, T_in, 1, H, W)
    Output: (B, T_out, 1, H, W)
"""

import torch
import torch.nn as nn


# ============================
# UNet 기본 블록 (네가 준 코드 기반)
# ============================

class Convblock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convblock, self).__init__()
        self.out_c = out_channel
        self.in_c = in_channel
        self.conv2d = nn.Conv2d(self.in_c, self.out_c, 3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        # 안정성 위해 간단한 초기화 추가(옵션)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity="relu")
        if self.conv2d.bias is not None:
            nn.init.zeros_(self.conv2d.bias)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class ContractingPath(nn.Module):
    def __init__(self, input_window):
        super(ContractingPath, self).__init__()
        self.convb1_1 = Convblock(input_window, 32)
        self.convb1_2 = Convblock(32, 32)
        self.convb2_1 = Convblock(32, 64)
        self.convb2_2 = Convblock(64, 64)
        self.convb3_1 = Convblock(64, 128)
        self.convb3_2 = Convblock(128, 128)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.convb1_1(x)
        x1 = self.convb1_2(x1)
        p1 = self.maxpool(x1)

        x2 = self.convb2_1(p1)
        x2 = self.convb2_2(x2)
        p2 = self.maxpool(x2)

        x3 = self.convb3_1(p2)
        x3 = self.convb3_2(x3)
        p3 = self.maxpool(x3)

        return x1, x2, x3, p3


class BottleNeck(nn.Module):
    def __init__(self):
        super(BottleNeck, self).__init__()
        self.convb1 = Convblock(128, 256)
        self.convb2 = Convblock(256, 256)

    def forward(self, x):
        x = self.convb1(x)
        x = self.convb2(x)
        return x


class ExpandingPath(nn.Module):
    def __init__(self, output_window):
        super(ExpandingPath, self).__init__()
        self.convb1 = Convblock(256, 128)
        self.convb2 = Convblock(128, 64)
        self.convb3 = Convblock(64, 32)
        self.convb1_2 = Convblock(128, 128)
        self.convb2_2 = Convblock(64, 64)
        self.convb3_2 = Convblock(32, 32)
        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)
        self.upconv3 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0)
        self.upconv4 = nn.ConvTranspose2d(32, output_window, 3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3, d):
        d = self.upconv1(d)
        d1 = torch.concat([d, x3], dim=1)
        d1 = self.convb1(d1)
        d1 = self.convb1_2(d1)

        d2 = self.upconv2(d1)
        d2 = torch.concat([d2, x2], dim=1)
        d2 = self.convb2(d2)
        d2 = self.convb2_2(d2)

        d3 = self.upconv3(d2)
        d3 = torch.concat([d3, x1], dim=1)
        d3 = self.convb3(d3)
        d3 = self.convb3_2(d3)

        out = self.upconv4(d3)
        out = self.sigmoid(out)
        return out


class UNetCore(nn.Module):
    """
    원래 UNet:
        Input : (B, input_window, H, W)
        Output: (B, output_window, H, W)
    """

    def __init__(self, input_window, output_window):
        super(UNetCore, self).__init__()
        self.contract = ContractingPath(input_window)
        self.bottleneck = BottleNeck()
        self.expand = ExpandingPath(output_window)

    def forward(self, x):
        x1, x2, x3, p3 = self.contract(x)
        p4 = self.bottleneck(p3)
        out = self.expand(x1, x2, x3, p4)
        return out


# ============================
# Forecaster 래퍼
# ============================

class UNetForecaster(nn.Module):
    """
    UNetCore를 (B, T_in, 1, H, W) → (B, T_out, 1, H, W) 형식으로 래핑
    """

    def __init__(
        self,
        input_size=(448, 304),
        in_time_points=30,
        out_time_points=7,
    ):
        super().__init__()
        self.input_size = input_size
        self.in_time_points = in_time_points
        self.out_time_points = out_time_points

        self.core = UNetCore(
            input_window=in_time_points,
            output_window=out_time_points,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T_in, 1, H, W)
        """
        B, T_in, C, H, W = x.shape
        assert C == 1, "UNetForecaster는 채널 1(SIC) 입력만 지원합니다."
        assert (
            T_in == self.in_time_points
        ), f"T_in({T_in}) != in_time_points({self.in_time_points})"

        # (B, T_in, 1, H, W) -> (B, T_in, H, W)
        x_2d = x.squeeze(2)

        # UNetCore 통과 → (B, T_out, H, W)
        out_2d = self.core(x_2d)

        # 다시 채널 차원 추가 → (B, T_out, 1, H, W)
        out = out_2d.unsqueeze(2)
        return out

    def get_model_info(self):
        return {
            "model_name": "UNet_CV_Forecaster",
            "input_size": self.input_size,
            "in_time_points": self.in_time_points,
            "out_time_points": self.out_time_points,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


def create_unet_model(
    input_size=(448, 304),
    seq_input=30,
    seq_output=7,
    device="cuda",
):
    print("\n" + "=" * 60)
    print("Creating UNet CV Forecaster Model")
    print("=" * 60)

    model = UNetForecaster(
        input_size=input_size,
        in_time_points=seq_input,
        out_time_points=seq_output,
    ).to(torch.device(device))

    info = model.get_model_info()
    print("\nModel Information:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")

    return model
