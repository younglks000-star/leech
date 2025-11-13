# -*- coding: utf-8 -*-
"""
DUNet (Unet + Decomposition) 기반 Sea-Ice Forecaster

입출력 형식:
    Input : (B, T_in, 1, H, W)
    Output: (B, T_out, 1, H, W)
"""

import torch
import torch.nn as nn


# ============================
# 기본 블록들 (네가 준 코드 그대로)
# ============================

class Convblock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.out_c = out_channel
        self.in_c = in_channel
        self.conv2d = nn.Conv2d(self.in_c, self.out_c, 3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class ContractingPath(nn.Module):
    def __init__(self, input_window):
        super().__init__()
        # input_window = 시간 길이(T_in)를 채널로 사용
        self.convb1_1 = Convblock(input_window, 32)
        self.convb1_2 = Convblock(32, 32)
        self.convb2_1 = Convblock(32, 64)
        self.convb2_2 = Convblock(64, 64)
        self.convb3_1 = Convblock(64, 128)
        self.convb3_2 = Convblock(128, 128)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        # x: (B, C=input_window, H, W)
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
        super().__init__()
        self.convb1 = Convblock(128, 256)
        self.convb2 = Convblock(256, 256)

    def forward(self, x):
        x = self.convb1(x)
        x = self.convb2(x)
        return x


class ExpandingPath(nn.Module):
    def __init__(self, output_window):
        super().__init__()
        # output_window = 예측 길이(T_out)를 채널로 사용
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

    def forward(self, x1, x2, x3, d):
        d = self.upconv1(d)
        d1 = torch.cat([d, x3], dim=1)
        d1 = self.convb1(d1)
        d1 = self.convb1_2(d1)

        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.convb2(d2)
        d2 = self.convb2_2(d2)

        d3 = self.upconv3(d2)
        d3 = torch.cat([d3, x1], dim=1)
        d3 = self.convb3(d3)
        d3 = self.convb3_2(d3)

        out = self.upconv4(d3)
        return out


class Unet(nn.Module):
    def __init__(self, input_window, output_window):
        super().__init__()
        self.contract = ContractingPath(input_window)
        self.bottleneck = BottleNeck()
        self.expand = ExpandingPath(output_window)

    def forward(self, x):
        # x: (B, C=input_window, H, W)
        x1, x2, x3, p3 = self.contract(x)
        p4 = self.bottleneck(p3)
        out = self.expand(x1, x2, x3, p4)
        # out: (B, C=output_window, H, W)
        return out


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (B, C, H, W)  여기서 C=시간길이
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)

        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = x.reshape(B, C, H, W)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class UnetDLinear(nn.Module):
    def __init__(self, input_window, output_window, de):
        super().__init__()
        self.decomp = series_decomp(de)
        self.unetT = Unet(input_window, output_window)
        self.unetR = Unet(input_window, output_window)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C=input_window, H, W)
        res, moving_mean = self.decomp(x)
        res = self.unetR(res)
        moving_mean = self.unetT(moving_mean)
        x = res + moving_mean
        x = self.sigmoid(x)
        # (B, C=output_window, H, W)
        return x


# ============================
# Forecaster 래퍼
# ============================

class DUNetForecaster(nn.Module):
    """
    UnetDLinear를 (B, T_in, 1, H, W) ↔ (B, T_out, 1, H, W) 형식으로 감싸는 래퍼
    """

    def __init__(
        self,
        input_size=(448, 304),
        in_time_points=30,
        out_time_points=7,
        de=25,  # decomposition kernel size
    ):
        super().__init__()
        self.input_size = input_size
        self.in_time_points = in_time_points
        self.out_time_points = out_time_points
        self.de = de

        self.core = UnetDLinear(
            input_window=in_time_points,
            output_window=out_time_points,
            de=de,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T_in, 1, H, W)
        """
        B, T_in, C, H, W = x.shape
        assert C == 1, "DUNetForecaster는 입력 채널 1(SIC)을 가정합니다."
        assert (
            T_in == self.in_time_points
        ), f"T_in({T_in}) != in_time_points({self.in_time_points})"

        # (B, T_in, 1, H, W) -> (B, T_in, H, W)
        x_2d = x.squeeze(2)

        # UnetDLinear 적용: (B, T_in, H, W) -> (B, T_out, H, W)
        out_2d = self.core(x_2d)

        # 다시 채널 축 추가: (B, T_out, H, W) -> (B, T_out, 1, H, W)
        out = out_2d.unsqueeze(2)
        return out

    def get_model_info(self):
        H, W = self.input_size
        return {
            "model_name": "DUNetForecaster",
            "input_size": self.input_size,
            "in_time_points": self.in_time_points,
            "out_time_points": self.out_time_points,
            "de": self.de,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


def create_dunet_model(
    input_size=(448, 304),
    seq_input=30,
    seq_output=7,
    de=25,
    device="cuda",
):
    print("\n" + "=" * 60)
    print("Creating DUNet Forecaster Model")
    print("=" * 60)

    model = DUNetForecaster(
        input_size=input_size,
        in_time_points=seq_input,
        out_time_points=seq_output,
        de=de,
    ).to(torch.device(device))

    info = model.get_model_info()
    print("\nModel Information:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")

    return model
