# -*- coding: utf-8 -*-
"""
Unicorn-based Sea-Ice Forecaster (wrapped for (B, T, 1, H, W) I/O)

원본 Unicorn 구조는 최대한 유지하고,
입력/출력 shape만 SeaIceDataset에 맞게 래핑한 모델입니다.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint


# ============================
# 기본 블록들 (원본 구조 유지)
# ============================

class Convblock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convblock, self).__init__()
        self.out_c = out_channel
        self.in_c = in_channel
        self.conv2d = nn.Conv2d(self.in_c, self.out_c, 3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class ContractingPathSkip(nn.Module):
    def __init__(self, input_window):
        super(ContractingPathSkip, self).__init__()
        self.convb1_1 = Convblock(input_window, 16)
        self.convb1_2 = Convblock(16, 16)
        self.convb2_1 = Convblock(16, 32)
        self.convb2_2 = Convblock(32, 32)
        self.convb3_1 = Convblock(32, 64)
        self.convb3_2 = Convblock(64, 64)
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

        return x1, x2, x3


class ContractingPathValue(nn.Module):
    def __init__(self, input_window):
        super(ContractingPathValue, self).__init__()
        self.convb1_1 = Convblock(input_window, 32)
        self.convb1_2 = Convblock(32, 32)
        self.convb2_1 = Convblock(32, 64)
        self.convb2_2 = Convblock(64, 64)
        self.convb3_1 = Convblock(64, 128)
        self.convb3_2 = Convblock(128, 128)
        self.maxpool = nn.MaxPool2d(2)

        self.imagefusion1 = nn.Conv2d(32 + 16, 32, 1, 1)
        self.imagefusion2 = nn.Conv2d(64 + 32, 64, 1, 1)
        self.imagefusion3 = nn.Conv2d(128 + 64, 128, 1, 1)

    def forward(self, x, sub1, sub2, sub3):
        x1 = self.convb1_1(x)
        x1 = self.convb1_2(x1)
        x1 = torch.cat([x1, sub1], dim=1)
        x1 = self.imagefusion1(x1)
        p1 = self.maxpool(x1)

        x2 = self.convb2_1(p1)
        x2 = self.convb2_2(x2)
        x2 = torch.cat([x2, sub2], dim=1)
        x2 = self.imagefusion2(x2)
        p2 = self.maxpool(x2)

        x3 = self.convb3_1(p2)
        x3 = self.convb3_2(x3)
        x3 = torch.cat([x3, sub3], dim=1)
        x3 = self.imagefusion3(x3)
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
        # 원 코드에서는 output_window를 인자로 받지만 실제로는 1채널 출력
        self.last = nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1)

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

        out = self.last(d3)
        return out


class ODEBlock(nn.Module):
    def __init__(self, ode_func):
        super(ODEBlock, self).__init__()
        self.ode_func = ode_func

    def forward(self, x, t):
        return odeint(self.ode_func, x, t, method='rk4')


class ConvODEFunc(nn.Module):
    def __init__(self, channel):
        super(ConvODEFunc, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, t, x):
        return self.conv(x)


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = x.reshape(B, C, H, W)
        return x


class DCMP_block(nn.Module):
    """
    D-CMP 블록
    - 원 코드와 동일하게 moving_avg만 사용.
    - kernel_size만 사용하므로, 호출부를 DCMP_block(de)로 맞춤.
    """
    def __init__(self, kernel_size):
        super(DCMP_block, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        T = self.moving_avg(x)
        R = x - T
        return T, R


class Unet(nn.Module):
    def __init__(self, input_window, output_window):
        super(Unet, self).__init__()
        self.output_window = output_window
        self.contract_sub = ContractingPathSkip(3)
        self.contract_sic = ContractingPathValue(input_window)
        self.bottleneck = BottleNeck()
        self.expand = nn.ModuleList(
            [ExpandingPath(output_window) for _ in range(self.output_window)]
        )
        self.ode = ODEBlock(ConvODEFunc(256))
        self.t = torch.linspace(0, 3, 4)

    def forward(self, x, sub):
        t = self.t.to(x.device)

        # 보조 입력 경로
        x1_sub, x2_sub, x3_sub = self.contract_sub(sub)

        # 메인 SIC 경로
        x1, x2, x3, p3 = self.contract_sic(x, x1_sub, x2_sub, x3_sub)
        p3 = self.bottleneck(p3)
        p3 = self.ode(p3, t)  # (len(t), B, C, H, W) 형태

        # 출력 시퀀스 조립
        B, _, H, W = x.shape
        out = torch.zeros(
            [B, self.output_window, H, W],
            dtype=x.dtype,
            device=x.device,
        )

        for i in range(self.output_window):
            # p3[i] : (B, C, H, W)
            out[:, i, :, :] = torch.squeeze(self.expand[i](x1, x2, x3, p3[i]), dim=1)

        return out


class UnetNODECombined(nn.Module):
    def __init__(self, input_window, output_window, de):
        super(UnetNODECombined, self).__init__()
        self.unet1 = Unet(input_window, output_window)
        self.unet2 = Unet(input_window, output_window)
        self.dcmp = DCMP_block(de)   # ← 원본의 DCMP_block(de, pe) 문제를 안전하게 수정
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4):
        # x1 : (B, T_in, H, W)
        T, R = self.dcmp(x1)

        # 마지막 시점 3개를 보조 채널로 사용
        sub = torch.cat(
            [
                x2[:, -1, :, :].unsqueeze(1),
                x3[:, -1, :, :].unsqueeze(1),
                x4[:, -1, :, :].unsqueeze(1),
            ],
            dim=1,
        )  # (B, 3, H, W)

        T = self.unet1(T, sub)
        R = self.unet2(R, sub)

        return self.sigmoid(T + R)   # (B, T_out, H, W)


# ============================
# Sea-ice용 래퍼 모델
# ============================

class UnicornForecaster(nn.Module):
    """
    Sea-ice 전용 Unicorn 래퍼
    Input : (B, T_in, 1, H, W)
    Output: (B, T_out, 1, H, W)
    """

    def __init__(
        self,
        input_size=(448, 304),
        seq_input=30,
        seq_output=7,
        de=25,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_input = seq_input
        self.seq_output = seq_output

        self.core = UnetNODECombined(
            input_window=seq_input,
            output_window=seq_output,
            de=de,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T_in, 1, H, W)
        """
        B, T_in, C, H, W = x.shape
        assert C == 1, "Input channel must be 1 (SIC)."

        # (B, T_in, H, W)
        x_main = x.squeeze(2)

        # 보조 입력이 따로 없으므로,
        # 간단히 같은 시퀀스를 복제해서 x2, x3, x4로 사용
        x2 = x_main
        x3 = x_main
        x4 = x_main

        out = self.core(x_main, x2, x3, x4)   # (B, T_out, H, W)
        out = out.unsqueeze(2)                # (B, T_out, 1, H, W)
        return out

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": "UnicornForecaster",
            "input_size": self.input_size,
            "seq_input": self.seq_input,
            "seq_output": self.seq_output,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }


def create_unicorn_model(
    input_size=(448, 304),
    seq_input=30,
    seq_output=7,
    de=25,
    device="cuda",
):
    print("\n" + "=" * 60)
    print("Creating Unicorn Forecaster Model")
    print("=" * 60)

    model = UnicornForecaster(
        input_size=input_size,
        seq_input=seq_input,
        seq_output=seq_output,
        de=de,
    ).to(torch.device(device))

    info = model.get_model_info()
    print("\nModel Information:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")
    return model
