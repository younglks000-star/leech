# -*- coding: utf-8 -*-
"""
ETTh1 기반 시계열 모델 효율성 측정 스크립트 (멀티 모델/멀티 설정 지원)

측정 지표:
    1) #Params (M, 소수점 4자리)
    2) MACs (M, 소수점 4자리)
    3) Inference Time (ms)  - 1 batch 기준
    4) Max Memory (MB)      - Inference 시 peak
    5) Epoch Time (s)       - train 1 epoch 기준

사용 방법:
    - TARGET_MODELS에 측정할 모델 이름들을 나열
    - BENCHMARK_JOBS에 (seq_len, pred_len, batch_size 등) 설정을 여러 개 넣어두면 순회하며 모두 측정
    - build_model_and_config() 안에 모델 분기를 추가/수정
"""

import time
import warnings
from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn as nn
from torch import optim
from thop import profile

import src.losses as losses
import src.data_factory as data_factory

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ========================== 0. 어떤 모델을 벤치마크할지 선택 ==========================
TARGET_MODELS: List[str] = [
    "Autoformer",
    "DLinear",
    "PatchTST",
    "iTransformer",
    "TimeCMA",
    "FITS",
    "SparseTSF",
    "F4SSM1",
    "F4SSM2",
    "new3_f4_ssm",
]


@dataclass
class BenchmarkJob:
    """벤치마크에서 사용할 입력 설정."""

    name: str
    seq_len: int = 96
    pred_len: int = 96
    batch_size: int = 32
    d_state: int = 3
    lambda_spec: float = 0.1
    lr: float = 0.01
    dataset: str = "ETTh1"
    root_path: str = "data/"
    starting_percent: int = 0
    percent: int = 100


BENCHMARK_JOBS: List[BenchmarkJob] = [
    BenchmarkJob(name="96to96-bs32"),
    BenchmarkJob(name="96to192-bs16", pred_len=192, batch_size=16),
]


# ========================== 1. 공통 유틸 ==========================
def make_dummy_embeddings(batch_x: torch.Tensor, d_llm: int, num_nodes: int):
    """TimeCMA용 더미 임베딩 생성."""
    B, _L, _D = batch_x.shape
    return torch.zeros(B, d_llm, num_nodes, device=batch_x.device, dtype=batch_x.dtype)


# ========================== 2. DiagonalSSM용 MACs hook ==========================
def count_diagonal_ssm(m, x, _y):
    """F4SSM 계열 MACs 근사 계산 함수."""
    inp = x[0]
    B, L, D = inp.shape

    if hasattr(m, "A_logit") and hasattr(m, "B") and hasattr(m, "C"):
        N = m.d_state
        macs = 2 * B * L * D * N
    elif hasattr(m, "kernel") and hasattr(m, "in_proj") and hasattr(m, "out_proj"):
        S = m.d_state
        K = m.kernel_size
        macs_in = B * L * D * S
        macs_conv = B * S * L * K
        macs_out = B * L * S * D
        macs = macs_in + macs_conv + macs_out
    else:
        macs = 0

    if not hasattr(m, "total_ops"):
        m.total_ops = torch.zeros(1, dtype=torch.float64)

    macs_tensor = m.total_ops.new_tensor([macs])
    m.total_ops += macs_tensor


# ========================== 3. 모델 생성 + 설정 함수 ==========================
def build_model_and_config(
    model_name: str,
    n_features: int,
    seq_len: int,
    pred_len: int,
    d_state: int,
    device: torch.device,
):
    from types import SimpleNamespace

    if model_name == "F4SSM1":
        import models.new1_f4_ssm.modules.new1_f4_ssm as F4SSM

        model = F4SSM.F4SSM(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=n_features,
            d_state=d_state,
            use_norm=True,
            period=24,
        )

        model_info = {
            "name": "F4SSM1",
            "returns_components": True,
            "use_spec_loss": True,
            "DiagonalSSM": F4SSM.DiagonalSSM,
            "forward_with_marks": False,
            "needs_embeddings": False,
        }

    elif model_name == "F4SSM2":
        import models.new2_f4_ssm.modules.new2_f4_ssm as F4SSM

        model = F4SSM.F4SSM(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=n_features,
            d_state=d_state,
            use_norm=True,
            period=24,
        )

        model_info = {
            "name": "F4SSM2",
            "returns_components": True,
            "use_spec_loss": True,
            "DiagonalSSM": F4SSM.DiagonalSSM,
            "forward_with_marks": False,
            "needs_embeddings": False,
        }

    elif model_name == "new3_f4_ssm":
        import models.new3_f4_ssm.modules.new3_f4_ssm as F4SSM

        model = F4SSM.F4SSM(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=n_features,
            d_state=d_state,
            use_norm=True,
            period=12,
        )

        model_info = {
            "name": "new3_f4_ssm",
            "returns_components": True,
            "use_spec_loss": True,
            "DiagonalSSM": F4SSM.DiagonalSSM,
            "forward_with_marks": False,
            "needs_embeddings": False,
        }

    elif model_name == "iTransformer":
        import models.iTransformer.modules.iTransformer as ITR

        args = SimpleNamespace(
            enc_in=n_features,
            dec_in=n_features,
            c_out=n_features,
            seq_len=seq_len,
            label_len=seq_len,
            pred_len=pred_len,
            d_model=512,
            n_heads=8,
            e_layers=2,
            d_layers=1,
            d_ff=2048,
            moving_avg=25,
            factor=1,
            distil=True,
            dropout=0.01,
            embed="timeF",
            activation="gelu",
            output_attention=False,
            do_predict=False,
            learning_rate=0.0001,
            lradj="type1",
            exp_name="MTSF",
            channel_independence=False,
            inverse=False,
            class_strategy="projection",
            use_norm=True,
            partial_start_index=0,
            freq="h",
            features="M",
        )

        model = ITR.Model(args)

        model_info = {
            "name": "iTransformer",
            "returns_components": False,
            "use_spec_loss": False,
            "DiagonalSSM": None,
            "forward_with_marks": True,
            "needs_embeddings": False,
        }

    elif model_name == "Autoformer":
        import models.Autoformer.layers.Autoformer as Autoformer

        args = SimpleNamespace(
            features="M",
            freq="h",
            seq_len=seq_len,
            label_len=seq_len,
            pred_len=pred_len,
            bucket_size=4,
            n_hashes=4,
            enc_in=n_features,
            dec_in=n_features,
            c_out=n_features,
            d_model=256,
            n_heads=8,
            e_layers=2,
            d_layers=1,
            d_ff=1024,
            moving_avg=25,
            factor=1,
            distil=True,
            dropout=0.01,
            embed="timeF",
            activation="gelu",
            output_attention=False,
            do_predict=False,
            learning_rate=0.0001,
            lradj="type1",
            exp_name="MTSF",
            channel_independence=False,
            inverse=False,
            class_strategy="projection",
            use_norm=True,
            partial_start_index=0,
        )

        model = Autoformer.Model(args)

        model_info = {
            "name": "Autoformer",
            "returns_components": False,
            "use_spec_loss": False,
            "DiagonalSSM": None,
            "forward_with_marks": True,
            "needs_embeddings": False,
        }

    elif model_name == "DLinear":
        import models.DLinear.modules.DLinear as DLinear

        args = SimpleNamespace(
            features="M",
            freq="h",
            seq_len=seq_len,
            label_len=seq_len,
            pred_len=pred_len,
            individual=False,
            embed_type=0,
            enc_in=n_features,
            dec_in=n_features,
            c_out=n_features,
            moving_avg=25,
            factor=1,
            distil=True,
            dropout=0.05,
            embed="timeF",
            activation="gelu",
            output_attention=False,
            do_predict=False,
            learning_rate=0.0001,
            lradj="type1",
            exp_name="MTSF",
        )

        model = DLinear.Model(args)

        model_info = {
            "name": "DLinear",
            "returns_components": False,
            "use_spec_loss": False,
            "DiagonalSSM": None,
            "forward_with_marks": False,
            "needs_embeddings": False,
        }

    elif model_name == "PatchTST":
        import models.PatchTST.modules.PatchTST as PatchTST

        args = SimpleNamespace(
            features="M",
            freq="h",
            seq_len=seq_len,
            label_len=seq_len,
            pred_len=pred_len,
            embed_type=0,
            enc_in=n_features,
            dec_in=n_features,
            c_out=n_features,
            d_model=512,
            n_heads=8,
            e_layers=2,
            d_layers=1,
            d_ff=2048,
            moving_avg=25,
            factor=1,
            distil=True,
            dropout=0.05,
            embed="timeF",
            activation="gelu",
            output_attention=False,
            do_predict=False,
            learning_rate=0.0001,
            lradj="type3",
            exp_name="MTSF",
            channel_independence=False,
            inverse=False,
            class_strategy="projection",
            use_norm=1,
            partial_start_index=0,
            fc_dropout=0.05,
            head_dropout=0.0,
            patch_len=16,
            stride=8,
            padding_patch="end",
            revin=1,
            affine=0,
            subtract_last=0,
            decomposition=0,
            kernel_size=25,
            individual=0,
        )

        model = PatchTST.Model(args)

        model_info = {
            "name": "PatchTST",
            "returns_components": False,
            "use_spec_loss": False,
            "DiagonalSSM": None,
            "forward_with_marks": False,
            "needs_embeddings": False,
        }

    elif model_name == "TimeCMA":
        import models.TimeCMA.models.TimeCMA as TimeCMA

        args = SimpleNamespace(
            device=device,
            features="M",
            freq="h",
            seq_len=seq_len,
            label_len=0,
            pred_len=pred_len,
            channel=32,
            num_nodes=n_features,
            dropout_n=0.2,
            d_llm=768,
            e_layers=1,
            d_layers=1,
            d_ff=32,
            head=8,
            activation="gelu",
            output_attention=False,
            do_predict=False,
            learning_rate=1e-4,
            lradj="type1",
            exp_name="MTSF",
            channel_independence=False,
            inverse=False,
            class_strategy="projection",
            use_norm=1,
            partial_start_index=0,
        )

        model = TimeCMA.Dual(args)

        if hasattr(model, "prompt_encoder"):
            model.prompt_encoder = nn.Identity()

        model_info = {
            "name": "TimeCMA",
            "returns_components": False,
            "use_spec_loss": False,
            "DiagonalSSM": None,
            "forward_with_marks": True,
            "needs_embeddings": True,
            "d_llm": args.d_llm,
            "num_nodes": args.num_nodes,
        }

    elif model_name == "FITS":
        import models.FITS.models.Real_FITS as FITS_mod

        args = SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=n_features,
            individual=False,
            cut_freq=24,
        )

        core_model = FITS_mod.Model(args)

        class FITSWrapper(nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core

            def forward(self, x):
                y, _ = self.core(x)
                return y

        model = FITSWrapper(core_model)

        model_info = {
            "name": "FITS",
            "returns_components": False,
            "use_spec_loss": False,
            "DiagonalSSM": None,
            "forward_with_marks": False,
            "needs_embeddings": False,
        }

    elif model_name == "SparseTSF":
        import models.SparseTSF.models.SparseTSF as SparseTSF

        args = SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=n_features,
            period_len=24,
            d_model=64,
            model_type="mlp",
        )

        model = SparseTSF.Model(args)

        model_info = {
            "name": "SparseTSF",
            "returns_components": False,
            "use_spec_loss": False,
            "DiagonalSSM": None,
            "forward_with_marks": False,
            "needs_embeddings": False,
        }

    else:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}")

    model = model.float().to(device)
    return model, model_info


# ========================== 4. 파라미터 수 계산 함수 ==========================
def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    return total, total / 1e6


# ========================== 5. MACs 프로파일 함수 ==========================
def profile_macs(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_x_mark: torch.Tensor,
    batch_y_mark: torch.Tensor,
    custom_ops: dict,
    device=device,
    model_info=None,
):
    model = model.to(device).float().eval()
    batch_x = batch_x.to(device).float()
    batch_y = batch_y.to(device).float()
    batch_x_mark = batch_x_mark.to(device).float()
    batch_y_mark = batch_y_mark.to(device).float()

    name = model_info.get("name", "")
    forward_with_marks = model_info.get("forward_with_marks", False)
    needs_embeddings = model_info.get("needs_embeddings", False)
    d_llm = model_info.get("d_llm", 768)
    num_nodes = model_info.get("num_nodes", batch_x.shape[-1])

    if name == "Autoformer":
        inputs = (batch_x, batch_x_mark, batch_y, batch_y_mark)
    elif name == "TimeCMA":
        embeddings = make_dummy_embeddings(batch_x, d_llm, num_nodes)
        inputs = (batch_x, batch_x_mark, embeddings)
    elif forward_with_marks:
        inputs = (batch_x, None, None, None)
    else:
        inputs = (batch_x,)

    try:
        with torch.no_grad():
            macs, _params_thop = profile(
                model,
                inputs=inputs,
                custom_ops=custom_ops,
                verbose=False,
            )
    except Exception as exc:  # pylint: disable=broad-except
        shapes_str = (
            f"x={tuple(batch_x.shape)}, "
            f"y={tuple(batch_y.shape)}, "
            f"x_mark={tuple(batch_x_mark.shape)}, "
            f"y_mark={tuple(batch_y_mark.shape)}"
        )
        raise RuntimeError(
            f"[profile_macs] 모델 '{name}'에서 THOP profiling 실패."
            f"  - 입력 shape: {shapes_str}\n"
            f"  - 원인: {exc!r}"
        ) from exc

    return macs


# ========================== 6. Inference Time + Max Memory ==========================
def measure_inference_time_and_memory(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_x_mark: torch.Tensor,
    batch_y_mark: torch.Tensor,
    device=device,
    warmup=10,
    repeat=50,
    model_info=None,
):
    model = model.to(device).float().eval()
    batch_x = batch_x.to(device).float()
    batch_y = batch_y.to(device).float()
    batch_x_mark = batch_x_mark.to(device).float()
    batch_y_mark = batch_y_mark.to(device).float()

    is_cuda = (
        isinstance(device, torch.device) and device.type == "cuda"
    ) or (
        isinstance(device, str) and device.startswith("cuda")
    )

    name = model_info.get("name", "")
    forward_with_marks = model_info.get("forward_with_marks", False)
    needs_embeddings = model_info.get("needs_embeddings", False)
    d_llm = model_info.get("d_llm", 768)
    num_nodes = model_info.get("num_nodes", batch_x.shape[-1])

    if name == "Autoformer":
        def _forward():
            return model(batch_x, batch_x_mark, batch_y, batch_y_mark)
    elif name == "TimeCMA":
        def _forward():
            embeddings = make_dummy_embeddings(batch_x, d_llm, num_nodes)
            return model(batch_x, batch_x_mark, embeddings)
    elif forward_with_marks:
        def _forward():
            return model(batch_x, None, None, None)
    else:
        def _forward():
            return model(batch_x)

    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(warmup):
            _ = _forward()
            if is_cuda:
                torch.cuda.synchronize()

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    with torch.no_grad():
        for _ in range(repeat):
            if is_cuda:
                torch.cuda.synchronize()
            start = time.time()
            _ = _forward()
            if is_cuda:
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000.0)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    if is_cuda:
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_mb = max_mem_bytes / (1024 ** 2)
    else:
        max_mem_mb = 0.0

    return avg_time, std_time, max_mem_mb


# ========================== 7. Epoch Time (ETTh1 train 1 epoch) ==========================
def measure_epoch_time_etth1_train(
    model: nn.Module,
    train_loader,
    pred_len: int,
    model_info: dict,
    lambda_spec: float = 0.2,
    lr: float = 0.01,
    device=device,
):
    model = model.to(device).float()
    model.train()

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    is_cuda = (
        isinstance(device, torch.device) and device.type == "cuda"
    ) or (
        isinstance(device, str) and device.startswith("cuda")
    )

    name = model_info.get("name", "")
    returns_components = model_info.get("returns_components", False)
    use_spec_loss = model_info.get("use_spec_loss", False)
    forward_with_marks = model_info.get("forward_with_marks", False)
    needs_embeddings = model_info.get("needs_embeddings", False)
    d_llm = model_info.get("d_llm", 768)
    num_nodes = model_info.get("num_nodes", None)

    use_amp = is_cuda and (name not in ["Autoformer", "FITS"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.time()
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        batch_y = batch_y.to(device).float()
        batch_x = batch_x.to(device).float()
        batch_x_mark = batch_x_mark.to(device).float()
        batch_y_mark = batch_y_mark.to(device).float()

        if num_nodes is None:
            num_nodes = batch_x.shape[-1]

        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.cuda.amp.autocast(enabled=use_amp):
                if returns_components:
                    output, H = model(batch_x, return_components=True)
                else:
                    if name == "Autoformer":
                        output = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    elif name == "TimeCMA":
                        embeddings = make_dummy_embeddings(batch_x, d_llm, num_nodes)
                        output = model(batch_x, batch_x_mark, embeddings)
                    elif forward_with_marks:
                        output = model(batch_x, None, None, None)
                    else:
                        output = model(batch_x)
                    H = None

                if output.ndim != 3 or batch_y.ndim != 3:
                    raise RuntimeError(
                        f"[measure_epoch_time] '{name}'의 출력/정답 차원이 3이 아닙니다. "
                        f"output.ndim={output.ndim}, batch_y.ndim={batch_y.ndim}"
                    )

                B_o, L_o, D_o = output.shape
                B_y, L_y, D_y = batch_y.shape

                if B_o != B_y or D_o != D_y:
                    raise RuntimeError(
                        f"[measure_epoch_time] '{name}'의 출력/정답 배치/채널 크기가 다릅니다. "
                        f"output.shape={output.shape}, batch_y.shape={batch_y.shape}"
                    )

                if L_o < pred_len or L_y < pred_len:
                    raise RuntimeError(
                        f"[measure_epoch_time] '{name}'에서 pred_len={pred_len}에 비해 "
                        f"output_len={L_o}, target_len={L_y}가 너무 짧습니다."
                    )

                output = output[:, -pred_len:]
                main_loss = criterion(output, batch_y[:, -pred_len:])

                if use_spec_loss and (H is not None):
                    spec_loss = losses.spectral_separation_loss_scales(H)
                    loss = main_loss + lambda_spec * spec_loss
                else:
                    loss = main_loss

        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"[measure_epoch_time] 모델 '{name}'의 forward/손실 계산 중 오류 발생.\n"
                f"  - batch_x.shape={tuple(batch_x.shape)}, "
                f"batch_y.shape={tuple(batch_y.shape)}, "
                f"batch_x_mark.shape={tuple(batch_x_mark.shape)}, "
                f"batch_y_mark.shape={tuple(batch_y_mark.shape)}\n"
                f"  - pred_len={pred_len}\n"
                f"  - 원인: {exc!r}"
            ) from exc

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    if is_cuda:
        torch.cuda.synchronize()
    end = time.time()

    epoch_time = end - start
    return epoch_time


# ========================== 8. 전체 벤치마크 (데이터 + 모델 + 측정) ==========================
def benchmark_etth1_model(
    model_name: str,
    job: BenchmarkJob,
    device=device,
):
    root_path = job.root_path
    data = job.dataset
    features = "M"
    starting_percent = job.starting_percent
    percent = job.percent
    seq_len = job.seq_len
    pred_len = job.pred_len
    batch_size = job.batch_size
    d_state = job.d_state

    print(f"[DEVICE] Using: {device}")
    print(f"[MODEL] {model_name}")
    print(f"[JOB] {job.name} | seq_len={seq_len}, pred_len={pred_len}, batch_size={batch_size}")

    df, freq, embed = data_factory.data_select(data, root_path)
    n_features = df.shape[1] - 1

    train_data, train_loader = data_factory.data_provider(
        root_path,
        data,
        features,
        batch_size,
        seq_len,
        seq_len,
        pred_len,
        "train",
        starting_percent=starting_percent,
        percent=percent,
    )
    test_data, test_loader = data_factory.data_provider(
        root_path,
        data,
        features,
        batch_size,
        seq_len,
        seq_len,
        pred_len,
        "test",
        starting_percent=starting_percent,
        percent=percent,
    )

    print(f"[INFO] train samples: {len(train_data)} / test samples: {len(test_data)}")

    if len(train_data) == 0 or len(test_data) == 0:
        raise RuntimeError(
            "[ERROR] train_data 또는 test_data의 길이가 0입니다. 데이터 구성 및 percent/starting_percent를 확인하세요."
        )

    if len(train_loader) == 0 or len(test_loader) == 0:
        raise RuntimeError(
            "[ERROR] train_loader 또는 test_loader에서 배치를 생성할 수 없습니다. batch_size 및 데이터 길이를 확인하세요."
        )

    model, model_info = build_model_and_config(
        model_name=model_name,
        n_features=n_features,
        seq_len=seq_len,
        pred_len=pred_len,
        d_state=d_state,
        device=device,
    )

    n_params, n_params_m = count_parameters(model)
    n_params_m = round(n_params_m, 4)
    print(f"[INFO] #Params: {n_params_m:.4f} M")

    try:
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))
    except StopIteration as exc:
        raise RuntimeError("[ERROR] train_loader에서 배치를 꺼낼 수 없습니다. 데이터/배치 구성을 다시 확인하세요.") from exc

    DiagonalSSM_cls = model_info.get("DiagonalSSM", None)
    if DiagonalSSM_cls is not None:
        custom_ops = {DiagonalSSM_cls: count_diagonal_ssm}
    else:
        custom_ops = {}

    macs = profile_macs(
        model,
        batch_x,
        batch_y,
        batch_x_mark,
        batch_y_mark,
        custom_ops=custom_ops,
        device=device,
        model_info=model_info,
    )
    macs_m = round(macs / 1e6, 4)
    print(f"[INFO] MACs (per batch): {macs_m:.4f} M")

    try:
        test_batch_x, test_batch_y, test_x_mark, test_y_mark = next(iter(test_loader))
    except StopIteration as exc:
        raise RuntimeError(
            "[ERROR] test_loader에서 배치를 꺼낼 수 없습니다. 데이터/배치 구성을 다시 확인하세요."
        ) from exc

    infer_ms, infer_std, max_mem_mb = measure_inference_time_and_memory(
        model,
        test_batch_x,
        test_batch_y,
        test_x_mark,
        test_y_mark,
        device=device,
        model_info=model_info,
    )
    print(f"[INFO] Inference Time (1 batch): {infer_ms:.3f} ± {infer_std:.3f} ms")
    print(f"[INFO] Max Memory (inference): {max_mem_mb:.2f} MB")

    epoch_time_s = measure_epoch_time_etth1_train(
        model,
        train_loader,
        pred_len=pred_len,
        model_info=model_info,
        lambda_spec=job.lambda_spec,
        lr=job.lr,
        device=device,
    )
    print(f"[INFO] Epoch Time (train 1 epoch): {epoch_time_s:.3f} s")

    result = {
        "Job": job.name,
        "Model": model_info.get("name", model_name),
        "Params(M)": n_params_m,
        "MACs(M)": macs_m,
        "Infer(ms)": round(infer_ms, 4),
        "Infer(std_ms)": round(infer_std, 4),
        "MaxMem(MB)": round(max_mem_mb, 4),
        "EpochTime(s)": round(epoch_time_s, 4),
        "Config": {
            "dataset": data,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "d_state": d_state,
            "lambda_spec": job.lambda_spec,
            "lr": job.lr,
        },
    }
    return result


# ========================== 9. 멀티 모델/설정 실행기 ==========================
def benchmark_suite(
    models: Iterable[str],
    jobs: Iterable[BenchmarkJob],
    device: torch.device = device,
):
    all_results = []
    for job in jobs:
        print("\n" + "=" * 80)
        print(f"[JOB START] {job.name}: seq_len={job.seq_len}, pred_len={job.pred_len}, batch_size={job.batch_size}")
        for model_name in models:
            print("-" * 40)
            try:
                res = benchmark_etth1_model(model_name=model_name, job=job, device=device)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[ERROR] {model_name} 실패: {exc}")
                res = {
                    "Job": job.name,
                    "Model": model_name,
                    "Params(M)": None,
                    "MACs(M)": None,
                    "Infer(ms)": None,
                    "Infer(std_ms)": None,
                    "MaxMem(MB)": None,
                    "EpochTime(s)": None,
                    "Config": {
                        "dataset": job.dataset,
                        "batch_size": job.batch_size,
                        "seq_len": job.seq_len,
                        "pred_len": job.pred_len,
                        "d_state": job.d_state,
                        "lambda_spec": job.lambda_spec,
                        "lr": job.lr,
                    },
                    "Error": str(exc),
                }
            all_results.append(res)
    return all_results


def print_summary_table(results: List[dict]):
    if not results:
        print("[WARN] 요약할 결과가 없습니다.")
        return

    header = [
        "Job",
        "Model",
        "Params(M)",
        "MACs(M)",
        "Infer(ms)",
        "Infer(std_ms)",
        "MaxMem(MB)",
        "EpochTime(s)",
        "Error",
    ]
    col_widths = {h: max(len(h), 12) for h in header}

    for res in results:
        for h in header:
            value = res.get(h, "")
            if value is None:
                value = "-"
            value_str = f"{value}"
            col_widths[h] = max(col_widths[h], len(value_str))

    def _format_row(row_dict):
        return " | ".join(
            f"{str(row_dict.get(h, '-')):<{col_widths[h]}}" for h in header
        )

    print("\n=== Benchmark Summary ===")
    print(_format_row({h: h for h in header}))
    print("-" * (sum(col_widths.values()) + 3 * (len(header) - 1)))
    for res in results:
        print(_format_row(res))


# ========================== 10. 메인 ==========================
if __name__ == "__main__":
    results = benchmark_suite(TARGET_MODELS, BENCHMARK_JOBS, device=device)
    print_summary_table(results)
