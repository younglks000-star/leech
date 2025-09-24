# train_timecma_experiment.py

import os
import gc
import time
import random
import argparse
import datetime as dt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_loader_emb_0911 import Dataset_Custom
from models.TimeCMA_0912 import Dual
import utils.Metric_node as Metric

import faulthandler
faulthandler.enable()

# -------------------------
# 환경 / 기본 설정
# -------------------------
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"

# Spyder에서 CLI 없이 기본값 주입용
OVERRIDE = {
    "enable": True,
    "csv_path": r"C:\Users\USER\Desktop\ice\data\cluster_0813_v4\cluster_daily_means_e1_k32_ws0_sorted.csv",
    "device": "cuda",
    "train_ratio": 0.7,
    "val_ratio": 0.1,
    "seq_input": 180,
    "batch_size": 24,
    "epochs": 50,
    "learning_rate": 1e-4,
    "dropout_n": 0.2,
    "d_llm": 4096,
    "e_layer": 1,
    "d_layer": 1,
    "head": 8,
    "weight_decay": 1e-3,
    "num_workers": 0,  # Windows/Spyder에서는 0 권장
    "seed": 2024,
    # 여러 horizon
    "output_lens": [180, 360, 720, 1080, 1440, 1800],
}

BASE_DATE = dt.datetime(2013, 1, 1)  # 기준 날짜

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--csv_path", type=str, required=False)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seq_input", type=int, default=180)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--dropout_n", type=float, default=0.2)
    p.add_argument("--d_llm", type=int, default=4096)
    p.add_argument("--e_layer", type=int, default=1)
    p.add_argument("--d_layer", type=int, default=1)
    p.add_argument("--head", type=int, default=8)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=2024)
    # 콤마로 구분된 pred_len 리스트를 받을 수도 있게
    p.add_argument("--output_lens", type=str, default="")
    args = p.parse_args()

    # OVERRIDE 주입
    if OVERRIDE.get("enable", False):
        for k, v in OVERRIDE.items():
            if k == "enable":
                continue
            if k == "output_lens":
                if args.output_lens == "":
                    setattr(args, "output_lens", v)
                else:
                    parsed = [int(x) for x in args.output_lens.split(",") if x.strip()]
                    setattr(args, "output_lens", parsed)
            else:
                if getattr(args, k, None) in [None, ""]:
                    setattr(args, k, v)
    else:
        # CLI에서 output_lens 파싱
        if isinstance(args.output_lens, str) and args.output_lens:
            args.output_lens = [int(x) for x in args.output_lens.split(",") if x.strip()]
        elif args.output_lens == "":
            args.output_lens = [180]

    if not args.csv_path:
        raise ValueError("csv_path가 비었습니다. OVERRIDE.enable=True 또는 --csv_path 로 지정하세요.")
    return args

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

def build_loaders(args, pred_len):
    """
    pred_len마다 Dataset_Custom/Loader 재생성
    """
    base = os.path.splitext(os.path.basename(args.csv_path))[0]
    # ✅ 이미지 임베딩 위치 사용
    embed_root = os.path.join("./Prompt_Embeddings_llama", base, f"out{pred_len}")

    ds_kwargs = dict(
        csv_path=args.csv_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        features='M',
        scale=True,
        timeenc=1,
        freq='D',
        d_llm=args.d_llm,
        expect_num_nodes=None,     # 채널 자동 확인
        embed_root=embed_root,     # 이미지 임베딩 폴더
        missing_embedding='error', # 임베딩 없으면 에러
    )

    # size = [seq_input, 0, pred_len]
    train_set = Dataset_Custom(flag='train', size=[args.seq_input, 0, pred_len], **ds_kwargs)
    test_set  = Dataset_Custom(flag='test',  size=[args.seq_input, 0, pred_len], **ds_kwargs)

    channel = train_set.C
    num_nodes = channel

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers
    )
    test_loader  = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=args.num_workers
    )

    return train_set, test_set, train_loader, test_loader, channel, num_nodes

def build_model(device, channel, num_nodes, seq_input, pred_len, args):
    model = Dual(
        device=device,
        channel=channel,
        num_nodes=num_nodes,
        seq_len=seq_input,
        pred_len=pred_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        head=args.head
    ).to(device)
    return model

# -------------------------
# 이미지 임베딩(dict) → 모델이 기대하는 텐서로 강제 변환 어댑터
# -------------------------
def move_emb_to_device(emb, device):
    """텐서/딕셔너리 모두 device로 이동"""
    if isinstance(emb, dict):
        out = {}
        for k, v in emb.items():
            if torch.is_tensor(v):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out
    return emb.to(device)

def expand_to_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    x: [B, d] → repeat해서 target_dim으로 확장 뒤 슬라이스
    """
    B, d = x.shape
    if d == target_dim:
        return x
    r = (target_dim + d - 1) // d
    x_rep = x.repeat(1, r)       # [B, d*r]
    return x_rep[:, :target_dim] # [B, target_dim]

def coerce_image_kv_to_prompt_tensor(emb: dict, d_llm: int, C: int) -> torch.Tensor:
    """
    emb: {'K':[B,N,dk], 'V':[B,N,dv], ...}
    반환: [B, d_llm, C, 1]  (기존 프롬프트 임베딩 텐서 형태)
    - 간단: K를 N축 평균 → d_llm 크기로 확장 → 채널 C로 타일링
    """
    if "K" not in emb:
        raise KeyError("Image embedding dict must contain key 'K'.")
    K = emb["K"]                  # [B, N, dk]
    if K.ndim != 3:
        raise ValueError(f"Expected K shape [B,N,dk], got {K.shape}")
    B, N, dk = K.shape
    k_mean = K.mean(dim=1)        # [B, dk]
    k_llm  = expand_to_dim(k_mean, d_llm)           # [B, d_llm]
    k_llm  = k_llm.view(B, d_llm, 1, 1).repeat(1, 1, C, 1)  # [B, d_llm, C, 1]
    return k_llm

def main():
    args = parse_args()
    seed_it(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("===== Experiment Config =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=============================")

    results = {}  # pred_len -> best dict

    for seq_output in args.output_lens:
        print(f"\n>>> Running experiment with pred_len = {seq_output} days")

        # 1) 데이터/로더
        train_set, test_set, train_loader, test_loader, channel, num_nodes = build_loaders(args, seq_output)
        n_features = channel
        print(f"[Data] channels(C)={n_features}, train_windows={len(train_set)}, test_windows={len(test_set)}")

        # 2) 모델/옵티마이저
        model = build_model(device, channel, num_nodes, args.seq_input, seq_output, args)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        print("The number of trainable parameters:", model.count_trainable_params())
        print("The number of parameters:", model.param_num())

        # 3) Best tracking
        best = [1e5, 1e5, -1e5]  # [mse, mae, corr]
        now = dt.datetime.now()
        model_tag = f"TimeCMA_{seq_output}"

        # 4) 학습 루프
        for epoch in range(args.epochs):
            model.train()
            for batch in train_loader:
                x, y, x_mark, y_mark, emb = batch
                x = x.to(device)           # [B, L, C]
                y = y.to(device)           # [B, O, C]
                x_mark = x_mark.to(device) # [B, L, d_time]

                # 이미지 임베딩(dict) → device → 텐서 강제 변환
                emb = move_emb_to_device(emb, device)
                if isinstance(emb, dict):
                    emb = coerce_image_kv_to_prompt_tensor(emb, args.d_llm, channel)  # [B, d_llm, C, 1]

                optimizer.zero_grad()
                out = model(x, x_mark, emb)  # [B, O, C]
                loss = F.mse_loss(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            # 5) 평가
            model.eval()
            outputs, actuals = [], []
            with torch.no_grad():
                for batch in test_loader:
                    x, y, x_mark, y_mark, emb = batch
                    x = x.to(device)
                    y = y.to(device)
                    x_mark = x_mark.to(device)

                    emb = move_emb_to_device(emb, device)
                    if isinstance(emb, dict):
                        emb = coerce_image_kv_to_prompt_tensor(emb, args.d_llm, channel)

                    pred = model(x, x_mark, emb)
                    outputs.append(pred.cpu())
                    actuals.append(y.cpu())

            outputs = torch.cat(outputs, dim=0)  # [Nt, O, C]
            actuals = torch.cat(actuals, dim=0)  # [Nt, O, C]

            mse, mae, corr = Metric.metric(outputs, actuals, n_features)
            best = Metric.update(
                now, save=True, model=model, best=best,
                metric=[mse, mae, corr],
                model_name=model_tag, seq_output=seq_output, epoch=epoch
            )

            if (epoch % 10 == 0) or (epoch == args.epochs - 1):
                Metric.plot(outputs, actuals, model_tag, seq_output, now)

            print(f"[Model: {model_tag} / Epoch: {epoch}]")
            print(f"[B_MSE: {best[0]:.9f} / B_MAE: {best[1]:.8f} / B_COR: {best[2]:.4f}]")
            print(f"[MSE: {mse:.9f} / MAE: {mae:.8f} / COR: {corr:.4f}]\n")

        # 6) 예측 결과 CSV 저장 (첫 배치 샘플)
        if len(test_set) > 0:
            first_batch_index = int(test_set.indices[0])
            predict_dates = [
                (BASE_DATE + dt.timedelta(days=int(first_batch_index + args.seq_input + i))).strftime("%Y-%m-%d")
                for i in range(seq_output)
            ]
            pred_np = outputs[0].numpy()  # [O, C]
            true_np = actuals[0].numpy()  # [O, C]

            out_dir = f'./STMA_node/{model_tag}/models/{model_tag}_{now.month}{now.day}{now.hour}{now.minute}'
            os.makedirs(out_dir, exist_ok=True)

            for r in range(pred_np.shape[1]):  # 각 region(feature)
                df = pd.DataFrame({
                    "Date": predict_dates,
                    "Prediction": pred_np[:, r],
                    "Actual": true_np[:, r]
                })
                df.to_csv(os.path.join(out_dir, f"region_{r}_pred_{seq_output}.csv"), index=False)

        # 7) 결과 집계
        results[seq_output] = {'b_mse': best[0], 'b_mae': best[1], 'b_cor': best[2]}

        # 메모리 정리
        gc.collect()
        torch.cuda.empty_cache()

    # 8) 최종 요약
    print("\n====== Final Summary ======")
    for key, val in results.items():
        print(f"[{key}] B_MSE: {val['b_mse']:.6f}, B_MAE: {val['b_mae']:.6f}, B_COR: {val['b_cor']:.4f}")

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
