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
from torch.utils.data import DataLoader 1234511

from data_provider.data_loader_emb_0922 import Dataset_Custom
from models.TimeCMA_0924v7 import Dual
import utils.Metric_node as Metric

import faulthandler
faulthandler.enable()

# -------------------------
# 환경 / 기본 설정
# -------------------------
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"

OVERRIDE = {
    "enable": True,  # True면 아래 값들이 argparse 기본값으로 주입됨. CLI가 있으면 그 항목만 덮어씀.
    "csv_path": r"C:\Users\USER\Desktop\ice\data\cluster_0813_v4\cluster_daily_means_e1_k32_ws0_sorted.csv",
    "device": "cuda",
    "train_ratio": 0.7,
    "val_ratio": 0.1,
    "seq_input": 360,
    "batch_size": 16,
    "epochs": 30,
    "learning_rate": 1e-4,
    "dropout_n": 0.2,
    "d_llm": 4096,
    "e_layer": 1,
    "d_layer": 1,
    "head": 8,
    "weight_decay": 1e-3,
    "num_workers": 0,  # Windows/Spyder에서는 0 권장
    "seed": 2024,
    # 경로(당신 환경에 맞춰 기본값 세팅)
    "prompt_emb_base": r".\Prompt_Embeddings_llama360",
    "image_emb_base":  r"C:\Users\USER\Desktop\baseline\ICECMA\Image_Embeddings_cluster_768",
    # 여러 horizon
    "output_lens": [180, 360, 720, 1080, 1440, 1800],
}

BASE_DATE = dt.datetime(2013, 1, 1)  # 기준 날짜


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str)
    p.add_argument("--csv_path", type=str)
    p.add_argument("--train_ratio", type=float)
    p.add_argument("--val_ratio", type=float)
    p.add_argument("--seq_input", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--dropout_n", type=float)
    p.add_argument("--d_llm", type=int)
    p.add_argument("--e_layer", type=int)
    p.add_argument("--d_layer", type=int)
    p.add_argument("--head", type=int)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--num_workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--prompt_emb_base")
    p.add_argument("--image_emb_base")
    p.add_argument("--output_lens")

    if OVERRIDE.get("enable", False):
        defaults = {k: v for k, v in OVERRIDE.items() if k != "enable"}
        p.set_defaults(**defaults)

    args = p.parse_args()

    # output_lens normalize
    if isinstance(args.output_lens, str):
        s = args.output_lens.strip()
        args.output_lens = [int(x) for x in s.split(",") if x.strip()] if s else [180]
    elif args.output_lens is None:
        args.output_lens = [180]

    if not args.csv_path:
        raise ValueError("csv_path가 비었습니다.")

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
    ds_kwargs = dict(
        csv_path=args.csv_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        features='M',
        scale=True,
        timeenc=1,
        freq='D',
        d_llm=args.d_llm,
        expect_num_nodes=None,               # CSV 채널과 일치 여부는 내부에서 검증
        prompt_embed_base=args.prompt_emb_base,
        image_embed_base=args.image_emb_base,
        # missing 정책은 필요시 Dataset_Custom 내부 기본값 사용
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
                x, y, x_mark, y_mark, emb_prompt, emb_image = batch
                x = x.to(device)                     # [B, L, C]
                y = y.to(device)                     # [B, O, C]
                x_mark = x_mark.to(device)           # [B, L, d_time]
                emb_prompt = emb_prompt.to(device)   # [B, E, C] or [B, E, C, 1]
                # emb_image는 dict(K,V,meta)로 collate되어 K:[B,C,dk], V:[B,C,dv]
                # 모델 내부에서 device/shape를 정리합니다.

                optimizer.zero_grad()
                out = model(x, x_mark, emb_prompt, emb_image)  # [B, O, C]
                loss = F.mse_loss(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            # 5) 평가
            model.eval()
            outputs, actuals = [], []
            with torch.no_grad():
                for batch in test_loader:
                    x, y, x_mark, y_mark, emb_prompt, emb_image = batch
                    x = x.to(device)
                    y = y.to(device)
                    x_mark = x_mark.to(device)
                    emb_prompt = emb_prompt.to(device)

                    pred = model(x, x_mark, emb_prompt, emb_image)
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
        if len(test_set) > 0 and outputs.size(0) > 0:
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
