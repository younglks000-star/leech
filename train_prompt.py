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
import matplotlib.pyplot as plt  # ★ 화면 표시용(저장 안 함)

from data_provider.data_loader_emb_0911 import Dataset_Custom
from models.TimeCMA import Dual
import utils.Metric_node2 as Metric

import faulthandler
faulthandler.enable()

# -------------------------
# 환경 / 기본 설정
# -------------------------
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"

OVERRIDE = {
    "enable": True,
    "csv_path": r"C:\Users\USER\Desktop\ice\data\cluster_0813_v4\cluster_daily_means_e1_k32_ws0_sorted.csv",
    "device": "cuda",
    "train_ratio": 0.7,
    "val_ratio": 0.1,
    "seq_input": 360,
    "batch_size": 24,
    "epochs": 30,
    "learning_rate": 1e-4,
    "dropout_n": 0.2,
    "d_llm": 4096,
    "e_layer": 1,
    "d_layer": 1,
    "head": 8,
    "weight_decay": 1e-3,
    "num_workers": 0,
    "seed": 2024,
    "output_lens": [180, 360, 720, 1080, 1440, 1800],
}

BASE_DATE = dt.datetime(2013, 1, 1)

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
    p.add_argument("--output_lens")
    
    if OVERRIDE.get("enable", False):
        defaults = {k: v for k, v in OVERRIDE.items() if k != "enable"}
        p.set_defaults(**defaults)

    args = p.parse_args()

    if isinstance(args.output_lens, str):
        s = args.output_lens.strip()
        if s == "":
            args.output_lens = OVERRIDE.get("output_lens", [180]) if OVERRIDE.get("enable", False) else [180]
        else:
            args.output_lens = [int(x) for x in s.split(",") if x.strip()]
    elif args.output_lens is None:
        args.output_lens = OVERRIDE.get("output_lens", [180]) if OVERRIDE.get("enable", False) else [180]

    if not args.csv_path:
        raise ValueError("csv_path가 비었습니다. OVERRIDE.enable=True로 기본값을 쓰거나, --csv_path로 지정하세요.")

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
    base = os.path.splitext(os.path.basename(args.csv_path))[0]
    embed_root = os.path.join("./Prompt_Embeddings_llama360", base, f"out{pred_len}")

    ds_kwargs = dict(
        csv_path=args.csv_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        features='M',
        scale=True,
        timeenc=1,
        freq='D',
        d_llm=args.d_llm,
        expect_num_nodes=None,
        embed_root=embed_root,
        missing_embedding='error',
    )

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

def move_emb_to_device(emb, device):
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
    B, d = x.shape
    if d == target_dim:
        return x
    r = (target_dim + d - 1) // d
    x_rep = x.repeat(1, r)
    return x_rep[:, :target_dim]

def coerce_image_kv_to_prompt_tensor(emb: dict, d_llm: int, C: int) -> torch.Tensor:
    if "K" not in emb:
        raise KeyError("Image embedding dict must contain key 'K'.")
    K = emb["K"]
    if K.ndim != 3:
        raise ValueError(f"Expected K shape [B,N,dk], got {K.shape}")
    B, N, dk = K.shape
    k_mean = K.mean(dim=1)
    k_llm  = expand_to_dim(k_mean, d_llm)
    k_llm  = k_llm.view(B, d_llm, 1, 1).repeat(1, 1, C, 1)
    return k_llm

def main():
    args = parse_args()
    seed_it(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("===== Experiment Config =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=============================")

    results = {}

    for seq_output in args.output_lens:
        print(f"\n>>> Running experiment with pred_len = {seq_output} days")

        train_set, test_set, train_loader, test_loader, channel, num_nodes = build_loaders(args, seq_output)
        n_features = channel
        print(f"[Data] channels(C)={n_features}, train_windows={len(train_set)}, test_windows={len(test_set)}")

        model = build_model(device, channel, num_nodes, args.seq_input, seq_output, args)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        print("The number of trainable parameters:", model.count_trainable_params())
        print("The number of parameters:", model.param_num())

        best = [1e5, 1e5, -1e5]
        now = dt.datetime.now()
        model_tag = f"TimeCMA_{seq_output}"

        for epoch in range(args.epochs):
            model.train()
            for batch in train_loader:
                x, y, x_mark, y_mark, emb = batch
                x = x.to(device)
                y = y.to(device)
                x_mark = x_mark.to(device)

                emb = move_emb_to_device(emb, device)
                if isinstance(emb, dict):
                    emb = coerce_image_kv_to_prompt_tensor(emb, args.d_llm, channel)

                optimizer.zero_grad()
                out = model(x, x_mark, emb)
                loss = F.mse_loss(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

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

            outputs = torch.cat(outputs, dim=0)
            actuals = torch.cat(actuals, dim=0)

            mse, mae, corr = Metric.metric(outputs, actuals, n_features)
            best = Metric.update(
                now, save=True, model=model, best=best,
                metric=[mse, mae, corr],
                model_name=model_tag, seq_output=seq_output, epoch=epoch
            )

            # ---------------- 화면 출력만(저장 X) ----------------
            if (epoch % 10 == 0) or (epoch == args.epochs - 1):
                pred_np = outputs.numpy()   # [Nb, O, C]
                true_np = actuals.numpy()
                b = pred_np.shape[0] // 2   # 배치 중간 샘플
                C = pred_np.shape[-1]

                # 4개씩 묶어(2x2) 모든 클러스터 출력
                for start in range(0, C, 4):
                    end = min(start + 4, C)
                    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
                    axes = axs.flatten()
                    for j, ch in enumerate(range(start, end)):
                        ax = axes[j]
                        ax.plot(pred_np[b, :, ch], label="prediction", color="r")
                        ax.plot(true_np[b, :, ch], label="actual", color="b")
                        ax.set_title(f"Cluster {ch}")
                    # 남는 서브플롯은 제거
                    for k in range(end - start, 4):
                        fig.delaxes(axes[k])
                    # 공통 범례
                    handles, labels = axes[0].get_legend_handles_labels()
                    fig.legend(handles, labels, loc="upper right")
                    fig.suptitle(f"Clusters {start}–{end - 1}")
                    plt.tight_layout()
                    plt.show()
            # ---------------------------------------------------

            print(f"[Model: {model_tag} / Epoch: {epoch}]")
            print(f"[B_MSE: {best[0]:.9f} / B_MAE: {best[1]:.8f} / B_COR: {best[2]:.4f}]")
            print(f"[MSE: {mse:.9f} / MAE: {mae:.8f} / COR: {corr:.4f}]\n")

        if len(test_set) > 0:
            first_batch_index = int(test_set.indices[0])
            predict_dates = [
                (BASE_DATE + dt.timedelta(days=int(first_batch_index + args.seq_input + i))).strftime("%Y-%m-%d")
                for i in range(seq_output)
            ]
            pred_np = outputs[0].numpy()
            true_np = actuals[0].numpy()

            out_dir = f'./STMA_node/{model_tag}/models/{model_tag}_{now.month}{now.day}{now.hour}{now.minute}'
            os.makedirs(out_dir, exist_ok=True)

            for r in range(pred_np.shape[1]):
                df = pd.DataFrame({
                    "Date": predict_dates,
                    "Prediction": pred_np[:, r],
                    "Actual": true_np[:, r]
                })
                df.to_csv(os.path.join(out_dir, f"region_{r}_pred_{seq_output}.csv"), index=False)

        results[seq_output] = {'b_mse': best[0], 'b_mae': best[1], 'b_cor': best[2]}

        gc.collect()
        torch.cuda.empty_cache()

    print("\n====== Final Summary ======")
    for key, val in results.items():
        print(f"[{key}] B_MSE: {val['b_mse']:.6f}, B_MAE: {val['b_mae']:.6f}, B_COR: {val['b_cor']:.4f}")

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
