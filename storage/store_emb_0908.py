# -*- coding: utf-8 -*-
"""
프롬프트 임베딩 저장 스크립트 (윈도우 단위)
- 저장 경로를 ./Prompt_Embeddings/... 로 변경
- Image_Embeddings와 확실히 구분
"""

import os
import time
import h5py
import torch
import argparse
from argparse import Namespace

# --- 프로젝트 경로 맞추기 ---
from data_provider.data_loader_save_0908 import Dataset_Custom
from storage.gen_CTprompt_emb import GenPromptEmb

# =========================
# Spyder 실행을 위한 OVERRIDE
# =========================
OVERRIDE = {
    "enable": True,
    "device": "cuda",
    "csv_path": r"C:\Users\USER\Desktop\ice\data\cluster_0813_v4\cluster_daily_means_e1_k32_ws0_sorted.csv",
    "input_len": 180,         
    "output_lens": [180, 360, 720, 1080, 1440, 1800],
    "train_ratio": 0.7,
    "val_ratio": 0.1,
    "model_name": "gpt2",
    "d_model": 768,
    "save_root": "./CTPrompt_Embeddings",   # ✅ 경로 변경 (원래는 ./Embeddings)
}

def parse_args_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--csv_path", type=str, required=False)
    p.add_argument("--input_len", type=int, default=180)
    p.add_argument("--output_lens", type=int, nargs="+", default=[180, 360, 720, 1080, 1440, 1800])
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--save_root", type=str, default="./CTPrompt_Embeddings")  # ✅ 기본값도 변경
    return p.parse_args()

def get_args():
    if OVERRIDE.get("enable", False):
        defaults = parse_args_cli().__dict__
        defaults.update({k:v for k,v in OVERRIDE.items() if k != "enable"})
        return Namespace(**defaults)
    return parse_args_cli()

def process_split(args, divide, output_len):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 데이터셋
    ds = Dataset_Custom(
        csv_path=args.csv_path,
        flag=divide,
        size=[args.input_len, 0, output_len],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        features='M',
        scale=False,
        timeenc=0,
        freq='D'
    )

    # 프롬프트 임베딩 생성기
    gen = GenPromptEmb(
        data_path='FRED',
        model_name=args.model_name,
        device=device,
        input_len=args.input_len,
        d_model=args.d_model,
        layer=12,
        divide=divide
    ).to(device)

    # 저장 경로
    base = os.path.splitext(os.path.basename(args.csv_path))[0]
    save_dir = os.path.join(args.save_root, base, f"out{output_len}", divide)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[INFO] divide={divide}, output_len={output_len}, windows={len(ds)}, save_dir={save_dir}")

    t1 = time.time()
    for idx in range(len(ds)):
        x, y, x_mark, y_mark = ds[idx]
        x_b = x.unsqueeze(0).to(device)
        x_mark_b = x_mark.unsqueeze(0).to(device)

        emb = gen.generate_embeddings(x_b, x_mark_b).squeeze(0).detach().cpu().numpy()

        s = int(ds.indices[idx])
        fpath = os.path.join(save_dir, f"{s}.h5")
        with h5py.File(fpath, "w") as hf:
            hf.create_dataset("embeddings", data=emb)

    print(f"[DONE] {divide} {output_len} → {len(ds)} windows saved in {(time.time()-t1)/60:.2f} min")

def main():
    args = get_args()
    for output_len in args.output_lens:
        for divide in ["train", "val", "test"]:
            process_split(args, divide, output_len)

if __name__ == "__main__":
    main()
