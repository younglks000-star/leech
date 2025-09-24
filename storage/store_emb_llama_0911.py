# store_emb_llama_0911.py
# -*- coding: utf-8 -*-

import os
import time
import h5py
import torch
import argparse
from argparse import Namespace

# --- 프로젝트 경로 맞추기 ---
from data_provider.data_loader_save_0908 import Dataset_Custom
from storage.gen_prompt_llama_emb import GenPromptEmb

# =========================
# Spyder 실행을 위한 OVERRIDE
# =========================
OVERRIDE = {
    "enable": True,
    "device": "cuda",  # "cuda:0"도 OK
    "csv_path": r"C:\Users\USER\Desktop\ice\data\cluster_0813_v4\cluster_daily_means_e1_k32_ws0_sorted.csv",
    "input_len": 180,
    "output_lens": [180, 360, 720, 1080, 1440, 1800],
    "train_ratio": 0.7,
    "val_ratio": 0.1,

    # ---- LLaMA 관련 (권장: FP16 단일 GPU) ----
    "model_name": "meta-llama/Llama-2-7b-hf",
    "load_in_4bit": False,
    "load_in_8bit": False,
    "torch_dtype": "float16",   # FP16 강제
    # FP16 전체 탑재이므로 반드시 None (auto 금지)
    "device_map": None,

    # 저장 루트
    "save_root": "./Prompt_Embeddings_llama",
    # 템플릿 선택용 data_path (표현만; 실제 CSV 도메인과 무관)
    "data_path_name": "FRED",
}

def parse_args_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--csv_path", type=str, required=False)
    p.add_argument("--input_len", type=int, default=180)
    p.add_argument("--output_lens", type=int, nargs="+",
                   default=[180, 360, 720, 1080, 1440, 1800])
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)

    # LLaMA / 임베딩
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--torch_dtype", type=str, default="float16", help="float16|bfloat16|float32")
    p.add_argument("--device_map", type=str, default=None, help='None or "auto" (FP16 단일 GPU는 None 권장)')

    # 저장
    p.add_argument("--save_root", type=str, default="./Prompt_Embeddings_llama")
    p.add_argument("--data_path_name", type=str, default="FRED")
    return p.parse_args()

def to_dtype(s):
    if s is None: return None
    s = s.lower()
    if s == "float16": return torch.float16
    if s == "bfloat16": return torch.bfloat16
    if s == "float32": return torch.float32
    return None

def get_args():
    if OVERRIDE.get("enable", False):
        defaults = parse_args_cli().__dict__
        defaults.update({k:v for k,v in OVERRIDE.items() if k != "enable"})
        return Namespace(**defaults)
    return parse_args_cli()

def process_split(args, divide, output_len):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- 데이터셋 ---
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

    # --- 프롬프트 임베딩 생성기 (LLaMA, FP16 단일 GPU) ---
    gen = GenPromptEmb(
        data_path=args.data_path_name,
        model_name=args.model_name,
        device=device,
        input_len=args.input_len,
        d_model=None,                    # LLaMA hidden_size 자동(4096)
        layer=None,
        divide=divide,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=to_dtype(args.torch_dtype),  # torch.float16
        device_map=args.device_map              # None (중요)
    )
    # ⚠️ 중요: device_map=None 모드에서는 GenPromptEmb 내부에서 model.to(device) 수행함.
    # 따라서 여기서 gen 자체에 .to(device)를 다시 호출하지 않습니다.

    # --- 저장 경로 ---
    base = os.path.splitext(os.path.basename(args.csv_path))[0]
    save_dir = os.path.join(args.save_root, base, f"out{output_len}", divide)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[INFO] divide={divide}, output_len={output_len}, windows={len(ds)}, save_dir={save_dir}")

    t1 = time.time()
    for idx in range(len(ds)):
        x, y, x_mark, y_mark = ds[idx]     # x: [T,C], x_mark: [T, ...]
        x_b = x.unsqueeze(0).to(device)    # [1,T,C]
        x_mark_b = x_mark.unsqueeze(0).to(device)

        # GenPromptEmb.generate_embeddings → [B, d_dim, C] (d_dim=4096)
        with torch.no_grad():
            emb = gen.generate_embeddings(x_b, x_mark_b).squeeze(0).detach().cpu().numpy()

        s = int(ds.indices[idx])
        fpath = os.path.join(save_dir, f"{s}.h5")
        with h5py.File(fpath, "w") as hf:
            hf.create_dataset("embeddings", data=emb)  # shape: [d_dim, C]

    print(f"[DONE] {divide} {output_len} → {len(ds)} windows saved in {(time.time()-t1)/60:.2f} min")

def main():
    args = get_args()
    for output_len in args.output_lens:
        for divide in ["train", "val", "test"]:
            process_split(args, divide, output_len)

if __name__ == "__main__":
    main()
