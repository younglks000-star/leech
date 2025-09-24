# -*- coding: utf-8 -*-
"""
윈도우 정렬 보장 + 이미지 전용 저장소 + 진단 로그
Dataset_Custom 윈도우와 1:1로 NSIDC concentration 이미지 패치 K/V 저장
- 저장 규약: ./Image_Embeddings_patch/<csv_basename>/out{output_len}/{divide}/{s}.h5
- 각 파일: datasets 'K'(N,dk), 'V'(N,dv), 'meta' attrs
"""

import os, re, math, h5py, time
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import rasterio
from torchvision import transforms

# === 프로젝트 로더 ===
from data_provider.data_loader_save_0908 import Dataset_Custom

# === OVERRIDE ===
OVERRIDE = {
    "device": "cuda",
    "csv_path": r"C:\Users\USER\Desktop\ice\data\cluster_0813_v4\cluster_daily_means_e1_k32_ws0_sorted.csv",
    "input_len": 180,
    "output_lens": [180, 360, 720, 1080, 1440, 1800],
    "train_ratio": 0.7,
    "val_ratio": 0.1,

    # NSIDC GeoTIFF 루트 (연/월/파일 구조)
    "nsidc_root": r"C:\Users\USER\Desktop\ice\data\NSIDC_Data",

    # 백본 (timm: dinov2 ViT-B/14)
    "timm_model": "vit_base_patch14_dinov2",

    # 투영 차원 (프롬프트 K,V 차원과 동일하게)
    "dk": 256, "dv": 256,

    # 전처리
    "keep_aspect": True,
    "fp16": True,

    # 이미 존재하면 건너뜀(False), 강제 재계산(True)
    "overwrite": False,

    # ✅ 이미지 전용 저장 루트 (프롬프트와 분리)
    "save_root": r".\Image_Embeddings_patch",
}

DATE_RE = re.compile(r"N_(\d{8})_concentration_v3\.0\.tif$", re.IGNORECASE)

def list_concentration_files(root_dir):
    files = glob(os.path.join(root_dir, "**", "*_concentration_v3.0.tif"), recursive=True)
    ymd2path = {}
    for f in files:
        m = DATE_RE.search(os.path.basename(f))
        if m:
            ymd2path[m.group(1)] = f
    return ymd2path   # {"YYYYMMDD": full_path}

def to_3ch_resize(img2d, H, W, keep_aspect=False):
    """img2d: (H0,W0) float32 [0,1] -> torch (1,3,H,W)"""
    img3 = np.repeat(img2d[..., None], 3, axis=-1)  # (H0,W0,3)
    if keep_aspect:
        H0, W0 = img3.shape[:2]
        scale = min(H / max(H0,1), W / max(W0,1))
        newH, newW = max(1, int(round(H0*scale))), max(1, int(round(W0*scale)))
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Resize((newH, newW))])
        t = tfm(img3)  # (3,newH,newW)
        padH, padW = H-newH, W-newW
        pad = (padW//2, padW-padW//2, padH//2, padH-padH//2)  # L,R,T,B
        t = torch.nn.functional.pad(t, pad, mode="constant", value=0.0)
        return t.unsqueeze(0)  # (1,3,H,W)
    else:
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Resize((H, W))])
        return tfm(img3).unsqueeze(0)


# ----- timm ViT 패치 토큰 추출기 -----
class TimmViTPatchBackbone:
    def __init__(self, model_name, device):
        import timm
        self.model = timm.create_model(model_name, pretrained=True).to(device).eval()
        self.device = device

        # 입력 해상도 / 정규화 파라미터를 timm에서 직접 가져오기
        cfg = getattr(self.model, "pretrained_cfg", getattr(self.model, "default_cfg", {}))
        in_size = cfg.get("input_size", (3, 224, 224))   # (C,H,W)
        self.in_h = in_size[1]
        self.in_w = in_size[2]
        mean = cfg.get("mean", (0.5,0.5,0.5))
        std  = cfg.get("std",  (0.5,0.5,0.5))
        self.mean = torch.tensor(mean, device=device).view(1,3,1,1)
        self.std  = torch.tensor(std,  device=device).view(1,3,1,1)

        self.dim = int(getattr(self.model, "num_features", 768))
        self.patch = int(getattr(self.model.patch_embed, "patch_size", (14,14))[0])

    @torch.no_grad()
    def patch_tokens(self, img_tensor_HW3):  # (1,3,H,W)
        x = img_tensor_HW3.clone().to(self.device)
        x = (x - self.mean) / self.std   # timm cfg 기준 정규화
        out = self.model.forward_features(x)
        if isinstance(out, dict):
            if "x_norm_patchtokens" in out:
                pt = out["x_norm_patchtokens"]          # (B,N,D)
            elif "x" in out and out["x"].ndim == 3:
                pt = out["x"][:, 1:, :]                  # (B,1+N,D) → patch-only
            else:
                raise RuntimeError("패치 토큰을 찾을 수 없습니다.")
        else:
            if out.ndim == 3 and out.shape[1] > 1:
                pt = out[:, 1:, :]
            else:
                raise RuntimeError("forward_features 결과에서 패치 토큰을 해석할 수 없습니다.")
        return pt.squeeze(0)  # (N,D)


def process_split(ops, divide, output_len, ymd2path):
    device = torch.device(ops["device"] if torch.cuda.is_available() else "cpu")

    ds = Dataset_Custom(
        csv_path=ops["csv_path"],
        flag=divide,
        size=[ops["input_len"], 0, output_len],
        train_ratio=ops["train_ratio"],
        val_ratio=ops["val_ratio"],
        features='M',
        scale=False,
        timeenc=0,
        freq='D'
    )

    backbone = TimmViTPatchBackbone(ops["timm_model"], device)
    d_img = backbone.dim
    Wk = nn.Linear(d_img, ops["dk"]).to(device).eval()
    Wv = nn.Linear(d_img, ops["dv"]).to(device).eval()
    nn.init.xavier_uniform_(Wk.weight); nn.init.zeros_(Wk.bias)
    nn.init.xavier_uniform_(Wv.weight); nn.init.zeros_(Wv.bias)

    base = os.path.splitext(os.path.basename(ops["csv_path"]))[0]
    save_dir = os.path.join(ops["save_root"], base, f"out{output_len}", divide)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[INFO] divide={divide}, out_len={output_len}, windows={len(ds)}, save_dir={save_dir}")
    t0 = time.time()
    n_saved = n_skip_exist = n_skip_missing = 0

    for idx in tqdm(range(len(ds))):
        x, y, x_mark, y_mark = ds[idx]

        s = int(ds.indices[idx])
        fpath = os.path.join(save_dir, f"{s}.h5")
        if (not ops["overwrite"]) and os.path.exists(fpath):
            n_skip_exist += 1
            continue

        # 윈도우 마지막 입력 시점 날짜
        year  = int(x_mark[ops["input_len"]-1, 0])
        month = int(x_mark[ops["input_len"]-1, 1])
        day   = int(x_mark[ops["input_len"]-1, 2])
        ymd = f"{year:04d}{month:02d}{day:02d}"

        tif_path = ymd2path.get(ymd, None)
        if tif_path is None:
            n_skip_missing += 1
            continue

        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            if arr.max() > 1.5:
                arr = arr / 100.0
            arr = np.clip(arr, 0.0, 1.0)

        # ✅ 백본 입력 크기로 리사이즈
        imgHW = to_3ch_resize(arr, backbone.in_h, backbone.in_w, keep_aspect=ops["keep_aspect"])

        with torch.no_grad():
            patches = backbone.patch_tokens(imgHW)    # (N, d_img)
            K = Wk(patches.to(device))                 # (N, dk)
            V = Wv(patches.to(device))                 # (N, dv)
            if ops["fp16"]:
                K = K.half(); V = V.half()

        with h5py.File(fpath, "w") as hf:
            hf.create_dataset("K", data=K.detach().cpu().numpy())
            hf.create_dataset("V", data=V.detach().cpu().numpy())
            meta = hf.create_group("meta")
            meta.attrs["window_start_idx"] = s
            meta.attrs["t_end_ymd"] = ymd
            meta.attrs["src_path"] = tif_path
            meta.attrs["model"] = ops["timm_model"]

            grid_h = backbone.in_h // backbone.patch
            grid_w = backbone.in_w // backbone.patch
            meta.attrs["grid_h"] = grid_h
            meta.attrs["grid_w"] = grid_w
            meta.attrs["patch"]  = backbone.patch

            meta.attrs["d_img"]  = int(d_img)
            meta.attrs["dk"] = int(ops["dk"]); meta.attrs["dv"] = int(ops["dv"])
            meta.attrs["keep_aspect"] = int(ops["keep_aspect"])
            meta.attrs["dtype"] = "fp16" if ops["fp16"] else "fp32"

        n_saved += 1

    dt_min = (time.time()-t0)/60.0
    print(f"[DONE] {divide} out{output_len}: windows={len(ds)} | saved={n_saved} | "
          f"skip_exists={n_skip_exist} | skip_missing_tif={n_skip_missing} | time={dt_min:.2f} min")


def run(ops):
    ymd2path = list_concentration_files(ops["nsidc_root"])
    for out_len in ops["output_lens"]:
        for divide in ["train", "val", "test"]:
            process_split(ops, divide, out_len, ymd2path)

if __name__ == "__main__":
    run(OVERRIDE)
