# -*- coding: utf-8 -*-
"""
(속도 최적화 버전) ViT 패치 임베딩 → 클러스터 풀링 저장
- 윈도우 단위 처리이지만, 배치 추론(DataLoader)로 속도 개선
- 근본 최적화는 '날짜별 1회 추론 캐시' 권장(별도 파이프라인)
"""

# ==== ⚠️ Spyder/Windows에서 OpenMP 중복 충돌 회피 (가장 먼저!) ====
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 임시 회피(안전X)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("GDAL_NUM_THREADS", "1")
os.environ.setdefault("CPL_CPU_COUNT", "1")
# ===================================================================

import re, h5py
from glob import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import rasterio
from torch.utils.data import DataLoader, Dataset
from contextlib import contextmanager

# PyTorch도 CPU 스레드 1로 제한 (전처리 안정화)
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True

# ==================== 설정 ====================
CFG = {
    # --- 데이터 ---
    "csv_path": r"C:\Users\USER\Desktop\ice\data\cluster_0813_v4\cluster_daily_means_e1_k32_ws0_sorted.csv",
    "cluster_map_sorted_npy": r"C:\Users\USER\Desktop\ice\data\cluster_0813_v4\cluster_map_e1_k32_ws0_sorted.npy",
    "mapping_json":           r"C:\Users\USER\Desktop\ice\data\cluster_0813_v4\cluster_id_order_by_center_e1_k32_ws0.json",
    "params_json":            r"C:\Users\USER\Desktop\ice\data\cluster_0813_v4\params_e1_k32_ws0.json",
    "nsidc_root":             r"C:\Users\USER\Desktop\ice\data\NSIDC_Data",
    "use_tif": "concentration",   # "concentration" | "extent"

    # --- 윈도우/분할 ---
    "input_len": 360,
    "output_lens": [1440, 1800],
    "train_ratio": 0.7,
    "val_ratio": 0.1,
    "splits": ["train", "val", "test"],

    # --- ViT 백본(timm DINOv2-B/14) ---
    "timm_model": "vit_base_patch14_dinov2",
    "keep_aspect": True,

    # --- 차원/정밀도 ---
    "dk": 768, "dv": 768,      # 4096 꼭 필요 없으면 768 권장(저장/속도/메모리 효율↑)
    "fp16": True,              # GPU면 True 권장(autocast)

    # --- 시간 집계 ---
    "temporal_mode": "ewa",    # "last" | "mean" | "ewa" | "stack"
    "ewa_gamma": 0.05,

    # --- 저장 ---
    "save_root": r".\Image_Embeddings_cluster_768",
    "overwrite": False,
    "save_patch_last": False,   # 패치 레벨 저장 X 권장
    "save_cluster_agg": True,
    "save_cluster_seq": False,

    # --- ⚡️ 성능 최적화 설정 ---
    "batch_size": 64,          # VRAM에 맞춰 조절
    "num_workers": 0,          # Spyder/Windows면 0 권장(콘솔 실행 시 늘려도 됨)

    # --- 장치 ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
# ============================================

# ========== 유틸: 날짜→GeoTIFF 경로 ==========
DATE_RE     = re.compile(r"N_(\d{8})_concentration_v3\.0\.tif$", re.IGNORECASE)
DATE_RE_EXT = re.compile(r"N_(\d{8})_extent_v3\.0\.tif$",        re.IGNORECASE)

def index_nsidc(root, which="concentration"):
    pat = "*_concentration_v3.0.tif" if which == "concentration" else "*_extent_v3.0.tif"
    files = glob(os.path.join(root, "**", pat), recursive=True)
    ymd2path = {}
    for f in files:
        base = os.path.basename(f)
        m = DATE_RE.search(base) if which=="concentration" else DATE_RE_EXT.search(base)
        if m:
            ymd2path[m.group(1)] = f
    # 날짜순 정렬된 dict
    return dict(sorted(ymd2path.items()))

# ========== 라벨맵 → M, A ==========
def build_patch2cluster_weights(label_map: np.ndarray, num_clusters: int,
                                target_hw: int, patch: int,
                                keep_aspect: bool=True, invalid_label: int=-1, eps=1e-8):
    """라벨맵(정렬된 0..C-1)을 ViT 캔버스에 맞춰 최근접 리사이즈+중앙패딩 → 패치별 라벨 분포 M, 평균풀링행렬 A 생성"""
    H0, W0 = label_map.shape
    if keep_aspect:
        s = min(target_hw / max(H0,1), target_hw / max(W0,1))
        newH, newW = max(1, int(round(H0*s))), max(1, int(round(W0*s)))
    else:
        newH, newW = target_hw, target_hw

    # 최근접 리사이즈
    lab = Image.fromarray(label_map.astype(np.int32), mode='I')
    lab_rz = np.array(lab.resize((newW,newH), resample=Image.NEAREST), dtype=np.int32)

    # 중앙 패딩
    canvas = np.full((target_hw, target_hw), invalid_label, dtype=np.int32)
    top  = (target_hw - newH) // 2
    left = (target_hw - newW) // 2
    canvas[top:top+newH, left:left+newW] = lab_rz

    grid_h = target_hw // patch
    grid_w = target_hw // patch
    Npatch = grid_h * grid_w
    M = np.zeros((Npatch, num_clusters), dtype=np.float32)

    pid = 0
    for gy in range(grid_h):
        for gx in range(grid_w):
            y0,y1 = gy*patch, (gy+1)*patch
            x0,x1 = gx*patch, (gx+1)*patch
            block = canvas[y0:y1, x0:x1]
            valid = block != invalid_label
            if valid.sum() == 0:
                pid += 1; continue
            vals = block[valid]
            vals = vals[(vals >= 0) & (vals < num_clusters)]
            if vals.size > 0:
                cnt = np.bincount(vals, minlength=num_clusters).astype(np.float32)
                M[pid, :] = cnt / (vals.size + eps)
            pid += 1

    # A: 클러스터별 평균 풀링 (행합=1)
    colsum = M.sum(axis=0, keepdims=True) + eps
    A = (M / colsum).T  # [C, Npatch]
    meta = dict(grid_h=grid_h, grid_w=grid_w, patch=patch, top=top, left=left, newH=newH, newW=newW)
    return M, A, meta

# ========== ViT 백본 ==========
class TimmViTPatchBackbone:
    def __init__(self, model_name, device):
        try:
            import timm
        except Exception as e:
            raise RuntimeError("timm가 설치되어 있어야 합니다. pip install timm") from e
        self.model  = timm.create_model(model_name, pretrained=True).to(device).eval()
        self.device = device
        cfg = getattr(self.model, "pretrained_cfg", getattr(self.model, "default_cfg", {}))
        in_size = cfg.get("input_size", (3, 224, 224))  # (C,H,W)
        self.in_h, self.in_w = int(in_size[1]), int(in_size[2])
        mean = cfg.get("mean", (0.5,0.5,0.5))
        std  = cfg.get("std",  (0.5,0.5,0.5))
        self.mean = torch.tensor(mean, device=device).view(1,3,1,1)
        self.std  = torch.tensor(std,  device=device).view(1,3,1,1)
        self.dim  = int(getattr(self.model, "num_features", 768))
        ps = getattr(self.model.patch_embed, "patch_size", (14,14))
        self.patch = int(ps if isinstance(ps, int) else ps[0])

    @torch.no_grad()
    def patch_tokens_batch(self, imgs_b3hw):   # [B,3,H,W]
        x = (imgs_b3hw.to(self.device) - self.mean) / self.std
        out = self.model.forward_features(x)
        if isinstance(out, dict):
            if   "x_norm_patchtokens" in out: pt = out["x_norm_patchtokens"]   # [B,N,dim]
            elif "x" in out and out["x"].ndim == 3: pt = out["x"][:,1:,:]      # [B,1+N,dim]→patch
            else: raise RuntimeError("patch tokens not found")
        else:
            if out.ndim == 3 and out.shape[1] > 1: pt = out[:,1:,:]
            else: raise RuntimeError("unexpected ViT output")
        return pt  # [B, Npatch, dim]

# ========== 전처리: TIFF → [1,3,H,W] ==========
def load_concentration_as_tensor(tif_path, H, W, keep_aspect=True):
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)          # [H0,W0]
    arr = np.where(np.isfinite(arr), arr, 0.0)
    if arr.max() > 1.5:  # 파일 스케일이 0..100인 경우
        arr = arr / 100.0
    arr = np.clip(arr, 0.0, 1.0)
    # 1채널 → 3채널
    t = torch.from_numpy(arr).unsqueeze(0).repeat(3,1,1).float()  # [3,H0,W0]
    if keep_aspect:
        H0, W0 = t.shape[-2:]
        scale = min(H / max(H0,1), W / max(W0,1))
        newH, newW = max(1, int(round(H0*scale))), max(1, int(round(W0*scale)))
        t = F.interpolate(t.unsqueeze(0), size=(newH,newW), mode="bilinear", align_corners=False).squeeze(0)
        padH, padW = H-newH, W-newW
        pad = (padW//2, padW-padW//2, padH//2, padH-padH//2)  # L,R,T,B
        t = F.pad(t, pad, mode="constant", value=0.0)
    else:
        t = F.interpolate(t.unsqueeze(0), size=(H,W), mode="bilinear", align_corners=False).squeeze(0)
    return t.unsqueeze(0)  # [1,3,H,W]

# ========== A 한번 생성 ==========
def build_and_save_A_once(cfg, backbone):
    label_map = np.load(cfg["cluster_map_sorted_npy"]).astype(np.int32)  # 정렬된 0..C-1 라벨
    C = 32
    target_hw = backbone.in_h
    patch = backbone.patch
    M, A, info = build_patch2cluster_weights(label_map, C, target_hw, patch, keep_aspect=cfg["keep_aspect"])
    out_dir = os.path.dirname(cfg["cluster_map_sorted_npy"])
    np.save(os.path.join(out_dir, "patch2cluster_M.npy"), M)
    np.save(os.path.join(out_dir, "cluster_avg_A.npy"), A)
    print(f"[A/M] saved at {out_dir} | M:{M.shape} A:{A.shape} info:{info}")
    return A

# ===================================================================
# 1) 병렬 데이터 로딩용 Dataset
# ===================================================================
class TifDataset(Dataset):
    def __init__(self, file_paths, target_h, target_w, keep_aspect):
        self.file_paths = file_paths
        self.target_h = target_h
        self.target_w = target_w
        self.keep_aspect = keep_aspect

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        tif_path = self.file_paths[idx]
        try:
            img_tensor = load_concentration_as_tensor(
                tif_path, self.target_h, self.target_w, self.keep_aspect
            )
            return img_tensor.squeeze(0)  # [3,H,W]
        except Exception:
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.stack(batch, dim=0)  # [B,3,H,W]

# ---- 작은 헬퍼: nullcontext (py<3.7 호환용)
@contextmanager
def nullcontext():
    yield

# ========== 단계 B: 임베딩 추출/저장 ==========
def process_windows(cfg):
    device = torch.device(cfg["device"])
    from data_provider.data_loader_save_0908 import Dataset_Custom

    ymd2path = index_nsidc(cfg["nsidc_root"], which=cfg["use_tif"])
    vit = TimmViTPatchBackbone(cfg["timm_model"], device)

    A_path = os.path.join(os.path.dirname(cfg["cluster_map_sorted_npy"]), "cluster_avg_A.npy")
    if not os.path.exists(A_path):
        A_np = build_and_save_A_once(cfg, vit)
    else:
        A_np = np.load(A_path).astype(np.float32)
    A = torch.from_numpy(A_np).to(device)  # [C, Npatch]
    A_f = A.float()                        # einsum용 float32 고정
    C, Npatch = A.shape

    # (K,V 차원 설정)
    dk, dv = int(cfg["dk"]), int(cfg["dv"])
    dim_vit = vit.dim
    if (dk, dv) == (dim_vit, dim_vit):
        Wk = nn.Identity().to(device)
        Wv = nn.Identity().to(device)
        print(f"[INFO] Using identity projection: patch_dim={dim_vit} → dk={dk}, dv={dv}")
    else:
        Wk = nn.Linear(dim_vit, dk, bias=False).to(device)
        Wv = nn.Linear(dim_vit, dv, bias=False).to(device)
        with torch.no_grad():
            # QR 초기화 (안정)
            if dk >= dim_vit:
                qk, _ = torch.linalg.qr(torch.randn(dk, dim_vit, device=device), mode='reduced')
                Wk.weight.copy_(qk)
            else:
                qk, _ = torch.linalg.qr(torch.randn(dim_vit, dk, device=device), mode='reduced')
                Wk.weight.copy_(qk.T)
            if dv >= dim_vit:
                qv, _ = torch.linalg.qr(torch.randn(dv, dim_vit, device=device), mode='reduced')
                Wv.weight.copy_(qv)
            else:
                qv, _ = torch.linalg.qr(torch.randn(dim_vit, dv, device=device), mode='reduced')
                Wv.weight.copy_(qv.T)
        print(f"[WARN] Projecting patch dim {dim_vit} → dk={dk}, dv={dv} via Linear; QR init.")

    use_autocast = (device.type == "cuda") and bool(cfg["fp16"])

    for out_len in cfg["output_lens"]:
        for divide in cfg["splits"]:
            ds = Dataset_Custom(
                csv_path=cfg["csv_path"], flag=divide,
                size=[cfg["input_len"], 0, out_len],
                train_ratio=cfg["train_ratio"], val_ratio=cfg["val_ratio"],
                features='M', scale=False, timeenc=0, freq='D'
            )
            base = os.path.splitext(os.path.basename(cfg["csv_path"]))[0]
            save_dir = os.path.join(cfg["save_root"], base, f"out{out_len}", divide)
            os.makedirs(save_dir, exist_ok=True)
            print(f"\n[INFO] {divide} out{out_len}: windows={len(ds)} → {save_dir}")

            # stack 모드면 시퀀스 저장 플래그 강제
            if cfg["temporal_mode"].lower() == "stack":
                cfg["save_cluster_seq"] = True

            for idx in tqdm(range(len(ds)), mininterval=0.3):
                x, y, x_mark, y_mark = ds[idx]
                s = int(ds.indices[idx])
                fpath = os.path.join(save_dir, f"{s}.h5")
                if (not cfg["overwrite"]) and os.path.exists(fpath):
                    continue

                # 윈도우 날짜들
                L = cfg["input_len"]
                ymd_list = [f"{int(x_mark[t,0]):04d}{int(x_mark[t,1]):02d}{int(x_mark[t,2]):02d}" for t in range(L)]

                tif_paths = [ymd2path.get(ymd) for ymd in ymd_list]
                if any(p is None for p in tif_paths):
                    # 빠진 날짜 있으면 스킵
                    continue

                # DataLoader로 배치 전처리
                tif_dataset = TifDataset(tif_paths, vit.in_h, vit.in_w, cfg["keep_aspect"])
                data_loader = DataLoader(
                    tif_dataset,
                    batch_size=cfg["batch_size"],
                    shuffle=False,
                    num_workers=cfg["num_workers"],     # Spyder면 0 권장
                    pin_memory=(device.type == "cuda"),
                    collate_fn=collate_fn
                )

                Kc_seq_list, Vc_seq_list = [], []

                with torch.inference_mode():
                    cm = torch.cuda.amp.autocast(enabled=use_autocast, dtype=torch.float16) if use_autocast else nullcontext()
                    with cm:
                        for img_batch in data_loader:
                            if img_batch is None:
                                continue  # 전부 None이면 skip
                            # [B,3,H,W] → [B,N,dim]
                            pt_batch = vit.patch_tokens_batch(img_batch)

                            # 투영
                            Kp_batch = Wk(pt_batch)  # [B,N,dk]
                            Vp_batch = Wv(pt_batch)  # [B,N,dv]

                            # 공간 풀링: [B,C,D] = [C,N] @ [B,N,D]
                            Kc_batch = torch.einsum("cn,bnd->bcd", A_f, Kp_batch.float())
                            Vc_batch = torch.einsum("cn,bnd->bcd", A_f, Vp_batch.float())

                            Kc_seq_list.append(Kc_batch.cpu())
                            Vc_seq_list.append(Vc_batch.cpu())

                if not Kc_seq_list:
                    continue

                # [L, C, D]로 이어 붙이기
                Kc_seq = torch.cat(Kc_seq_list, dim=0).numpy()
                Vc_seq = torch.cat(Vc_seq_list, dim=0).numpy()

                # 시간 집계
                mode = cfg["temporal_mode"].lower()
                if mode == "last":
                    Kc_agg, Vc_agg = Kc_seq[-1], Vc_seq[-1]
                elif mode == "mean":
                    Kc_agg, Vc_agg = Kc_seq.mean(axis=0), Vc_seq.mean(axis=0)
                elif mode == "ewa":
                    t_idx = np.arange(L, dtype=np.float32)
                    w = np.exp(-cfg["ewa_gamma"] * (L-1 - t_idx))
                    w = (w / (w.sum() + 1e-8)).astype(np.float32)
                    Kc_agg = np.tensordot(w, Kc_seq, axes=(0,0))  # [C,D]
                    Vc_agg = np.tensordot(w, Vc_seq, axes=(0,0))
                elif mode == "stack":
                    Kc_agg, Vc_agg = None, None
                else:
                    raise ValueError(f"unknown temporal_mode={mode}")

                # 저장
                with h5py.File(fpath, "w") as hf:
                    meta = hf.create_group("meta")
                    meta.attrs["window_start_idx"] = s
                    meta.attrs["dates"] = np.array(ymd_list, dtype='S8')
                    meta.attrs["model"] = cfg["timm_model"]
                    meta.attrs["grid_h"] = vit.in_h // vit.patch
                    meta.attrs["grid_w"] = vit.in_w // vit.patch
                    meta.attrs["patch"]  = vit.patch
                    meta.attrs["vit_dim"] = int(dim_vit)
                    meta.attrs["dk"] = int(dk); meta.attrs["dv"] = int(dv)
                    meta.attrs["keep_aspect"] = int(cfg["keep_aspect"])
                    meta.attrs["dtype"] = "fp16" if cfg["fp16"] else "fp32"
                    meta.attrs["temporal_mode"] = cfg["temporal_mode"]
                    meta.attrs["ewa_gamma"] = float(cfg["ewa_gamma"])
                    meta.attrs["C"] = int(C); meta.attrs["Npatch"] = int(Npatch)
                    meta.attrs["source"] = cfg["use_tif"]

                    if cfg["save_cluster_agg"] and (Kc_agg is not None):
                        hf.create_dataset("Kc", data=Kc_agg)   # [C, dk]
                        hf.create_dataset("Vc", data=Vc_agg)   # [C, dv]

                    if cfg["save_cluster_seq"]:
                        hf.create_dataset("Kc_seq", data=Kc_seq)   # [L, C, dk]
                        hf.create_dataset("Vc_seq", data=Vc_seq)   # [L, C, dv]

    print("[DONE] all splits/output_lens")


# ========== 실행 ==========
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    process_windows(CFG)
