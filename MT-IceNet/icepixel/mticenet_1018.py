# mt_icenet_sequential_prediction.py
# MT-IceNet: 연속 예측 (예: +7일 예측 → +1일, +2일, ..., +7일 모두 예측)

import os, re, glob, warnings, sys
import numpy as np
import pandas as pd
import tifffile as tiff
import tensorflow as tf
from datetime import datetime, timedelta
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Metric_node import (torch 의존)
sys.path.append(r"C:\Users\USER\Desktop\baseline\MT-IceNet\utils")
import Metric_node2 as Metric  # ✅ Metric_node2로 변경
import torch

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Conv2D, ConvLSTM2D, BatchNormalization,
                                     MaxPooling2D, UpSampling2D, concatenate, Activation)
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# =========================
# 설정
# =========================
DATA_ROOT     = r"C:\Users\USER\Desktop\ice\data\NSIDC_Data"
FILE_REGEX    = r"N_(\d{8})_concentration.*\.tif$"
IMG_SHAPE     = (448, 304)

# ✅ 연속 예측 설정
output_lens   = [7, 14, 30]  # +7일(1~7), +14일(1~14), +30일(1~30) 연속 예측
seq_input     = 30           # 입력: 과거 30일

BATCH_SIZE    = 4
Epoch         = 50
LEARNING_RATE = 1e-4
SEED          = 42
STRIDE        = 7
DOWNSAMPLE    = 2

USE_MIXED_PRECISION = True
USE_XLA             = False

model_name   = "MT-IceNet"
save         = True
base_date    = datetime(2013, 1, 1)

np.random.seed(SEED)
tf.random.set_seed(SEED)

if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
if USE_XLA:
    tf.config.optimizer.set_jit(True)

for g in tf.config.list_physical_devices('GPU'):
    try: 
        tf.config.experimental.set_memory_growth(g, True)
    except: 
        pass

# 연도 분할
TRAIN_YEARS = list(range(2013, 2020))
VAL_YEARS   = [2020]
TEST_YEARS  = [2021, 2022]

# =========================
# 유틸/데이터
# =========================
def list_tif_paths(root): 
    return sorted(glob.glob(os.path.join(root, "*", "*", "*.tif")))

def parse_date(p):
    m = re.search(FILE_REGEX, os.path.basename(p))
    return None if not m else datetime.strptime(m.group(1), "%Y%m%d")

def read_one_tif(path): 
    return tiff.imread(path).astype(np.float32)

def load_daily_stack(root, target_hw=IMG_SHAPE):
    recs = []
    for p in list_tif_paths(root):
        d = parse_date(p)
        if d is None: 
            continue
        recs.append((pd.Timestamp(d), p))
    
    if not recs: 
        raise RuntimeError("GeoTIFF 파일을 찾지 못했습니다.")
    
    recs.sort(key=lambda x: x[0])
    dates, frames = [], []
    
    for d, p in recs:
        a = read_one_tif(p)
        if a.shape != target_hw: 
            raise ValueError(f"크기 불일치 {p} {a.shape}!={target_hw}")
        frames.append(a)
        dates.append(d)
    
    X = np.stack(frames, axis=0)  # [T,H,W]
    X = np.nan_to_num(X, nan=0.0) / 100.0  # 0~1
    idx = pd.DatetimeIndex(dates)
    
    return idx, X

def make_land_mask(daily_stack):
    valid = np.isfinite(daily_stack)
    ocean = (valid.sum(axis=0) > 0).astype(np.float32)
    return ocean

def build_index_splits(daily_idx, t_days, max_lead, split_years, stride=1):
    T = len(daily_idx)
    ii, yrs = [], []
    
    for t in range(t_days, T - max_lead, stride):
        ii.append(t)
        yrs.append(daily_idx[t-1].year)
    
    ii  = np.array(ii,  dtype=np.int32)
    yrs = np.array(yrs, dtype=np.int32)
    
    tr = ii[np.isin(yrs, split_years[0])]
    va = ii[np.isin(yrs, split_years[1])]
    te = ii[np.isin(yrs, split_years[2])]
    
    return tr, va, te

def _maybe_downsample_3d(x, new_hw):
    x = tf.expand_dims(x, -1)
    x = tf.image.resize(x, new_hw, method='area')
    return tf.squeeze(x, -1)

def make_dataset(daily_stack, indices, t_days, lead_days, 
                 batch_size=1, shuffle=False, seed=42, downsample=1):
    """
    ✅ 수정: lead_days가 [1,2,3,...,N]처럼 연속된 리스트
    """
    ds_x = tf.convert_to_tensor(daily_stack, dtype=tf.float32)  # [T,H,W]
    lead_days_tf = tf.constant(list(lead_days), dtype=tf.int32)
    
    H, W = ds_x.shape[1], ds_x.shape[2]
    new_hw = (H // downsample, W // downsample) if downsample > 1 else None
    
    ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle: 
        ds = ds.shuffle(buffer_size=min(4096, len(indices)), 
                       seed=seed, reshuffle_each_iteration=True)
    
    @tf.function
    def _slice_one(t):
        x = ds_x[t - t_days : t]  # [T,H,W]
        if new_hw is not None: 
            x = _maybe_downsample_3d(x, new_hw)
        x = tf.expand_dims(x, -1)  # [T,h,w,1]
        
        # ✅ 연속 예측: t+1, t+2, ..., t+N
        ys = tf.stack([ds_x[t + L] for L in tf.unstack(lead_days_tf)], axis=-1)  # [H,W,C]
        if new_hw is not None: 
            ys = tf.image.resize(ys, new_hw, method='area')
        
        return x, ys
    
    ds = ds.map(_slice_one, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    return ds

# 모델
def build_model_daily(t_days, H, W, n_out, lr=LEARNING_RATE, filt=3):
    """
    n_out: 출력 채널 수 (예: 7이면 +1~+7일 예측)
    """
    inp = Input(shape=(t_days, H, W, 1))
    
    x = ConvLSTM2D(8, (5,5), padding="same", return_sequences=False,
                   data_format="channels_last", activation="tanh",
                   recurrent_activation="sigmoid")(inp)
    
    c1 = Conv2D(16, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(x)
    c1 = Conv2D(16, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(c1)
    b1 = BatchNormalization(axis=-1)(c1)
    p1 = MaxPooling2D(pool_size=(2,2))(b1)
    
    b  = Conv2D(64, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(p1)
    b  = Conv2D(64, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(b)
    bN = BatchNormalization(axis=-1)(b)
    
    u1 = UpSampling2D(size=(2,2))(bN)
    u1 = Conv2D(32, 2, activation='relu', padding='same', 
                kernel_initializer='he_normal')(u1)
    m1 = concatenate([b1, u1], axis=3)
    
    c9 = Conv2D(16, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(m1)
    c9 = Conv2D(16, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(c9)
    
    raw_out = Conv2D(n_out, 1, activation='linear')(c9)
    out = Activation('linear', dtype='float32')(raw_out)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    
    return model

# 평가
@torch.no_grad()
def evaluate_for_metric_node(pred_maps, true_maps, lead_days, 
                             first_batch_index, seq_input, model_name, tag_time, land_mask):
    """
    pred_maps/true_maps: [N, h, w, C] 
    C = len(lead_days) = 연속 예측 길이
    land_mask: [h, w] 바다=1, 육지=0
    """
    N, h, w, C = pred_maps.shape
    
    # [N, h*w, C] -> [N, C, h*w]
    pred = pred_maps.reshape(N, h*w, C).transpose(0, 2, 1)
    true = true_maps.reshape(N, h*w, C).transpose(0, 2, 1)
    
    pred_t = torch.from_numpy(pred).float()
    true_t = torch.from_numpy(true).float()
    
    n_features = h * w
    metric = Metric.metric(pred_t, true_t, n_features)  # [MSE, MAE, COR]
    
    # ✅ 플롯 저장 (land_mask 전달)
    Metric.plot(pred_t, true_t, f"{model_name}_{C}", C, tag_time, land_mask=land_mask)
    
    # CSV 저장
    out_dir = f'./STMA_node/{model_name}/models/{model_name}_{C}_{tag_time.month}{tag_time.day}{tag_time.hour}{tag_time.minute}'
    os.makedirs(out_dir, exist_ok=True)
    
    # 날짜 생성: 입력 마지막 시점 + 각 리드타임
    predict_dates = [(base_date + timedelta(days=int(first_batch_index + seq_input + int(L))))\
                     .strftime("%Y-%m-%d") for L in lead_days]
    
    # 첫 샘플, 상위 K 픽셀만 저장
    pred_first = pred[0]  # [C, h*w]
    true_first = true[0]
    topK = min(32, pred_first.shape[1])
    
    for r in range(topK):
        df = pd.DataFrame({
            "Date": predict_dates, 
            "Prediction": pred_first[:, r], 
            "Actual": true_first[:, r]
        })
        df.to_csv(os.path.join(out_dir, f"region_{r}_pred_{C}.csv"), index=False)
    
    return metric

# =========================
# 메인
# =========================
def main():
    # 데이터 로드
    daily_idx, daily_stack = load_daily_stack(DATA_ROOT, IMG_SHAPE)
    H, W = daily_stack.shape[1], daily_stack.shape[2]
    land_mask = make_land_mask(daily_stack)
    
    if DOWNSAMPLE > 1:
        lm = tf.convert_to_tensor(land_mask[...,None], tf.float32)
        lm = tf.image.resize(lm, (H//DOWNSAMPLE, W//DOWNSAMPLE), method='nearest')
        land_mask_d = tf.squeeze(lm, -1).numpy().astype(np.float32)
        H_eff, W_eff = land_mask_d.shape
    else:
        land_mask_d = land_mask
        H_eff, W_eff = H, W
    
    T = len(daily_idx)
    
    # ✅ 실험 루프: 연속 예측
    results = {}
    
    for seq_output in output_lens:
        print(f"\n{'='*70}")
        print(f"실험: +{seq_output}일 연속 예측 (1일~{seq_output}일)")
        print(f"{'='*70}")
        
        # ✅ 연속 리드타임 생성: [1, 2, 3, ..., seq_output]
        lead_days = list(range(1, seq_output + 1))
        max_lead = seq_output
        
        print(f"리드타임: {lead_days[:5]}...{lead_days[-3:]} (총 {len(lead_days)}개)")
        
        # 인덱스 분할
        tr_idx, va_idx, te_idx = build_index_splits(
            daily_idx, seq_input, max_lead,
            (TRAIN_YEARS, VAL_YEARS, TEST_YEARS), 
            stride=STRIDE
        )
        
        print(f"데이터: Train={len(tr_idx)}, Val={len(va_idx)}, Test={len(te_idx)}")
        
        # 데이터셋 생성
        train_ds = make_dataset(daily_stack, tr_idx, seq_input, tuple(lead_days),
                               batch_size=BATCH_SIZE, shuffle=True, 
                               seed=SEED, downsample=DOWNSAMPLE)
        val_ds   = make_dataset(daily_stack, va_idx, seq_input, tuple(lead_days),
                               batch_size=BATCH_SIZE, shuffle=False, 
                               downsample=DOWNSAMPLE)
        test_ds  = make_dataset(daily_stack, te_idx, seq_input, tuple(lead_days),
                               batch_size=BATCH_SIZE, shuffle=False, 
                               downsample=DOWNSAMPLE)
        
        # ✅ 모델: 출력 채널 = seq_output (예: 7일이면 7개 채널)
        model = build_model_daily(seq_input, H_eff, W_eff, n_out=seq_output)
        now = datetime.now()
        best = [1e5, 1e5, -1e5]  # [MSE, MAE, COR]
        
        print(f"\n모델 구조:")
        print(model.summary())
        
        # 학습 루프
        print(f"\n학습 시작...")
        for epoch in range(Epoch):
            # 학습
            history = model.fit(train_ds, validation_data=val_ds, 
                              epochs=1, verbose=2)
            
            # 테스트 예측
            preds, trues = [], []
            for xb, yb in test_ds:
                pb = model.predict(xb, verbose=0)  # [B, h, w, seq_output]
                preds.append(pb)
                trues.append(yb.numpy())
            
            pred = np.concatenate(preds, axis=0)  # [N, h, w, seq_output]
            true = np.concatenate(trues, axis=0)
            
            # 0~100 스케일 + 마스크
            pred = np.clip(pred * 100.0, 0, 100) * land_mask_d[..., None]
            true = np.clip(true * 100.0, 0, 100) * land_mask_d[..., None]
            
            # Metric 평가 (✅ land_mask 전달)
            first_batch_index = int(te_idx[0]) if len(te_idx) else 0
            metric = evaluate_for_metric_node(
                pred, true, lead_days, 
                first_batch_index, seq_input, model_name, now, land_mask_d  # ✅ land_mask 추가
            )
            
            # Best 업데이트
            best = Metric.update(
                now, save, model, best, metric, 
                f"{model_name}_{seq_output}", seq_output, epoch
            )
            
            # 출력
            print(f"[Epoch {epoch:02d}] "
                  f"MSE: {metric[0]:.6f} | MAE: {metric[1]:.6f} | COR: {metric[2]:.4f}")
            print(f"[Best]     "
                  f"MSE: {best[0]:.6f} | MAE: {best[1]:.6f} | COR: {best[2]:.4f}\n")
            
            # 메모리 정리
            del preds, trues
            torch.cuda.empty_cache()
        
        # 결과 저장
        results[seq_output] = {
            'b_mse': best[0], 
            'b_mae': best[1], 
            'b_cor': best[2]
        }
    
    # 최종 요약
    print("\n" + "="*70)
    print("최종 결과 요약")
    print("="*70)
    for k, v in results.items():
        print(f"[+{k}일 연속] MSE={v['b_mse']:.6f} | MAE={v['b_mae']:.6f} | COR={v['b_cor']:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()