# baseline_unet.py
# ✅ 베이스라인: 순수 CNN U-Net + Metric_node3 공간 시각화 연동

import os, re, glob, warnings, sys
import numpy as np
import pandas as pd
import tifffile as tiff
import tensorflow as tf
from datetime import datetime, timedelta
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


sys.path.append(r"C:\Users\USER\Desktop\baseline\leech\MT-IceNet\utils")
import Metric_node3 as Metric  # ← 변경
import torch

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Conv2D, BatchNormalization,
                                     MaxPooling2D, UpSampling2D, concatenate, 
                                     Activation, Dropout)
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# =============================================================================
# 설정 (기존과 동일)
# =============================================================================
DATA_ROOT     = r"C:\Users\USER\Desktop\ice\data\NSIDC_Data"
FILE_REGEX    = r"N_(\d{8})_concentration.*\.tif$"
IMG_SHAPE     = (448, 304)

output_lens   = [7, 14, 21]
seq_input     = 30

BATCH_SIZE    = 2
Epoch         = 50
LEARNING_RATE = 1e-4
SEED          = 42
STRIDE        = 7
DOWNSAMPLE    = 1

USE_MIXED_PRECISION = False
USE_XLA             = False

model_name   = "CNN-UNet-Baseline"
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

TRAIN_YEARS = list(range(2013, 2020))
VAL_YEARS   = [2020]
TEST_YEARS  = [2021, 2022]

# =============================================================================
# 유틸/데이터 (기존 코드 유지)
# =============================================================================
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
        
        a[a >= 2500] = np.nan
        a = a / 1000.0
        
        frames.append(a)
        dates.append(d)
    
    X = np.stack(frames, axis=0)
    idx = pd.DatetimeIndex(dates)
    
    return idx, X

def make_land_mask(daily_stack):
    valid = np.isfinite(daily_stack)
    ocean = (valid.sum(axis=0) > 0).astype(np.float32)
    return ocean

def build_index_splits(daily_idx, seq_input, max_lead, split_years, stride=1):
    T = len(daily_idx)
    ii, yrs = [], []
    
    for t in range(seq_input, T - max_lead, stride):
        ii.append(t)
        yrs.append(daily_idx[t-1].year)
    
    ii  = np.array(ii,  dtype=np.int32)
    yrs = np.array(yrs, dtype=np.int32)
    
    tr = ii[np.isin(yrs, split_years[0])]
    va = ii[np.isin(yrs, split_years[1])]
    te = ii[np.isin(yrs, split_years[2])]
    
    return tr, va, te

def _maybe_downsample_2d(x, new_hw):
    x = tf.expand_dims(x, 0)
    x = tf.expand_dims(x, -1)
    x = tf.image.resize(x, new_hw, method='area')
    x = tf.squeeze(x, [0, -1])
    return x

def make_unet_dataset(daily_stack, indices, seq_input, 
                      lead_days, batch_size=2, shuffle=False, 
                      seed=42, downsample=1):
    daily_stack_clean = np.nan_to_num(daily_stack, nan=0.0)
    ds_x = tf.convert_to_tensor(daily_stack_clean, dtype=tf.float32)
    
    lead_days_tf = tf.constant(list(lead_days), dtype=tf.int32)
    
    H, W = ds_x.shape[1], ds_x.shape[2]
    new_hw = (H // downsample, W // downsample) if downsample > 1 else None
    
    ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle: 
        ds = ds.shuffle(buffer_size=min(4096, len(indices)), 
                       seed=seed, reshuffle_each_iteration=True)
    
    @tf.function
    def _slice_unet(t):
        x_frames = []
        for i in range(seq_input):
            frame = ds_x[t - seq_input + i]
            if new_hw is not None:
                frame = _maybe_downsample_2d(frame, new_hw)
            x_frames.append(frame)
        
        x = tf.stack(x_frames, axis=-1)
        
        y_frames = []
        for L in tf.unstack(lead_days_tf):
            frame = ds_x[t + L]
            if new_hw is not None:
                frame = _maybe_downsample_2d(frame, new_hw)
            y_frames.append(frame)
        
        ys = tf.stack(y_frames, axis=-1)
        
        return x, ys
    
    ds = ds.map(_slice_unet, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    return ds

# =============================================================================
# 모델 (기존 코드 유지)
# =============================================================================
def build_unet_baseline(seq_input, H, W, n_out, 
                       land_mask=None,
                       lr=LEARNING_RATE):
    input_layer = Input(shape=(H, W, seq_input), name='input')
    
    # Encoder
    c1 = Conv2D(32, 3, activation='relu', padding='same', 
                kernel_initializer='he_normal')(input_layer)
    c1 = Conv2D(32, 3, activation='relu', padding='same', 
                kernel_initializer='he_normal')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    
    c2 = Conv2D(64, 3, activation='relu', padding='same', 
                kernel_initializer='he_normal')(p1)
    c2 = Conv2D(64, 3, activation='relu', padding='same', 
                kernel_initializer='he_normal')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    
    c3 = Conv2D(128, 3, activation='relu', padding='same', 
                kernel_initializer='he_normal')(p2)
    c3 = Conv2D(128, 3, activation='relu', padding='same', 
                kernel_initializer='he_normal')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    
    c4 = Conv2D(256, 3, activation='relu', padding='same', 
                kernel_initializer='he_normal')(p3)
    c4 = Conv2D(256, 3, activation='relu', padding='same', 
                kernel_initializer='he_normal')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.3)(c4)
    
    # Decoder
    u5 = UpSampling2D(size=(2, 2))(c4)
    u5 = Conv2D(128, 2, activation='relu', padding='same', 
                kernel_initializer='he_normal')(u5)
    m5 = concatenate([c3, u5], axis=3)
    c5 = Conv2D(128, 3, activation='relu', padding='same', 
                kernel_initializer='he_normal')(m5)
    c5 = Conv2D(128, 3, activation='relu', padding='same', 
                kernel_initializer='he_normal')(c5)
    c5 = BatchNormalization()(c5)
    
    u6 = UpSampling2D(size=(2, 2))(c5)
    u6 = Conv2D(64, 2, activation='relu', padding='same', 
                kernel_initializer='he_normal')(u6)
    m6 = concatenate([c2, u6], axis=3)
    c6 = Conv2D(64, 3, activation='relu', padding='same', 
                kernel_initializer='he_normal')(m6)
    c6 = Conv2D(64, 3, activation='relu', padding='same', 
                kernel_initializer='he_normal')(c6)
    c6 = BatchNormalization()(c6)
    
    u7 = UpSampling2D(size=(2, 2))(c6)
    u7 = Conv2D(32, 2, activation='relu', padding='same', 
                kernel_initializer='he_normal')(u7)
    m7 = concatenate([c1, u7], axis=3)
    c7 = Conv2D(32, 3, activation='relu', padding='same', 
                kernel_initializer='he_normal')(m7)
    c7 = Conv2D(32, 3, activation='relu', padding='same', 
                kernel_initializer='he_normal')(c7)
    
    raw_out = Conv2D(n_out, 1, activation='linear')(c7)
    out = Activation('linear', dtype='float32')(raw_out)
    
    model = Model(inputs=input_layer, outputs=out)
    
    if land_mask is not None:
        mask_4d = tf.constant(
            land_mask.reshape(1, H, W, 1), 
            dtype=tf.float32
        )
        
        def masked_mse(y_true, y_pred):
            y_true_masked = y_true * mask_4d
            y_pred_masked = y_pred * mask_4d
            
            squared_diff = tf.square(y_true_masked - y_pred_masked)
            
            batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
            n_channels = tf.cast(tf.shape(y_true)[3], tf.float32)
            n_ocean_per_sample = tf.reduce_sum(mask_4d)
            n_total_ocean = batch_size * n_channels * n_ocean_per_sample
            
            loss = tf.reduce_sum(squared_diff) / n_total_ocean
            return loss
        
        model.compile(optimizer=Adam(learning_rate=lr), loss=masked_mse)
        print("✅ Masked Loss 활성화: 육지 제외하고 학습")
    else:
        model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
        print("⚠️  일반 MSE 사용: 육지 포함")
    
    return model

# =============================================================================
# ✅ 평가 함수 - 공간 시각화 추가
# =============================================================================
@torch.no_grad()
def evaluate_for_metric_node(pred_maps, true_maps, lead_days, 
                             first_batch_index, seq_input, model_name, 
                             tag_time, land_mask):
    """
    ✅ Metric_node3의 공간 시각화 함수 추가
    """
    N, h, w, C = pred_maps.shape
    
    # 1. 기존 시계열 평가
    pred = pred_maps.reshape(N, h*w, C).transpose(0, 2, 1)
    true = true_maps.reshape(N, h*w, C).transpose(0, 2, 1)
    
    pred_t = torch.from_numpy(pred).float()
    true_t = torch.from_numpy(true).float()
    
    n_features = h * w
    metric = Metric.metric(pred_t, true_t, n_features)
    
    # 기존 시계열 플롯
    Metric.plot(pred_t, true_t, f"{model_name}_{C}", C, tag_time)
    
    # 2. ✅ 새로운 공간 시각화 추가
    print("\n[공간 시각화 생성 중...]")
    
    # 2-1. 예측 vs 실제 비교 (Day 1, 7, 14)
    vis_days = [d for d in [1, 7, 14, 21] if d <= C]
    Metric.plot_spatial_comparison(
        pred_maps, true_maps, land_mask,
        model_name=model_name,
        seq_output=C,
        now=tag_time,
        lead_days=vis_days,
        sample_idx=0
    )
    
    # 2-2. 시간 진화 시각화
    Metric.plot_spatial_temporal(
        pred_maps, true_maps, land_mask,
        model_name=model_name,
        seq_output=C,
        now=tag_time,
        sample_idx=0,
        n_timesteps=min(7, C)
    )
    
    # 2-3. 오차 통계 시각화
    Metric.plot_error_statistics(
        pred_maps, true_maps, land_mask,
        model_name=model_name,
        seq_output=C,
        now=tag_time
    )
    
    print("[공간 시각화 완료]")
    
    # 3. CSV 저장 (기존)
    out_dir = f'./STMA_node/{model_name}/models/{model_name}_{C}_{tag_time.month}{tag_time.day}{tag_time.hour}{tag_time.minute}'
    os.makedirs(out_dir, exist_ok=True)
    
    predict_dates = [(base_date + timedelta(days=int(first_batch_index + seq_input + int(L))))\
                     .strftime("%Y-%m-%d") for L in lead_days]
    
    pred_first = pred[0]
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

# =============================================================================
# 메인 (기존 코드와 동일)
# =============================================================================
def main():
    print("="*70)
    print("베이스라인: 순수 CNN U-Net + 공간 시각화")
    print("="*70)
    
    # 데이터 로드
    daily_idx, daily_stack = load_daily_stack(DATA_ROOT, IMG_SHAPE)
    H, W = daily_stack.shape[1], daily_stack.shape[2]
    land_mask = make_land_mask(daily_stack)
    
    print(f"데이터 shape: {daily_stack.shape}")
    print(f"날짜 범위: {daily_idx[0]} ~ {daily_idx[-1]}")
    print(f"총 {len(daily_idx)}일 데이터")
    
    nan_count = np.isnan(daily_stack).sum()
    total = daily_stack.size
    ocean_pixels = (land_mask == 1).sum()
    land_pixels = (land_mask == 0).sum()
    
    print(f"\n[데이터 통계]")
    print(f"NaN 개수: {nan_count:,} ({nan_count/total*100:.2f}%)")
    print(f"바다 픽셀: {ocean_pixels:,} ({ocean_pixels/(ocean_pixels+land_pixels)*100:.2f}%)")
    print(f"육지 픽셀: {land_pixels:,} ({land_pixels/(ocean_pixels+land_pixels)*100:.2f}%)")
    print(f"값 범위 (NaN 제외): {np.nanmin(daily_stack):.4f} ~ {np.nanmax(daily_stack):.4f}\n")
    
    if DOWNSAMPLE > 1:
        lm = tf.convert_to_tensor(land_mask[...,None], tf.float32)
        lm = tf.image.resize(lm, (H//DOWNSAMPLE, W//DOWNSAMPLE), method='nearest')
        land_mask_d = tf.squeeze(lm, -1).numpy().astype(np.float32)
        H_eff, W_eff = land_mask_d.shape
    else:
        land_mask_d = land_mask
        H_eff, W_eff = H, W
    
    results = {}
    
    for seq_output in output_lens:
        print(f"\n{'='*70}")
        print(f"실험: +{seq_output}일 연속 예측")
        print(f"{'='*70}")
        
        lead_days = list(range(1, seq_output + 1))
        max_lead = seq_output
        
        tr_idx, va_idx, te_idx = build_index_splits(
            daily_idx, seq_input, max_lead,
            (TRAIN_YEARS, VAL_YEARS, TEST_YEARS), 
            stride=STRIDE
        )
        
        print(f"데이터: Train={len(tr_idx)}, Val={len(va_idx)}, Test={len(te_idx)}")
        
        train_ds = make_unet_dataset(
            daily_stack, tr_idx, seq_input, tuple(lead_days),
            batch_size=BATCH_SIZE, shuffle=True, seed=SEED, downsample=DOWNSAMPLE
        )
        val_ds = make_unet_dataset(
            daily_stack, va_idx, seq_input, tuple(lead_days),
            batch_size=BATCH_SIZE, shuffle=False, downsample=DOWNSAMPLE
        )
        test_ds = make_unet_dataset(
            daily_stack, te_idx, seq_input, tuple(lead_days),
            batch_size=BATCH_SIZE, shuffle=False, downsample=DOWNSAMPLE
        )
        
        model = build_unet_baseline(
            seq_input, H_eff, W_eff, 
            n_out=seq_output,
            land_mask=land_mask_d
        )
        
        now = datetime.now()
        best = [1e5, 1e5, -1e5]
        
        print(f"\n학습 시작...")
        for epoch in range(Epoch):
            history = model.fit(train_ds, validation_data=val_ds, 
                              epochs=1, verbose=2)
            
            preds, trues = [], []
            for xb, yb in test_ds:
                pb = model.predict(xb, verbose=0)
                preds.append(pb)
                trues.append(yb.numpy())
            
            pred = np.concatenate(preds, axis=0)
            true = np.concatenate(trues, axis=0)
            
            pred = np.clip(pred * 100.0, 0, 100) * land_mask_d[..., None]
            true = np.clip(true * 100.0, 0, 100) * land_mask_d[..., None]
            
            first_batch_index = int(te_idx[0]) if len(te_idx) else 0
            
            # ✅ 공간 시각화 포함된 평가 호출
            metric = evaluate_for_metric_node(
                pred, true, lead_days, 
                first_batch_index, seq_input, model_name, now, land_mask_d
            )
            
            best = Metric.update(
                now, save, model, best, metric, 
                f"{model_name}_{seq_output}", seq_output, epoch
            )
            
            print(f"[Epoch {epoch:02d}] "
                  f"MSE: {metric[0]:.6f} | MAE: {metric[1]:.6f} | COR: {metric[2]:.4f}")
            print(f"[Best]     "
                  f"MSE: {best[0]:.6f} | MAE: {best[1]:.6f} | COR: {best[2]:.4f}\n")
            
            del preds, trues
        
        results[seq_output] = {
            'b_mse': best[0], 
            'b_mae': best[1], 
            'b_cor': best[2]
        }
    
    print("\n" + "="*70)
    print("최종 결과 요약")
    print("="*70)
    for k, v in results.items():
        print(f"[+{k}일] MSE={v['b_mse']:.6f} | MAE={v['b_mae']:.6f} | COR={v['b_cor']:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()