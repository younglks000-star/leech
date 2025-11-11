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
import Metric_node5 as Metric  # ← 새로운 시각화와 평가 지표 사용
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
# 기존 evaluate_for_metric_node 함수는 제거됨
# 새로운 직관적 시각화가 메인 루프에서 직접 호출됨

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
        # 새로운 평가 지표 순서: [rmse, bacc, ssim, mse, mae, cor]
        best = [1e5, 0.0, 0.0, 1e5, 1e5, 0.0]
        
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
            
            # ✅ 새로운 평가 지표 사용
            # 시계열 데이터로 변환 (기존 metric 함수용)
            pred_ts = pred.reshape(pred.shape[0], pred.shape[1]*pred.shape[2], pred.shape[3]).transpose(0, 2, 1)
            true_ts = true.reshape(true.shape[0], true.shape[1]*true.shape[2], true.shape[3]).transpose(0, 2, 1)
            
            pred_t = torch.from_numpy(pred_ts).float()
            true_t = torch.from_numpy(true_ts).float()
            
            n_features = pred.shape[1] * pred.shape[2]
            metric = Metric.metric(pred_t, true_t, n_features)
            
            # ✅ 새로운 직관적 시각화 사용 (Spyder에서 바로 표시)
            if epoch % 5 == 0 or epoch == Epoch - 1:  # 5 epoch마다 또는 마지막에 시각화
                print(f"\n[Epoch {epoch:02d}] 직관적 시각화 생성 중...")
                
                # 1. 성능 대시보드
                Metric.plot_ice_prediction_dashboard(
                    pred, true, land_mask_d,
                    model_name, seq_output, now,
                    lead_days=[1, 7, 14] if seq_output >= 14 else [1, 3, 7] if seq_output >= 7 else [1],
                    sample_idx=0
                )
                
                # 2. 해빙 경계선 비교
                Metric.plot_ice_edge_comparison(
                    pred, true, land_mask_d,
                    model_name, seq_output, now,
                    lead_days=[1, 7, 14] if seq_output >= 14 else [1, 3, 7] if seq_output >= 7 else [1],
                    sample_idx=0
                )
                
                # 3. 성능 변화 시각화 (마지막에만)
                if epoch == Epoch - 1:
                    Metric.plot_performance_evolution(
                        pred, true, land_mask_d,
                        model_name, seq_output, now
                    )
                
                print(f"[Epoch {epoch:02d}] 직관적 시각화 완료")
            
            best = Metric.update(
                now, save, model, best, metric, 
                f"{model_name}_{seq_output}", seq_output, epoch
            )
            
            print(f"[Epoch {epoch:02d}] "
                  f"RMSE: {metric[0]:.4f} | BACC: {metric[1]:.4f} | SSIM: {metric[2]:.4f}")
            print(f"[Epoch {epoch:02d}] "
                  f"MSE: {metric[3]:.4f} | MAE: {metric[4]:.4f} | COR: {metric[5]:.4f}")
            print(f"[Best]     "
                  f"RMSE: {best[0]:.4f} | BACC: {best[1]:.4f} | SSIM: {best[2]:.4f}")
            print(f"[Best]     "
                  f"MSE: {best[3]:.4f} | MAE: {best[4]:.4f} | COR: {best[5]:.4f}\n")
            
            del preds, trues
        
        results[seq_output] = {
            'b_rmse': best[0], 'b_bacc': best[1], 'b_ssim': best[2],
            'b_mse': best[3], 'b_mae': best[4], 'b_cor': best[5]
        }
    
    print("\n" + "="*70)
    print("🧊 최종 결과 요약 - CNN-UNet-Baseline")
    print("="*70)
    print("새로운 평가 지표 (해빙 예측에 최적화):")
    print("RMSE: Root Mean Square Error (낮을수록 좋음)")
    print("BACC: Binary Accuracy 15% SIC threshold (높을수록 좋음)")
    print("SSIM: Structural Similarity Index (높을수록 좋음)")
    print("="*70)
    
    for k, v in results.items():
        print(f"[+{k}일 예측]")
        print(f"  🎯 RMSE: {v['b_rmse']:.4f} | BACC: {v['b_bacc']:.4f} | SSIM: {v['b_ssim']:.4f}")
        print(f"  📊 MSE: {v['b_mse']:.4f} | MAE: {v['b_mae']:.4f} | COR: {v['b_cor']:.4f}")
        print()
    
    print("="*70)
    print("✅ SIFNet 수준 목표:")
    print("   MAE: 4.69% | BACC: 95.16% | SSIM: 95.13%")
    print("="*70)

if __name__ == "__main__":
    main()