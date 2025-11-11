# mt_icenet_dual_temporal_corrected.py
# MT-IceNet: Dual-Temporal 구조 + 단변량 해빙 농도 예측
# ✅ 수정사항:
# 1. 정규화: /100 → /1000 (0~1000 → 0~1)
# 2. 특수값 처리: ≥2500 → NaN (육지/극점/해안선)
# 3. Masked Loss: 육지 제외하고 학습

import os, re, glob, warnings, sys
import numpy as np
import pandas as pd
import tifffile as tiff
import tensorflow as tf
from datetime import datetime, timedelta
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Metric_node import
sys.path.append(r"C:\Users\USER\Desktop\baseline\MT-IceNet\utils")
import Metric_node as Metric
import torch

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Conv2D, ConvLSTM2D, BatchNormalization,
                                     MaxPooling2D, UpSampling2D, concatenate, Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# =============================================================================
# 설정
# =============================================================================
DATA_ROOT     = r"C:\Users\USER\Desktop\ice\data\NSIDC_Data"
FILE_REGEX    = r"N_(\d{8})_concentration.*\.tif$"
IMG_SHAPE     = (448, 304)

# 연속 예측 설정
output_lens   = [7, 14, 21]

# Dual-Temporal 설정
seq_short     = 30
seq_long      = 90

BATCH_SIZE    = 2
Epoch         = 50
LEARNING_RATE = 1e-4
SEED          = 42
STRIDE        = 7
DOWNSAMPLE    = 1

USE_MIXED_PRECISION = False
USE_XLA             = False

model_name   = "MT-IceNet-Dual"
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

# =============================================================================
# 유틸/데이터 (✅ 수정됨)
# =============================================================================
def list_tif_paths(root): 
    return sorted(glob.glob(os.path.join(root, "*", "*", "*.tif")))

def parse_date(p):
    m = re.search(FILE_REGEX, os.path.basename(p))
    return None if not m else datetime.strptime(m.group(1), "%Y%m%d")

def read_one_tif(path): 
    return tiff.imread(path).astype(np.float32)

def load_daily_stack(root, target_hw=IMG_SHAPE):
    """
    ✅ 수정: 올바른 정규화 + 특수값 처리
    
    Returns:
        idx: 날짜 인덱스
        X: 해빙 농도 [T, H, W]
           - 바다: 0~1 (정규화된 해빙 농도)
           - 육지/극점: NaN
    """
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
        
        # ✅ 1단계: 특수값 → NaN
        # 2510 (극점), 2530 (해안선), 2540 (육지)
        a[a >= 2500] = np.nan
        
        # ✅ 2단계: 0~1000 → 0~1 정규화 (수정!)
        a = a / 1000.0
        
        frames.append(a)
        dates.append(d)
    
    # [T, H, W] 형태로 스택
    X = np.stack(frames, axis=0)
    # ⚠️ NaN은 유지! (np.nan_to_num 사용 안 함)
    
    idx = pd.DatetimeIndex(dates)
    
    return idx, X

def make_land_mask(daily_stack):
    """
    육지/바다 마스크 생성
    
    Returns:
        mask: [H, W]
            - 1: 바다 (유효한 데이터)
            - 0: 육지/극점 (NaN)
    """
    valid = np.isfinite(daily_stack)  # NaN이 아닌 픽셀
    ocean = (valid.sum(axis=0) > 0).astype(np.float32)  # 한 번이라도 유효하면 바다
    return ocean

def build_index_splits(daily_idx, seq_long, max_lead, split_years, stride=1):
    """인덱스 분할 (긴 윈도우 기준)"""
    T = len(daily_idx)
    ii, yrs = [], []
    
    for t in range(seq_long, T - max_lead, stride):
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

def make_dual_temporal_dataset(daily_stack, indices, seq_short, seq_long, 
                                lead_days, batch_size=2, shuffle=False, 
                                seed=42, downsample=1):
    """
    Dual-Temporal Dataset 생성
    
    ✅ NaN → 0 변환은 여기서만 (TensorFlow 계산용)
    """
    # ✅ NaN → 0 변환 (학습용)
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
    def _slice_dual(t):
        # 짧은 윈도우
        x1 = ds_x[t - seq_short : t]
        if new_hw is not None: 
            x1 = _maybe_downsample_3d(x1, new_hw)
        x1 = tf.expand_dims(x1, -1)
        
        # 긴 윈도우
        x2 = ds_x[t - seq_long : t]
        if new_hw is not None: 
            x2 = _maybe_downsample_3d(x2, new_hw)
        x2 = tf.expand_dims(x2, -1)
        
        # 연속 예측
        ys = tf.stack([ds_x[t + L] for L in tf.unstack(lead_days_tf)], axis=-1)
        if new_hw is not None: 
            ys = tf.image.resize(ys, new_hw, method='area')
        
        return (x1, x2), ys
    
    ds = ds.map(_slice_dual, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    return ds

# =============================================================================
# 모델: Dual-Temporal MT-IceNet (✅ Masked Loss 추가)
# =============================================================================
def build_mt_icenet_dual(seq_short, seq_long, H, W, n_out, 
                         land_mask=None,  # ✅ 추가
                         lr=LEARNING_RATE, filt=3):
    """
    Dual-Temporal MT-IceNet
    
    ✅ 수정: Masked Loss 적용 (육지 제외)
    ⚠️ 모델 구조는 그대로 유지 (공정한 비교를 위해)
    """
    input1 = Input(shape=(seq_short, H, W, 1), name='input_short')
    input2 = Input(shape=(seq_long, H, W, 1), name='input_long')
    
    # ===== Branch 1: 짧은 윈도우 =====
    convlstm1 = ConvLSTM2D(8, (5,5), padding="same", return_sequences=False, 
                           data_format="channels_last", activation="tanh",
                           recurrent_activation="sigmoid")(input1)
    
    c1 = Conv2D(16, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(convlstm1)
    c1 = Conv2D(16, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(c1)
    b1 = BatchNormalization(axis=-1)(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(b1)
    
    # ===== Branch 2: 긴 윈도우 =====
    convlstm2 = ConvLSTM2D(8, (5,5), padding="same", return_sequences=False, 
                           data_format="channels_last", activation="tanh",
                           recurrent_activation="sigmoid")(input2)
    
    c2 = Conv2D(32, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(convlstm2)
    c2 = Conv2D(32, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(c2)
    b2 = BatchNormalization(axis=-1)(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(b2)
    
    # ===== Encoder =====
    c3 = Conv2D(64, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(p2)
    c3 = Conv2D(64, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(c3)
    b3 = BatchNormalization(axis=-1)(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(b3)
    
    # ===== Decoder =====
    u8 = UpSampling2D(size=(2,2))(p3)
    u8 = Conv2D(32, 2, activation='relu', padding='same', 
                kernel_initializer='he_normal')(u8)
    m8 = concatenate([b3, u8], axis=3)
    c8 = Conv2D(32, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(m8)
    c8 = Conv2D(32, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(c8)
    b8 = BatchNormalization(axis=-1)(c8)
    
    u9 = UpSampling2D(size=(2,2))(b8)
    u9 = Conv2D(16, 2, activation='relu', padding='same', 
                kernel_initializer='he_normal')(u9)
    m9 = concatenate([b1, u9], axis=3)
    c9 = Conv2D(16, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(m9)
    c9 = Conv2D(16, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(c9)
    c9 = Conv2D(16, filt, activation='relu', padding='same', 
                kernel_initializer='he_normal')(c9)
    
    # ===== Output =====
    raw_out = Conv2D(n_out, 1, activation='linear')(c9)
    out = Activation('linear', dtype='float32')(raw_out)
    
    model = Model(inputs=[input1, input2], outputs=out)
    
    # ✅ Masked Loss 적용
    if land_mask is not None:
        mask_4d = tf.constant(
            land_mask.reshape(1, H, W, 1), 
            dtype=tf.float32
        )
        
        def masked_mse(y_true, y_pred):
            """
            바다 영역만 Loss 계산
            육지(mask=0)는 완전히 무시
            """
            # 마스크 적용
            y_true_masked = y_true * mask_4d
            y_pred_masked = y_pred * mask_4d
            
            # MSE 계산
            squared_diff = tf.square(y_true_masked - y_pred_masked)
            
            # 바다 픽셀 수로 정규화
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
# 평가
# =============================================================================
@torch.no_grad()
def evaluate_for_metric_node(pred_maps, true_maps, lead_days, 
                             first_batch_index, seq_input, model_name, 
                             tag_time, land_mask):
    """
    Metric_node를 사용한 평가
    """
    N, h, w, C = pred_maps.shape
    
    # [N, h*w, C] -> [N, C, h*w]
    pred = pred_maps.reshape(N, h*w, C).transpose(0, 2, 1)
    true = true_maps.reshape(N, h*w, C).transpose(0, 2, 1)
    
    pred_t = torch.from_numpy(pred).float()
    true_t = torch.from_numpy(true).float()
    
    n_features = h * w
    metric = Metric.metric(pred_t, true_t, n_features)
    
    # 플롯 저장
    Metric.plot(pred_t, true_t, f"{model_name}_{C}", C, tag_time)
    
    # CSV 저장
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
# 메인
# =============================================================================
def main():
    # 데이터 로드
    print("="*70)
    print("데이터 로딩 시작...")
    print("="*70)
    daily_idx, daily_stack = load_daily_stack(DATA_ROOT, IMG_SHAPE)
    H, W = daily_stack.shape[1], daily_stack.shape[2]
    land_mask = make_land_mask(daily_stack)
    
    print(f"데이터 shape: {daily_stack.shape}")
    print(f"날짜 범위: {daily_idx[0]} ~ {daily_idx[-1]}")
    print(f"총 {len(daily_idx)}일 데이터")
    
    # ✅ 데이터 통계 출력
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
    
    T = len(daily_idx)
    
    # 실험 루프
    results = {}
    
    for seq_output in output_lens:
        print(f"\n{'='*70}")
        print(f"실험: +{seq_output}일 연속 예측 (1일~{seq_output}일)")
        print(f"단기 윈도우: {seq_short}일, 장기 윈도우: {seq_long}일")
        print(f"{'='*70}")
        
        lead_days = list(range(1, seq_output + 1))
        max_lead = seq_output
        
        # 인덱스 분할
        tr_idx, va_idx, te_idx = build_index_splits(
            daily_idx, seq_long, max_lead,
            (TRAIN_YEARS, VAL_YEARS, TEST_YEARS), 
            stride=STRIDE
        )
        
        print(f"데이터: Train={len(tr_idx)}, Val={len(va_idx)}, Test={len(te_idx)}")
        
        # 데이터셋 생성
        train_ds = make_dual_temporal_dataset(
            daily_stack, tr_idx, seq_short, seq_long, tuple(lead_days),
            batch_size=BATCH_SIZE, shuffle=True, seed=SEED, downsample=DOWNSAMPLE
        )
        val_ds = make_dual_temporal_dataset(
            daily_stack, va_idx, seq_short, seq_long, tuple(lead_days),
            batch_size=BATCH_SIZE, shuffle=False, downsample=DOWNSAMPLE
        )
        test_ds = make_dual_temporal_dataset(
            daily_stack, te_idx, seq_short, seq_long, tuple(lead_days),
            batch_size=BATCH_SIZE, shuffle=False, downsample=DOWNSAMPLE
        )
        
        # ✅ 모델 생성 (land_mask 전달)
        model = build_mt_icenet_dual(
            seq_short, seq_long, H_eff, W_eff, 
            n_out=seq_output,
            land_mask=land_mask_d  # ✅ 마스크 전달
        )
        
        now = datetime.now()
        best = [1e5, 1e5, -1e5]
        
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
            for (x1b, x2b), yb in test_ds:
                pb = model.predict([x1b, x2b], verbose=0)
                preds.append(pb)
                trues.append(yb.numpy())
            
            pred = np.concatenate(preds, axis=0)
            true = np.concatenate(trues, axis=0)
            
            # ✅ 0~1 → 0~100% 변환 (수정!)
            pred = np.clip(pred * 100.0, 0, 100) * land_mask_d[..., None]
            true = np.clip(true * 100.0, 0, 100) * land_mask_d[..., None]
            
            # Metric 평가
            first_batch_index = int(te_idx[0]) if len(te_idx) else 0
            metric = evaluate_for_metric_node(
                pred, true, lead_days, 
                first_batch_index, seq_long, model_name, now, land_mask_d
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
            
            del preds, trues
        
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