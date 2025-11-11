# mt_icenet_daily_multihead_stream.py
# MT-IceNet (일별 단일 피처, 다중 리드타임: +1/+3/+6일) — 메모리 세이프(tf.data) + GPU 최적화
# - 일별 GeoTIFF SIC → 시퀀스(과거 90일) → ConvLSTM+U-Net → 3개 리드타임 동시 예측

import os, re, glob, warnings
import numpy as np
import pandas as pd
import tifffile as tiff
import tensorflow as tf

from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Conv2D, ConvLSTM2D, BatchNormalization, MaxPooling2D,
                                     UpSampling2D, concatenate, Activation)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# -----------------------------
# 0) 설정
# -----------------------------
DATA_ROOT     = r"C:\Users\USER\Desktop\ice\data\NSIDC_Data"   # 연/월/일 경로 루트
FILE_REGEX    = r"N_(\d{8})_concentration.*\.tif$"             # 날짜 파싱
IMG_SHAPE     = (448, 304)                                     # NSIDC 25km 그리드 예시

TIMESTEP_DAYS = 30            # 입력 길이(과거 90일)
LEAD_DAYS     = [1, 3, 6]     # 예측 리드타임(일)
BATCH_SIZE    = 4             # 메모리 여유되면 2
EPOCHS        = 50
LEARNING_RATE = 1e-4
SEED          = 42
STRIDE        = 7             # 윈도우 간격(>1로 줄이면 샘플↓ 메모리/시간↓)
DOWNSAMPLE    = 2             # 1=원본, 2/4=해상도 축소(속도↑ 메모리↑)

USE_MIXED_PRECISION = True    # GPU면 True 권장
USE_XLA             = True    # XLA JIT

# 연도 분할(입력 마지막 날의 연도 기준)
TRAIN_YEARS = list(range(2013, 2020))   # 2013~2019
VAL_YEARS   = [2020]                    # 2020
TEST_YEARS  = [2021, 2022]              # 2021~2022

np.random.seed(SEED)
tf.random.set_seed(SEED)

# 혼합정밀/XLA/메모리 성장
if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
if USE_XLA:
    tf.config.optimizer.set_jit(True)
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

print("CUDA build?:", tf.test.is_built_with_cuda())
print("Physical GPU:", tf.config.list_physical_devices("GPU"))
print("Logical  GPU:", tf.config.list_logical_devices("GPU"))

# -----------------------------
# 1) 유틸: 파일 스캔 & 읽기
# -----------------------------
def list_tif_paths(data_root):
    paths = glob.glob(os.path.join(data_root, "*", "*", "*.tif"))
    return sorted(paths)

def parse_date_from_name(path):
    m = re.search(FILE_REGEX, os.path.basename(path))
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d")

def read_one_tif(path):
    arr = tiff.imread(path).astype(np.float32)
    return arr

# -----------------------------
# 2) 일별 스택 로드 → DatetimeIndex
# -----------------------------
def load_daily_stack(data_root, target_shape=IMG_SHAPE):
    recs = []
    for p in list_tif_paths(data_root):
        d = parse_date_from_name(p)
        if d is None:
            continue
        recs.append((pd.Timestamp(d), p))
    if not recs:
        raise RuntimeError("GeoTIFF 파일을 찾지 못했습니다.")

    recs.sort(key=lambda x: x[0])
    dates, arrays = [], []
    for d, p in recs:
        a = read_one_tif(p)
        if a.shape != target_shape:
            raise ValueError(f"이미지 크기 불일치: {p} shape={a.shape}, expected={target_shape}")
        arrays.append(a)
        dates.append(d)

    X = np.stack(arrays, axis=0)  # [T,H,W]
    idx = pd.DatetimeIndex(dates)
    return idx, X

# -----------------------------
# 3) 마스크(바다=1, 육지/결측=0)
# -----------------------------
def make_land_mask(daily_stack):
    valid = np.isfinite(daily_stack)
    ocean = (valid.sum(axis=0) > 0).astype(np.float32)
    return ocean  # [H,W]

# -----------------------------
# 4) 인덱스 분할 + tf.data 스트리밍
# -----------------------------
def build_index_splits(daily_idx, t_days, lead_days, split_years, stride=1):
    T = len(daily_idx)
    max_lead = max(lead_days)
    ii, yrs = [], []
    for t in range(t_days, T - max_lead, stride):
        ii.append(t)
        yrs.append(daily_idx[t-1].year)   # 입력 마지막 날의 연도
    ii  = np.array(ii,  dtype=np.int32)
    yrs = np.array(yrs, dtype=np.int32)
    tr = ii[np.isin(yrs, split_years[0])]
    va = ii[np.isin(yrs, split_years[1])]
    te = ii[np.isin(yrs, split_years[2])]
    return tr, va, te

def _maybe_downsample_3d(x, new_hw):
    # x: [T,H,W] or [H,W] (최종에서 맞춰서 호출)
    x = tf.expand_dims(x, -1)  # [...,1]
    x = tf.image.resize(x, new_hw, method='area')
    return tf.squeeze(x, -1)

def make_dataset(daily_stack, indices, t_days=90, lead_days=(1,3,6),
                 batch_size=1, shuffle=False, seed=42, downsample=1):
    # 0~1 스케일(복사 최소화를 위해 float32 텐서로 1회 변환)
    ds_x = tf.convert_to_tensor(np.nan_to_num(daily_stack, nan=0.0) / 100.0, dtype=tf.float32)  # [T,H,W]
    lead_days = tf.constant(list(lead_days), dtype=tf.int32)
    H, W = ds_x.shape[1], ds_x.shape[2]
    if downsample > 1:
        new_hw = (H // downsample, W // downsample)
    else:
        new_hw = None

    ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(4096, len(indices)), seed=seed, reshuffle_each_iteration=True)

    @tf.function  # 성능↑
    def _slice_one(t):
        x = ds_x[t - t_days : t]              # [T,H,W]
        if new_hw is not None:
            x = _maybe_downsample_3d(x, new_hw)  # [T,h,w]
        x = tf.expand_dims(x, -1)             # [T,h,w,1] or [T,H,W,1]
        ys = tf.stack([ds_x[t + L] for L in tf.unstack(lead_days)], axis=-1)  # [H,W,C]
        if new_hw is not None:
            ys = tf.image.resize(ys, new_hw, method='area')  # [h,w,C]
        return x, ys

    ds = ds.map(_slice_one, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# -----------------------------
# 5) 모델(ConvLSTM + U-Net, 출력 C=len(LEAD_DAYS))
# -----------------------------
def build_model_daily(t_days, H, W, n_out, lr=LEARNING_RATE, filt=3):
    inp = Input(shape=(t_days, H, W, 1))
    # cuDNN 경로로 강제(매우 중요)
    x = ConvLSTM2D(
        8, (5,5), padding="same", return_sequences=False, data_format="channels_last",
        activation="tanh", recurrent_activation="sigmoid")(inp)

    # Encoder
    c1 = Conv2D(16, filt, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    c1 = Conv2D(16, filt, activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    b1 = BatchNormalization(axis=-1)(c1)
    p1 = MaxPooling2D(pool_size=(2,2))(b1)

    # Bottleneck
    b  = Conv2D(64, filt, activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    b  = Conv2D(64, filt, activation='relu', padding='same', kernel_initializer='he_normal')(b)
    bN = BatchNormalization(axis=-1)(b)

    # Decoder
    u1 = UpSampling2D(size=(2,2))(bN)
    u1 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(u1)
    m1 = concatenate([b1, u1], axis=3)

    c9 = Conv2D(16, filt, activation='relu', padding='same', kernel_initializer='he_normal')(m1)
    c9 = Conv2D(16, filt, activation='relu', padding='same', kernel_initializer='he_normal')(c9)

    raw_out = Conv2D(n_out, 1, activation='linear')(c9)   # mixed precision이면 float16일 수 있음
    out = Activation('linear', dtype='float32')(raw_out)  # 손실/지표 안정화를 위해 float32로 캐스트

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse",
                  metrics=[RootMeanSquaredError()])
    return model

# -----------------------------
# 6) 메인
# -----------------------------
def main():
    # 1) 로드
    daily_idx, daily_stack = load_daily_stack(DATA_ROOT)  # [T,H,W]
    H, W = daily_stack.shape[1], daily_stack.shape[2]
    print("Daily stack:", daily_stack.shape)

    # 2) 마스크
    land_mask = make_land_mask(daily_stack)  # [H,W]
    if DOWNSAMPLE > 1:
        # 평가/저장과 동일 해상도 맞추기
        lm = tf.convert_to_tensor(land_mask[...,None], dtype=tf.float32)
        lm = tf.image.resize(lm, (H//DOWNSAMPLE, W//DOWNSAMPLE), method='nearest')  # 마스크는 nearest
        land_mask_d = tf.squeeze(lm, -1).numpy().astype(np.float32)
        H_eff, W_eff = land_mask_d.shape
    else:
        land_mask_d = land_mask
        H_eff, W_eff = H, W

    # 3) 인덱스 분할 + 데이터셋
    tr_idx, va_idx, te_idx = build_index_splits(
        daily_idx, TIMESTEP_DAYS, LEAD_DAYS, (TRAIN_YEARS, VAL_YEARS, TEST_YEARS), stride=STRIDE
    )
    print(f"indices  train/val/test: {len(tr_idx)} / {len(va_idx)} / {len(te_idx)}")

    train_ds = make_dataset(daily_stack, tr_idx, TIMESTEP_DAYS, LEAD_DAYS,
                            batch_size=BATCH_SIZE, shuffle=True, seed=SEED, downsample=DOWNSAMPLE)
    val_ds   = make_dataset(daily_stack, va_idx, TIMESTEP_DAYS, LEAD_DAYS,
                            batch_size=BATCH_SIZE, shuffle=False, downsample=DOWNSAMPLE)
    test_ds  = make_dataset(daily_stack, te_idx, TIMESTEP_DAYS, LEAD_DAYS,
                            batch_size=BATCH_SIZE, shuffle=False, downsample=DOWNSAMPLE)

    # 4) 모델
    model = build_model_daily(TIMESTEP_DAYS, H_eff, W_eff, n_out=len(LEAD_DAYS))
    print(model.summary())

    # 5) 학습
    es = EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2, callbacks=[es])

    # 6) 예측(스트리밍)
    preds, trues = [], []
    for xb, yb in test_ds:
        pb = model.predict(xb, verbose=0)  # [B,h,w,C]
        preds.append(pb)
        trues.append(yb.numpy())
    pred = np.concatenate(preds, axis=0)  # [N,h,w,C], 스케일 0-1
    true = np.concatenate(trues, axis=0)  # [N,h,w,C]

    # 7) 0-100 복원 + 마스크 + 클립
    pred = np.clip(pred * 100.0, 0, 100) * land_mask_d[..., None]
    true = np.clip(true * 100.0, 0, 100) * land_mask_d[..., None]

    # 8) 지표(리드타임별)
    for i, L in enumerate(LEAD_DAYS):
        p = pred[..., i].ravel()
        t = true[..., i].ravel()
        mse  = mean_squared_error(t, p)
        rmse = sqrt(mse)
        mae  = mean_absolute_error(t, p)
        r2   = r2_score(t, p)
        print(f"[+{L}d] MSE={mse:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")

    # 9) 저장
    os.makedirs("./outputs", exist_ok=True)
    np.save("./outputs/pred_sic_mt_icenet_daily_ldays.npy", pred.astype(np.float32))  # [N,h,w,3]
    np.save("./outputs/true_sic_mt_icenet_daily_ldays.npy", true.astype(np.float32))  # [N,h,w,3]
    np.save("./outputs/land_mask.npy", land_mask_d.astype(np.float32))
    print("Saved: ./outputs/pred_sic_mt_icenet_daily_ldays.npy (N,h,w,3)")

if __name__ == "__main__":
    main()
