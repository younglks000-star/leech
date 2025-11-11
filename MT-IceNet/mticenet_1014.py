# mt_icenet_spyder_from_geotiff.py
# MT-IceNet (SIC 단일 피처) — Spyder 실행용
# - 일별 GeoTIFF SIC → 월/15일 평균 → 시퀀스 구성 → ConvLSTM+UNet 학습
# ------------------------------------------------------------------------------

import os, re, glob, math, warnings
import numpy as np
import pandas as pd
import tifffile as tiff
import tensorflow as tf

from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Conv2D, ConvLSTM2D, BatchNormalization, MaxPooling2D,
                                     UpSampling2D, concatenate)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# -----------------------------
# 0) 설정
# -----------------------------
DATA_ROOT = r"C:\Users\USER\Desktop\ice\data\NSIDC_Data"  # 연/월/일 경로 루트
FILE_REGEX = r"N_(\d{8})_concentration.*\.tif$"            # 날짜 파싱
IMG_SHAPE = (448, 304)                                     # NSIDC 25km 그리드 예시
FREQ_MONTH = "M"       # 월평균
FREQ_15D   = "15D"     # 15일 평균

# 시퀀스/학습 설정
TIMESTEP_MONTH = 12     # 과거 12개월
TIMESTEP_15D   = 24     # 과거 24*15일
LAG_MONTHS     = 6      # 타깃: lag개월 뒤의 월평균
BATCH_SIZE     = 2
EPOCHS         = 50
LEARNING_RATE  = 1e-4
VAL_SPLIT      = 0.2
SEED           = 42

# 데이터 분할(연도 기준): 필요에 맞게 조정
TRAIN_YEARS = list(range(2013, 2020))   # 2013~2019
VAL_YEARS   = [2020]                    # 2020
TEST_YEARS  = [2021, 2022]              # 2021~2022

np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# 1) 유틸: 파일 스캔 & 읽기
# -----------------------------
def list_tif_paths(data_root):
    paths = glob.glob(os.path.join(data_root, "*", "*", "*.tif"))
    paths = sorted(paths)
    return paths

def parse_date_from_name(path):
    # 파일명에서 YYYYMMDD 추출
    m = re.search(FILE_REGEX, os.path.basename(path))
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d")

def read_one_tif(path):
    arr = tiff.imread(path)
    arr = np.array(arr, dtype=np.float32)
    return arr

# -----------------------------
# 2) 일별 스택 로드 → pandas 시계열 정렬
# -----------------------------
def load_daily_stack(data_root, target_shape=IMG_SHAPE):
    recs = []
    for p in list_tif_paths(data_root):
        d = parse_date_from_name(p)
        if d is None:
            continue
        recs.append((d, p))
    if not recs:
        raise RuntimeError("GeoTIFF 파일을 찾지 못했습니다.")

    recs.sort(key=lambda x: x[0])
    dates = []
    arrays = []
    for d, p in recs:
        a = read_one_tif(p)
        # 크기 확인
        if a.shape != target_shape:
            raise ValueError(f"이미지 크기 불일치: {p} shape={a.shape}, expected={target_shape}")
        # SIC 스케일(0~100) 가정. 필요시 스케일 변환/마스킹 추가
        # NaN 처리: NaN 유지 후 나중에 평균/마스크에 반영
        arrays.append(a)
        dates.append(pd.Timestamp(d))
    X = np.stack(arrays, axis=0)  # [T, H, W]
    idx = pd.to_datetime(dates)
    return pd.Series(list(range(len(idx))), index=idx), X

# -----------------------------
# 3) 월/15일 평균 시계열 생성
# -----------------------------
def time_average_series(indexer, data_stack, rule):
    # indexer: pd.Series(values=position, index=datetime)
    # rule: 'M' or '15D'
    # 반환: (new_index, new_stack) where new_stack [T2, H, W]
    df = pd.DataFrame({"pos": indexer.values}, index=indexer.index)
    groups = df.resample(rule).mean()  # pos 평균(정수 아님) → 아래에서 직접 평균 계산
    out_idx = []
    out_list = []
    # 각 구간의 실제 날짜 범위로 평균
    for ts in groups.index:
        # resample 결과의 기간 범위 얻기
        # pandas는 레이블만 줌 → 윈도우 범위를 idx 기반으로 추정
        # 간단히: 해당 bin에 포함된 원시 타임스탬프 선택
        mask = (indexer.index >= ts - pd.tseries.frequencies.to_offset(rule) + pd.Timedelta(days=1)) & (indexer.index <= ts)
        select_idx = np.where(mask)[0]
        if len(select_idx) == 0:
            continue
        # 평균(시간축)
        block = data_stack[select_idx, ...]  # [n, H, W]
        with np.errstate(invalid='ignore'):
            mean_map = np.nanmean(block, axis=0)
        out_idx.append(ts)
        out_list.append(mean_map.astype(np.float32))
    if not out_list:
        raise RuntimeError("해당 주기로 평균한 결과가 없습니다.")
    out_stack = np.stack(out_list, axis=0)
    return pd.Index(out_idx), out_stack  # [T2, H, W]

# -----------------------------
# 4) 마스크 생성(바다=1, 육지/결측=0)
# -----------------------------
def make_land_mask(daily_stack):
    # 기간 중 한 번이라도 관측(비NaN)이면 바다로 간주 (1), 전기간 NaN이면 0
    valid = np.isfinite(daily_stack)
    ocean = (valid.sum(axis=0) > 0).astype(np.float32)
    return ocean  # [H, W], (1=해역)

# -----------------------------
# 5) 시퀀스 생성(정렬)
# -----------------------------
def build_sequences(month_idx, month_stack, bi15_idx, bi15_stack,
                    t_month=12, t_15d=24, lag=6,
                    split_by_year=(TRAIN_YEARS, VAL_YEARS, TEST_YEARS)):
    """
    - 입력1: 과거 t_month 개(월평균)  @ 시점 t-1 까지
    - 입력2: 과거 t_15d   개(15일평균) @ 시점 t-1(15일 단위) 까지
    - 타깃 : month_stack 의 (t + lag) 시점 맵
    기준 타임라인은 month_idx(월말 타임스탬프)로 잡음.
    """
    # 월 타임라인에서 샘플 가능한 t 인덱스 집합 생성
    X1, X2, Y  = [], [], []
    T1, H, W = month_stack.shape
    T2       = bi15_stack.shape[0]

    # 빠른 매칭을 위한 dict
    bi_map = {ts: i for i, ts in enumerate(bi15_idx)}
    # 15일 인덱스를 월(ts) 레이블에 맞춰 "ts 이전"까지 갖도록 ts별로 마지막 bi 인덱스를 찾음
    bi_ts = list(bi_map.keys())

    # 월 인덱스에서 유효한 중앙 시점 t를 순회
    for t in range(t_month, T1 - lag):  # t-1까지 입력 확보, t+lag 타깃 존재
        ts_t   = month_idx[t]          # 기준 월 타임스탬프
        ts_inp = month_idx[t-1]        # 입력 마지막 월

        # 입력1: month_stack[t - t_month : t]
        x1 = month_stack[t - t_month : t, ...]  # [t_month, H, W]

        # 입력2: 15일 인덱스에서 ts_inp 이전까지의 마지막 위치 찾기
        # bi15는 규칙적이므로 ts_inp 이하의 타임스탬프 중 최댓값 위치를 사용
        bi_valid = [b for b in bi_ts if b <= ts_inp]
        if len(bi_valid) < t_15d:
            continue
        end_bi = bi_valid[-1]
        end_bi_idx = bi_map[end_bi]
        start_bi_idx = end_bi_idx - (t_15d - 1)
        if start_bi_idx < 0:
            continue
        x2 = bi15_stack[start_bi_idx:end_bi_idx+1, ...]  # [t_15d, H, W]

        # 타깃: month_stack[t + lag]
        y  = month_stack[t + lag, ...]  # [H, W]

        X1.append(x1)
        X2.append(x2)
        Y.append(y)

    X1 = np.stack(X1, axis=0)  # [N, t_month, H, W]
    X2 = np.stack(X2, axis=0)  # [N, t_15d, H, W]
    Y  = np.stack(Y,  axis=0)  # [N, H, W]

    # 피처 차원 추가(단일 피처=1)
    X1 = X1[..., np.newaxis]  # [N, t_month, H, W, 1]
    X2 = X2[..., np.newaxis]  # [N, t_15d,   H, W, 1]
    Y  = Y[...,  np.newaxis]  # [N, H, W, 1]

    # 연도별 분할
    years = np.array([ts.year for ts in month_idx[t_month: len(month_idx)-lag]])
    # 위 years는 X/Y에 대응되는 t 축의 실제 연도. 상단 루프 로직과 맞추기 위해 재계산:
    years = []
    for t in range(t_month, T1 - lag):
        years.append(month_idx[t].year)
    years = np.array(years)

    train_mask = np.isin(years, split_by_year[0])
    val_mask   = np.isin(years, split_by_year[1])
    test_mask  = np.isin(years, split_by_year[2])

    def split(mask):
        return X1[mask], X2[mask], Y[mask]

    return split(train_mask) + split(val_mask) + split(test_mask)

# -----------------------------
# 6) 스케일러 (SIC 단일 피처)
# -----------------------------
def fit_transform_scalers(x1_tr, x2_tr, y_tr, x1_va, x2_va, y_va, x1_te, x2_te, y_te):
    sc_x1 = StandardScaler()
    sc_x2 = StandardScaler()
    sc_y  = StandardScaler()

    def scale_5d(sc, arr):
        N, T, H, W, C = arr.shape
        flat = arr.reshape(-1, 1)  # 단일 피처
        flat = sc.fit_transform(flat)
        return flat.reshape(N, T, H, W, C)

    def scale_4d(sc, arr, fit=False):
        N, H, W, C = arr.shape
        flat = arr.reshape(-1, 1)
        if fit:
            flat = sc.fit_transform(flat)
        else:
            flat = sc.transform(flat)
        return flat.reshape(N, H, W, C)

    x1_tr_s = scale_5d(sc_x1, x1_tr)
    x2_tr_s = scale_5d(sc_x2, x2_tr)
    y_tr_s  = scale_4d(sc_y,  y_tr, fit=True)

    def trans(sc, arr5, arr4):
        x1_s = sc_x1.transform(arr5.reshape(-1,1)).reshape(arr5.shape)
        y_s  = sc_y.transform(arr4.reshape(-1,1)).reshape(arr4.shape)
        return x1_s, y_s

    x1_va_s = sc_x1.transform(x1_va.reshape(-1,1)).reshape(x1_va.shape)
    x2_va_s = sc_x2.transform(x2_va.reshape(-1,1)).reshape(x2_va.shape)
    y_va_s  = sc_y.transform(y_va.reshape(-1,1)).reshape(y_va.shape)

    x1_te_s = sc_x1.transform(x1_te.reshape(-1,1)).reshape(x1_te.shape)
    x2_te_s = sc_x2.transform(x2_te.reshape(-1,1)).reshape(x2_te.shape)
    y_te_s  = sc_y.transform(y_te.reshape(-1,1)).reshape(y_te.shape)

    return (x1_tr_s, x2_tr_s, y_tr_s,
            x1_va_s, x2_va_s, y_va_s,
            x1_te_s, x2_te_s, y_te_s,
            sc_x1, sc_x2, sc_y)

# -----------------------------
# 7) 모델 (MT-IceNet 단순화, 피처=1)
# -----------------------------
def build_model(input1_shape, input2_shape, lr=LEARNING_RATE, filt=3, n_out=1):
    input1 = Input(shape=input1_shape)  # (12, H, W, 1)
    input2 = Input(shape=input2_shape)  # (24, H, W, 1)

    convlstm1 = ConvLSTM2D(8, (5,5), padding="same", return_sequences=False, data_format="channels_last")(input1)
    conv1 = Conv2D(16, filt, activation='relu', padding='same', kernel_initializer='he_normal')(convlstm1)
    conv1 = Conv2D(16, filt, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    bn1   = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    convlstm2 = ConvLSTM2D(8, (5,5), padding="same", return_sequences=False, data_format="channels_last")(input2)
    conv2 = Conv2D(32, filt, activation='relu', padding='same', kernel_initializer='he_normal')(convlstm2)
    conv2 = Conv2D(32, filt, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    bn2   = BatchNormalization(axis=-1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    # bottleneck
    conv3 = Conv2D(64, filt, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, filt, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    bn3   = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    # up1
    up8   = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(pool3))
    merge8= concatenate([bn3, up8], axis=3)
    conv8 = Conv2D(32, filt, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(32, filt, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    bn8   = BatchNormalization(axis=-1)(conv8)

    # up2
    up9   = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(bn8))
    merge9= concatenate([bn1, up9], axis=3)
    conv9 = Conv2D(16, filt, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(16, filt, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(16, filt, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    output = Conv2D(n_out, 1, activation='linear')(conv9)

    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=[RootMeanSquaredError()])
    return model

# -----------------------------
# 8) 메인 파이프라인
# -----------------------------
def main():
    # GPU 메모리 점유 완화
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass

    # 1) 일별 SIC 로드
    daily_indexer, daily_stack = load_daily_stack(DATA_ROOT)  # indexer: date->pos
    H, W = daily_stack.shape[1], daily_stack.shape[2]
    print("Daily stack:", daily_stack.shape)

    # 2) 마스크
    land_mask = make_land_mask(daily_stack)  # ocean=1, land=0
    y_land_mask = land_mask[np.newaxis, ...]  # [1, H, W] (추후 브로드캐스팅에 사용)

    # 3) 월/15일 평균
    month_idx, month_stack = time_average_series(daily_indexer, daily_stack, FREQ_MONTH)
    bi15_idx,   bi15_stack = time_average_series(daily_indexer, daily_stack, FREQ_15D)
    print("Monthly:", month_stack.shape, "15D:", bi15_stack.shape)

    # 4) 시퀀스 생성
    (x1_tr, x2_tr, y_tr,
     x1_va, x2_va, y_va,
     x1_te, x2_te, y_te) = build_sequences(month_idx, month_stack, bi15_idx, bi15_stack,
                                           t_month=TIMESTEP_MONTH, t_15d=TIMESTEP_15D, lag=LAG_MONTHS,
                                           split_by_year=(TRAIN_YEARS, VAL_YEARS, TEST_YEARS))
    print("Train:", x1_tr.shape, x2_tr.shape, y_tr.shape)
    print("Val  :", x1_va.shape, x2_va.shape, y_va.shape)
    print("Test :", x1_te.shape, x2_te.shape, y_te.shape)

    # 5) 스케일
    (x1_tr_s, x2_tr_s, y_tr_s,
     x1_va_s, x2_va_s, y_va_s,
     x1_te_s, x2_te_s, y_te_s,
     sc_x1, sc_x2, sc_y) = fit_transform_scalers(x1_tr, x2_tr, y_tr, x1_va, x2_va, y_va, x1_te, x2_te, y_te)

    # 6) 모델
    input1_shape = (TIMESTEP_MONTH, H, W, 1)
    input2_shape = (TIMESTEP_15D,   H, W, 1)
    model = build_model(input1_shape, input2_shape)
    print(model.summary())

    # 7) 학습
    es = EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
    history = model.fit([x1_tr_s, x2_tr_s], y_tr_s,
                        validation_data=([x1_va_s, x2_va_s], y_va_s),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, callbacks=[es])

    # 8) 예측 & 역스케일 & 마스크
    y_pred_s = model.predict([x1_te_s, x2_te_s], verbose=0)
    inv_y_pred = sc_y.inverse_transform(y_pred_s.reshape(-1,1)).reshape(y_pred_s.shape)  # [N,H,W,1]
    inv_y_test = sc_y.inverse_transform(y_te_s.reshape(-1,1)).reshape(y_te_s.shape)

    inv_y_pred = inv_y_pred.squeeze(-1)  # [N,H,W]
    inv_y_test = inv_y_test.squeeze(-1)

    # 마스킹 (해역만 평가)
    inv_y_pred = inv_y_pred * land_mask
    inv_y_test = inv_y_test * land_mask

    # 후처리 (0~100 clip)
    post_y = np.clip(inv_y_pred, 0, 100)

    # 9) 지표
    rmse = sqrt(mean_squared_error(inv_y_test.flatten(), inv_y_pred.flatten()))
    r_sq = r2_score(inv_y_test.flatten(), inv_y_pred.flatten())
    rmse_post = sqrt(mean_squared_error(inv_y_test.flatten(), post_y.flatten()))
    r_sq_post = r2_score(inv_y_test.flatten(), post_y.flatten())
    mae_post  = mean_absolute_error(inv_y_test.flatten(), post_y.flatten())

    print(f"Test RMSE(raw): {rmse:.3f}")
    print(f"Test R2  (raw): {r_sq:.3f}")
    print(f"Post-Processed RMSE: {rmse_post:.3f}")
    print(f"Post-Processed R2  : {r_sq_post:.3f}")
    print(f"Post-Processed MAE : {mae_post:.3f}")

    # 10) 저장
    os.makedirs("./outputs", exist_ok=True)
    np.save("./outputs/pred_sic_mt_icenet_lag{}_test.npy".format(LAG_MONTHS), post_y.astype(np.float32))
    np.save("./outputs/true_sic_mt_icenet_lag{}_test.npy".format(LAG_MONTHS), inv_y_test.astype(np.float32))
    np.save("./outputs/land_mask.npy", land_mask.astype(np.float32))
    print("Saved: ./outputs/*.npy")

if __name__ == "__main__":
    main()
