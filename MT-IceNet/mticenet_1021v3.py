# mticenet_corrected.py
# ✅ MT-IceNet 공식 코드 기반, 당신의 실험 설정만 반영
# ✅ 정규화: StandardScaler (공식 코드와 동일)
# ✅ 데이터 로딩: 당신의 NSIDC 데이터
# ✅ 평가: Metric_node 사용

import os, re, glob, warnings, sys
import numpy as np
import pandas as pd
import tifffile as tiff
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

sys.path.append(r"C:\Users\USER\Desktop\baseline\MT-IceNet\utils")
import Metric_node as Metric
import torch

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Conv2D, ConvLSTM2D, BatchNormalization,
                                     MaxPooling2D, UpSampling2D, concatenate, Activation)
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# =============================================================================
# 설정
# =============================================================================
DATA_ROOT     = r"C:\Users\USER\Desktop\ice\data\NSIDC_Data"
FILE_REGEX    = r"N_(\d{8})_concentration.*\.tif$"
IMG_SHAPE     = (448, 304)

# 실험 설정 (당신이 수정 가능)
output_lens   = [7, 14, 21]
seq_short     = 30   # 12 → 30 (당신 설정)
seq_long      = 90   # 24 → 90 (당신 설정)

BATCH_SIZE    = 2
Epoch         = 50
LEARNING_RATE = 1e-4
SEED          = 42
STRIDE        = 7

model_name   = "MT-IceNet-Dual"
save         = True
base_date    = datetime(2013, 1, 1)

np.random.seed(SEED)
tf.random.set_seed(SEED)

# GPU 설정 (공식 코드 방식)
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

TRAIN_YEARS = list(range(2013, 2020))
VAL_YEARS   = [2020]
TEST_YEARS  = [2021, 2022]

# =============================================================================
# 데이터 로딩 (✅ 공식 코드 방식)
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
    ✅ 공식 코드와 동일: NaN → 0, 원본 값 그대로 유지
    (정규화는 나중에 StandardScaler로)
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
        
        # ✅ 공식 코드: 특수값 처리
        # 2510 (극점), 2530 (해안선), 2540 (육지)
        a[a >= 2500] = 0  # 공식 코드는 0으로 처리
        
        frames.append(a)
        dates.append(d)
    
    X = np.stack(frames, axis=0)  # [T, H, W]
    X = np.nan_to_num(X, nan=0.0)  # ✅ 공식 코드: NaN → 0
    
    idx = pd.DatetimeIndex(dates)
    return idx, X

def make_land_mask(daily_stack):
    """육지 마스크 (공식 코드 방식)"""
    # 모든 시점에서 0인 픽셀 = 육지
    land = (daily_stack == 0).all(axis=0)
    ocean = (~land).astype(np.float32)
    return ocean

def build_index_splits(daily_idx, seq_long, max_lead, split_years, stride=1):
    """인덱스 분할"""
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

# =============================================================================
# 데이터셋 생성
# =============================================================================
def create_sequences(data, indices, seq_short, seq_long, lead_days):
    """
    ✅ 공식 코드 방식: NumPy로 시퀀스 생성
    """
    X1_list, X2_list, Y_list = [], [], []
    
    for idx in indices:
        # 짧은 윈도우
        x1 = data[idx - seq_short : idx]  # [seq_short, H, W]
        
        # 긴 윈도우
        x2 = data[idx - seq_long : idx]   # [seq_long, H, W]
        
        # 출력
        y = np.stack([data[idx + L] for L in lead_days], axis=-1)  # [H, W, n_leads]
        
        X1_list.append(x1)
        X2_list.append(x2)
        Y_list.append(y)
    
    X1 = np.array(X1_list)  # [N, seq_short, H, W]
    X2 = np.array(X2_list)  # [N, seq_long, H, W]
    Y = np.array(Y_list)    # [N, H, W, n_leads]
    
    return X1, X2, Y

# =============================================================================
# 모델: MT-IceNet (✅ 공식 코드 그대로)
# =============================================================================
def build_mt_icenet(seq_short, seq_long, H, W, n_features, n_out, 
                    lr=LEARNING_RATE, filt=3):
    """
    ✅ 공식 코드의 MT-IceNet 구조 그대로
    """
    input1 = Input(shape=(seq_short, H, W, n_features), name='input_short')
    input2 = Input(shape=(seq_long, H, W, n_features), name='input_long')
    
    # Branch 1: 짧은 윈도우
    convlstm1 = ConvLSTM2D(8, (5,5), padding="same", return_sequences=False, 
                           data_format="channels_last")(input1)
    
    conv1 = Conv2D(16, filt, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(convlstm1)
    conv1 = Conv2D(16, filt, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(conv1)
    bn1 = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
    
    # Branch 2: 긴 윈도우
    convlstm2 = ConvLSTM2D(8, (5,5), padding="same", return_sequences=False, 
                           data_format="channels_last")(input2)
    
    conv2 = Conv2D(32, filt, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(convlstm2)
    conv2 = Conv2D(32, filt, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(conv2)
    bn2 = BatchNormalization(axis=-1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
    
    # Encoder
    conv3 = Conv2D(64, filt, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, filt, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(conv3)
    bn3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)
    
    # Decoder
    up8 = Conv2D(32, 2, activation='relu', padding='same', 
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(pool3))
    merge8 = concatenate([bn3, up8], axis=3)
    conv8 = Conv2D(32, filt, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(32, filt, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(conv8)
    bn8 = BatchNormalization(axis=-1)(conv8)
    
    up9 = Conv2D(16, 2, activation='relu', padding='same', 
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(bn8))
    merge9 = concatenate([bn1, up9], axis=3)
    conv9 = Conv2D(16, filt, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(16, filt, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(16, filt, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(conv9)
    
    # Output
    output = Conv2D(n_out, 1, activation='linear')(conv9)
    
    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer=Adam(lr=lr), loss='mse', 
                  metrics=['RootMeanSquaredError'])
    
    return model

# =============================================================================
# 평가
# =============================================================================
@torch.no_grad()
def evaluate_for_metric_node(pred_maps, true_maps, lead_days, 
                             first_batch_index, seq_input, model_name, 
                             tag_time, land_mask):
    """Metric_node 평가"""
    N, h, w, C = pred_maps.shape
    
    pred = pred_maps.reshape(N, h*w, C).transpose(0, 2, 1)
    true = true_maps.reshape(N, h*w, C).transpose(0, 2, 1)
    
    pred_t = torch.from_numpy(pred).float()
    true_t = torch.from_numpy(true).float()
    
    n_features = h * w
    metric = Metric.metric(pred_t, true_t, n_features)
    
    Metric.plot(pred_t, true_t, f"{model_name}_{C}", C, tag_time)
    
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
    print("="*70)
    print("MT-IceNet: 공식 코드 기반 실험")
    print("="*70)
    
    # 1. 데이터 로드
    daily_idx, daily_stack = load_daily_stack(DATA_ROOT, IMG_SHAPE)
    H, W = daily_stack.shape[1], daily_stack.shape[2]
    land_mask = make_land_mask(daily_stack)
    
    print(f"데이터 shape: {daily_stack.shape}")
    print(f"날짜 범위: {daily_idx[0]} ~ {daily_idx[-1]}")
    print(f"총 {len(daily_idx)}일 데이터")
    
    ocean_pixels = (land_mask == 1).sum()
    land_pixels = (land_mask == 0).sum()
    print(f"바다 픽셀: {ocean_pixels:,} ({ocean_pixels/(ocean_pixels+land_pixels)*100:.2f}%)")
    print(f"육지 픽셀: {land_pixels:,}\n")
    
    results = {}
    
    for seq_output in output_lens:
        print(f"\n{'='*70}")
        print(f"실험: +{seq_output}일 연속 예측")
        print(f"단기 윈도우: {seq_short}일, 장기 윈도우: {seq_long}일")
        print(f"{'='*70}")
        
        lead_days = list(range(1, seq_output + 1))
        max_lead = seq_output
        
        # 2. 인덱스 분할
        tr_idx, va_idx, te_idx = build_index_splits(
            daily_idx, seq_long, max_lead,
            (TRAIN_YEARS, VAL_YEARS, TEST_YEARS), 
            stride=STRIDE
        )
        
        print(f"데이터: Train={len(tr_idx)}, Val={len(va_idx)}, Test={len(te_idx)}")
        
        # 3. 시퀀스 생성
        X1_train, X2_train, Y_train = create_sequences(
            daily_stack, tr_idx, seq_short, seq_long, lead_days)
        X1_val, X2_val, Y_val = create_sequences(
            daily_stack, va_idx, seq_short, seq_long, lead_days)
        X1_test, X2_test, Y_test = create_sequences(
            daily_stack, te_idx, seq_short, seq_long, lead_days)
        
        print(f"X1_train: {X1_train.shape}, X2_train: {X2_train.shape}, Y_train: {Y_train.shape}")
        
        # 4. ✅ 정규화 (공식 코드 방식: StandardScaler)
        print("\n✅ StandardScaler 정규화 중...")
        
        scaler_x1 = StandardScaler()
        X1_train = scaler_x1.fit_transform(X1_train.reshape(-1, 1)).reshape(X1_train.shape)
        X1_val = scaler_x1.transform(X1_val.reshape(-1, 1)).reshape(X1_val.shape)
        X1_test = scaler_x1.transform(X1_test.reshape(-1, 1)).reshape(X1_test.shape)
        
        scaler_x2 = StandardScaler()
        X2_train = scaler_x2.fit_transform(X2_train.reshape(-1, 1)).reshape(X2_train.shape)
        X2_val = scaler_x2.transform(X2_val.reshape(-1, 1)).reshape(X2_val.shape)
        X2_test = scaler_x2.transform(X2_test.reshape(-1, 1)).reshape(X2_test.shape)
        
        scaler_y = StandardScaler()
        Y_train = scaler_y.fit_transform(Y_train.reshape(-1, 1)).reshape(Y_train.shape)
        Y_val = scaler_y.transform(Y_val.reshape(-1, 1)).reshape(Y_val.shape)
        Y_test = scaler_y.transform(Y_test.reshape(-1, 1)).reshape(Y_test.shape)
        
        # 5. Reshape: [N, T, H, W] → [N, T, H, W, 1]
        X1_train = X1_train[..., np.newaxis]
        X2_train = X2_train[..., np.newaxis]
        X1_val = X1_val[..., np.newaxis]
        X2_val = X2_val[..., np.newaxis]
        X1_test = X1_test[..., np.newaxis]
        X2_test = X2_test[..., np.newaxis]
        
        print(f"정규화 완료: X1_train {X1_train.shape}, Y_train {Y_train.shape}")
        
        # 6. 모델 생성
        model = build_mt_icenet(
            seq_short, seq_long, H, W, 
            n_features=1,  # 단변량
            n_out=seq_output
        )
        
        print(f"\n모델 구조:")
        print(model.summary())
        
        now = datetime.now()
        best = [1e5, 1e5, -1e5]
        
        # 7. 학습
        print(f"\n학습 시작...")
        for epoch in range(Epoch):
            history = model.fit(
                x=[X1_train, X2_train], 
                y=Y_train,
                validation_data=([X1_val, X2_val], Y_val),
                epochs=1, 
                batch_size=BATCH_SIZE, 
                verbose=2
            )
            
            # 테스트 예측
            Y_pred = model.predict([X1_test, X2_test], verbose=0)
            
            # ✅ 역정규화 (공식 코드 방식)
            Y_pred_inv = scaler_y.inverse_transform(Y_pred.reshape(-1, 1)).reshape(Y_pred.shape)
            Y_test_inv = scaler_y.inverse_transform(Y_test.reshape(-1, 1)).reshape(Y_test.shape)
            
            # ✅ 육지 마스크 적용 + Clipping
            Y_pred_masked = np.clip(Y_pred_inv, 0, 100) * land_mask[..., None]
            Y_test_masked = np.clip(Y_test_inv, 0, 100) * land_mask[..., None]
            
            # Metric 평가
            first_batch_index = int(te_idx[0]) if len(te_idx) else 0
            metric = evaluate_for_metric_node(
                Y_pred_masked, Y_test_masked, lead_days, 
                first_batch_index, seq_long, model_name, now, land_mask
            )
            
            best = Metric.update(
                now, save, model, best, metric, 
                f"{model_name}_{seq_output}", seq_output, epoch
            )
            
            print(f"[Epoch {epoch:02d}] "
                  f"MSE: {metric[0]:.6f} | MAE: {metric[1]:.6f} | COR: {metric[2]:.4f}")
            print(f"[Best]     "
                  f"MSE: {best[0]:.6f} | MAE: {best[1]:.6f} | COR: {best[2]:.4f}\n")
        
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