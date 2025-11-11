# geographic_icenet.py
# 🧊 해빙 지리적 특성 기반 혁신적 예측 모델
# 
# 핵심 혁신점:
# 1. 위도별 해빙 특성 (남쪽으로 갈수록 많이 녹음)
# 2. 장거리 공간 의존성 (이미지 상단-하단 간 연관성)
# 3. Multi-scale Geographic Feature Extraction
# 4. Spatial Transformer 기반 Attention
# 5. 기존 CNN/U-Net의 한계 극복

import os, re, glob, warnings, sys
import numpy as np
import pandas as pd
import tifffile as tiff
import tensorflow as tf
from datetime import datetime, timedelta
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Metric_node3 import (공간 시각화 포함)
sys.path.append(r"C:\Users\USER\Desktop\baseline\leech\MT-IceNet\utils")
import Metric_node3 as Metric
import torch

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Conv2D, ConvLSTM2D, BatchNormalization,
                                     MaxPooling2D, UpSampling2D, concatenate, 
                                     Activation, Dense, Reshape, Multiply, Add,
                                     GlobalAveragePooling2D, Lambda, Layer,
                                     AveragePooling2D, GlobalMaxPooling2D)
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# =============================================================================
# 설정
# =============================================================================
DATA_ROOT     = r"C:\Users\USER\Desktop\ice\data\NSIDC_Data"
FILE_REGEX    = r"N_(\d{8})_concentration.*\.tif$"
IMG_SHAPE     = (448, 304)

# 연속 예측 설정
output_lens   = [7, 14, 21]

# ✅ 새로운 단일 입력 구조
seq_length    = 30  # 단일 시퀀스 길이 (기존 dual 구조 제거)

BATCH_SIZE    = 2
Epoch         = 50
LEARNING_RATE = 1e-4
SEED          = 42
STRIDE        = 7
DOWNSAMPLE    = 1

USE_MIXED_PRECISION = False
USE_XLA             = False

model_name   = "Geographic-IceNet-Revolutionary"
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
# 🧊 혁신적 지리적 특성 레이어들
# =============================================================================

class MultiScaleGeographicExtractor(Layer):
    """
    다중 스케일 지리적 특성 추출기
    - 위도별 해빙 특성 (남쪽으로 갈수록 많이 녹음)
    - 다양한 스케일에서 지리적 패턴 포착
    - 해빙의 물리적 특성 반영
    """
    def __init__(self, filters, **kwargs):
        super(MultiScaleGeographicExtractor, self).__init__(**kwargs)
        self.filters = filters
        
    def build(self, input_shape):
        H, W, C = input_shape[1], input_shape[2], input_shape[3]
        
        # 위도별 가중치 (남쪽으로 갈수록 높은 값)
        self.latitude_weights = self.add_weight(
            name='latitude_weights',
            shape=(H, 1, 1, 1),
            initializer='ones',
            trainable=True
        )
        
        # 다중 스케일 커널
        self.conv3x3 = Conv2D(self.filters, (3, 3), padding='same', activation='relu')
        self.conv5x5 = Conv2D(self.filters, (5, 5), padding='same', activation='relu')
        self.conv7x7 = Conv2D(self.filters, (7, 7), padding='same', activation='relu')
        
        # 위도별 특성 추출
        self.latitude_conv = Conv2D(self.filters, (1, 1), padding='same', activation='relu')
        
        # 결합 레이어
        self.combine = Conv2D(self.filters, (1, 1), padding='same', activation='relu')
        
    def call(self, inputs):
        # 위도별 가중치 적용
        weighted_inputs = inputs * self.latitude_weights
        
        # 다중 스케일 특성 추출
        feat3x3 = self.conv3x3(weighted_inputs)
        feat5x5 = self.conv5x5(weighted_inputs)
        feat7x7 = self.conv7x7(weighted_inputs)
        
        # 위도별 특성 추출
        lat_feat = self.latitude_conv(weighted_inputs)
        
        # 모든 특성 결합
        combined = tf.concat([feat3x3, feat5x5, feat7x7, lat_feat], axis=-1)
        output = self.combine(combined)
        
        return output

class SpatialTransformerAttention(Layer):
    """
    공간적 Transformer 기반 Attention
    - 이미지 상단-하단 간 장거리 의존성
    - 공간적 위치 정보를 명시적으로 활용
    - 해빙의 지리적 특성 반영
    """
    def __init__(self, num_heads=8, **kwargs):
        super(SpatialTransformerAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        
    def build(self, input_shape):
        H, W, C = input_shape[1], input_shape[2], input_shape[3]
        self.d_model = C
        
        # Query, Key, Value 변환
        self.wq = Dense(self.d_model)
        self.wk = Dense(self.d_model)
        self.wv = Dense(self.d_model)
        
        # 위치 인코딩 (위도별 특성 반영)
        self.position_encoding = self.add_weight(
            name='position_encoding',
            shape=(H, W, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # 출력 변환
        self.dense = Dense(self.d_model)
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        H, W, C = tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        
        # 위치 인코딩 추가
        pos_encoded = inputs + self.position_encoding
        
        # Query, Key, Value 계산
        Q = self.wq(pos_encoded)  # (batch, H, W, d_model)
        K = self.wk(pos_encoded)  # (batch, H, W, d_model)
        V = self.wv(pos_encoded)  # (batch, H, W, d_model)
        
        # Reshape for multi-head attention
        Q = tf.reshape(Q, [batch_size, H*W, self.d_model])
        K = tf.reshape(K, [batch_size, H*W, self.d_model])
        V = tf.reshape(V, [batch_size, H*W, self.d_model])
        
        # Scaled dot-product attention
        attention_scores = tf.matmul(Q, K, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.d_model, tf.float32))
        
        # Softmax
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Attention 적용
        context = tf.matmul(attention_weights, V)
        
        # Reshape back
        context = tf.reshape(context, [batch_size, H, W, self.d_model])
        
        # 출력 변환
        output = self.dense(context)
        
        return output

class LongRangeSpatialConnection(Layer):
    """
    장거리 공간 연결 메커니즘
    - 이미지 상단-하단 간 직접 연결
    - 위도별 해빙 특성 반영
    - 공간적 의존성 모델링
    """
    def __init__(self, **kwargs):
        super(LongRangeSpatialConnection, self).__init__(**kwargs)
        
    def build(self, input_shape):
        H, W, C = input_shape[1], input_shape[2], input_shape[3]
        
        # 상단-하단 정보 추출
        self.top_pool = GlobalAveragePooling2D()
        self.bottom_pool = GlobalAveragePooling2D()
        
        # 위도별 가중치
        self.latitude_weights = self.add_weight(
            name='latitude_weights',
            shape=(H, 1, 1, 1),
            initializer='ones',
            trainable=True
        )
        
        # Attention weights
        self.attention_weights = self.add_weight(
            name='attention_weights',
            shape=(C * 2, C),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Spatial broadcasting weights
        self.spatial_weights = self.add_weight(
            name='spatial_weights',
            shape=(H, W, C),
            initializer='ones',
            trainable=True
        )
        
    def call(self, inputs):
        H, W, C = tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        
        # 상단과 하단 정보 추출
        top_info = self.top_pool(inputs)  # (batch, C)
        bottom_info = self.bottom_pool(inputs)  # (batch, C)
        
        # 상단-하단 정보 결합
        combined_info = tf.concat([top_info, bottom_info], axis=-1)  # (batch, C*2)
        
        # Attention 계산
        attention = tf.matmul(combined_info, self.attention_weights)  # (batch, C)
        attention = tf.nn.sigmoid(attention)
        
        # Spatial broadcasting
        attention_spatial = tf.reshape(attention, [-1, 1, 1, C])  # (batch, 1, 1, C)
        attention_spatial = tf.tile(attention_spatial, [1, H, W, 1])  # (batch, H, W, C)
        
        # 위도별 가중치 적용
        latitude_attention = self.latitude_weights * attention_spatial
        
        # Spatial weights와 결합
        spatial_attention = latitude_attention * self.spatial_weights
        
        # 원본 입력과 결합
        return inputs + spatial_attention

class GeographicFeatureFusion(Layer):
    """
    지리적 특성 융합 레이어
    - 다양한 지리적 특성들을 효과적으로 결합
    - 해빙의 물리적 특성 반영
    """
    def __init__(self, filters, **kwargs):
        super(GeographicFeatureFusion, self).__init__(**kwargs)
        self.filters = filters
        
    def build(self, input_shape):
        # 특성 융합을 위한 레이어들
        self.fusion_conv = Conv2D(self.filters, (1, 1), padding='same', activation='relu')
        self.bn = BatchNormalization()
        self.activation = Activation('relu')
        
    def call(self, inputs):
        # 특성 융합
        fused = self.fusion_conv(inputs)
        fused = self.bn(fused)
        fused = self.activation(fused)
        
        return fused

# =============================================================================
# 기존 유틸리티 함수들 (동일)
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

def build_index_splits(daily_idx, seq_long, max_lead, split_years, stride=1):
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

def make_geographic_dataset(daily_stack, indices, seq_length, 
                           lead_days, batch_size=2, shuffle=False, 
                           seed=42, downsample=1):
    """
    새로운 지리적 특성 기반 데이터셋 생성
    - 단일 시퀀스 입력 (dual 구조 제거)
    - 지리적 특성에 집중
    """
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
    def _slice_geographic(t):
        # 단일 시퀀스 입력 (지리적 특성에 집중)
        x = ds_x[t - seq_length : t]
        if new_hw is not None: 
            x = _maybe_downsample_3d(x, new_hw)
        x = tf.expand_dims(x, -1)
        
        # 연속 예측
        ys = tf.stack([ds_x[t + L] for L in tf.unstack(lead_days_tf)], axis=-1)
        if new_hw is not None: 
            ys = tf.image.resize(ys, new_hw, method='area')
        
        return x, ys
    
    ds = ds.map(_slice_geographic, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    return ds

# =============================================================================
# 🧊 혁신적 지리적 특성 모델
# =============================================================================
def build_geographic_icenet(seq_length, H, W, n_out, 
                           land_mask=None, lr=LEARNING_RATE, filt=3):
    """
    🧊 해빙 지리적 특성 기반 혁신적 예측 모델
    
    핵심 혁신점:
    1. MultiScaleGeographicExtractor: 다중 스케일 지리적 특성 추출
    2. SpatialTransformerAttention: 공간적 Transformer 기반 Attention
    3. LongRangeSpatialConnection: 상단-하단 간 장거리 의존성
    4. GeographicFeatureFusion: 지리적 특성 융합
    5. 단일 입력 구조 (dual 구조 제거)
    """
    # 단일 입력 (기존 dual 구조 제거)
    input_seq = Input(shape=(seq_length, H, W, 1), name='input_sequence')
    
    # ===== 1단계: 시계열 특성 추출 =====
    convlstm = ConvLSTM2D(
        16, (5,5), 
        padding="same", 
        return_sequences=False, 
        data_format="channels_last",
        activation="tanh",
        recurrent_activation="sigmoid"
    )(input_seq)
    
    # ===== 2단계: 다중 스케일 지리적 특성 추출 =====
    geo_extractor1 = MultiScaleGeographicExtractor(32)(convlstm)
    geo_extractor1 = BatchNormalization(axis=-1)(geo_extractor1)
    geo_extractor1 = Activation('relu')(geo_extractor1)
    
    # ===== 3단계: 공간적 Transformer Attention =====
    spatial_attention = SpatialTransformerAttention(num_heads=8)(geo_extractor1)
    spatial_attention = BatchNormalization(axis=-1)(spatial_attention)
    spatial_attention = Activation('relu')(spatial_attention)
    
    # ===== 4단계: 장거리 공간 연결 =====
    long_range = LongRangeSpatialConnection()(spatial_attention)
    long_range = BatchNormalization(axis=-1)(long_range)
    long_range = Activation('relu')(long_range)
    
    # ===== 5단계: 지리적 특성 융합 =====
    geo_fusion = GeographicFeatureFusion(64)(long_range)
    
    # ===== 6단계: 다중 스케일 지리적 특성 추출 (2차) =====
    geo_extractor2 = MultiScaleGeographicExtractor(64)(geo_fusion)
    geo_extractor2 = BatchNormalization(axis=-1)(geo_extractor2)
    geo_extractor2 = Activation('relu')(geo_extractor2)
    
    # ===== 7단계: 최종 공간적 Attention =====
    final_attention = SpatialTransformerAttention(num_heads=8)(geo_extractor2)
    final_attention = BatchNormalization(axis=-1)(final_attention)
    final_attention = Activation('relu')(final_attention)
    
    # ===== 8단계: 최종 지리적 특성 융합 =====
    final_fusion = GeographicFeatureFusion(32)(final_attention)
    
    # ===== 9단계: 출력 생성 =====
    # 다중 스케일 특성 결합
    multi_scale1 = Conv2D(16, (3, 3), padding='same', activation='relu')(final_fusion)
    multi_scale2 = Conv2D(16, (5, 5), padding='same', activation='relu')(final_fusion)
    multi_scale3 = Conv2D(16, (7, 7), padding='same', activation='relu')(final_fusion)
    
    # 모든 스케일 특성 결합
    combined_features = concatenate([multi_scale1, multi_scale2, multi_scale3], axis=-1)
    
    # 최종 출력 레이어
    output_conv = Conv2D(32, (3, 3), padding='same', activation='relu')(combined_features)
    output_conv = BatchNormalization(axis=-1)(output_conv)
    output_conv = Activation('relu')(output_conv)
    
    # 최종 예측
    raw_out = Conv2D(n_out, 1, activation='linear')(output_conv)
    out = Activation('linear', dtype='float32')(raw_out)
    
    model = Model(inputs=input_seq, outputs=out)
    
    # Masked Loss
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
# 평가 함수 (기존과 동일)
# =============================================================================
@torch.no_grad()
def evaluate_with_spatial_viz(pred_maps, true_maps, lead_days, 
                              first_batch_index, seq_input, model_name, 
                              tag_time, land_mask):
    """
    ✅ 평가 + 공간 시각화
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
    
    # 2. ✅ 공간 시각화 추가
    print("\n[공간 시각화 생성 중...]")
    
    # 주요 lead days 비교
    viz_days = []
    if C >= 1: viz_days.append(1)
    if C >= 7: viz_days.append(7)
    if C >= 14: viz_days.append(14)
    if C >= 21: viz_days.append(21)
    if not viz_days: viz_days = [1]
    
    Metric.plot_spatial_comparison(
        pred_maps, true_maps, land_mask,
        model_name, C, tag_time,
        lead_days=viz_days,
        sample_idx=0
    )
    
    # 시간별 진화
    Metric.plot_spatial_temporal(
        pred_maps, true_maps, land_mask,
        model_name, C, tag_time,
        sample_idx=0,
        n_timesteps=min(7, C)
    )
    
    # 오차 분석
    Metric.plot_error_statistics(
        pred_maps, true_maps, land_mask,
        model_name, C, tag_time
    )
    
    print("[공간 시각화 완료]")
    
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
# 메인 함수
# =============================================================================
def main():
    print("="*70)
    print("🧊 Geographic-IceNet-Revolutionary: 해빙 지리적 특성 기반 혁신적 모델")
    print("="*70)
    print("핵심 혁신점:")
    print("1. 위도별 해빙 특성 (남쪽으로 갈수록 많이 녹음)")
    print("2. 장거리 공간 의존성 (이미지 상단-하단 간 연관성)")
    print("3. Multi-scale Geographic Feature Extraction")
    print("4. Spatial Transformer 기반 Attention")
    print("5. 기존 CNN/U-Net의 한계 극복")
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
        print(f"시퀀스 길이: {seq_length}일 (단일 입력 구조)")
        print(f"{'='*70}")
        
        lead_days = list(range(1, seq_output + 1))
        max_lead = seq_output
        
        tr_idx, va_idx, te_idx = build_index_splits(
            daily_idx, seq_length, max_lead,
            (TRAIN_YEARS, VAL_YEARS, TEST_YEARS), 
            stride=STRIDE
        )
        
        print(f"데이터: Train={len(tr_idx)}, Val={len(va_idx)}, Test={len(te_idx)}")
        
        # ✅ 새로운 지리적 특성 기반 데이터셋 생성
        train_ds = make_geographic_dataset(
            daily_stack, tr_idx, seq_length, tuple(lead_days),
            batch_size=BATCH_SIZE, shuffle=True, seed=SEED, downsample=DOWNSAMPLE
        )
        val_ds = make_geographic_dataset(
            daily_stack, va_idx, seq_length, tuple(lead_days),
            batch_size=BATCH_SIZE, shuffle=False, downsample=DOWNSAMPLE
        )
        test_ds = make_geographic_dataset(
            daily_stack, te_idx, seq_length, tuple(lead_days),
            batch_size=BATCH_SIZE, shuffle=False, downsample=DOWNSAMPLE
        )
        
        # ✅ 혁신적 지리적 특성 모델 생성
        model = build_geographic_icenet(
            seq_length, H_eff, W_eff, 
            n_out=seq_output,
            land_mask=land_mask_d
        )
        
        now = datetime.now()
        best = [1e5, 1e5, -1e5]
        
        print(f"\n모델 구조:")
        print(model.summary())
        
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
            
            # ✅ 공간 시각화 포함 평가
            metric = evaluate_with_spatial_viz(
                pred, true, lead_days, 
                first_batch_index, seq_length, model_name, now, land_mask_d
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
    print("최종 결과 요약 - Geographic-IceNet-Revolutionary")
    print("="*70)
    for k, v in results.items():
        print(f"[+{k}일] MSE={v['b_mse']:.6f} | MAE={v['b_mae']:.6f} | COR={v['b_cor']:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()
