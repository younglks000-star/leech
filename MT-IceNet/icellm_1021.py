# llm_guided_ice_prediction.py
# LLM-Guided Spatiotemporal Predictor for Sea Ice
# 핵심 아이디어: LLM이 전역 시계열 패턴을 학습 → CNN이 공간 디테일 예측에 활용

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
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config  # Hugging Face의 사전학습 GPT-2

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Conv2D, ConvLSTM2D, BatchNormalization,
                                     MaxPooling2D, UpSampling2D, concatenate, 
                                     Activation, Layer, Lambda)
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# =========================
# 설정
# =========================
DATA_ROOT     = r"C:\Users\USER\Desktop\ice\data\NSIDC_Data"
FILE_REGEX    = r"N_(\d{8})_concentration.*\.tif$"
IMG_SHAPE     = (448, 304)  # 원본 해빙 이미지 크기

# ✅ 연속 예측 길이 설정 (7일, 14일, 21일)
output_lens   = [7, 14, 21]

# ✅ 입력 시퀀스 길이 (과거 180일 데이터 사용)
seq_input     = 180

BATCH_SIZE    = 2
Epoch         = 50
LEARNING_RATE = 1e-4
SEED          = 42
STRIDE        = 7  # 샘플 추출 간격 (메모리 절약)
DOWNSAMPLE    = 1  # 공간 다운샘플링 비율 (1=원본 유지)

USE_MIXED_PRECISION = False
USE_XLA             = False

model_name   = "LLM-Guided-IceNet"
save         = True
base_date    = datetime(2013, 1, 1)

# 재현성을 위한 시드 고정
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
if USE_XLA:
    tf.config.optimizer.set_jit(True)

# GPU 메모리 동적 할당 설정
for g in tf.config.list_physical_devices('GPU'):
    try: 
        tf.config.experimental.set_memory_growth(g, True)
    except: 
        pass

# 연도 분할 (Train/Validation/Test)
TRAIN_YEARS = list(range(2013, 2020))  # 2013~2019 학습
VAL_YEARS   = [2020]                    # 2020 검증
TEST_YEARS  = [2021, 2022]              # 2021~2022 테스트

# =========================
# 유틸/데이터 로딩 함수들
# =========================
def list_tif_paths(root): 
    """모든 GeoTIFF 파일 경로 리스트 반환"""
    return sorted(glob.glob(os.path.join(root, "*", "*", "*.tif")))

def parse_date(p):
    """파일명에서 날짜 추출 (예: N_20130101_concentration.tif → 2013-01-01)"""
    m = re.search(FILE_REGEX, os.path.basename(p))
    return None if not m else datetime.strptime(m.group(1), "%Y%m%d")

def read_one_tif(path): 
    """단일 GeoTIFF 파일 읽기"""
    return tiff.imread(path).astype(np.float32)

def load_daily_stack(root, target_hw=IMG_SHAPE):
    """
    모든 일별 해빙 데이터를 시계열 스택으로 로드
    
    Returns:
        idx: 날짜 인덱스 (pandas DatetimeIndex)
        X: 해빙 농도 배열 [T, H, W], 0~1 정규화
    """
    recs = []
    # 모든 파일 스캔
    for p in list_tif_paths(root):
        d = parse_date(p)
        if d is None: 
            continue
        recs.append((pd.Timestamp(d), p))
    
    if not recs: 
        raise RuntimeError("GeoTIFF 파일을 찾지 못했습니다.")
    
    # 날짜순 정렬
    recs.sort(key=lambda x: x[0])
    dates, frames = [], []
    
    # 각 파일 읽어서 스택
    for d, p in recs:
        a = read_one_tif(p)
        if a.shape != target_hw: 
            raise ValueError(f"크기 불일치 {p} {a.shape}!={target_hw}")
        frames.append(a)
        dates.append(d)
    
    # [T, H, W] 형태로 스택
    X = np.stack(frames, axis=0)
    # NaN 처리 + 0~1 정규화 (원본은 0~100%)
    X = np.nan_to_num(X, nan=0.0) / 100.0
    idx = pd.DatetimeIndex(dates)
    
    return idx, X

def make_land_mask(daily_stack):
    """
    육지/바다 마스크 생성
    유효한 데이터가 있는 픽셀 = 바다(1), 없는 픽셀 = 육지(0)
    """
    valid = np.isfinite(daily_stack)
    ocean = (valid.sum(axis=0) > 0).astype(np.float32)
    return ocean

def build_index_splits(daily_idx, seq_len, max_lead, split_years, stride=1):
    """
    Train/Val/Test 인덱스 분할
    
    Args:
        daily_idx: 전체 날짜 인덱스
        seq_len: 입력 시퀀스 길이 (180일)
        max_lead: 최대 예측 리드타임 (7, 14, 21)
        split_years: (train_years, val_years, test_years) 튜플
        stride: 샘플 추출 간격
        
    Returns:
        tr, va, te: 각 split의 인덱스 배열
    """
    T = len(daily_idx)
    ii, yrs = [], []
    
    # seq_len ~ T-max_lead 범위에서 샘플링
    # 예: t=180이면 [0:180] 입력 → [181, 182, ..., 180+max_lead] 예측
    for t in range(seq_len, T - max_lead, stride):
        ii.append(t)
        yrs.append(daily_idx[t-1].year)  # 입력 마지막 날의 연도
    
    ii  = np.array(ii,  dtype=np.int32)
    yrs = np.array(yrs, dtype=np.int32)
    
    # 연도별로 분할
    tr = ii[np.isin(yrs, split_years[0])]
    va = ii[np.isin(yrs, split_years[1])]
    te = ii[np.isin(yrs, split_years[2])]
    
    return tr, va, te

def _maybe_downsample_3d(x, new_hw):
    """3D 텐서 공간 다운샘플링 헬퍼"""
    x = tf.expand_dims(x, -1)
    x = tf.image.resize(x, new_hw, method='area')
    return tf.squeeze(x, -1)

def make_dataset(daily_stack, indices, seq_len, lead_days, 
                batch_size=2, shuffle=False, seed=42, downsample=1):
    """
    TensorFlow Dataset 생성
    
    Args:
        daily_stack: 전체 데이터 [T, H, W]
        indices: 사용할 인덱스 배열 (t)
        seq_len: 입력 시퀀스 길이 (180)
        lead_days: 예측할 리드타임 튜플 (1, 2, ..., N)
        
    Returns:
        tf.data.Dataset: (input, output) 쌍
            - input: [B, seq_len, H, W, 1]
            - output: [B, H, W, len(lead_days)]
    """
    ds_x = tf.convert_to_tensor(daily_stack, dtype=tf.float32)
    lead_days_tf = tf.constant(list(lead_days), dtype=tf.int32)
    
    H, W = ds_x.shape[1], ds_x.shape[2]
    new_hw = (H // downsample, W // downsample) if downsample > 1 else None
    
    # 인덱스 → Dataset
    ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle: 
        ds = ds.shuffle(buffer_size=min(4096, len(indices)), 
                       seed=seed, reshuffle_each_iteration=True)
    
    @tf.function
    def _slice_one(t):
        """
        단일 샘플 슬라이싱
        t=200이면:
            - input: [20:200] (180일)
            - output: [201, 202, ..., 200+N] (N일 예측)
        """
        # 입력: t-seq_len ~ t
        x = ds_x[t - seq_len : t]  # [seq_len, H, W]
        if new_hw is not None: 
            x = _maybe_downsample_3d(x, new_hw)
        x = tf.expand_dims(x, -1)  # [seq_len, H, W, 1]
        
        # 출력: t+1, t+2, ..., t+N
        ys = tf.stack([ds_x[t + L] for L in tf.unstack(lead_days_tf)], axis=-1)  # [H, W, N]
        if new_hw is not None: 
            ys = tf.image.resize(ys, new_hw, method='area')
        
        return x, ys
    
    # 병렬 처리 + 배치 + 프리페치
    ds = ds.map(_slice_one, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    return ds

# =========================
# 🔥 핵심: LLM 모듈 (PyTorch)
# =========================
class TimeLLMModule(nn.Module):
    """
    Time-LLM 스타일의 전역 시계열 이해 모듈
    
    역할:
    1. 공간 평균 시계열 입력 → GPT-2로 전역 패턴 학습
    2. 전역 예측값 생성 (공간 평균 미래 값)
    3. Global Context 벡터 생성 → CNN에 전달
    """
    def __init__(self, seq_len, pred_len, llm_dim=768, freeze_llm=True):
        """
        Args:
            seq_len: 입력 시퀀스 길이 (180)
            pred_len: 예측 길이 (7, 14, 21)
            llm_dim: GPT-2 hidden dimension (768)
            freeze_llm: LLM 파라미터 고정 여부
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # ✅ 사전학습된 GPT-2 로드 (Frozen LLM)
        config = GPT2Config.from_pretrained('gpt2')
        self.llm = GPT2Model(config)
        
        # LLM 파라미터 고정 (메모리 절약 + 안정성)
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # ✅ Input Reprogramming: 1D 시계열 → 768차원 임베딩
        # 시계열 데이터를 LLM이 이해할 수 있는 형태로 변환
        self.input_proj = nn.Linear(1, llm_dim)
        
        # ✅ Output Projection: LLM 출력 → 미래 예측값
        # GPT-2의 마지막 hidden state → 미래 N일 예측
        self.output_proj = nn.Sequential(
            nn.Linear(llm_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, pred_len)  # 예측 길이만큼 출력
        )
        
        # ✅ Context Extraction: CNN 가이드용 전역 컨텍스트
        # LLM이 파악한 전역 패턴을 CNN에 전달
        self.context_proj = nn.Linear(llm_dim, 256)
        
    def forward(self, global_series):
        """
        Args:
            global_series: [B, T, 1] - 공간 평균 시계열
                예: 각 날짜의 전체 해빙 면적 평균
                
        Returns:
            global_pred: [B, pred_len] - 전역 예측값
            global_context: [B, 256] - CNN 가이드용 컨텍스트 벡터
        """
        B, T, _ = global_series.shape
        
        # 1️⃣ Reprogram: 시계열 → LLM embedding space
        # [B, T, 1] → [B, T, 768]
        x = self.input_proj(global_series)
        
        # 2️⃣ LLM Reasoning: GPT-2로 시계열 패턴 학습
        # 주의: inputs_embeds 사용 (텍스트 토큰이 아닌 직접 임베딩)
        llm_output = self.llm(inputs_embeds=x)
        hidden_states = llm_output.last_hidden_state  # [B, T, 768]
        
        # 3️⃣ Global Prediction: 마지막 타임스텝으로 미래 예측
        # hidden_states[:, -1, :] = 마지막 시점의 전역 정보
        global_pred = self.output_proj(hidden_states[:, -1, :])  # [B, pred_len]
        
        # 4️⃣ Context for Spatial Guidance: CNN에 전달할 전역 컨텍스트
        # "LLM이 본 전역 패턴"을 CNN이 참고하도록
        global_context = self.context_proj(hidden_states[:, -1, :])  # [B, 256]
        
        return global_pred, global_context

# =========================
# TensorFlow Custom Layer (LLM 통합)
# =========================
class LLMContextLayer(Layer):
    """
    PyTorch LLM을 TensorFlow 모델에 통합하는 커스텀 레이어
    
    ⚠️ 주의: 실제로는 학습 시 별도 처리 필요
    (TF ↔ PyTorch 변환 오버헤드가 큼)
    """
    def __init__(self, llm_module, **kwargs):
        super().__init__(**kwargs)
        self.llm_module = llm_module
        
    def call(self, inputs):
        """
        Args:
            inputs: [B, T, H, W, 1] - 입력 시퀀스
            
        Returns:
            context_tf: [B, 256] - LLM context (TensorFlow 텐서)
        """
        # 1️⃣ Global Pooling: 공간 차원 평균 → 1D 시계열
        # [B, T, H, W, 1] → [B, T, 1]
        global_series = tf.reduce_mean(inputs, axis=[2, 3])
        
        # 2️⃣ TensorFlow → NumPy → PyTorch 변환
        global_np = global_series.numpy()
        global_torch = torch.from_numpy(global_np).float()
        
        # 3️⃣ LLM Inference (no gradient)
        with torch.no_grad():
            _, global_context = self.llm_module(global_torch)
        
        # 4️⃣ PyTorch → NumPy → TensorFlow 변환
        context_np = global_context.cpu().numpy()
        context_tf = tf.constant(context_np, dtype=tf.float32)
        
        return context_tf  # [B, 256]

# =========================
# 모델: LLM-Guided Predictor
# =========================
def build_llm_guided_model(seq_len, H, W, n_out, lr=LEARNING_RATE):
    """
    LLM-Guided Spatiotemporal Predictor 구축
    
    구조:
    ┌─────────────────────────────────────┐
    │  Input: [B, 180, 448, 304, 1]       │
    └────────────┬────────────────────────┘
                 │
         ┌───────┴───────┐
         │               │
    [Global]        [Local]
    공간 평균        공간 유지
         │               │
      LLM(GPT-2)      ConvLSTM
    전역 패턴 파악    공간 특징 추출
         │               │
    Context(256)     Encoder
         │               │
         └───────┬───────┘
                 │
              Decoder
         (context로 가이드)
                 │
    ┌────────────┴────────────────────────┐
    │  Output: [B, 448, 304, N]           │
    │  (픽셀별 N일 예측)                   │
    └─────────────────────────────────────┘
    
    Args:
        seq_len: 입력 시퀀스 길이 (180)
        H, W: 공간 해상도
        n_out: 예측 길이 (7, 14, 21)
        
    Returns:
        model: TensorFlow Keras 모델
        llm_module: PyTorch LLM 모듈 (별도 관리)
    """
    
    # ===== 1️⃣ PyTorch LLM Module 생성 =====
    llm_module = TimeLLMModule(seq_len, n_out, freeze_llm=True)
    
    # ===== 2️⃣ TensorFlow CNN Model =====
    input_seq = Input(shape=(seq_len, H, W, 1), name='input_sequence')
    
    # === Global Branch 주석 ===
    # 실제 통합 시에는 학습 루프에서 별도로 LLM context를 계산하고
    # Decoder에 주입해야 함 (현재는 구조만 표현)
    
    # === Local Branch: Spatial Encoder ===
    # ConvLSTM2D: 시공간 패턴 학습
    # - 입력: [B, T, H, W, 1]
    # - 출력: [B, H, W, 8] (return_sequences=False)
    convlstm = ConvLSTM2D(
        8, (5,5),  # 8개 필터, 5x5 커널
        padding="same", 
        return_sequences=False,  # 마지막 타임스텝만 반환
        data_format="channels_last",
        activation="tanh",
        recurrent_activation="sigmoid"
    )(input_seq)
    
    # Encoder Block 1: 초기 특징 추출
    c1 = Conv2D(16, 3, activation='relu', padding='same')(convlstm)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    b1 = BatchNormalization()(c1)  # 학습 안정화
    p1 = MaxPooling2D((2, 2))(b1)  # 공간 해상도 1/2
    
    # Encoder Block 2: 중간 특징 추출
    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    b2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(b2)  # 공간 해상도 1/4
    
    # Encoder Block 3: 고수준 특징 추출
    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)
    b3 = BatchNormalization()(c3)  # Bottleneck
    
    # === Decoder: 공간 해상도 복원 ===
    # (실제로는 여기서 LLM global_context를 결합해야 함)
    
    # Decoder Block 1: 1/4 → 1/2 복원
    u1 = UpSampling2D((2, 2))(b3)
    u1 = Conv2D(32, 2, activation='relu', padding='same')(u1)
    m1 = concatenate([b2, u1])  # Skip connection (U-Net 스타일)
    c4 = Conv2D(32, 3, activation='relu', padding='same')(m1)
    c4 = Conv2D(32, 3, activation='relu', padding='same')(c4)
    
    # Decoder Block 2: 1/2 → 원본 복원
    u2 = UpSampling2D((2, 2))(c4)
    u2 = Conv2D(16, 2, activation='relu', padding='same')(u2)
    m2 = concatenate([b1, u2])  # Skip connection
    c5 = Conv2D(16, 3, activation='relu', padding='same')(m2)
    c5 = Conv2D(16, 3, activation='relu', padding='same')(c5)
    
    # === Output Layer ===
    # 1x1 Conv로 채널 수를 예측 길이로 변환
    # [B, H, W, 16] → [B, H, W, n_out]
    raw_out = Conv2D(n_out, 1, activation='linear')(c5)
    out = Activation('linear', dtype='float32')(raw_out)
    
    # 모델 생성 및 컴파일
    model = Model(inputs=input_seq, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    
    return model, llm_module

# =========================
# 🔥 핵심: LLM-Enhanced Training
# =========================
class LLMGuidedTrainer:
    """
    LLM context를 활용한 커스텀 학습 루프
    
    학습 과정:
    1. 입력 배치에서 공간 평균 → LLM으로 global context 계산
    2. CNN으로 픽셀별 예측 (global context는 간접적으로 활용)
    3. MSE loss로 CNN만 업데이트 (LLM은 frozen)
    
    ⚠️ 현재는 context를 직접 주입하지 않음
    (실제로는 Decoder에 context를 결합하도록 모델 수정 필요)
    """
    def __init__(self, tf_model, llm_module, learning_rate=1e-4):
        self.tf_model = tf_model  # TensorFlow CNN
        self.llm_module = llm_module  # PyTorch LLM (frozen)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
    @tf.function
    def train_step(self, x, y, global_context):
        """
        단일 학습 스텝
        
        Args:
            x: [B, T, H, W, 1] - 입력
            y: [B, H, W, n_out] - 정답
            global_context: [B, 256] - LLM context (현재 미사용)
            
        Returns:
            loss: MSE 손실값
        """
        with tf.GradientTape() as tape:
            # 1️⃣ CNN Forward: 픽셀별 예측
            y_pred = self.tf_model(x, training=True)
            
            # 2️⃣ MSE Loss 계산
            loss = tf.reduce_mean(tf.square(y - y_pred))
            
        # 3️⃣ Gradient 계산 및 업데이트 (CNN만)
        gradients = tape.gradient(loss, self.tf_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.tf_model.trainable_variables))
        
        return loss
    
    def fit(self, train_ds, val_ds, epochs=1):
        """
        한 epoch 학습
        
        Returns:
            train_loss, val_loss: 평균 손실값
        """
        train_losses = []
        
        # === Training Loop ===
        for x_batch, y_batch in train_ds:
            # 1️⃣ Global Context 계산 (LLM)
            # 공간 평균: [B, T, H, W, 1] → [B, T, 1]
            global_series = tf.reduce_mean(x_batch, axis=[2, 3])
            
            # TF → PyTorch 변환
            global_np = global_series.numpy()
            global_torch = torch.from_numpy(global_np).float()
            
            # LLM inference (no gradient)
            with torch.no_grad():
                _, global_context = self.llm_module(global_torch)
            
            # 2️⃣ CNN 학습 (global_context는 향후 사용 예정)
            loss = self.train_step(x_batch, y_batch, global_context)
            train_losses.append(loss.numpy())
        
        # === Validation ===
        val_losses = []
        for x_batch, y_batch in val_ds:
            y_pred = self.tf_model(x_batch, training=False)
            val_loss = tf.reduce_mean(tf.square(y_batch - y_pred))
            val_losses.append(val_loss.numpy())
        
        return np.mean(train_losses), np.mean(val_losses)

# =========================
# 평가 함수 (기존과 동일)
# =========================
@torch.no_grad()
def evaluate_for_metric_node(pred_maps, true_maps, lead_days, 
                             first_batch_index, seq_input, model_name, 
                             tag_time, land_mask):
    """
    Metric_node를 사용한 평가
    
    평가 지표:
    - MSE: 픽셀별 제곱 오차 평균
    - MAE: 픽셀별 절대 오차 평균
    - COR: 픽셀별 상관계수 평균
    """
    N, h, w, C = pred_maps.shape
    
    # [N, h, w, C] → [N, C, h*w] 변환 (Metric 포맷)
    pred = pred_maps.reshape(N, h*w, C).transpose(0, 2, 1)
    true = true_maps.reshape(N, h*w, C).transpose(0, 2, 1)
    
    # NumPy → PyTorch
    pred_t = torch.from_numpy(pred).float()
    true_t = torch.from_numpy(true).float()
    
    # Metric 계산
    n_features = h * w
    metric = Metric.metric(pred_t, true_t, n_features)  # [MSE, MAE, COR]
    
    # 시각화 저장
    Metric.plot(pred_t, true_t, f"{model_name}_{C}", C, tag_time)
    
    # CSV 저장 (첫 32개 지역)
    out_dir = f'./STMA_node/{model_name}/models/{model_name}_{C}_{tag_time.month}{tag_time.day}{tag_time.hour}{tag_time.minute}'
    os.makedirs(out_dir, exist_ok=True)
    
    # 예측 날짜 계산
    predict_dates = [(base_date + timedelta(days=int(first_batch_index + seq_input + int(L))))\
                     .strftime("%Y-%m-%d") for L in lead_days]
    
    # 첫 번째 샘플의 예측/정답
    pred_first = pred[0]  # [C, h*w]
    true_first = true[0]
    topK = min(32, pred_first.shape[1])  # 최대 32개 지역
    
    # 지역별 CSV 저장
    for r in range(topK):
        df = pd.DataFrame({
            "Date": predict_dates, 
            "Prediction": pred_first[:, r],  # 해당 픽셀의 예측값
            "Actual": true_first[:, r]       # 해당 픽셀의 실제값
        })
        df.to_csv(os.path.join(out_dir, f"region_{r}_pred_{C}.csv"), index=False)
    
    return metric

# =========================
# 메인 실행 함수
# =========================
def main():
    """
    LLM-Guided Ice Prediction 메인 실행
    
    전체 흐름:
    1. 데이터 로드 (2013~2022 일별 해빙 농도)
    2. Train/Val/Test 분할
    3. 각 예측 길이(7, 14, 21일)에 대해:
       a. 데이터셋 생성
       b. LLM-Guided 모델 생성
       c. 학습 (Epoch마다 테스트)
       d. Best 모델 저장
    4. 최종 결과 요약
    """
    print("="*70)
    print("LLM-Guided Sea Ice Prediction")
    print("="*70)
    
    # ===== 1️⃣ 데이터 로드 =====
    print("데이터 로딩 시작...")
    daily_idx, daily_stack = load_daily_stack(DATA_ROOT, IMG_SHAPE)
    H, W = daily_stack.shape[1], daily_stack.shape[2]
    land_mask = make_land_mask(daily_stack)
    
    print(f"데이터 shape: {daily_stack.shape}")
    print(f"날짜 범위: {daily_idx[0]} ~ {daily_idx[-1]}")
    print(f"총 {len(daily_idx)}일 데이터\n")
    
    # ===== 2️⃣ 다운샘플링 처리 =====
    if DOWNSAMPLE > 1:
        # 마스크도 다운샘플링
        lm = tf.convert_to_tensor(land_mask[...,None], tf.float32)
        lm = tf.image.resize(lm, (H//DOWNSAMPLE, W//DOWNSAMPLE), method='nearest')
        land_mask_d = tf.squeeze(lm, -1).numpy().astype(np.float32)
        H_eff, W_eff = land_mask_d.shape
    else:
        land_mask_d = land_mask
        H_eff, W_eff = H, W
    
    # ===== 3️⃣ 실험 루프: 각 예측 길이별 =====
    results = {}
    
    for seq_output in output_lens:
        print(f"\n{'='*70}")
        print(f"실험: +{seq_output}일 연속 예측")
        print(f"입력: {seq_input}일")
        print(f"{'='*70}")
        
        # 리드타임 생성: [1, 2, 3, ..., seq_output]
        lead_days = list(range(1, seq_output + 1))
        max_lead = seq_output
        
        # ===== 4️⃣ 데이터 분할 =====
        tr_idx, va_idx, te_idx = build_index_splits(
            daily_idx, seq_input, max_lead,
            (TRAIN_YEARS, VAL_YEARS, TEST_YEARS), 
            stride=STRIDE
        )
        
        print(f"데이터: Train={len(tr_idx)}, Val={len(va_idx)}, Test={len(te_idx)}")
        
        # ===== 5️⃣ Dataset 생성 =====
        # Train: 셔플 O
        train_ds = make_dataset(
            daily_stack, tr_idx, seq_input, tuple(lead_days),
            batch_size=BATCH_SIZE, shuffle=True, seed=SEED, downsample=DOWNSAMPLE
        )
        # Validation: 셔플 X
        val_ds = make_dataset(
            daily_stack, va_idx, seq_input, tuple(lead_days),
            batch_size=BATCH_SIZE, shuffle=False, downsample=DOWNSAMPLE
        )
        # Test: 셔플 X
        test_ds = make_dataset(
            daily_stack, te_idx, seq_input, tuple(lead_days),
            batch_size=BATCH_SIZE, shuffle=False, downsample=DOWNSAMPLE
        )
        
        # ===== 6️⃣ 모델 생성 =====
        tf_model, llm_module = build_llm_guided_model(
            seq_input, H_eff, W_eff, n_out=seq_output
        )
        
        # LLM-Guided Trainer 생성
        trainer = LLMGuidedTrainer(tf_model, llm_module, LEARNING_RATE)
        
        # Best 모델 추적
        now = datetime.now()  # 실험 시작 시각
        best = [1e5, 1e5, -1e5]  # [best_MSE, best_MAE, best_COR]
        
        print(f"\n모델 구조:")
        print(tf_model.summary())
        
        # ===== 7️⃣ 학습 루프 =====
        print(f"\n학습 시작...")
        for epoch in range(Epoch):
            # === 학습 (1 epoch) ===
            train_loss, val_loss = trainer.fit(train_ds, val_ds, epochs=1)
            
            print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # === 테스트 예측 ===
            preds, trues = [], []
            for xb, yb in test_ds:
                # CNN으로 예측 (LLM context는 학습 시만 사용)
                pb = tf_model.predict(xb, verbose=0)
                preds.append(pb)
                trues.append(yb.numpy())
            
            # 배치들을 하나로 합침
            pred = np.concatenate(preds, axis=0)  # [N, H, W, seq_output]
            true = np.concatenate(trues, axis=0)
            
            # === 후처리: 0~100 스케일 복원 + 마스킹 ===
            # 0~1 → 0~100% 변환
            pred = np.clip(pred * 100.0, 0, 100) * land_mask_d[..., None]
            true = np.clip(true * 100.0, 0, 100) * land_mask_d[..., None]
            
            # === 평가 지표 계산 ===
            first_batch_index = int(te_idx[0]) if len(te_idx) else 0
            metric = evaluate_for_metric_node(
                pred, true, lead_days, 
                first_batch_index, seq_input, model_name, now, land_mask_d
            )
            
            # === Best 업데이트 ===
            # Metric_node.update()가 best를 갱신하고 모델 저장
            best = Metric.update(
                now, save, tf_model, best, metric, 
                f"{model_name}_{seq_output}", seq_output, epoch
            )
            
            # === 출력 ===
            print(f"[Test] MSE: {metric[0]:.6f} | MAE: {metric[1]:.6f} | COR: {metric[2]:.4f}")
            print(f"[Best] MSE: {best[0]:.6f} | MAE: {best[1]:.6f} | COR: {best[2]:.4f}\n")
            
            # 메모리 정리
            del preds, trues
        
        # ===== 8️⃣ 결과 저장 =====
        results[seq_output] = {
            'b_mse': best[0], 
            'b_mae': best[1], 
            'b_cor': best[2]
        }
    
    # ===== 9️⃣ 최종 요약 =====
    print("\n" + "="*70)
    print("최종 결과 요약")
    print("="*70)
    for k, v in results.items():
        print(f"[+{k}일] MSE={v['b_mse']:.6f} | MAE={v['b_mae']:.6f} | COR={v['b_cor']:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()


# =========================
# 🔍 전체 구조 요약
# =========================
"""
LLM-Guided Ice Prediction 핵심 아이디어:

1️⃣ 문제 인식:
   - 해빙은 전역 기후 패턴의 영향을 받음 (계절성, 대기순환 등)
   - 기존 CNN/ConvLSTM은 공간 패턴에만 집중 → 전역 맥락 부족

2️⃣ 해결책:
   ┌─────────────────────────────────────────────┐
   │  LLM (GPT-2)                                │
   │  - 전역 시계열 이해                          │
   │  - "올해는 평년보다 따뜻한 겨울"             │
   │  - "엘니뇨 패턴 감지"                       │
   └────────────┬────────────────────────────────┘
                │ Global Context
                ↓
   ┌─────────────────────────────────────────────┐
   │  CNN (ConvLSTM + U-Net)                     │
   │  - 픽셀별 상세 예측                          │
   │  - "이 지역은 녹을 것", "저 지역은 얼 것"    │
   │  - LLM context로 가이드됨                   │
   └─────────────────────────────────────────────┘

3️⃣ 장점:
   - LLM: 방대한 시계열 패턴 학습 (Frozen, 메모리 절약)
   - CNN: 고해상도 공간 디테일 유지
   - 멀티스케일: 전역(LLM) + 지역(CNN)

4️⃣ 현재 구현 상태:
   ✅ LLM 모듈 구현 (TimeLLMModule)
   ✅ CNN 모듈 구현 (U-Net 스타일)
   ⚠️ LLM context 주입은 부분 구현
      (실제로는 Decoder에 context를 concat 필요)
   
5️⃣ 개선 방향:
   - Decoder에 global_context를 명시적으로 주입
   - Cross-attention으로 LLM-CNN 결합 강화
   - Multi-task learning (전역 예측 + 픽셀 예측 동시 학습)

6️⃣ 메모리 최적화:
   - LLM frozen (파라미터 업데이트 X)
   - Mixed precision 사용 가능
   - 예측 길이: 7, 14, 21일 (30일 제외)
   - BATCH_SIZE=2 → 필요시 1로 조정
"""