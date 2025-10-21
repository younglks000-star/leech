# mt_icenet_optimized_hybrid.py
# 메모리 최적화된 계층적 Soft Clustering + Transformer
# RTX 4090 24GB에 최적화

import os, re, glob, warnings, sys
import numpy as np
import pandas as pd
import tifffile as tiff
import tensorflow as tf
from datetime import datetime, timedelta
import torch

sys.path.append(r"C:\Users\USER\Desktop\baseline\MT-IceNet\utils")
import Metric_node as Metric

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Conv2D, LayerNormalization, Add, 
                                     Concatenate, Dense, Dropout)
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# =========================
# 설정
# =========================
DATA_ROOT     = r"C:\Users\USER\Desktop\ice\data\NSIDC_Data"
FILE_REGEX    = r"N_(\d{8})_concentration.*\.tif$"
IMG_SHAPE     = (448, 304)

output_lens   = [7, 14, 21, 30]
seq_input     = 30
BATCH_SIZE    = 2  # 메모리 최적화
EPOCHS        = 50
LEARNING_RATE = 1e-4
SEED          = 42
STRIDE        = 7
DOWNSAMPLE    = 2

model_name = "OptimizedHybridClusterTrans"
save = True
base_date = datetime(2013, 1, 1)

np.random.seed(SEED)
tf.random.set_seed(SEED)

for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

TRAIN_YEARS = list(range(2013, 2020))
VAL_YEARS   = [2020]
TEST_YEARS  = [2021, 2022]

# =========================
# 데이터 로딩
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
    
    recs.sort(key=lambda x: x[0])
    dates, frames = [], []
    
    print("GeoTIFF 로딩 중...")
    for i, (d, p) in enumerate(recs):
        if i % 500 == 0:
            print(f"  {i}/{len(recs)} 완료...")
        a = read_one_tif(p)
        if a.shape != target_hw:
            raise ValueError(f"크기 불일치 {p}")
        frames.append(a)
        dates.append(d)
    
    X = np.stack(frames, axis=0)
    X = np.nan_to_num(X, nan=0.0) / 100.0
    idx = pd.DatetimeIndex(dates)
    
    print(f"로딩 완료: {X.shape}")
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
    
    ii = np.array(ii, dtype=np.int32)
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
    ds_x = tf.convert_to_tensor(daily_stack, dtype=tf.float32)
    lead_days = tf.constant(list(lead_days), dtype=tf.int32)
    
    H, W = ds_x.shape[1], ds_x.shape[2]
    new_hw = (H//downsample, W//downsample) if downsample > 1 else None
    
    ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(4096, len(indices)), 
                       seed=seed, reshuffle_each_iteration=True)
    
    @tf.function
    def _slice_one(t):
        x = ds_x[t - t_days : t]
        if new_hw is not None:
            x = _maybe_downsample_3d(x, new_hw)
        x = tf.expand_dims(x, -1)
        
        ys = tf.stack([ds_x[t + L] for L in tf.unstack(lead_days)], axis=-1)
        if new_hw is not None:
            ys = tf.image.resize(ys, new_hw, method='area')
        
        return x, ys
    
    ds = ds.map(_slice_one, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    return ds

# =========================
# 커스텀 레이어
# =========================
class SoftClusteringLayer(tf.keras.layers.Layer):
    """Soft Clustering with temperature control"""
    def __init__(self, K, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.temperature = temperature
    
    def build(self, input_shape):
        D = input_shape[-1]
        
        self.query = self.add_weight(
            shape=(D, self.K),
            initializer="glorot_uniform",
            trainable=True,
            name="cluster_query"
        )
        self.bias = self.add_weight(
            shape=(self.K,),
            initializer="zeros",
            trainable=True,
            name="cluster_bias"
        )
        
        super().build(input_shape)
    
    def call(self, feats):
        # feats: (B, T, N, D)
        logits = tf.tensordot(feats, self.query, axes=[-1, 0]) + self.bias
        assign = tf.nn.softmax(logits / self.temperature, axis=-1)
        return assign

# =========================
# 메인 모델 (메모리 최적화)
# =========================
def build_optimized_hybrid_model(t_days, H, W, in_ch, K1, K2, D, out_len):
    """
    메모리 최적화된 하이브리드 모델
    - 시간축 평균으로 메모리 절약
    - 채널 수 감소
    - Skip connections 유지
    """
    inp = Input(shape=(t_days, H, W, in_ch), name="input")
    
    # ========== Temporal Encoder ==========
    # 최근 N일 가중 평균 (메모리 절약)
    recent_days = 7
    recent_frames = inp[:, -recent_days:, :, :, :]
    
    # 가중 평균 (최근일수록 높은 가중치)
    weights = tf.nn.softmax(tf.range(recent_days, dtype=tf.float32))
    weights = tf.reshape(weights, (1, recent_days, 1, 1, 1))
    x = tf.reduce_sum(recent_frames * weights, axis=1)  # (B, H, W, 1)
    
    # ========== Spatial Encoder ==========
    x = Conv2D(16, 3, padding="same", activation="relu", name="enc_conv1")(x)
    skip1 = x  # Skip connection 1 (16 channels)
    
    x = Conv2D(32, 3, padding="same", activation="relu", name="enc_conv2")(x)
    skip2 = x  # Skip connection 2 (32 channels)
    
    x = Conv2D(D, 3, padding="same", activation="relu", name="enc_conv3")(x)
    
    # Flatten for clustering
    B = tf.shape(x)[0]
    N = H * W
    feats_flat = tf.reshape(x, (B, 1, N, D))  # (B, 1, N, D)
    
    # ========== Hierarchical Soft Clustering ==========
    # Level 1: Fine-grained (K1 clusters)
    assign1 = SoftClusteringLayer(K1, temperature=0.5, 
                                   name="cluster_level1")(feats_flat)
    # assign1: (B, 1, N, K1)
    
    assign1_T = tf.transpose(assign1, perm=(0, 1, 3, 2))  # (B, 1, K1, N)
    cluster1 = tf.matmul(assign1_T, feats_flat)  # (B, 1, K1, D)
    cluster1 = tf.squeeze(cluster1, axis=1)  # (B, K1, D)
    
    # Level 2: Coarse-grained (K2 clusters)
    cluster1_expanded = tf.expand_dims(cluster1, 1)  # (B, 1, K1, D)
    assign2 = SoftClusteringLayer(K2, temperature=0.5, 
                                   name="cluster_level2")(cluster1_expanded)
    # assign2: (B, 1, K1, K2)
    
    assign2_T = tf.transpose(assign2, perm=(0, 1, 3, 2))  # (B, 1, K2, K1)
    cluster2 = tf.matmul(assign2_T, cluster1_expanded)  # (B, 1, K2, D)
    cluster2 = tf.squeeze(cluster2, axis=1)  # (B, K2, D)
    
    # ========== Transformer Attention ==========
    flat2 = tf.reshape(cluster2, (B, K2 * D))
    flat2_expanded = tf.expand_dims(flat2, 1)  # (B, 1, K2*D)
    
    # Multi-head self-attention
    att = tf.keras.layers.MultiHeadAttention(
        num_heads=4, 
        key_dim=D,
        dropout=0.1,
        name="cluster_attention"
    )(flat2_expanded, flat2_expanded)
    
    att = tf.squeeze(att, axis=1)  # (B, K2*D)
    res = Add()([flat2, att])
    res = LayerNormalization()(res)
    
    # Feed-forward network
    ff = Dense(K2 * D * 2, activation="relu")(res)
    ff = Dropout(0.1)(ff)
    ff = Dense(K2 * D)(ff)
    
    res = Add()([res, ff])
    res = LayerNormalization()(res)
    
    cluster2_out = tf.reshape(res, (B, K2, D))
    
    # ========== Decoder: Reconstruction ==========
    # Level 2 → Level 1
    assign2_squeeze = tf.squeeze(assign2, axis=1)  # (B, K1, K2)
    recon1 = tf.matmul(assign2_squeeze, cluster2_out)  # (B, K1, D)
    
    # Level 1 → Pixels
    assign1_squeeze = tf.squeeze(assign1, axis=1)  # (B, N, K1)
    recon0 = tf.matmul(assign1_squeeze, recon1)  # (B, N, D)
    
    recon0_resh = tf.reshape(recon0, (B, H, W, D))
    
    # ========== Skip Connections + Refinement ==========
    # Concatenate with skip2
    feat = Concatenate(name="concat_skip2")([recon0_resh, skip2])
    
    # Refinement block 1
    y = Conv2D(64, 3, padding="same", activation="relu", name="refine1")(feat)
    y = Conv2D(64, 3, padding="same", activation="relu", name="refine2")(y)
    
    # Concatenate with skip1
    y = Concatenate(name="concat_skip1")([y, skip1])
    
    # Refinement block 2
    y = Conv2D(32, 3, padding="same", activation="relu", name="refine3")(y)
    y = Conv2D(32, 3, padding="same", activation="relu", name="refine4")(y)
    
    # ========== Output Head ==========
    out = Conv2D(out_len, 1, padding="same", activation="linear", 
                 dtype='float32', name="output")(y)
    
    model = Model(inputs=inp, outputs=out, name=model_name)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )
    
    return model

# =========================
# 평가
# =========================
@torch.no_grad()
def evaluate_for_metric_node(pred_maps, true_maps, lead_days, 
                             first_batch_index, seq_input, model_name, tag_time):
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

# =========================
# 메인
# =========================
def main():
    print("="*70)
    print("메모리 최적화 Hierarchical Cluster-Transformer")
    print("="*70)
    
    # 데이터 로드
    daily_idx, daily_stack = load_daily_stack(DATA_ROOT, IMG_SHAPE)
    H, W = daily_stack.shape[1], daily_stack.shape[2]
    land_mask = make_land_mask(daily_stack)
    
    if DOWNSAMPLE > 1:
        lm = tf.convert_to_tensor(land_mask[..., None], tf.float32)
        lm = tf.image.resize(lm, (H//DOWNSAMPLE, W//DOWNSAMPLE), method="nearest")
        land_mask_d = tf.squeeze(lm, -1).numpy().astype(np.float32)
        H_eff, W_eff = land_mask_d.shape
    else:
        land_mask_d = land_mask
        H_eff, W_eff = H, W
    
    print(f"유효 이미지 크기: {H_eff} × {W_eff}")
    
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
        
        print(f"Train: {len(tr_idx)}, Val: {len(va_idx)}, Test: {len(te_idx)}")
        
        train_ds = make_dataset(daily_stack, tr_idx, seq_input, lead_days,
                               batch_size=BATCH_SIZE, shuffle=True, 
                               seed=SEED, downsample=DOWNSAMPLE)
        val_ds = make_dataset(daily_stack, va_idx, seq_input, lead_days,
                             batch_size=BATCH_SIZE, shuffle=False, 
                             downsample=DOWNSAMPLE)
        test_ds = make_dataset(daily_stack, te_idx, seq_input, lead_days,
                              batch_size=BATCH_SIZE, shuffle=False, 
                              downsample=DOWNSAMPLE)
        
        # 모델 생성
        print(f"\n모델 생성 중...")
        model = build_optimized_hybrid_model(
            seq_input, H_eff, W_eff, in_ch=1,
            K1=64, K2=16, D=32, out_len=seq_output
        )
        
        print(model.summary())
        
        best = [1e9, 1e9, -1e9]
        now = datetime.now()
        
        print(f"\n학습 시작...")
        for epoch in range(EPOCHS):
            history = model.fit(train_ds, validation_data=val_ds, 
                              epochs=1, verbose=2)
            
            # 테스트 예측
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
            metric = evaluate_for_metric_node(
                pred, true, lead_days, 
                first_batch_index, seq_input, model_name, now
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
            torch.cuda.empty_cache()
        
        results[seq_output] = {
            'mse': best[0],
            'mae': best[1],
            'cor': best[2]
        }
    
    print("\n" + "="*70)
    print("최종 결과 요약")
    print("="*70)
    for k, v in results.items():
        print(f"[+{k}일] MSE={v['mse']:.6f} | MAE={v['mae']:.6f} | COR={v['cor']:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()