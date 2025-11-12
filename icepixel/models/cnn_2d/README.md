# CNN 2D Forecaster

해빙 농도 예측을 위한 CNN 2D 베이스라인 모델

## 개요

간단하지만 효과적인 CNN 기반 시공간 예측 모델입니다. 
PyTorch 기본 모듈만 사용하여 구현되었으며, 외부 라이브러리 의존성이 없습니다.

**Architecture**: CNN Encoder + LSTM Temporal Processing + CNN Decoder

**Reference**: Inspired by sea_ice_transformers paper CNN baseline

## 필수 조건

### 기본 의존성

```bash
pip install torch numpy scikit-learn matplotlib seaborn pandas rasterio scikit-image
```

**Note**: `torchcnnbuilder`는 필요 없습니다. PyTorch 기본 모듈만 사용합니다.

## 파일 구조

```
cnn_2d/
├── __init__.py          # 모듈 export
├── model.py             # CNN2DForecaster 모델 정의
├── train.py             # 학습 스크립트
└── README.md            # 이 문서
```

## 모델 구조 (Pure CNN, NO LSTM)

```
Input: (B, T_in, 1, H, W)
    ↓
Spatial Encoder (각 timestep 독립적으로)
  - 5개 Conv2D layers
  - BatchNorm + ReLU  
  - MaxPool (공간 축소: H→H/16, W→W/16)
  → (B, T_in, C', H', W')
    ↓
Flatten Spatial
  → (B, T_in, C'×H'×W')
    ↓
Temporal Conv1D (시간 차원 처리)
  - Conv1D(features, 512, kernel=3)
  - Conv1D(512, 512, kernel=3)
  → (B, T_in, 512)
    ↓
Temporal Projection (T_in → T_out)
  - Linear per feature dimension
  → (B, T_out, 512)
    ↓
Spatial Projection
  - Linear(512, C'×H'×W')
  → (B, T_out, C', H', W')
    ↓
Spatial Decoder (각 timestep 독립적으로)
  - 4개 TransposeConv2D layers (upsampling)
  - BatchNorm + ReLU
  - Conv2D(→1) + Sigmoid
  → (B, T_out, 1, H, W)
```

**핵심**: LSTM 없이 Conv + Linear만 사용하는 순수 CNN 구조

## 하이퍼파라미터

### 기본 설정

- **Input size**: (448, 304) - 원본 NSIDC 해상도
- **Conv layers**: 5 (encoder/decoder 각각)
- **Hidden dim**: 64
- **Temporal conv**: Conv1D (NO LSTM)
- **Input sequence**: 30 days
- **Output sequences**: [7, 14, 21] days
- **Batch size**: 2 (메모리 제약)
- **Learning rate**: 0.001
- **Optimizer**: Adam
- **Loss**: MSE (masked)
- **Epochs**: 30

**Note**: 순수 CNN 구조 (LSTM 없음)

### 메모리 절약 옵션

원본 크기(448×304)가 메모리를 많이 사용하면:

```python
input_size = (224, 152)  # 원본의 절반
```

## 사용 방법

### 1. 모델만 사용

```python
from models.cnn_2d import create_cnn2d_model

# 모델 생성
model = create_cnn2d_model(
    input_size=(448, 304),
    seq_input=360,
    seq_output=180,
    n_layers=5,
    device='cuda'
)

# Forward
import torch
x = torch.randn(2, 360, 1, 448, 304).cuda()
output = model(x)
print(output.shape)  # (2, 180, 1, 448, 304)
```

### 2. 학습 스크립트 실행

```bash
cd C:\Users\USER\Desktop\baseline\icepixel
python -m models.cnn_2d.train
```

또는:

```bash
cd C:\Users\USER\Desktop\baseline\icepixel\models\cnn_2d
python train.py
```

### 3. 설정 수정

`train.py`의 `get_config()` 함수에서 설정 변경:

```python
config = SimpleNamespace(
    # 데이터 경로
    root_path="your/data/path",
    
    # 시퀀스 길이
    seq_input=360,
    output_lens=[180, 360, 720],
    
    # 모델 설정
    input_size=(448, 304),  # or (224, 152)
    n_layers=5,
    
    # 학습 설정
    batch_size=2,
    Epoch=30,
    lr=0.001,
)
```

## 학습 과정

학습 스크립트는 iTransformer 스타일을 따릅니다:

1. **데이터 로드**: data_provider 사용
2. **모델 생성**: CNN2DForecaster
3. **Epoch 루프**:
   - Train epoch
   - Evaluate (바로)
   - 평가지표 계산 (RMSE, MAE, R²)
   - Best 모델 업데이트
   - 주기적 시각화 (10 epoch마다)
4. **결과 저장**: metrics, plots, model

## 출력 결과

### 디렉토리 구조

```
results/
└── CNN_2D/
    └── seq_180_MMDD_HHMM/
        ├── best_model.pt
        ├── best_metrics.csv
        └── plots/
            ├── epoch_000_spatial.png
            ├── epoch_000_timeseries.png
            ├── epoch_010_spatial.png
            ├── epoch_010_timeseries.png
            └── ...
```

### 평가지표 (best_metrics.csv)

```csv
RMSE,MAE,R2,SIE_pred,SIE_true,SIE_error_pct,best_epoch
0.0234,0.0189,0.856,12.34,12.56,-1.75,15
```

### 시각화

1. **spatial_*.png**: 예측/실제/오차 3개 비교
2. **timeseries_*.png**: 전체 평균 시계열

## 예상 결과

### 메모리 사용량

- **원본 크기 (448×304)**:
  - GPU 메모리: ~8GB (batch_size=2)
  - 학습 시간: ~2-3분/epoch

- **절반 크기 (224×152)**:
  - GPU 메모리: ~2GB (batch_size=8)
  - 학습 시간: ~30초/epoch

### 성능 (대략적)

| Output Length | RMSE | MAE | R² |
|--------------|------|-----|----|
| 180 days | ~0.02-0.04 | ~0.01-0.03 | ~0.85-0.90 |
| 360 days | ~0.03-0.05 | ~0.02-0.04 | ~0.75-0.85 |
| 720 days | ~0.04-0.07 | ~0.03-0.05 | ~0.65-0.75 |

*실제 값은 데이터셋에 따라 달라질 수 있습니다.*

## 문제 해결

### 1. torchcnnbuilder 설치 실패

```bash
pip install --upgrade pip
pip install torchcnnbuilder
```

또는:

```bash
pip install torchcnnbuilder --no-cache-dir
```

### 2. CUDA Out of Memory

**해결 1**: Batch size 줄이기
```python
batch_size = 1
```

**해결 2**: 입력 크기 줄이기
```python
input_size = (224, 152)
```

**해결 3**: Mixed precision 사용
```python
# train.py에 추가
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(batch_x)
    loss = F.mse_loss(output_valid, target_valid)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. 학습이 너무 느림

- `num_workers=0` 유지 (Windows)
- `input_size` 줄이기
- `batch_size` 늘리기 (메모리 허용 시)

## 논문 작성 시 인용

```
Baseline Model:
We employed CNN-based forecaster [1] as a baseline model, which has been 
widely used for spatiotemporal prediction in Arctic sea ice forecasting.
The model consists of 5 convolutional layers for spatial feature extraction
and temporal transformation.

Configuration:
- Input: 360 days × 448 × 304 pixels (25km resolution)
- Output: 180/360/720 days
- Optimizer: Adam (lr=0.001)
- Loss: Masked MSE

[1] sea_ice_transformers paper reference
```

## 추가 정보

- 데이터로더: `data_provider/`
- 평가지표: `src/metric_visualizer.py`
- 원본 구현: torchcnnbuilder.ForecasterBase

## 라이선스

Research Use Only

