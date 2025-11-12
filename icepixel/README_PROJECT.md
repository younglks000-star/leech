# 해빙 농도 예측 프로젝트

NSIDC 해빙 농도 데이터 기반 시계열 예측 비교 실험 프로젝트

## 프로젝트 구조

```
icepixel/
├── data_provider/              # 데이터로더
│   ├── __init__.py
│   ├── seaice_dataset.py       # NSIDC 데이터셋
│   └── data_factory.py         # data_provider 팩토리
│
├── src/                        # 평가 및 시각화
│   ├── __init__.py
│   ├── metric_visualizer.py    # 평가지표 & 시각화
│   ├── test_metrics.py
│   └── README.md
│
├── models/                     # 예측 모델들
│   ├── __init__.py
│   ├── cnn_2d/                 # ✅ CNN 2D Forecaster
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── README.md
│   ├── cnn_3d/                 # 🚧 CNN 3D (TODO)
│   ├── timesformer/            # 🚧 TimeSformer (TODO)
│   └── swinlstm/               # 🚧 SwinLSTM (TODO)
│
├── results/                    # 실험 결과 (자동 생성)
│   └── [model_name]/
│       └── seq_[length]/
│           ├── best_model.pt
│           ├── best_metrics.csv
│           └── plots/
│
├── test_dataloader.py          # 데이터로더 테스트
├── requirements.txt            # 의존성
├── README.md                   # 메인 문서
└── README_PROJECT.md           # 이 문서
```

## 완료된 컴포넌트

### ✅ 1. 데이터로더 (data_provider/)

- **SeaIceDataset**: NSIDC TIFF 파일 로드
- **data_provider**: Dataset + DataLoader 생성
- **기능**:
  - 날짜별 파일 자동 수집
  - Train/Val/Test 연도별 분할
  - 전처리 (정규화, 특수값 처리)
  - 유효 영역 마스크 생성

**사용법**:
```python
from data_provider import data_provider
from types import SimpleNamespace

args = SimpleNamespace(
    root_path="C:/Users/USER/Desktop/ice/data/NSIDC_Data",
    seq_len=360,
    pred_len=180,
    batch_size=2,
    num_workers=0,
)

train_dataset, train_loader = data_provider(args, split="train")
```

### ✅ 2. 평가 및 시각화 (src/)

- **평가지표**: RMSE, MAE, R², SIE
- **시각화**:
  - 공간 맵 비교 (Pred/True/Error)
  - Error Map
  - RMSE Map
  - Time-series
  - SIE Trend

**사용법**:
```python
from src import calc_metrics, plot_spatial_comparison

metrics = calc_metrics(pred, true, mask)
plot_spatial_comparison(pred, true, save_path='result.png')
```

### ✅ 3. CNN 2D 모델 (models/cnn_2d/)

- **모델**: torchcnnbuilder.ForecasterBase 기반
- **학습**: iTransformer 스타일 (epoch마다 train→eval)
- **기능**:
  - 자동 평가지표 계산
  - Best 모델 저장
  - 주기적 시각화

**사용법**:
```bash
cd C:\Users\USER\Desktop\baseline\icepixel
python -m models.cnn_2d.train
```

## 설치

### 1. 기본 의존성

```bash
pip install -r requirements.txt
```

### 2. torchcnnbuilder (CNN 2D/3D용)

```bash
pip install torchcnnbuilder
```

## 빠른 시작

### 1. 데이터 확인

```bash
python test_dataloader.py
```

### 2. 평가 모듈 테스트

```bash
cd src
python test_metrics.py
```

### 3. CNN 2D 학습

```bash
python -m models.cnn_2d.train
```

## 실험 설정

### 데이터 설정

- **데이터 경로**: `C:/Users/USER/Desktop/ice/data/NSIDC_Data`
- **Train**: 2013-2020
- **Val**: 2021
- **Test**: 2022
- **해상도**: 448 × 304 (25km)

### 시퀀스 설정

- **입력 길이**: 360 days
- **출력 길이**: [180, 360, 720] days

### 학습 설정

- **Batch size**: 2 (메모리 제약)
- **Epochs**: 30
- **Learning rate**: 0.001
- **Optimizer**: Adam
- **Loss**: MSE (masked)

## 실험 결과

### 출력 구조

```
results/
└── CNN_2D/
    ├── seq_180_MMDD_HHMM/
    │   ├── best_model.pt
    │   ├── best_metrics.csv
    │   └── plots/
    ├── seq_360_MMDD_HHMM/
    └── seq_720_MMDD_HHMM/
```

### 평가지표

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **SIE**: Sea Ice Extent (백만 km²)
- **SIE Error**: SIE 오차 (%)

## 다음 단계

### 🚧 구현 예정

1. **CNN 3D Forecaster**
   - torchcnnbuilder 3D 버전
   - 시공간 동시 처리

2. **TimeSformer**
   - Vision Transformer 기반
   - 시간 attention

3. **SwinLSTM**
   - Swin Transformer + LSTM
   - 계층적 구조

### 확장 가능성

- [ ] 앙상블 모델
- [ ] Transfer learning
- [ ] Multi-task learning
- [ ] 불확실성 추정
- [ ] 실시간 예측

## 문제 해결

### CUDA Out of Memory

```python
# train.py에서 설정 변경
batch_size = 1
input_size = (224, 152)  # 원본의 절반
```

### Windows num_workers 문제

```python
num_workers = 0  # Windows는 항상 0
```

### 데이터 경로 오류

`train.py`의 `get_config()`에서 경로 수정:
```python
root_path = "your/data/path"
```

## 인용

### 데이터

```
NSIDC Sea Ice Index, Version 3
Fetterer, F., K. Knowles, W. N. Meier, M. Savoie, and A. K. Windnagel. 
2017, updated daily. Sea Ice Index, Version 3.
```

### 모델

```
CNN Forecaster:
- torchcnnbuilder library
- sea_ice_transformers paper
```

## 라이선스

Research Use Only

## 개발 노트

- **데이터로더**: 2024-11-06 완료
- **평가 모듈**: 2024-11-06 완료
- **CNN 2D**: 2024-11-06 완료
- **CNN 3D**: 진행 예정
- **TimeSformer**: 진행 예정
- **SwinLSTM**: 진행 예정

## 참고 자료

- [NSIDC Data](https://nsidc.org/data/g02135)
- [torchcnnbuilder](https://github.com/...)
- [sea_ice_transformers](C:\Users\USER\Desktop\sea_ice_transformers-main)

