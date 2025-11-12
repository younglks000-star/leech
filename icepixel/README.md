# NSIDC 해빙 농도 DataLoader

NSIDC 일별 GeoTIFF 해빙 농도 데이터 전용 PyTorch Dataset 및 DataLoader

## 파일 구조

```
icepixel/
├── data_provider/
│   ├── __init__.py           # 모듈 export
│   ├── seaice_dataset.py     # SeaIceDataset 클래스
│   └── data_factory.py       # data_provider 팩토리 함수
├── test_dataloader.py        # 테스트 스크립트
└── README.md                 # 이 문서
```

## 주요 기능

### 1. SeaIceDataset

NSIDC 일별 해빙 농도 데이터를 로드하는 PyTorch Dataset 클래스

**특징:**
- 자동 파일 수집 및 날짜순 정렬
- Train/Val/Test 연도별 자동 분할
- 특수값 처리 (육지, 해안선, 극점 구멍 등)
- 유효 영역 마스크 자동 생성
- 선택적 리사이즈 (최근접 보간)
- 메모리 캐싱 옵션

**전처리 규칙:**
- `0~1000` → `/1000.0` → `[0, 1]` (해빙 농도)
- `2510` → `-1` (극점 구멍)
- `2530` → `-2` (해안선)
- `2540` → `-3` (육지)
- `2550` → `-4` (결측)

**반환 형태:**
```python
{
    "input": torch.FloatTensor,   # (T_in, 1, H, W)
    "target": torch.FloatTensor,  # (T_out, 1, H, W)
    "mask": torch.BoolTensor,     # (H, W) - 유효 영역 (True)
    "dates_in": List[str],        # ['YYYY-MM-DD', ...]
    "dates_out": List[str],       # ['YYYY-MM-DD', ...]
}
```

### 2. data_provider

Dataset과 DataLoader를 함께 생성하는 팩토리 함수

**배치 후 형태:**
```python
batch = {
    "input": (B, T_in, 1, H, W),
    "target": (B, T_out, 1, H, W),
    "mask": (B, H, W),
    "dates_in": List[List[str]],
    "dates_out": List[List[str]],
}
```

## 사용 방법

### 기본 사용

```python
from types import SimpleNamespace
from data_provider import data_provider

# 설정
args = SimpleNamespace(
    root_path="C:/Users/USER/Desktop/ice/data/NSIDC_Data",
    seq_len=16,           # 입력 시퀀스 길이
    pred_len=7,           # 출력 시퀀스 길이
    batch_size=2,
    num_workers=4,
    stride=1,
    resize_hw=None,       # 원본 크기 사용
    cache_in_memory=False,
    verbose=True,
    train_years=(2013, 2020),
    val_years=(2021, 2021),
    test_years=(2022, 2022),
)

# Train 데이터
train_dataset, train_loader = data_provider(args, split="train")

# 학습 루프
for batch in train_loader:
    input_seq = batch["input"]      # (B, T_in, 1, H, W)
    target_seq = batch["target"]    # (B, T_out, 1, H, W)
    mask = batch["mask"]            # (B, H, W)
    
    # 모델 학습
    # ...
```

### 직접 Dataset 사용

```python
from data_provider import SeaIceDataset

dataset = SeaIceDataset(
    root="C:/Users/USER/Desktop/ice/data/NSIDC_Data",
    seq_input=16,
    seq_output=7,
    split="train",
    verbose=True,
)

# 단일 샘플 로드
sample = dataset[0]
print(sample["input"].shape)    # (16, 1, 448, 304)
print(sample["target"].shape)   # (7, 1, 448, 304)
print(sample["mask"].shape)     # (448, 304)
```

### 리사이즈 사용

```python
args = SimpleNamespace(
    # ... 기타 설정 ...
    resize_hw=(224, 152),  # 원본의 절반 크기
)

dataset, loader = data_provider(args, split="train")
# input shape: (B, T_in, 1, 224, 152)
```

### 메모리 캐싱

```python
args = SimpleNamespace(
    # ... 기타 설정 ...
    cache_in_memory=True,  # 파일을 메모리에 캐싱
)

# 첫 번째 epoch은 느리지만, 이후 빠름
```

## 테스트

```bash
cd C:\Users\USER\Desktop\baseline\icepixel
python test_dataloader.py
```

## 데이터 구조 요구사항

```
NSIDC_Data/
├── 2013/
│   ├── 01_Jan/
│   │   ├── N_20130101_concentration_v3.0.tif
│   │   ├── N_20130102_concentration_v3.0.tif
│   │   └── ...
│   ├── 02_Feb/
│   └── ...
├── 2014/
└── ...
```

파일명 패턴: `N_YYYYMMDD_concentration_v3.0.tif`

## 파라미터 설명

### SeaIceDataset

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `root` | str | 필수 | NSIDC_Data 루트 경로 |
| `seq_input` | int | 필수 | 입력 시퀀스 길이 |
| `seq_output` | int | 필수 | 출력 시퀀스 길이 |
| `split` | str | "train" | "train", "val", "test" 중 선택 |
| `train_years` | tuple | (2013, 2020) | 학습 데이터 연도 범위 |
| `val_years` | tuple | (2021, 2021) | 검증 데이터 연도 범위 |
| `test_years` | tuple | (2022, 2022) | 테스트 데이터 연도 범위 |
| `stride` | int | 1 | 샘플링 stride |
| `resize_hw` | tuple\|None | None | 리사이즈 크기 (H, W) |
| `cache_in_memory` | bool | False | 메모리 캐싱 여부 |
| `verbose` | bool | False | 로깅 출력 여부 |

### data_provider

args 객체에 포함되어야 할 속성:

**필수:**
- `root_path`: NSIDC_Data 경로
- `seq_len`: 입력 길이
- `pred_len`: 출력 길이
- `batch_size`: 배치 크기
- `num_workers`: DataLoader worker 수

**선택 (기본값 있음):**
- `stride`: 1
- `resize_hw`: None
- `cache_in_memory`: False
- `verbose`: False
- `train_years`: (2013, 2020)
- `val_years`: (2021, 2021)
- `test_years`: (2022, 2022)

## 주의사항

1. **메모리:** 원본 크기(448×304)는 메모리를 많이 사용합니다. 필요시 `resize_hw` 사용을 권장합니다.

2. **Windows num_workers:** Windows에서는 `num_workers=0` 권장 (multiprocessing 문제)

3. **특수값:** 육지/해안선 등은 음수로 인코딩되며, `mask`를 사용하여 loss 계산에서 제외해야 합니다.

4. **날짜 연속성:** 데이터에 결측일이 있으면 시퀀스가 불연속적일 수 있습니다.

## Loss 계산 예시

```python
import torch.nn.functional as F

# Forward
pred = model(batch["input"])  # (B, T_out, 1, H, W)
target = batch["target"]      # (B, T_out, 1, H, W)
mask = batch["mask"]          # (B, H, W)

# 마스크 적용하여 loss 계산
# 방법 1: 유효 영역만 선택
mask_expanded = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, H, W)
mask_expanded = mask_expanded.expand_as(pred)    # (B, T_out, 1, H, W)

pred_valid = pred[mask_expanded]
target_valid = target[mask_expanded]
loss = F.mse_loss(pred_valid, target_valid)

# 방법 2: 마스크를 가중치로 사용
loss = F.mse_loss(pred, target, reduction='none')  # (B, T_out, 1, H, W)
loss = loss * mask_expanded
loss = loss.sum() / mask_expanded.sum()
```

## 확장 가능성

다른 데이터셋(ERA5 등) 추가 시:
1. `BaseDataset` 추상 클래스 생성
2. `SeaIceDataset`, `ERA5Dataset` 등으로 상속
3. `data_factory.py`에서 데이터셋 타입별 분기

## 라이선스

Research Use Only

