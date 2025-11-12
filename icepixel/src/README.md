# 해빙 농도 예측 모델 평가 및 시각화 모듈

NSIDC 기반 해빙 농도 예측 결과를 평가하고 시각화하는 모듈

## 파일 구조

```
src/
├── __init__.py               # 모듈 export
├── metric_visualizer.py      # 평가지표 및 시각화 함수
├── test_metrics.py           # 테스트 스크립트
└── README.md                 # 이 문서
```

## 주요 기능

### 1. 평가지표 계산

#### 필수 지표
- **RMSE** (Root Mean Squared Error): 큰 오차에 민감
- **MAE** (Mean Absolute Error): 평균 절대 오차
- **R²** (Coefficient of Determination): 예측력 (-1~1)

#### 추가 지표
- **SIE** (Sea Ice Extent): 15% 이상 농도 영역 면적 (백만 km²)
- **SIE Error (%)**: 해빙 면적 오차 비율

### 2. 시각화

#### 공간 맵
- **Spatial Comparison**: 예측/실제/오차 3개 비교
- **Error Map**: 지역별 예측 오차 (-50% ~ +50%)
- **RMSE Map**: 시간 평균 RMSE 분포

#### 시계열
- **Time-series**: 전체 평균 해빙 농도 변화
- **SIE Trend**: 해빙 면적 변화 추이

## 사용 방법

### 기본 사용

```python
import torch
from src import calc_metrics, plot_spatial_comparison, plot_timeseries

# 데이터 준비
pred = torch.randn(2, 7, 1, 448, 304)  # (B, T, 1, H, W)
true = torch.randn(2, 7, 1, 448, 304)
mask = torch.ones(2, 448, 304).bool()  # (B, H, W)

# 평가지표 계산
metrics = calc_metrics(pred, true, mask, pixel_area_km2=625.0)
print(metrics)
# {'RMSE': 0.234, 'MAE': 0.189, 'R2': 0.856, 
#  'SIE_pred': 12.34, 'SIE_true': 12.56, 'SIE_error_pct': -1.75}
```

### 시각화

#### 1. 공간 맵 비교

```python
from src import plot_spatial_comparison

plot_spatial_comparison(
    pred[0, -1, 0],  # 첫 샘플, 마지막 timestep (H, W)
    true[0, -1, 0],
    date='2020-01-15',
    save_path='results/spatial_comparison.png'
)
```

#### 2. Error Map

```python
from src import plot_error_map

plot_error_map(
    pred[0, -1, 0],
    true[0, -1, 0],
    title="Day 7 Prediction Error",
    date='2020-01-15',
    save_path='results/error_map.png'
)
```

#### 3. RMSE Map (시간 평균)

```python
from src import plot_rmse_map

plot_rmse_map(
    pred[0, :, 0],  # (T, H, W)
    true[0, :, 0],
    save_path='results/rmse_map.png'
)
```

#### 4. Time-series

```python
from src import plot_timeseries

plot_timeseries(
    pred[0, :, 0],
    true[0, :, 0],
    model_name='CNN_2D',
    seq_output=7,
    mask=mask[0],
    dates=['2020-01-09', '2020-01-10', ..., '2020-01-15'],
    save_path='results/timeseries.png'
)
```

#### 5. SIE Trend

```python
from src import calc_sie, plot_sie_trend

# SIE 시계열 계산
sie_pred_list = [calc_sie(pred[0, t, 0]) for t in range(7)]
sie_true_list = [calc_sie(true[0, t, 0]) for t in range(7)]

plot_sie_trend(
    sie_pred_list,
    sie_true_list,
    time_axis=list(range(1, 8)),
    time_unit='days',
    save_path='results/sie_trend.png'
)
```

### 학습 루프에서 사용

```python
from src import calc_metrics, update_best_metrics, plot_spatial_comparison

# 초기화
best_metrics = {'RMSE': float('inf'), 'MAE': float('inf')}

for epoch in range(num_epochs):
    # 학습...
    
    # 평가
    model.eval()
    with torch.no_grad():
        pred = model(input_seq)
        
        # 평가지표 계산
        metrics = calc_metrics(pred, target, mask)
        
        # Best 업데이트
        best_metrics = update_best_metrics(
            metrics,
            best_metrics,
            save_dir=f'results/{model_name}',
            model=model,
            epoch=epoch
        )
        
        # 주기적 시각화 (10 epoch마다)
        if epoch % 10 == 0:
            plot_spatial_comparison(
                pred[0, -1, 0],
                target[0, -1, 0],
                save_path=f'results/{model_name}/epoch_{epoch:03d}_spatial.png'
            )
    
    print(f"Epoch {epoch}: RMSE={metrics['RMSE']:.4f}, Best={best_metrics['RMSE']:.4f}")
```

### Metric Summary Table

```python
from src import create_metric_table

# DataFrame 생성 및 저장
df = create_metric_table(
    metrics,
    save_path='results/metrics_summary.csv'
)

print(df)
#     RMSE       MAE        R2  SIE_pred  SIE_true  SIE_error_pct
# 0.234000  0.189000  0.856000    12.340    12.560         -1.750
```

## API 레퍼런스

### 평가지표 함수

#### `calc_metrics(pred, true, mask, pixel_area_km2)`

모든 평가지표 한번에 계산

**Parameters:**
- `pred` (torch.Tensor | np.ndarray): 예측값 (B, T, 1, H, W) 또는 (B, H, W)
- `true` (torch.Tensor | np.ndarray): 실제값
- `mask` (np.ndarray | None): 유효 영역 마스크 (B, H, W)
- `pixel_area_km2` (float): 픽셀당 면적 (기본 625 = 25km × 25km)

**Returns:**
- `dict`: {'RMSE', 'MAE', 'R2', 'SIE_pred', 'SIE_true', 'SIE_error_pct'}

---

#### `calc_rmse(pred, true, mask)`

RMSE 계산

**Returns:** `float`

---

#### `calc_mae(pred, true, mask)`

MAE 계산

**Returns:** `float`

---

#### `calc_r2(pred, true, mask)`

R² 계산

**Returns:** `float`

---

#### `calc_sie(sic_map, threshold, pixel_area_km2)`

SIE (해빙 면적) 계산

**Parameters:**
- `sic_map`: 해빙 농도 맵 (임의 shape)
- `threshold` (float): 해빙 존재 임계값 (기본 0.15)
- `pixel_area_km2` (float): 픽셀당 면적 (기본 625)

**Returns:** `float` (백만 km²)

---

### 시각화 함수

#### `plot_spatial_comparison(pred, true, date, save_path)`

3개 subplot (Prediction, Ground Truth, Error)

**Parameters:**
- `pred` (H, W): 예측 맵
- `true` (H, W): 실제 맵
- `date` (str | None): 날짜 (YYYY-MM-DD)
- `save_path` (str | None): 저장 경로

---

#### `plot_error_map(pred, true, title, date, save_path)`

Error Map (-50% ~ +50%)

---

#### `plot_rmse_map(pred_seq, true_seq, save_path)`

시간 평균 RMSE 맵

**Parameters:**
- `pred_seq` (T, H, W): 예측 시퀀스
- `true_seq` (T, H, W): 실제 시퀀스

---

#### `plot_timeseries(pred, true, model_name, seq_output, dates, mask, save_path)`

전체 평균 시계열 비교

---

#### `plot_sie_trend(sie_pred, sie_true, time_axis, time_unit, save_path)`

SIE 추이 비교

**Parameters:**
- `sie_pred` (list | np.ndarray): 예측 SIE 리스트
- `sie_true` (list | np.ndarray): 실제 SIE 리스트
- `time_axis` (list | None): 시간축
- `time_unit` (str): "days" 또는 "months"

---

### 유틸리티 함수

#### `create_metric_table(metrics_dict, save_path)`

Metric DataFrame 생성 및 CSV 저장

**Returns:** `pd.DataFrame`

---

#### `update_best_metrics(current_metrics, best_metrics, save_dir, model, epoch)`

Best metrics 업데이트 및 저장

**Returns:** `dict` (업데이트된 best_metrics)

---

## 테스트

```bash
cd C:\Users\USER\Desktop\baseline\icepixel\src
python test_metrics.py
```

테스트 결과는 `test_results/` 디렉토리에 저장됩니다.

## 주의사항

1. **마스크 처리**: 육지/극점/해안선은 반드시 마스크로 제외
2. **값 범위**: 해빙 농도는 0~1, 특수값은 음수
3. **픽셀 면적**: NSIDC 25km 해상도 기준 625 km²
4. **SIE 임계값**: 일반적으로 15% (0.15) 사용

## 예시 출력

### Metric 출력
```
{'RMSE': 0.0234, 'MAE': 0.0189, 'R2': 0.856, 
 'SIE_pred': 12.34, 'SIE_true': 12.56, 'SIE_error_pct': -1.75}
```

### 시각화 예시
- `spatial_comparison.png`: 3개 subplot (Pred, True, Error)
- `error_map.png`: 오차 분포 (-50% ~ +50%, RdBu_r colormap)
- `rmse_map.png`: RMSE 분포 (0 ~ 50%, YlOrRd colormap)
- `timeseries.png`: 선 그래프 (빨강=예측, 파랑=실제)
- `sie_trend.png`: SIE 추이 (빨강=모델, 파랑=NSIDC)

## 확장 가능성

- [ ] 지역별 평가 (카라해, 동시베리아해 등)
- [ ] Binary Accuracy (해빙 유/무)
- [ ] 해빙 경계선 비교
- [ ] 계절별 분석
- [ ] 월평균 집계

## 라이선스

Research Use Only

