# Spyder에서 CNN 2D 실행하기

Spyder IDE에서 CNN 2D Forecaster를 실행하는 방법

## 실행 방법

### 1. Spyder에서 파일 열기

```
File → Open → train.py 선택
```

경로: `C:\Users\USER\Desktop\baseline\icepixel\models\cnn_2d\train.py`

### 2. 실행

**F5** 키 누르기 또는 **Run** 버튼 클릭

### 3. 설정 변경 (선택)

`train.py` 파일의 `get_config()` 함수 수정:

```python
def get_config():
    config = SimpleNamespace(
        # 데이터 경로 (필요시 수정)
        root_path="C:/Users/USER/Desktop/ice/data/NSIDC_Data",
        
        # 시퀀스 길이
        seq_input=30,
        output_lens=[7, 14, 21],
        
        # 학습 설정
        batch_size=2,
        Epoch=30,
        lr=0.001,
    )
    return config
```

## 코드 수정 사항 (Spyder 최적화)

### 경로 처리

기존 코드:
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
```

Spyder용 수정:
```python
try:
    # __file__이 정의된 경우
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../..'))
except NameError:
    # Spyder에서 실행 시
    current_dir = os.getcwd()
    # icepixel 디렉토리 찾기
    while not os.path.exists(os.path.join(current_dir, 'data_provider')):
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            # 못 찾으면 직접 지정
            project_root = r"C:\Users\USER\Desktop\baseline\icepixel"
            break
        current_dir = parent
    else:
        project_root = current_dir

sys.path.insert(0, project_root)
```

이제 Spyder에서도 정상 작동합니다!

## 실행 디렉토리

Spyder에서 실행할 때 현재 작업 디렉토리는 자동으로 처리됩니다.

**확인 방법:**
```python
import os
print(f"Current dir: {os.getcwd()}")
print(f"Project root: {project_root}")
```

## 결과 확인

학습이 완료되면 결과가 다음 경로에 저장됩니다:

```
C:\Users\USER\Desktop\baseline\icepixel\results\CNN_2D\
└── seq_7_MMDD_HHMM\
    ├── best_model.pt
    ├── best_metrics.csv
    └── plots\
```

## Spyder 설정 권장 사항

### 1. IPython Console 설정

**Tools → Preferences → IPython console → Graphics**
- Backend: `Automatic` 또는 `Inline`

### 2. Working Directory

**Tools → Preferences → Current working directory**
- 선택: "The directory of the file being executed"

### 3. 메모리 관리

긴 학습 시 메모리 정리:
```python
import gc
gc.collect()
torch.cuda.empty_cache()
```

## 디버깅

### Import 에러

```python
ModuleNotFoundError: No module named 'data_provider'
```

**해결:**
```python
import sys
print(sys.path)
# 프로젝트 루트가 포함되어 있는지 확인
```

### 경로 에러

```python
FileNotFoundError: [Errno 2] No such file or directory
```

**해결:**
```python
import os
print(f"Current dir: {os.getcwd()}")
print(f"Project root: {project_root}")
# 경로가 올바른지 확인
```

### CUDA Out of Memory

**해결 1:** Batch size 줄이기
```python
batch_size = 1
```

**해결 2:** 입력 크기 줄이기
```python
input_size = (224, 152)
```

## 단계별 실행 (디버깅용)

Spyder에서 단계별로 실행하려면:

```python
# 1. 설정만 확인
config = get_config()
print(config)

# 2. 데이터 로드만 테스트
from data_provider import data_provider
args = SimpleNamespace(
    root_path=config.root_path,
    seq_len=config.seq_input,
    pred_len=7,
    batch_size=2,
    num_workers=0,
    verbose=True,
)
train_dataset, train_loader = data_provider(args, split="train")
print(f"Samples: {len(train_dataset)}")

# 3. 배치 확인
batch = next(iter(train_loader))
print(batch["input"].shape)

# 4. 전체 학습
main()
```

## 빠른 테스트

전체 학습은 시간이 오래 걸리므로 빠른 테스트:

```python
config = get_config()
config.Epoch = 2  # 2 epoch만
config.output_lens = [7]  # 하나만
config.batch_size = 1  # 작게

main()
```

## 주의사항

1. **Spyder 재시작**: 모듈 수정 후에는 Spyder 커널 재시작
   - **Ctrl + .** (점)

2. **메모리**: GPU 메모리 부족 시 Spyder 재시작

3. **Plot 창**: `plot_interval=10` 설정으로 너무 많은 창이 안 뜸

4. **중단**: 학습 중단은 **Ctrl + C**

## 문제 해결

### 1. Spyder에서 실행이 안 됨

```python
# train.py 맨 위에 추가
import sys
sys.path.insert(0, r"C:\Users\USER\Desktop\baseline\icepixel")
```

### 2. 그래프가 안 보임

```python
# Spyder Preferences
# IPython console → Graphics → Backend → "Inline" 선택
```

### 3. 학습이 느림

```python
# 설정 변경
input_size = (224, 152)  # 원본의 절반
batch_size = 4  # 늘리기
```

## 추가 팁

### Variable Explorer 활용

Spyder의 Variable Explorer에서:
- `config`: 설정 확인
- `metrics`: 현재 성능 확인
- `best_metrics`: Best 성능 확인
- `outputs`, `actuals`: 예측/실제값 확인

### Console 활용

학습 중에 Console에서:
```python
# 현재 epoch 확인
print(f"Current epoch: {epoch}")

# 중간 결과 저장
torch.save(model.state_dict(), 'temp_checkpoint.pt')
```

## 성공적인 실행 예시

```
Project root: C:\Users\USER\Desktop\baseline\icepixel
================================================================================
CNN 2D Forecaster Training
================================================================================
Model: CNN_2D
Device: cuda
Input sequence: 30 days
Output sequences: [7, 14, 21]
...
[Epoch 1/30]
  Train Loss: 0.012345
  RMSE: 0.034567  |  Best: 0.034567
...
```

이렇게 출력되면 정상 작동 중입니다!

