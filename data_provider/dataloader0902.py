# dataloader_iTransformer_norm.py
import torch
import numpy as np
from torch.utils.data import Dataset

class TimeLLMDataset(Dataset):
    """
    iTransformer용 슬라이딩 윈도우 Dataset (+선택적 정규화)
    - file_path: npy 파일 경로 (shape: (T, C) or (C, T) -> 자동 정렬)
    - seq_input: 입력 길이
    - seq_output: 예측 길이
    - mode: 'train' | 'val' | 'test'
    - train_ratio/val_ratio: split 비율
    - normalize: True면 훈련 구간만으로 스케일러 fit 후 전체 transform
    - norm_type: 'standard' | 'minmax'
    - feature_range: minmax일 때 범위
    - reset_scaler: True면 이번 인스턴스에서 스케일러 새로 학습
    """
    scaler = None  # 클래스 레벨 캐시

    def __init__(
        self, file_path, seq_input, seq_output, mode='train',
        train_ratio=0.7, val_ratio=0.1,
        normalize=True, norm_type='standard', feature_range=(0.0, 1.0),
        reset_scaler=False,
    ):
        self.seq_input = int(seq_input)
        self.seq_output = int(seq_output)
        self.mode = mode
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)

        # ---- load & shape 정리 ----
        arr = np.load(file_path).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]             # (T,) -> (T, 1)
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T                    # (C, T) 였다면 (T, C)로
        self.raw = arr                     # 원본 (optional)
        T, C = arr.shape
        self.T, self.C = T, C

        # ---- 윈도우 시작 인덱스 ----
        max_start = T - (self.seq_input + self.seq_output)
        if max_start < 0:
            raise ValueError("데이터가 (seq_input + seq_output)보다 짧습니다.")
        all_starts = np.arange(max_start + 1, dtype=np.int64)

        # split
        n_total = len(all_starts)
        n_train = int(n_total * self.train_ratio)
        n_val   = int(n_total * self.val_ratio)
        if mode == 'train':
            self.indices = all_starts[:n_train]
        elif mode == 'val':
            self.indices = all_starts[n_train:n_train + n_val]
        elif mode == 'test':
            self.indices = all_starts[n_train + n_val:]
        else:
            raise ValueError("mode must be 'train' | 'val' | 'test'")

        # ---- 정규화 (선택) ----
        self.normalize = bool(normalize)
        self.norm_type = norm_type
        self.feature_range = feature_range

        if self.normalize:
            need_fit = (TimeLLMDataset.scaler is None) or reset_scaler
            if need_fit:
                # 훈련 split의 시작 인덱스 (현재 인스턴스가 train이 아니어도 훈련 영역으로 fit)
                train_starts = all_starts[:n_train]
                # 훈련에서 입력/미래 텍스트 구간 모두 모아서 분포 학습 (분포 shift 최소화)
                chunks = []
                for s in train_starts:
                    e = s + self.seq_input
                    f = e + self.seq_output
                    chunks.append(arr[s:e])   # 입력
                    chunks.append(arr[e:f])   # 미래
                X_train = np.concatenate(chunks, axis=0)  # (많은 시점, C)

                if self.norm_type == 'standard':
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                elif self.norm_type == 'minmax':
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler(feature_range=self.feature_range)
                else:
                    raise ValueError("norm_type must be 'standard' or 'minmax'")

                scaler.fit(X_train)
                TimeLLMDataset.scaler = scaler

            # 전체 시계열에 동일 변환 적용
            self.data = TimeLLMDataset.scaler.transform(arr)
        else:
            self.data = arr  # no scaling

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = int(self.indices[idx])
        e = s + self.seq_input
        f = e + self.seq_output

        x = self.data[s:e]       # (seq_input, C)
        y = self.data[e:f]       # (seq_output, C)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # iTransformer는 time-mark를 쓰니 placeholder라도 반환(0으로 유지)
        x_mark = torch.zeros_like(x)
        y_mark = torch.zeros_like(y)

        return x, x_mark, y, y_mark

    # ------ 역정규화: (..., C) 마지막 축이 채널이어야 함 ------
    def inverse_transform(self, arr_like):
        """
        arr_like: numpy array or torch.Tensor, 마지막 축이 C.
        반환 값 dtype/타입은 입력과 동일하게 맞춰줌.
        """
        if not self.normalize or TimeLLMDataset.scaler is None:
            return arr_like  # 변환 안 했으면 그대로

        to_torch = isinstance(arr_like, torch.Tensor)
        x = arr_like.detach().cpu().numpy() if to_torch else np.asarray(arr_like)

        orig_shape = x.shape
        x2 = x.reshape(-1, self.C)
        x2 = TimeLLMDataset.scaler.inverse_transform(x2)
        x2 = x2.reshape(orig_shape)

        if to_torch:
            return torch.from_numpy(x2).type_as(arr_like)
        return x2
