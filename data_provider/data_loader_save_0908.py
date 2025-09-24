# data_provider/data_loader_save.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    """
    - CSV에서 'date' 컬럼 + 나머지 채널들 읽음.
    - 윈도우 시작 인덱스를 전부 만든 뒤 → (train/val/test) 비율로 split.
    - scale=False 권장(프롬프트용 숫자 원본 유지). 필요하면 True로 바꿔도 됨.
    - 반환: (x, y, x_mark, y_mark)  (임베딩은 여기서 다루지 않음)
    - 저장 시 파일명 매칭을 위해 self.indices를 외부에서 참조할 수 있음.
    """
    def __init__(self,
        csv_path,               # ← 변경: 절대/상대 CSV 경로 받기
        flag='train',           # 'train' | 'val' | 'test'
        size=None,              # [seq_len, label_len(미사용), pred_len]
        train_ratio=0.7, val_ratio=0.1,
        features='M', target='OT',
        scale=False,            # 프롬프트 값은 보통 원본 쓰는 게 자연스러움
        timeenc=0,              # 날짜 텍스트 포맷에 맞추려고 0을 기본 (연/월/일 등)
        freq='D'                # 일단위
    ):
        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = bool(scale)
        self.timeenc = timeenc
        self.freq = freq

        if size is None:
            self.seq_len, self.label_len, self.pred_len = 96, 0, 96
        else:
            self.seq_len, self.label_len, self.pred_len = map(int, size)

        # CSV 로드
        df_raw = pd.read_csv(csv_path)
        if 'date' not in df_raw.columns:
            raise ValueError("CSV에 'date' 컬럼이 필요합니다.")

        if self.features in ['M', 'MS']:
            cols_data = [c for c in df_raw.columns if c != 'date']
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            if self.target not in df_raw.columns:
                raise ValueError(f"target='{self.target}' 컬럼이 없습니다.")
            df_data = df_raw[[self.target]]
        else:
            raise ValueError("features must be 'M' or 'S'.")

        # 타임마크(전구간)
        stamp = df_raw[['date']].copy()
        stamp['date'] = pd.to_datetime(stamp['date'])
        if self.timeenc == 0:
            stamp['year']    = stamp['date'].dt.year
            stamp['month']   = stamp['date'].dt.month
            stamp['day']     = stamp['date'].dt.day
            stamp['weekday'] = stamp['date'].dt.weekday
            stamp['hour']    = stamp['date'].dt.hour
            stamp['minute']  = stamp['date'].dt.minute
            data_stamp = stamp.drop(columns=['date']).values.astype(np.float32)
        else:
            tf = time_features(pd.to_datetime(stamp['date'].values), freq=self.freq)
            data_stamp = tf.transpose(1, 0).astype(np.float32)  # (T, d_time)

        arr = df_data.values.astype(np.float32)  # (T, C)
        self.C = arr.shape[1]
        T = arr.shape[0]

        # 윈도우 시작 인덱스 전체 생성 → 비율 split
        max_start = T - (self.seq_len + self.pred_len)
        if max_start < 0:
            raise ValueError("데이터 길이가 (seq_len + pred_len)보다 짧습니다.")
        all_starts = np.arange(max_start + 1, dtype=np.int64)
        n_total = len(all_starts)
        n_train = int(n_total * float(train_ratio))
        n_val   = int(n_total * float(val_ratio))

        if flag == 'train':
            self.indices = all_starts[:n_train]
        elif flag == 'val':
            self.indices = all_starts[n_train:n_train + n_val]
        else:
            self.indices = all_starts[n_train + n_val:]

        # 스케일러(옵션): train윈도우(과거+미래) 묶어서 fit → 전구간 transform
        self.scaler = StandardScaler()
        if self.scale:
            chunks = []
            train_starts = all_starts[:n_train]
            for s in train_starts:
                e, f = s + self.seq_len, s + self.seq_len + self.pred_len
                chunks.append(arr[s:e]); chunks.append(arr[e:f])
            X_train = np.concatenate(chunks, axis=0) if len(chunks) else arr
            self.scaler.fit(X_train)
            data = self.scaler.transform(arr)
        else:
            data = arr

        self.data_all  = data       # (T, C)
        self.stamp_all = data_stamp # (T, d_time)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = int(self.indices[idx])
        e, f = s + self.seq_len, s + self.seq_len + self.pred_len
        seq_x = torch.from_numpy(self.data_all[s:e]).float()      # (L_in, C)
        seq_y = torch.from_numpy(self.data_all[e:f]).float()      # (L_out, C)
        x_mark = torch.from_numpy(self.stamp_all[s:e]).float()    # (L_in, d_time)
        y_mark = torch.from_numpy(self.stamp_all[e:f]).float()    # (L_out, d_time)
        return seq_x, seq_y, x_mark, y_mark

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
