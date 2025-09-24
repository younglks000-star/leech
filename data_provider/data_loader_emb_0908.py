# data_provider/data_loader_emb_0908.py

import os
import numpy as np
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset
from utils.tools import StandardScaler
from utils.timefeatures import time_features

class Dataset_Custom(Dataset):
    def __init__(self,
        csv_path,                 # 절대/상대 csv 경로
        flag='train',             # 'train' | 'val' | 'test'
        size=None,                # [seq_len, label_len, pred_len]
        train_ratio=0.7, val_ratio=0.1,   # 윈도우 비율 split
        features='M', target='OT',
        scale=True, timeenc=1, freq='D',
        model_name="gpt2",
        d_llm=768,                        # 임베딩 차원 (검증용)
        expect_num_nodes=None,            # 채널 수 체크 (예: 32)
        embed_root=None,                  # 임베딩 경로 루트
        missing_embedding='zeros'         # 없으면 'zeros' or 'error'
    ):
        assert flag in ['train','val','test']
        self.flag = flag
        self.features, self.target = features, target
        self.scale, self.timeenc, self.freq = scale, timeenc, freq
        self.d_llm = int(d_llm)
        self.missing_embedding = missing_embedding

        # 길이 셋업
        if size is None:
            self.seq_len, self.label_len, self.pred_len = 96, 0, 96
        else:
            self.seq_len, self.label_len, self.pred_len = map(int, size)

        # CSV 로드 (date 컬럼 필수)
        df_raw = pd.read_csv(csv_path)
        if 'date' not in df_raw.columns:
            raise ValueError("CSV에 'date' 컬럼이 필요합니다.")

        if self.features in ['M','MS']:
            cols_data = [c for c in df_raw.columns if c != 'date']
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            if self.target not in df_raw.columns:
                raise ValueError(f"target='{self.target}' 없음")
            df_data = df_raw[[self.target]]
        else:
            raise ValueError("features must be 'M' or 'S'.")

        # 채널 수 확인
        self.C = df_data.shape[1]
        if expect_num_nodes is not None and expect_num_nodes != self.C:
            raise ValueError(f"expect_num_nodes({expect_num_nodes}) != CSV 채널 수({self.C})")

        # 타임마크(전 구간)
        stamp = df_raw[['date']].copy()
        stamp['date'] = pd.to_datetime(stamp['date'])
        if self.timeenc == 0:
            stamp['year'] = stamp['date'].dt.year
            stamp['month'] = stamp['date'].dt.month
            stamp['day'] = stamp['date'].dt.day
            stamp['weekday'] = stamp['date'].dt.weekday
            stamp['hour'] = stamp['date'].dt.hour
            stamp['minute'] = stamp['date'].dt.minute
            data_stamp = stamp.drop(columns=['date']).values.astype(np.float32)
        else:
            # 안전하게 time_features 출력 정규화
            tf = time_features(stamp[['date']], freq=self.freq)  # DataFrame 그대로 전달
            tf = np.asarray(tf)
            if tf.ndim != 2:
                raise ValueError(f"time_features returned shape {tf.shape}, expected 2D.")

            T = len(stamp)
            if tf.shape[1] == T:     # (n_feat, T)
                data_stamp = tf.T.astype(np.float32)  # (T, n_feat)
            elif tf.shape[0] == T:   # (T, n_feat)
                data_stamp = tf.astype(np.float32)
            else:
                raise ValueError(f"time_features shape mismatch: {tf.shape}, T={T}")

        arr = df_data.values.astype(np.float32)   # (T, C)
        T = arr.shape[0]

        # 윈도우 먼저 만들고 → 비율 split
        max_start = T - (self.seq_len + self.pred_len)
        if max_start < 0:
            raise ValueError("데이터가 (seq_len+pred_len)보다 짧습니다.")
        all_starts = np.arange(max_start + 1, dtype=np.int64)
        n_total = len(all_starts)
        n_train = int(n_total * float(train_ratio))
        n_val   = int(n_total * float(val_ratio))
        if flag == 'train':
            self.indices = all_starts[:n_train]
        elif flag == 'val':
            self.indices = all_starts[n_train:n_train+n_val]
        else:
            self.indices = all_starts[n_train+n_val:]

        # 스케일러: train 윈도우 past+future 합쳐 fit → 전구간 transform
        self.scaler = StandardScaler()
        if self.scale:
            train_starts = all_starts[:n_train]
            chunks = []
            for s in train_starts:
                e, f = s + self.seq_len, s + self.seq_len + self.pred_len
                chunks.append(arr[s:e]); chunks.append(arr[e:f])
            X_train = np.concatenate(chunks, axis=0) if len(chunks) else arr
            self.scaler.fit(X_train)
            data = self.scaler.transform(arr)
        else:
            data = arr

        self.data_all  = data        # (T, C)
        self.stamp_all = data_stamp  # (T, d_time)

        # 임베딩 루트 기본값: ./Embeddings/<csv베이스명>/
        base = os.path.splitext(os.path.basename(csv_path))[0]
        self.embed_root = embed_root or os.path.join("./Prompt_Embeddings", base)

    def __len__(self):
        return len(self.indices)

    def _load_embedding_or_dummy(self, s: int):
        """s(윈도우 시작) 기준 파일명 매칭 + 없으면 zeros 옵션"""
        E, C = self.d_llm, self.C
        fpath = os.path.join(self.embed_root, self.flag, f"{s}.h5")
        if not os.path.exists(fpath):
            if self.missing_embedding == 'zeros':
                return torch.zeros(E, C, 1, dtype=torch.float32)
            raise FileNotFoundError(f"Missing embedding: {fpath}")
        with h5py.File(fpath, 'r') as hf:
            emb = hf['embeddings'][:]
        ten = torch.from_numpy(emb)
        if ten.ndim == 2:
            ten = ten.unsqueeze(-1)  # (E,C) -> (E,C,1)
        if ten.shape[0] != E:
            raise ValueError(f"E mismatch: file {ten.shape[0]} vs {E}")
        if ten.shape[1] != C:
            raise ValueError(f"C mismatch: file {ten.shape[1]} vs {C}")
        if ten.shape[2] != 1:
            ten = ten[..., :1]
        return ten.float()

    def __getitem__(self, idx):
        s = int(self.indices[idx])
        e, f = s + self.seq_len, s + self.seq_len + self.pred_len
        x = torch.from_numpy(self.data_all[s:e]).float()     # (L_in, C)
        y = torch.from_numpy(self.data_all[e:f]).float()     # (L_out, C)
        x_mark = torch.from_numpy(self.stamp_all[s:e]).float()  # (L_in, d_time)
        y_mark = torch.from_numpy(self.stamp_all[e:f]).float()  # (L_out, d_time)
        embeddings = self._load_embedding_or_dummy(s)        # (E, C, 1)
        return x, y, x_mark, y_mark, embeddings

    def inverse_transform(self, arr_like):
        return self.scaler.inverse_transform(arr_like)
