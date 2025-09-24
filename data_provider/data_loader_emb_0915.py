# data_provider/data_loader_emb_dual.py  (원 파일 이름에 맞춰 저장해도 됨)

import os
import numpy as np
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset
from utils.tools import StandardScaler
from utils.timefeatures import time_features

class Dataset_Custom(Dataset):
    """
    CSV → 시계열 윈도우 생성 + 동일 시작 s로 저장된 임베딩 2종(프롬프트/이미지)을 동시 로드.

    반환:
      __getitem__ -> (x, y, x_mark, y_mark, prompt_emb, image_emb)

    - prompt_emb: torch.FloatTensor (E, C, 1) 또는 None
      (파일 형식: H5 내부에 'embeddings' dataset)
    - image_emb: dict {'K': FloatTensor(N, dk), 'V': FloatTensor(N, dv), 'meta': dict} 또는 None
      (파일 형식: H5 내부에 'K', 'V' dataset)
      ⚠ meta는 배치 시 리스트로 모일 수 있음. 모델에서 필요 없으면 무시 OK.

    경로 인식:
      - <root>/out{pred_len}/{flag}/{s}.h5 가 있으면 거기서 로드
      - 없으면 <root>/{flag}/{s}.h5
      - 그래도 없으면 <root>/{s}.h5
    """
    def __init__(self,
        csv_path,
        flag='train',              # 'train' | 'val' | 'test'
        size=None,                 # [seq_len, label_len, pred_len]
        train_ratio=0.7, val_ratio=0.1,
        features='M', target='OT',
        scale=False, timeenc=1, freq='D',
        d_llm=768,                         # 프롬프트 임베딩 E 차원
        expect_num_nodes=None,             # CSV 채널 수 체크 (예: 32)

        # 👇 새로 추가: 프롬프트/이미지 임베딩 루트 경로를 분리해 받음
        prompt_root=None,                  # e.g. r"C:\...\Prompt_Embeddings"
        image_root=None,                   # e.g. r"C:\...\Image_Embeddings_patch"

        # 누락 시 처리 정책
        missing_prompt='zeros',            # 'zeros' | 'error'
        missing_image='error',             # 'zeros' | 'error'  (이미지는 차원 불명이라 가급적 'error' 권장)
    ):
        assert flag in ['train','val','test']
        self.flag = flag
        self.features, self.target = features, target
        self.scale, self.timeenc, self.freq = scale, timeenc, freq
        self.d_llm = int(d_llm)
        self.missing_prompt = str(missing_prompt)
        self.missing_image = str(missing_image)

        # 길이 셋업
        if size is None:
            self.seq_len, self.label_len, self.pred_len = 96, 0, 96
        else:
            self.seq_len, self.label_len, self.pred_len = map(int, size)

        # CSV 로드 (date 필수)
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

        # 타임마크
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
            tf = time_features(stamp[['date']], freq=self.freq)
            tf = np.asarray(tf)
            if tf.ndim != 2:
                raise ValueError(f"time_features returned shape {tf.shape}, expected 2D.")

            T = len(stamp)
            if tf.shape[1] == T:
                data_stamp = tf.T.astype(np.float32)
            elif tf.shape[0] == T:
                data_stamp = tf.astype(np.float32)
            else:
                raise ValueError(f"time_features shape mismatch: {tf.shape}, T={T}")

        arr = df_data.values.astype(np.float32)   # (T, C)
        T = arr.shape[0]

        # 윈도우 시작 인덱스
        max_start = T - (self.seq_len + self.pred_len)
        if max_start < 0:
            raise ValueError("데이터가 (seq_len + pred_len)보다 짧습니다.")
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

        # 스케일러
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

        # ---- 임베딩 루트 보관 (그대로 경로를 받고, 내부에서 out{pred_len}/flag 등 자동 탐색) ----
        self.prompt_root = prompt_root  # e.g. r"C:\...\Prompt_Embeddings"
        self.image_root  = image_root   # e.g. r"C:\...\Image_Embeddings_patch"

        # 미리 풀 경로를 해석: <root>/out{pred_len}/{flag} or <root>/{flag} or <root>
        self.prompt_dir = self._resolve_embed_dir(self.prompt_root, self.pred_len, self.flag) if self.prompt_root else None
        self.image_dir  = self._resolve_embed_dir(self.image_root,  self.pred_len, self.flag) if self.image_root  else None

    # ------------------------------- helpers -------------------------------

    def _resolve_embed_dir(self, root: str, pred_len: int, flag: str) -> str:
        """
        <root>/out{pred_len}/{flag} -> <root>/{flag} -> <root>
        순으로 존재하는 폴더를 찾아 반환 (없으면 마지막 후보 반환)
        """
        if root is None:
            return None
        candidates = [
            os.path.join(root, f"out{pred_len}", flag),
            os.path.join(root, flag),
            root,
        ]
        for d in candidates:
            if os.path.isdir(d):
                return d
        return candidates[-1]

    def _load_prompt_embedding(self, s: int):
        """
        프롬프트 임베딩 로드: H5 파일에 'embeddings' dataset
        -> torch.FloatTensor shape (E, C, 1)
        """
        if not self.prompt_dir:
            return None

        E, C = self.d_llm, self.C
        fpath = os.path.join(self.prompt_dir, f"{s}.h5")
        if not os.path.exists(fpath):
            if self.missing_prompt == 'zeros':
                return torch.zeros(E, C, 1, dtype=torch.float32)
            raise FileNotFoundError(f"[prompt] Missing embedding: {fpath}")

        with h5py.File(fpath, 'r') as hf:
            if 'embeddings' not in hf:
                raise KeyError(f"[prompt] '{fpath}' has no dataset 'embeddings'")
            emb = hf['embeddings'][:]  # (E, C) or (E, C, 1)
        ten = torch.from_numpy(emb)
        if ten.ndim == 2:
            ten = ten.unsqueeze(-1)  # (E,C) -> (E,C,1)

        # 강제 shape 보정
        if ten.shape[0] != E:
            raise ValueError(f"[prompt] E mismatch: file {ten.shape[0]} vs expected {E}")
        if ten.shape[1] != C:
            raise ValueError(f"[prompt] C mismatch: file {ten.shape[1]} vs expected {C}")
        if ten.shape[2] != 1:
            ten = ten[..., :1]
        return ten.float()

    def _load_image_embedding(self, s: int):
        """
        이미지 임베딩 로드: 'K'/'V' dataset이 있는 H5
        -> dict {'K': FloatTensor(N, dk), 'V': FloatTensor(N, dv), 'meta': dict}
        """
        if not self.image_dir:
            return None

        fpath = os.path.join(self.image_dir, f"{s}.h5")
        if not os.path.exists(fpath):
            if self.missing_image == 'zeros':
                # dk/dv를 모르는 상황이라 안전한 제로 텐서를 만들기 어렵다.
                # 필요하다면 downstream에서 None 체크 후 대체 로직 사용 권장.
                return {'K': torch.zeros(self.C, 1), 'V': torch.zeros(self.C, 1), 'meta': {'note': 'zeros'}}
            raise FileNotFoundError(f"[image] Missing embedding: {fpath}")

        out = {'K': None, 'V': None, 'meta': {}}
        with h5py.File(fpath, 'r') as hf:
            if 'K' not in hf and 'V' not in hf:
                raise KeyError(f"[image] '{fpath}' must contain 'K' and/or 'V' datasets")

            if 'K' in hf:
                K = hf['K'][:]  # (N, dk)
                out['K'] = torch.from_numpy(K).float()
            if 'V' in hf:
                V = hf['V'][:]  # (N, dv)
                out['V'] = torch.from_numpy(V).float()

            if 'meta' in hf:
                for k, v in hf['meta'].attrs.items():
                    if isinstance(v, np.ndarray) and v.shape == ():
                        v = v.tolist()
                    out['meta'][k] = v

        # (K 또는 V 중 하나라도 있으면 OK) — 둘 다 없으면 위에서 막힘
        return out

    # ----------------------------------------------------------------------

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = int(self.indices[idx])
        e, f = s + self.seq_len, s + self.seq_len + self.pred_len

        x = torch.from_numpy(self.data_all[s:e]).float()        # (L_in, C)
        y = torch.from_numpy(self.data_all[e:f]).float()        # (L_out, C)
        x_mark = torch.from_numpy(self.stamp_all[s:e]).float()  # (L_in, d_time)
        y_mark = torch.from_numpy(self.stamp_all[e:f]).float()  # (L_out, d_time)

        prompt_emb = self._load_prompt_embedding(s)             # (E, C, 1) or None
        image_emb  = self._load_image_embedding(s)              # dict(...) or None

        return x, y, x_mark, y_mark, prompt_emb, image_emb

    # 역변환
    def inverse_transform(self, arr_like):
        return self.scaler.inverse_transform(arr_like)
