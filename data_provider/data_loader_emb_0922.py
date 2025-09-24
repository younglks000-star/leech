# data_provider/data_loader_emb_0922.py

import os
import numpy as np
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset
from utils.tools import StandardScaler
from utils.timefeatures import time_features


def _decode_meta_attrs(hf_meta) -> dict:
    """
    h5py attrs를 배치 안전한 파이썬 타입으로 변환:
    - bytes / np.bytes_ -> str
    - np.generic(스칼라) -> 파이썬 스칼라 .item()
    - 바이트 문자열 배열(dtype.kind=='S') -> 유니코드 str 리스트
    - 그 외 ndarray는 가능한 한 .tolist() (필요 없으면 그대로 둬도 됨)
    """
    meta = {}
    for k, v in hf_meta.attrs.items():
        # bytes -> str
        if isinstance(v, (bytes, np.bytes_)):
            meta[k] = v.decode("utf-8", errors="ignore")
            continue
        # numpy scalar -> python scalar
        if isinstance(v, np.generic):
            meta[k] = v.item()
            continue
        # numpy array
        if isinstance(v, np.ndarray):
            if v.dtype.kind == 'S':  # byte-strings array (ex: dates b'20130101')
                meta[k] = v.astype('U').tolist()  # to list[str]
            else:
                # 보통 숫자/혼합이면 리스트로 캐스팅
                try:
                    meta[k] = v.tolist()
                except Exception:
                    meta[k] = v  # 마지막 수단
            continue
        # 그 외는 그대로
        meta[k] = v
    return meta


class Dataset_Custom(Dataset):
    """
    CSV에서 (슬라이딩) 윈도우를 만들고, 같은 시작 인덱스 s로 저장된
    - 프롬프트 임베딩(HDF5): dataset 'embeddings'  -> (E, C) 또는 (E, C, 1)
    - 이미지 임베딩(HDF5):   dataset 'K'/'V' or 'Kc'/'Vc' -> (C, d)

    을 **동시에** 로드해 반환합니다.

    반환:
      x           : (L_in, C)
      y           : (L_out, C)
      x_mark      : (L_in, d_time)
      y_mark      : (L_out, d_time)
      emb_prompt  : (E, C, 1)   # 모델에서 squeeze(-1)해서 씀
      emb_image   : dict {'K': (C, dk), 'V': (C, dv)}          # 기본(return_meta=False)
                     or {'K': (C, dk), 'V': (C, dv), 'meta': dict} (return_meta=True)
    """
    def __init__(self,
        csv_path,
        flag='train',                   # 'train' | 'val' | 'test'
        size=None,                      # [seq_len, label_len, pred_len]
        train_ratio=0.7, val_ratio=0.1,
        features='M', target='OT',
        scale=True, timeenc=1, freq='D',
        d_llm=4096,                     # 프롬프트 임베딩 길이(검증용)
        expect_num_nodes=None,          # CSV 채널 수 체크 (예: 32)
        prompt_embed_base=None,         # 예: "./Prompt_Embeddings_llama360"
        image_embed_base=None,          # 예: r"C:\\...\\Image_Embeddings_cluster_768"
        missing_prompt='error',         # 'error' | 'zeros'
        missing_image='error',          # 'error' | 'skip' (skip이면 dummy zeros로 대체)
        return_meta=False,              # 배치에 meta를 포함할지 (기본 False 권장)
        prefer_cluster=True,            # Kc/Vc가 있으면 그것을 우선 사용
    ):
        assert flag in ['train','val','test']
        self.flag = flag
        self.features, self.target = features, target
        self.scale, self.timeenc, self.freq = scale, timeenc, freq
        self.d_llm = int(d_llm)
        self.missing_prompt = missing_prompt
        self.missing_image  = missing_image
        self.return_meta    = bool(return_meta)
        self.prefer_cluster = bool(prefer_cluster)

        # --- 길이 ---
        if size is None:
            self.seq_len, self.label_len, self.pred_len = 96, 0, 96
        else:
            self.seq_len, self.label_len, self.pred_len = map(int, size)

        # --- CSV ---
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

        # --- time mark 전체 만들기 ---
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
            if tf.shape[1] == T:     # (n_feat, T)
                data_stamp = tf.T.astype(np.float32)
            elif tf.shape[0] == T:   # (T, n_feat)
                data_stamp = tf.astype(np.float32)
            else:
                raise ValueError(f"time_features shape mismatch: {tf.shape}, T={T}")

        arr = df_data.values.astype(np.float32)   # (T, C)
        T = arr.shape[0]

        # --- 윈도우 인덱스 ---
        max_start = T - (self.seq_len + self.pred_len)
        if max_start < 0:
            raise ValueError("데이터가 (seq_len+pred_len)보다 짧습니다.")
        all_starts = np.arange(max_start + 1, dtype=np.int64)
        n_total = len(all_starts)
        n_train = int(n_total * float(train_ratio))
        n_val   = int(n_total * float(val_ratio))
        if flag == 'train':
            indices = all_starts[:n_train]
        elif flag == 'val':
            indices = all_starts[n_train:n_train+n_val]
        else:
            indices = all_starts[n_train+n_val:]

        # --- 스케일링 ---
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

        # --- 임베딩 경로 ---
        base = os.path.splitext(os.path.basename(csv_path))[0]

        # 프롬프트 임베
        if prompt_embed_base is None:
            raise ValueError("prompt_embed_base 경로를 지정하세요.")
        self.prompt_root = os.path.join(prompt_embed_base, base)  # 내부에 out{pred_len}/split/s.h5

        # 이미지 임베
        if image_embed_base is None:
            raise ValueError("image_embed_base 경로를 지정하세요.")
        self.image_root = os.path.join(image_embed_base, base)    # 내부에 out{pred_len}/split/s.h5

        # pred_len 보관 및 인덱스 저장
        self.pred_len = int(self.pred_len)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    # -------- 프롬프트 임베딩 로더 --------
    def _load_prompt_emb(self, start_idx: int, split: str, pred_len: int):
        """
        프롬프트 임베딩 파일:  {prompt_root}/out{pred_len}/{split}/{s}.h5  with dataset 'embeddings'
        반환: torch.FloatTensor (E, C, 1)
        """
        E, C = self.d_llm, self.C
        fpath = os.path.join(self.prompt_root, f"out{pred_len}", split, f"{start_idx}.h5")
        if not os.path.exists(fpath):
            if self.missing_prompt == 'zeros':
                return torch.zeros(E, C, 1, dtype=torch.float32)
            raise FileNotFoundError(f"[prompt] missing: {fpath}")

        with h5py.File(fpath, 'r') as hf:
            if 'embeddings' not in hf:
                raise KeyError(f"[prompt] no 'embeddings' in {fpath}")
            emb = hf['embeddings'][:]  # (E,C) or (E,C,1)

        ten = torch.from_numpy(emb)
        if ten.ndim == 2:           # (E,C) → (E,C,1)
            ten = ten.unsqueeze(-1)
        # 강제 형상 검사
        if ten.shape[1] != C:
            raise ValueError(f"[prompt] C mismatch: file {ten.shape[1]} vs CSV {C}")
        if ten.shape[0] != E:
            # E가 다르면 여기서 잘라/패드하지 않고 경고만 — 모델에서 d_model과 맞아야 하므로 보통 파일/설정 맞추는 게 정석
            print(f"[WARN] prompt E mismatch: file {ten.shape[0]} vs expected {E}")
        return ten.float()

    # -------- 이미지 임베딩 로더 --------
    def _load_image_emb(self, start_idx: int, split: str, pred_len: int):
        """
        이미지 임베딩 파일: {image_root}/out{pred_len}/{split}/{s}.h5
        지원 키: ('Kc','Vc') 또는 ('K','V')
        반환: dict {'K': (C, dk), 'V': (C, dv)} (+ 'meta' 옵션)
        """
        fpath = os.path.join(self.image_root, f"out{pred_len}", split, f"{start_idx}.h5")
        if not os.path.exists(fpath):
            if self.missing_image == 'skip':
                return None
            raise FileNotFoundError(f"[image] missing: {fpath}")

        with h5py.File(fpath, 'r') as hf:
            keys = list(hf.keys())

            K = V = None
            # 클러스터 풀링 우선
            if self.prefer_cluster and ('Kc' in keys and 'Vc' in keys):
                K = hf['Kc'][:]
                V = hf['Vc'][:]
            elif ('K' in keys and 'V' in keys):
                K = hf['K'][:]
                V = hf['V'][:]
            elif ('Kc' in keys and 'Vc' in keys):
                K = hf['Kc'][:]
                V = hf['Vc'][:]
            else:
                raise KeyError(f"[image] unknown datasets in {fpath}: {keys}")

            # meta는 옵션
            meta = None
            if self.return_meta and ('meta' in hf):
                meta = _decode_meta_attrs(hf['meta'])

        K = torch.from_numpy(K).float()  # (C, dk)
        V = torch.from_numpy(V).float()  # (C, dv)
        if K.shape[0] != self.C or V.shape[0] != self.C:
            raise ValueError(f"[image] C mismatch: file K/V leading dim {K.shape[0]}/{V.shape[0]} vs CSV {self.C}")

        if self.return_meta and (meta is not None):
            return {'K': K, 'V': V, 'meta': meta}
        else:
            return {'K': K, 'V': V}

    def __getitem__(self, idx):
        s = int(self.indices[idx])
        e, f = s + self.seq_len, s + self.seq_len + self.pred_len

        x = torch.from_numpy(self.data_all[s:e]).float()        # (L_in, C)
        y = torch.from_numpy(self.data_all[e:f]).float()        # (L_out, C)
        x_mark = torch.from_numpy(self.stamp_all[s:e]).float()  # (L_in, d_time)
        y_mark = torch.from_numpy(self.stamp_all[e:f]).float()  # (L_out, d_time)

        # 임베딩 로드
        split = self.flag
        pred_len = self.pred_len

        emb_prompt = self._load_prompt_emb(s, split, pred_len)          # (E, C, 1)
        emb_image  = self._load_image_emb(s, split, pred_len)           # dict or None

        if emb_image is None and self.missing_image == 'skip':
            # 드랍할 수 없으면 더미라도 내보냄(형상 보존)
            emb_image = {'K': torch.zeros(self.C, 1, dtype=torch.float32),
                         'V': torch.zeros(self.C, 1, dtype=torch.float32)}
            if self.return_meta:
                emb_image['meta'] = {}

        return x, y, x_mark, y_mark, emb_prompt, emb_image

    def inverse_transform(self, arr_like):
        return self.scaler.inverse_transform(arr_like)
