# -*- coding: utf-8 -*-
"""
NSIDC 해빙 농도 Dataset (리샘플링 제거 버전)
- 항상 원해상도(448×304)로 동작
"""

import os
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class SeaIceDataset(Dataset):
    """
    NSIDC 일별 GeoTIFF 해빙 농도 데이터셋 (원해상도 고정)

    Args:
        root (str): NSIDC_Data 루트 경로
        seq_input (int): 입력 시퀀스 길이
        seq_output (int): 출력 시퀀스 길이
        split (str): "train", "val", "test" 중 하나
        train_years (tuple): 학습 데이터 연도 범위 (시작, 끝)
        val_years (tuple): 검증 데이터 연도 범위 (시작, 끝)
        test_years (tuple): 테스트 데이터 연도 범위 (시작, 끝)
        stride (int): 샘플링 stride
        cache_in_memory (bool): 메모리 캐싱 여부
        verbose (bool): 로깅 출력 여부

    Returns (각 샘플):
        - input:  (T_in, 1, H, W)
        - target: (T_out, 1, H, W)
        - mask:   (H, W)  # target 마지막 프레임 기준 유효영역
        - dates_in:  List[str] (YYYY-MM-DD)
        - dates_out: List[str] (YYYY-MM-DD)
    """

    def __init__(
        self,
        root: str,
        seq_input: int,
        seq_output: int,
        split: str = "train",
        train_years: Tuple[int, int] = (2013, 2020),
        val_years: Tuple[int, int] = (2021, 2021),
        test_years: Tuple[int, int] = (2022, 2022),
        stride: int = 7,
        cache_in_memory: bool = False,
        verbose: bool = False,
    ):
        self.root = root
        self.seq_input = seq_input
        self.seq_output = seq_output
        self.split = split
        self.stride = stride
        self.cache_in_memory = cache_in_memory
        self.verbose = verbose

        # 캐시
        self._cache = {} if cache_in_memory else None

        # 파일 목록 수집 및 분할 필터
        self.file_list = self._collect_files(root)
        year_ranges = {"train": train_years, "val": val_years, "test": test_years}
        if split not in year_ranges:
            raise ValueError(f"split must be one of {list(year_ranges.keys())}, got {split}")
        start_year, end_year = year_ranges[split]
        self.file_list = self._filter_by_years(self.file_list, start_year, end_year)

        # 길이 검증
        min_required = seq_input + seq_output
        if len(self.file_list) < min_required:
            raise ValueError(
                f"Insufficient data for split '{split}': "
                f"found {len(self.file_list)} files, need >= {min_required} "
                f"(seq_input={seq_input} + seq_output={seq_output})"
            )

        # 시작 인덱스
        max_start = len(self.file_list) - (seq_input + seq_output) + 1
        self.start_indices = list(range(0, max_start, stride))

        if verbose:
            first_date = self.file_list[0][0]
            last_date = self.file_list[-1][0]
            print(
                f"[{split.upper()}] Files: {len(self.file_list)}, "
                f"Samples: {len(self.start_indices)}, "
                f"Date range: {first_date} to {last_date}"
            )

    def _collect_files(self, root: str) -> List[Tuple[str, str]]:
        """파일 목록 수집 및 날짜순 정렬"""
        pattern = os.path.join(root, "**", "N_????????_concentration_v3.0.tif")
        file_paths = glob.glob(pattern, recursive=True)
        if len(file_paths) == 0:
            raise FileNotFoundError(
                f"No concentration files found in {root}. "
                f"Expected pattern: **/N_YYYYMMDD_concentration_v3.0.tif"
            )
        files_with_dates = []
        for path in file_paths:
            filename = os.path.basename(path)
            # N_YYYYMMDD_concentration_v3.0.tif
            date_str = filename.split('_')[1]  # YYYYMMDD
            try:
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                date_formatted = date_obj.strftime('%Y-%m-%d')
                files_with_dates.append((date_formatted, path))
            except ValueError:
                continue
        files_with_dates.sort(key=lambda x: x[0])
        return files_with_dates

    def _filter_by_years(
        self,
        file_list: List[Tuple[str, str]],
        start_year: int,
        end_year: int
    ) -> List[Tuple[str, str]]:
        """연도 범위로 파일 필터링"""
        filtered = []
        for date_str, path in file_list:
            year = int(date_str.split('-')[0])
            if start_year <= year <= end_year:
                filtered.append((date_str, path))
        return filtered

    def _load_and_preprocess(self, filepath: str) -> np.ndarray:
        """
        TIFF 로드 및 전처리(원해상도 유지)
        - 0~1000: /1000.0 → [0,1]
        - 2510 → -1.0 (극점 구멍)
        - 2530 → -2.0 (해안선)
        - 2540 → -3.0 (육지)
        - 위 범주 외의 값 발견 시 즉시 에러로 중단
        """
        # 캐시
        if self._cache is not None and filepath in self._cache:
            return self._cache[filepath].copy()

        with rasterio.open(filepath) as src:
            data = src.read(1).astype(np.float32)

        processed = np.zeros_like(data, dtype=np.float32)

        # 정상 해빙 값 정규화
        mask_valid = (data >= 0) & (data <= 1000)
        processed[mask_valid] = data[mask_valid] / 1000.0

        # 특수코드 매핑
        processed[data == 2510] = -1.0  # 극점 구멍
        processed[data == 2530] = -2.0  # 해안선
        processed[data == 2540] = -3.0  # 육지

        # 허용되지 않은 값 체크 (디버그/안전)
        unknown_mask = (~mask_valid) & (data != 2510) & (data != 2530) & (data != 2540)
        if np.any(unknown_mask):
            bad_vals = np.unique(data[unknown_mask])
            raise ValueError(f"Unexpected code(s) in {filepath}: {bad_vals}")

        if self._cache is not None:
            self._cache[filepath] = processed.copy()

        return processed

    def _create_valid_mask(self, data: np.ndarray) -> np.ndarray:
        """유효 영역 마스크 생성: [0,1] 범위만 True"""
        return (data >= 0) & (data <= 1)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.start_indices[idx]

        # 입력 시퀀스
        input_data, dates_in = [], []
        for i in range(start, start + self.seq_input):
            date_str, filepath = self.file_list[i]
            data = self._load_and_preprocess(filepath)
            input_data.append(data)
            dates_in.append(date_str)

        # 출력 시퀀스
        target_data, dates_out = [], []
        for i in range(start + self.seq_input, start + self.seq_input + self.seq_output):
            date_str, filepath = self.file_list[i]
            data = self._load_and_preprocess(filepath)
            target_data.append(data)
            dates_out.append(date_str)

        # 스택 및 채널 차원 추가
        input_seq = np.stack(input_data, axis=0)[:, np.newaxis, :, :]    # (T_in, 1, H, W)
        target_seq = np.stack(target_data, axis=0)[:, np.newaxis, :, :]  # (T_out, 1, H, W)

        # 유효 마스크 (target 마지막 프레임 기준)
        mask = self._create_valid_mask(target_data[-1])

        return {
            "input": torch.FloatTensor(input_seq),
            "target": torch.FloatTensor(target_seq),
            "mask": torch.BoolTensor(mask),
            "dates_in": dates_in,
            "dates_out": dates_out,
        }
