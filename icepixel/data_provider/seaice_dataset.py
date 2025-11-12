"""
NSIDC 해빙 농도 Dataset 구현
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
    NSIDC 일별 GeoTIFF 해빙 농도 데이터셋
    
    Args:
        root (str): NSIDC_Data 루트 경로
        seq_input (int): 입력 시퀀스 길이
        seq_output (int): 출력 시퀀스 길이
        split (str): "train", "val", "test" 중 하나
        train_years (tuple): 학습 데이터 연도 범위 (시작, 끝)
        val_years (tuple): 검증 데이터 연도 범위 (시작, 끝)
        test_years (tuple): 테스트 데이터 연도 범위 (시작, 끝)
        stride (int): 샘플링 stride
        resize_hw (tuple | None): 리사이즈 크기 (H, W), None이면 원본 크기
        cache_in_memory (bool): 메모리 캐싱 여부
        verbose (bool): 로깅 출력 여부
    
    Returns:
        Dict containing:
            - input: (T_in, 1, H, W) 입력 시퀀스
            - target: (T_out, 1, H, W) 목표 시퀀스
            - mask: (H, W) 유효 영역 마스크 (target 기준)
            - dates_in: List[str] 입력 날짜들 (YYYY-MM-DD)
            - dates_out: List[str] 출력 날짜들 (YYYY-MM-DD)
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
        stride: int = 1,
        resize_hw: Optional[Tuple[int, int]] = None,
        cache_in_memory: bool = False,
        verbose: bool = False,
    ):
        self.root = root
        self.seq_input = seq_input
        self.seq_output = seq_output
        self.split = split
        self.stride = stride
        self.resize_hw = resize_hw
        self.cache_in_memory = cache_in_memory
        self.verbose = verbose
        
        # 캐시
        self._cache = {} if cache_in_memory else None
        
        # 파일 목록 수집
        self.file_list = self._collect_files(root)
        
        # Split별 연도 필터링
        year_ranges = {
            "train": train_years,
            "val": val_years,
            "test": test_years,
        }
        
        if split not in year_ranges:
            raise ValueError(f"split must be one of {list(year_ranges.keys())}, got {split}")
        
        start_year, end_year = year_ranges[split]
        self.file_list = self._filter_by_years(self.file_list, start_year, end_year)
        
        # 검증: 충분한 데이터가 있는지 확인
        min_required = seq_input + seq_output
        if len(self.file_list) < min_required:
            raise ValueError(
                f"Insufficient data for split '{split}': "
                f"found {len(self.file_list)} files, but need at least "
                f"{min_required} (seq_input={seq_input} + seq_output={seq_output})"
            )
        
        # 샘플 시작 인덱스 계산
        max_start = len(self.file_list) - (seq_input + seq_output) + 1
        self.start_indices = list(range(0, max_start, stride))
        
        # 로깅
        if verbose:
            first_date = self.file_list[0][0]
            last_date = self.file_list[-1][0]
            print(
                f"[{split.upper()}] Files: {len(self.file_list)}, "
                f"Samples: {len(self.start_indices)}, "
                f"Date range: {first_date} to {last_date}"
            )
    
    def _collect_files(self, root: str) -> List[Tuple[str, str]]:
        """
        파일 목록 수집 및 날짜순 정렬
        
        Returns:
            List of (date_str, filepath) tuples, sorted by date
        """
        pattern = os.path.join(root, "**", "N_????????_concentration_v3.0.tif")
        file_paths = glob.glob(pattern, recursive=True)
        
        if len(file_paths) == 0:
            raise FileNotFoundError(
                f"No concentration files found in {root}. "
                f"Expected pattern: **/N_YYYYMMDD_concentration_v3.0.tif"
            )
        
        # 파일명에서 날짜 추출 및 정렬
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
                # 날짜 파싱 실패 시 무시
                continue
        
        # 날짜순 정렬
        files_with_dates.sort(key=lambda x: x[0])
        
        return files_with_dates
    
    def _filter_by_years(
        self, 
        file_list: List[Tuple[str, str]], 
        start_year: int, 
        end_year: int
    ) -> List[Tuple[str, str]]:
        """
        연도 범위로 파일 필터링
        
        Args:
            file_list: (date_str, filepath) 리스트
            start_year: 시작 연도 (포함)
            end_year: 종료 연도 (포함)
        
        Returns:
            필터링된 파일 리스트
        """
        filtered = []
        for date_str, path in file_list:
            year = int(date_str.split('-')[0])
            if start_year <= year <= end_year:
                filtered.append((date_str, path))
        
        return filtered
    
    def _load_and_preprocess(self, filepath: str) -> np.ndarray:
        """
        TIFF 파일 로드 및 전처리
        
        전처리 규칙:
        - 0~1000: /1000.0 → [0, 1] (해빙 농도)
        - 2510 → -1 (극점 구멍)
        - 2530 → -2 (해안선)
        - 2540 → -3 (육지)
        - 2550 → -4 (결측)
        
        Args:
            filepath: TIFF 파일 경로
        
        Returns:
            전처리된 배열 (H, W), dtype=float32
        """
        # 캐시 확인
        if self._cache is not None and filepath in self._cache:
            return self._cache[filepath].copy()
        
        # TIFF 로드
        with rasterio.open(filepath) as src:
            data = src.read(1).astype(np.float32)
        
        # 전처리
        processed = np.zeros_like(data, dtype=np.float32)
        
        # 0~1000: 정규화
        mask_valid = (data >= 0) & (data <= 1000)
        processed[mask_valid] = data[mask_valid] / 1000.0
        
        # 특수값 매핑
        processed[data == 2510] = -1.0  # 극점 구멍
        processed[data == 2530] = -2.0  # 해안선
        processed[data == 2540] = -3.0  # 육지
        processed[data == 2550] = -4.0  # 결측
        
        # 리사이즈 (최근접 보간, 특수코드 보존)
        if self.resize_hw is not None:
            from skimage.transform import resize
            H_new, W_new = self.resize_hw
            # order=0: 최근접 보간
            processed = resize(
                processed, 
                (H_new, W_new), 
                order=0,  # nearest neighbor
                preserve_range=True,
                anti_aliasing=False
            ).astype(np.float32)
        
        # 캐시 저장
        if self._cache is not None:
            self._cache[filepath] = processed.copy()
        
        return processed
    
    def _create_valid_mask(self, data: np.ndarray) -> np.ndarray:
        """
        유효 영역 마스크 생성
        
        Args:
            data: 전처리된 데이터 (H, W)
        
        Returns:
            마스크 (H, W), dtype=bool, True=유효영역
        """
        # 유효값: 0~1 범위
        mask = (data >= 0) & (data <= 1)
        return mask
    
    def __len__(self) -> int:
        """샘플 개수"""
        return len(self.start_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        샘플 반환
        
        Args:
            idx: 샘플 인덱스
        
        Returns:
            Dict containing:
                - input: (T_in, 1, H, W)
                - target: (T_out, 1, H, W)
                - mask: (H, W)
                - dates_in: List[str]
                - dates_out: List[str]
        """
        start = self.start_indices[idx]
        
        # 입력 시퀀스
        input_data = []
        dates_in = []
        for i in range(start, start + self.seq_input):
            date_str, filepath = self.file_list[i]
            data = self._load_and_preprocess(filepath)
            input_data.append(data)
            dates_in.append(date_str)
        
        # 출력 시퀀스
        target_data = []
        dates_out = []
        for i in range(start + self.seq_input, start + self.seq_input + self.seq_output):
            date_str, filepath = self.file_list[i]
            data = self._load_and_preprocess(filepath)
            target_data.append(data)
            dates_out.append(date_str)
        
        # 스택 및 차원 추가
        input_seq = np.stack(input_data, axis=0)    # (T_in, H, W)
        target_seq = np.stack(target_data, axis=0)  # (T_out, H, W)
        
        # 채널 차원 추가: (T, H, W) → (T, 1, H, W)
        input_seq = input_seq[:, np.newaxis, :, :]   # (T_in, 1, H, W)
        target_seq = target_seq[:, np.newaxis, :, :]  # (T_out, 1, H, W)
        
        # 유효 마스크 (target의 마지막 프레임 기준)
        mask = self._create_valid_mask(target_data[-1])  # (H, W)
        
        return {
            "input": torch.FloatTensor(input_seq),
            "target": torch.FloatTensor(target_seq),
            "mask": torch.BoolTensor(mask),
            "dates_in": dates_in,
            "dates_out": dates_out,
        }



