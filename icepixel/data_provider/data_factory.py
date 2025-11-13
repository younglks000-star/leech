"""
Data Provider Factory
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from .seaice_dataset import SeaIceDataset


def _collate_fn(batch):
    """Batch dicts into tensors/lists (top-level for multiprocessing pickle)."""
    return {
        "input": torch.stack([item["input"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "dates_in": [item["dates_in"] for item in batch],
        "dates_out": [item["dates_out"] for item in batch],
    }


def data_provider(args, split: str) -> Tuple[SeaIceDataset, DataLoader]:
    """
    SeaIceDataset 및 DataLoader 생성
    
    Args:
        args: 설정 객체 (SimpleNamespace 또는 유사 객체)
            필수 속성:
                - root_path (str): NSIDC_Data 루트 경로
                - seq_len (int): 입력 시퀀스 길이
                - pred_len (int): 출력 시퀀스 길이
                - batch_size (int): 배치 크기
                - num_workers (int): DataLoader worker 수
            선택 속성:
                - stride (int): 샘플링 stride, 기본값 1
                - resize_hw (tuple | None): 리사이즈 크기, 기본값 None
                - cache_in_memory (bool): 메모리 캐싱 여부, 기본값 False
                - verbose (bool): 로깅 출력 여부, 기본값 False
                - train_years (tuple): 학습 연도 범위, 기본값 (2013, 2020)
                - val_years (tuple): 검증 연도 범위, 기본값 (2021, 2021)
                - test_years (tuple): 테스트 연도 범위, 기본값 (2022, 2022)
        
        split (str): "train", "val", "test" 중 하나
    
    Returns:
        (dataset, dataloader) 튜플
    
    Example:
        from types import SimpleNamespace
        
        args = SimpleNamespace(
            root_path="C:/Users/USER/Desktop/ice/data/NSIDC_Data",
            seq_len=16,
            pred_len=7,
            batch_size=2,
            num_workers=4,
            stride=1,
            resize_hw=None,
            cache_in_memory=False,
            verbose=True,
            train_years=(2013, 2020),
            val_years=(2021, 2021),
            test_years=(2022, 2022),
        )
        
        dataset, loader = data_provider(args, split="train")
        batch = next(iter(loader))
        # batch["input"]: (B, T_in, 1, H, W)
        # batch["target"]: (B, T_out, 1, H, W)
        # batch["mask"]: (B, H, W)
    """
    # 필수 파라미터
    root_path = args.root_path
    seq_input = args.seq_len
    seq_output = args.pred_len
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    # 선택 파라미터 (기본값 사용)
    stride = getattr(args, 'stride', 1)    
    cache_in_memory = getattr(args, 'cache_in_memory', False)
    verbose = getattr(args, 'verbose', False)
    train_years = getattr(args, 'train_years', (2013, 2020))
    val_years = getattr(args, 'val_years', (2021, 2021))
    test_years = getattr(args, 'test_years', (2022, 2022))
    
    # Dataset 생성
    dataset = SeaIceDataset(
        root=root_path,
        seq_input=seq_input,
        seq_output=seq_output,
        split=split,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        stride=stride,        
        cache_in_memory=cache_in_memory,
        verbose=verbose,
    )
    
    # DataLoader 설정
    shuffle = (split == "train")
    drop_last = False
    pin_memory = True
    persistent_workers = num_workers > 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=_collate_fn,
    )
    
    return dataset, dataloader


