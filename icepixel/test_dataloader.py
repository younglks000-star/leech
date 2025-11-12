"""
데이터로더 테스트 스크립트
"""

from types import SimpleNamespace
import sys

from data_provider import data_provider


def test_dataloader():
    """데이터로더 기본 동작 테스트"""
    
    print("=" * 80)
    print("NSIDC SeaIce DataLoader 테스트")
    print("=" * 80)
    
    # 설정
    args = SimpleNamespace(
        root_path="C:/Users/USER/Desktop/ice/data/NSIDC_Data",
        seq_len=16,
        pred_len=7,
        batch_size=2,
        num_workers=0,  # Windows에서는 0 권장
        stride=1,
        resize_hw=None,  # 원본 크기 사용
        cache_in_memory=False,
        verbose=True,
        train_years=(2013, 2020),
        val_years=(2021, 2021),
        test_years=(2022, 2022),
    )
    
    # Train 데이터셋
    print("\n[1] Train Dataset 생성")
    print("-" * 80)
    try:
        train_dataset, train_loader = data_provider(args, split="train")
        print(f"✓ Train dataset 생성 성공")
        print(f"  - 총 샘플 수: {len(train_dataset)}")
    except Exception as e:
        print(f"✗ Train dataset 생성 실패: {e}")
        return
    
    # Validation 데이터셋
    print("\n[2] Validation Dataset 생성")
    print("-" * 80)
    try:
        val_dataset, val_loader = data_provider(args, split="val")
        print(f"✓ Val dataset 생성 성공")
        print(f"  - 총 샘플 수: {len(val_dataset)}")
    except Exception as e:
        print(f"✗ Val dataset 생성 실패: {e}")
        return
    
    # Test 데이터셋
    print("\n[3] Test Dataset 생성")
    print("-" * 80)
    try:
        test_dataset, test_loader = data_provider(args, split="test")
        print(f"✓ Test dataset 생성 성공")
        print(f"  - 총 샘플 수: {len(test_dataset)}")
    except Exception as e:
        print(f"✗ Test dataset 생성 실패: {e}")
        return
    
    # 샘플 1개 로드 테스트
    print("\n[4] 단일 샘플 로드 테스트")
    print("-" * 80)
    try:
        sample = train_dataset[0]
        print(f"✓ 샘플 로드 성공")
        print(f"  - input shape: {sample['input'].shape}")
        print(f"  - target shape: {sample['target'].shape}")
        print(f"  - mask shape: {sample['mask'].shape}")
        print(f"  - dates_in (처음 3개): {sample['dates_in'][:3]}")
        print(f"  - dates_out (처음 3개): {sample['dates_out'][:3]}")
        
        # 값 범위 확인
        input_data = sample['input']
        print(f"\n  [값 범위 확인]")
        print(f"  - input min: {input_data.min():.4f}")
        print(f"  - input max: {input_data.max():.4f}")
        print(f"  - 유효값(0~1) 비율: {((input_data >= 0) & (input_data <= 1)).float().mean():.2%}")
        print(f"  - 특수값(-1~-4) 비율: {(input_data < 0).float().mean():.2%}")
        
        # 마스크 확인
        mask = sample['mask']
        print(f"\n  [마스크 확인]")
        print(f"  - 유효 픽셀 수: {mask.sum().item()} / {mask.numel()}")
        print(f"  - 유효 영역 비율: {mask.float().mean():.2%}")
        
    except Exception as e:
        print(f"✗ 샘플 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 배치 로드 테스트
    print("\n[5] 배치 로드 테스트")
    print("-" * 80)
    try:
        batch = next(iter(train_loader))
        print(f"✓ 배치 로드 성공")
        print(f"  - input shape: {batch['input'].shape}")
        print(f"  - target shape: {batch['target'].shape}")
        print(f"  - mask shape: {batch['mask'].shape}")
        print(f"  - dates_in length: {len(batch['dates_in'])}")
        print(f"  - dates_out length: {len(batch['dates_out'])}")
        
        # 배치 내 첫 샘플의 날짜 확인
        print(f"\n  [첫 번째 샘플의 날짜]")
        print(f"  - Input dates: {batch['dates_in'][0][:3]} ... {batch['dates_in'][0][-1]}")
        print(f"  - Output dates: {batch['dates_out'][0][:3]} ... {batch['dates_out'][0][-1]}")
        
    except Exception as e:
        print(f"✗ 배치 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 요약
    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)
    print(f"✓ Train: {len(train_dataset)} 샘플")
    print(f"✓ Val: {len(val_dataset)} 샘플")
    print(f"✓ Test: {len(test_dataset)} 샘플")
    print(f"✓ Batch shape: {batch['input'].shape}")
    print("=" * 80)


if __name__ == "__main__":
    test_dataloader()

