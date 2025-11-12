"""
Metric Visualizer 테스트 스크립트
"""

import numpy as np
import torch
import os

from metric_visualizer import (
    calc_metrics,
    calc_rmse,
    calc_mae,
    calc_r2,
    calc_sie,
    plot_spatial_comparison,
    plot_timeseries,
    plot_rmse_map,
    plot_error_map,
    plot_sie_trend,
    create_metric_table,
    update_best_metrics,
)


def test_metrics():
    """평가지표 계산 테스트"""
    print("=" * 80)
    print("평가지표 계산 테스트")
    print("=" * 80)
    
    # 더미 데이터 생성 (해빙 농도 0~1 범위)
    np.random.seed(42)
    pred = np.random.rand(2, 7, 1, 100, 80) * 0.8  # (B, T, 1, H, W)
    true = np.random.rand(2, 7, 1, 100, 80) * 0.8
    mask = np.ones((2, 100, 80), dtype=bool)
    
    # 일부 영역을 육지로 (특수값)
    true[:, :, :, :20, :] = -3.0  # 육지
    pred[:, :, :, :20, :] = -3.0
    mask[:, :20, :] = False
    
    print("\n[1] 전체 평가지표 계산")
    print("-" * 80)
    metrics = calc_metrics(pred, true, mask, pixel_area_km2=625.0)
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n[2] 개별 지표 계산")
    print("-" * 80)
    rmse = calc_rmse(pred, true, mask)
    mae = calc_mae(pred, true, mask)
    r2 = calc_r2(pred, true, mask)
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    
    print("\n[3] SIE 계산")
    print("-" * 80)
    sie_map = pred[0, -1, 0, :, :]  # 마지막 timestep (H, W)
    sie = calc_sie(sie_map, threshold=0.15, pixel_area_km2=625.0)
    print(f"  SIE: {sie:.4f} 백만 km²")
    
    print("\n✓ 평가지표 계산 테스트 완료!\n")
    
    return pred, true, mask, metrics


def test_visualizations(pred, true, mask, metrics):
    """시각화 테스트"""
    print("=" * 80)
    print("시각화 테스트")
    print("=" * 80)
    
    # 결과 저장 디렉토리
    save_dir = './test_results'
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n[1] 공간 맵 비교")
    print("-" * 80)
    try:
        plot_spatial_comparison(
            pred[0, -1, 0],  # (H, W)
            true[0, -1, 0],
            date='2020-01-15',
            save_path=os.path.join(save_dir, 'spatial_comparison.png')
        )
        print("  ✓ spatial_comparison.png 저장 완료")
    except Exception as e:
        print(f"  ✗ 실패: {e}")
    
    print("\n[2] Error Map")
    print("-" * 80)
    try:
        plot_error_map(
            pred[0, -1, 0],
            true[0, -1, 0],
            title="Test Error Map",
            date='2020-01-15',
            save_path=os.path.join(save_dir, 'error_map.png')
        )
        print("  ✓ error_map.png 저장 완료")
    except Exception as e:
        print(f"  ✗ 실패: {e}")
    
    print("\n[3] RMSE Map")
    print("-" * 80)
    try:
        plot_rmse_map(
            pred[0, :, 0],  # (T, H, W)
            true[0, :, 0],
            save_path=os.path.join(save_dir, 'rmse_map.png')
        )
        print("  ✓ rmse_map.png 저장 완료")
    except Exception as e:
        print(f"  ✗ 실패: {e}")
    
    print("\n[4] Time-series")
    print("-" * 80)
    try:
        plot_timeseries(
            pred[0, :, 0],
            true[0, :, 0],
            model_name='Test_Model',
            seq_output=7,
            mask=mask[0],
            save_path=os.path.join(save_dir, 'timeseries.png')
        )
        print("  ✓ timeseries.png 저장 완료")
    except Exception as e:
        print(f"  ✗ 실패: {e}")
    
    print("\n[5] SIE Trend")
    print("-" * 80)
    try:
        # SIE 시계열 계산
        sie_pred_list = [calc_sie(pred[0, t, 0]) for t in range(7)]
        sie_true_list = [calc_sie(true[0, t, 0]) for t in range(7)]
        
        plot_sie_trend(
            sie_pred_list,
            sie_true_list,
            time_axis=list(range(1, 8)),
            time_unit='days',
            save_path=os.path.join(save_dir, 'sie_trend.png')
        )
        print("  ✓ sie_trend.png 저장 완료")
    except Exception as e:
        print(f"  ✗ 실패: {e}")
    
    print("\n[6] Metric Table")
    print("-" * 80)
    try:
        df = create_metric_table(
            metrics,
            save_path=os.path.join(save_dir, 'metrics.csv')
        )
        print("  ✓ metrics.csv 저장 완료")
        print("\n  테이블 내용:")
        print(df.to_string(index=False))
    except Exception as e:
        print(f"  ✗ 실패: {e}")
    
    print("\n[7] Best Metrics 업데이트")
    print("-" * 80)
    try:
        best = {'RMSE': float('inf'), 'MAE': float('inf')}
        best = update_best_metrics(
            metrics,
            best,
            save_dir,
            model=None,
            epoch=10
        )
        print("  ✓ Best metrics 업데이트 완료")
        print(f"  Best RMSE: {best['RMSE']:.6f}")
        print(f"  Best MAE: {best['MAE']:.6f}")
        print(f"  Best epoch: {best.get('best_epoch', 'N/A')}")
    except Exception as e:
        print(f"  ✗ 실패: {e}")
    
    print("\n✓ 시각화 테스트 완료!")
    print(f"\n모든 결과는 '{save_dir}' 디렉토리에 저장되었습니다.\n")


def test_torch_compatibility():
    """PyTorch Tensor 호환성 테스트"""
    print("=" * 80)
    print("PyTorch Tensor 호환성 테스트")
    print("=" * 80)
    
    # PyTorch Tensor 생성
    pred_torch = torch.randn(2, 7, 1, 50, 40) * 0.5 + 0.5  # 0~1 범위
    true_torch = torch.randn(2, 7, 1, 50, 40) * 0.5 + 0.5
    mask_torch = torch.ones(2, 50, 40).bool()
    
    print("\n[1] Tensor 입력 테스트")
    print("-" * 80)
    try:
        metrics = calc_metrics(pred_torch, true_torch, mask_torch)
        print("  ✓ PyTorch Tensor 입력 성공")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  MAE: {metrics['MAE']:.6f}")
    except Exception as e:
        print(f"  ✗ 실패: {e}")
    
    print("\n[2] GPU Tensor 테스트")
    print("-" * 80)
    if torch.cuda.is_available():
        try:
            pred_gpu = pred_torch.cuda()
            true_gpu = true_torch.cuda()
            mask_gpu = mask_torch.cuda()
            
            metrics = calc_metrics(pred_gpu, true_gpu, mask_gpu)
            print("  ✓ GPU Tensor 입력 성공")
            print(f"  RMSE: {metrics['RMSE']:.6f}")
        except Exception as e:
            print(f"  ✗ 실패: {e}")
    else:
        print("  (CUDA 사용 불가능 - 건너뜀)")
    
    print("\n✓ PyTorch 호환성 테스트 완료!\n")


def main():
    """메인 테스트 실행"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Metric Visualizer 테스트" + " " * 34 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # 1. 평가지표 테스트
    pred, true, mask, metrics = test_metrics()
    
    # 2. 시각화 테스트
    test_visualizations(pred, true, mask, metrics)
    
    # 3. PyTorch 호환성 테스트
    test_torch_compatibility()
    
    print("=" * 80)
    print("모든 테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()

