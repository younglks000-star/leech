"""
해빙 농도 예측 모델용 평가지표 및 시각화 모듈

주요 기능:
    - 평가지표: RMSE, MAE, R², SIE Error
    - 시각화: Error Map, RMSE Map, Time-series, SIE Trend
    - Metric Summary Table
"""

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================
# 유틸리티 함수
# ============================

def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    torch.Tensor 또는 numpy.ndarray를 numpy로 변환
    
    Args:
        x: 입력 데이터
    
    Returns:
        numpy.ndarray
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def apply_mask(pred: np.ndarray, true: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    마스크를 적용하여 유효 영역만 추출
    
    Args:
        pred: 예측값 (임의 shape)
        true: 실제값 (pred와 동일 shape)
        mask: 마스크 (True=유효, False=무효), None이면 자동 생성
    
    Returns:
        (pred_valid, true_valid) - 1D 배열
    """
    pred = to_numpy(pred)
    true = to_numpy(true)
    
    if mask is None:
        # 자동 마스크 생성: 유효값(0~1) 범위
        mask = (true >= 0) & (true <= 1)
    else:
        mask = to_numpy(mask)

    # mask를 bool 타입으로 변환하고, 불필요한 singleton 제거
    mask = np.asarray(mask).astype(bool, copy=False)
    mask = np.squeeze(mask)

    if mask.ndim == 0:
        # 단일 값인 경우 전체를 동일하게 적용
        mask = np.full(pred.shape, mask, dtype=bool)
    else:
        # pred 차원보다 mask가 더 많을 경우 -> 오류
        if mask.ndim > pred.ndim:
            raise ValueError(
                f"Mask dimension {mask.ndim} is greater than prediction dimension {pred.ndim}."
            )

        # 부족한 차원은 앞쪽(배치/시간축)부터 singleton으로 채움
        # 예: mask (B, H, W) -> (B, 1, 1, H, W)
        add_dims = pred.ndim - mask.ndim
        if add_dims > 0:
            if mask.ndim >= 2:
                prefix = mask.shape[:-2]
                spatial = mask.shape[-2:]
                mask = mask.reshape(prefix + (1,) * add_dims + spatial)
            else:
                mask = mask.reshape((1,) * add_dims + mask.shape)

        try:
            mask = np.broadcast_to(mask, pred.shape)
        except ValueError as e:
            raise ValueError(
                f"Mask shape {mask.shape} cannot be broadcast to prediction shape {pred.shape}: {e}"
            )

    pred_valid = pred[mask]
    true_valid = true[mask]
    
    # NaN/Inf 제거
    finite_mask = np.isfinite(pred_valid) & np.isfinite(true_valid)
    pred_valid = pred_valid[finite_mask]
    true_valid = true_valid[finite_mask]

    return pred_valid, true_valid


# ============================
# 평가지표 계산 함수
# ============================

def calc_rmse(pred: Union[torch.Tensor, np.ndarray], 
              true: Union[torch.Tensor, np.ndarray],
              mask: Optional[np.ndarray] = None) -> float:
    """
    RMSE (Root Mean Squared Error) 계산
    
    Args:
        pred: 예측값
        true: 실제값
        mask: 유효 영역 마스크 (True=유효)
    
    Returns:
        RMSE 값
    """
    pred_valid, true_valid = apply_mask(pred, true, mask)
    if pred_valid.size == 0:
        return 0.0
    mse = mean_squared_error(true_valid, pred_valid)
    rmse = np.sqrt(mse)
    return float(rmse)


def calc_mae(pred: Union[torch.Tensor, np.ndarray],
             true: Union[torch.Tensor, np.ndarray],
             mask: Optional[np.ndarray] = None) -> float:
    """
    MAE (Mean Absolute Error) 계산
    
    Args:
        pred: 예측값
        true: 실제값
        mask: 유효 영역 마스크
    
    Returns:
        MAE 값
    """
    pred_valid, true_valid = apply_mask(pred, true, mask)
    if pred_valid.size == 0:
        return 0.0
    mae = mean_absolute_error(true_valid, pred_valid)
    return float(mae)


def calc_r2(pred: Union[torch.Tensor, np.ndarray],
            true: Union[torch.Tensor, np.ndarray],
            mask: Optional[np.ndarray] = None) -> float:
    """
    R² (Coefficient of Determination) 계산
    
    Args:
        pred: 예측값
        true: 실제값
        mask: 유효 영역 마스크
    
    Returns:
        R² 값 (1에 가까울수록 좋음)
    """
    pred_valid, true_valid = apply_mask(pred, true, mask)

    if pred_valid.size == 0:
        return 0.0
    
    r2 = r2_score(true_valid, pred_valid)
    return float(r2)


def calc_sie(sic_map: Union[torch.Tensor, np.ndarray],
             threshold: float = 0.15,
             pixel_area_km2: float = 625.0) -> float:
    """
    SIE (Sea Ice Extent) 계산
    
    Args:
        sic_map: 해빙 농도 맵 (임의 shape, 값 범위 0~1)
        threshold: 해빙 존재 임계값 (기본 15%)
        pixel_area_km2: 픽셀당 면적 (km²), 기본 625 (25km × 25km)
    
    Returns:
        SIE (백만 km²)
    """
    sic_map = to_numpy(sic_map)
    
    # 유효 영역만 (0~1 범위)
    valid_mask = (sic_map >= 0) & (sic_map <= 1)
    
    # 해빙 존재 픽셀 수
    ice_mask = valid_mask & (sic_map >= threshold)
    num_ice_pixels = np.sum(ice_mask)
    
    # 면적 계산 (백만 km²)
    sie_km2 = num_ice_pixels * pixel_area_km2
    sie_million_km2 = sie_km2 / 1e6
    
    return float(sie_million_km2)


def calc_metrics(pred: Union[torch.Tensor, np.ndarray],
                 true: Union[torch.Tensor, np.ndarray],
                 mask: Optional[np.ndarray] = None,
                 pixel_area_km2: float = 625.0) -> Dict[str, float]:
    """
    모든 평가지표 한번에 계산
    
    Args:
        pred: 예측값 (B, T, 1, H, W) 또는 (B, H, W)
        true: 실제값 (pred와 동일 shape)
        mask: 유효 영역 마스크 (B, H, W)
        pixel_area_km2: 픽셀당 면적
    
    Returns:
        Dict containing:
            - RMSE: Root Mean Squared Error
            - MAE: Mean Absolute Error
            - R2: Coefficient of Determination
            - SIE_pred: 예측 해빙 면적 (백만 km²)
            - SIE_true: 실제 해빙 면적 (백만 km²)
            - SIE_error_pct: SIE 오차 (%)
    """
    pred_np = to_numpy(pred)
    true_np = to_numpy(true)
    
    # RMSE, MAE, R²
    rmse = calc_rmse(pred_np, true_np, mask)
    mae = calc_mae(pred_np, true_np, mask)
    r2 = calc_r2(pred_np, true_np, mask)
    
    # SIE 계산 (마지막 timestep 기준)
    # pred shape이 (B, T, 1, H, W)인 경우 마지막 T 선택
    if pred_np.ndim == 5:  # (B, T, 1, H, W)
        pred_last = pred_np[:, -1, 0, :, :]  # (B, H, W)
        true_last = true_np[:, -1, 0, :, :]
    elif pred_np.ndim == 4:  # (B, 1, H, W)
        pred_last = pred_np[:, 0, :, :]  # (B, H, W)
        true_last = true_np[:, 0, :, :]
    else:  # (B, H, W) or (H, W)
        pred_last = pred_np
        true_last = true_np
    
    # 배치 평균 SIE
    sie_pred = calc_sie(pred_last, pixel_area_km2=pixel_area_km2)
    sie_true = calc_sie(true_last, pixel_area_km2=pixel_area_km2)
    
    # SIE Error (%)
    if sie_true > 0:
        sie_error_pct = (sie_pred - sie_true) / sie_true * 100
    else:
        sie_error_pct = 0.0
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'SIE_pred': sie_pred,
        'SIE_true': sie_true,
        'SIE_error_pct': sie_error_pct,
    }


# ============================
# 시각화 함수
# ============================

def plot_error_map(pred: Union[torch.Tensor, np.ndarray],
                   true: Union[torch.Tensor, np.ndarray],
                   title: str = "",
                   date: Optional[str] = None,
                   save_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (10, 8),
                   dpi: int = 150):
    """
    Error Map 시각화 (지역별 예측 오차)
    
    Args:
        pred: 예측값 (H, W) 또는 (1, H, W)
        true: 실제값 (H, W) 또는 (1, H, W)
        title: 그래프 제목
        date: 날짜 정보 (YYYY-MM-DD)
        save_path: 저장 경로 (None이면 plt.show())
        figsize: Figure 크기
        dpi: 해상도
    """
    pred = to_numpy(pred).squeeze()  # (H, W)
    true = to_numpy(true).squeeze()
    
    # Error 계산 (%)
    error = (pred - true) * 100  # -100 ~ 100
    
    # 유효 영역만 (특수값 마스킹)
    valid_mask = (true >= 0) & (true <= 1)
    error_masked = np.where(valid_mask, error, np.nan)
    
    # RMSE 계산
    rmse = calc_rmse(pred, true)
    
    # 플롯
    plt.figure(figsize=figsize, dpi=dpi)
    
    im = plt.imshow(error_masked, cmap='RdBu_r', vmin=-50, vmax=50, aspect='auto')
    plt.colorbar(im, label='Error (%)', fraction=0.046, pad=0.04)
    
    # 제목
    if title:
        plot_title = title
    else:
        plot_title = "Sea Ice Concentration Error Map"
    
    if date:
        plot_title += f"\nDate: {date}"
    
    plot_title += f" | RMSE: {rmse:.4f}"
    
    plt.title(plot_title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Longitude Index', fontsize=11)
    plt.ylabel('Latitude Index', fontsize=11)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_rmse_map(pred_seq: Union[torch.Tensor, np.ndarray],
                  true_seq: Union[torch.Tensor, np.ndarray],
                  save_path: Optional[str] = None,
                  figsize: Tuple[int, int] = (10, 8),
                  dpi: int = 150):
    """
    RMSE Map 시각화 (시간 평균 RMSE)
    
    Args:
        pred_seq: 예측 시퀀스 (T, H, W) 또는 (T, 1, H, W)
        true_seq: 실제 시퀀스 (T, H, W) 또는 (T, 1, H, W)
        save_path: 저장 경로
        figsize: Figure 크기
        dpi: 해상도
    """
    pred_seq = to_numpy(pred_seq).squeeze()  # (T, H, W)
    true_seq = to_numpy(true_seq).squeeze()
    
    if pred_seq.ndim == 2:  # (H, W) 단일 프레임
        pred_seq = pred_seq[np.newaxis, ...]  # (1, H, W)
        true_seq = true_seq[np.newaxis, ...]
    
    # 픽셀별 RMSE 계산
    # (T, H, W) → (H, W)
    squared_error = (pred_seq - true_seq) ** 2  # (T, H, W)
    mse_map = np.mean(squared_error, axis=0)  # (H, W)
    rmse_map = np.sqrt(mse_map) * 100  # 퍼센트로 변환
    
    # 유효 영역 마스킹
    valid_mask = (true_seq >= 0) & (true_seq <= 1)
    valid_mask_2d = np.all(valid_mask, axis=0)  # (H, W)
    rmse_map_masked = np.where(valid_mask_2d, rmse_map, np.nan)
    
    # 플롯
    plt.figure(figsize=figsize, dpi=dpi)
    
    im = plt.imshow(rmse_map_masked, cmap='YlOrRd', vmin=0, vmax=50, aspect='auto')
    plt.colorbar(im, label='RMSE (%)', fraction=0.046, pad=0.04)
    
    plt.title('Time-Averaged RMSE Map', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Longitude Index', fontsize=11)
    plt.ylabel('Latitude Index', fontsize=11)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_timeseries(pred: Union[torch.Tensor, np.ndarray],
                    true: Union[torch.Tensor, np.ndarray],
                    model_name: str = "Model",
                    seq_output: Optional[int] = None,
                    dates: Optional[list] = None,
                    mask: Optional[np.ndarray] = None,
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (12, 6),
                    dpi: int = 150):
    """
    Time-series Comparison Plot (전체 평균 SIC)
    
    Args:
        pred: 예측 시퀀스 (T, H, W) 또는 (T, 1, H, W)
        true: 실제 시퀀스
        model_name: 모델 이름
        seq_output: 예측 길이
        dates: 날짜 리스트 ['YYYY-MM-DD', ...]
        mask: 유효 영역 마스크 (H, W)
        save_path: 저장 경로
        figsize: Figure 크기
        dpi: 해상도
    """
    pred = to_numpy(pred).squeeze()  # (T, H, W)
    true = to_numpy(true).squeeze()
    
    if pred.ndim == 2:  # (H, W)
        pred = pred[np.newaxis, ...]
        true = true[np.newaxis, ...]
    
    # 유효 영역 평균 계산
    if mask is not None:
        mask = to_numpy(mask)
        pred_avg = []
        true_avg = []
        for t in range(pred.shape[0]):
            pred_valid = pred[t][mask]
            true_valid = true[t][mask]
            pred_avg.append(pred_valid.mean())
            true_avg.append(true_valid.mean())
        pred_avg = np.array(pred_avg)
        true_avg = np.array(true_avg)
    else:
        # 자동 마스킹 (0~1 범위)
        valid_mask = (true >= 0) & (true <= 1)
        pred_avg = np.array([pred[t][valid_mask[t]].mean() for t in range(pred.shape[0])])
        true_avg = np.array([true[t][valid_mask[t]].mean() for t in range(true.shape[0])])
    
    # RMSE 계산
    rmse = calc_rmse(pred, true, mask)
    
    # 시간축
    if dates is not None:
        x_axis = np.arange(len(dates))
        x_label = 'Time Step'
    else:
        x_axis = np.arange(len(pred_avg))
        x_label = 'Time Step'
    
    # 플롯 (seaborn 스타일)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize, dpi=dpi)
    
    plt.plot(x_axis, pred_avg, label='Prediction', 
             linestyle='-', color='#E74C3C', linewidth=2.0, alpha=0.8)
    plt.plot(x_axis, true_avg, label='Ground Truth', 
             linestyle='--', color='#3498DB', linewidth=2.0, alpha=0.8)
    
    # 제목
    title = f'{model_name}'
    if seq_output:
        title += f' (Prediction Length: {seq_output} days)'
    title += f'\nRMSE: {rmse:.4f}'
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(x_label, fontsize=12, labelpad=10)
    plt.ylabel('Mean Sea Ice Concentration', fontsize=12, labelpad=10)
    plt.legend(fontsize=11, loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_sie_trend(sie_pred: Union[list, np.ndarray],
                   sie_true: Union[list, np.ndarray],
                   time_axis: Optional[Union[list, np.ndarray]] = None,
                   time_unit: str = "days",
                   save_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (12, 6),
                   dpi: int = 150):
    """
    SIE Trend Plot (일별 또는 월평균)
    
    Args:
        sie_pred: 예측 SIE 리스트 (백만 km²)
        sie_true: 실제 SIE 리스트 (백만 km²)
        time_axis: 시간축 (days 또는 dates)
        time_unit: 시간 단위 ("days", "months")
        save_path: 저장 경로
        figsize: Figure 크기
        dpi: 해상도
    """
    sie_pred = np.array(sie_pred) if not isinstance(sie_pred, np.ndarray) else sie_pred
    sie_true = np.array(sie_true) if not isinstance(sie_true, np.ndarray) else sie_true
    
    if time_axis is None:
        time_axis = np.arange(len(sie_pred))
    else:
        time_axis = np.array(time_axis)
    
    # 플롯
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize, dpi=dpi)
    
    plt.plot(time_axis, sie_pred, label='Model', 
             linestyle='-', color='#E74C3C', linewidth=2.5, marker='o', 
             markersize=4, alpha=0.8)
    plt.plot(time_axis, sie_true, label='NSIDC', 
             linestyle='--', color='#3498DB', linewidth=2.5, marker='s', 
             markersize=4, alpha=0.8)
    
    plt.title('Sea Ice Extent (SIE) Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(f'Time ({time_unit})', fontsize=12, labelpad=10)
    plt.ylabel('SIE (Million km²)', fontsize=12, labelpad=10)
    plt.legend(fontsize=11, loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_spatial_comparison(pred: Union[torch.Tensor, np.ndarray],
                            true: Union[torch.Tensor, np.ndarray],
                            date: Optional[str] = None,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (15, 5),
                            dpi: int = 150):
    """
    공간 맵 3개 비교 (Prediction, Ground Truth, Error)
    
    Args:
        pred: 예측값 (H, W)
        true: 실제값 (H, W)
        date: 날짜 정보
        save_path: 저장 경로
        figsize: Figure 크기
        dpi: 해상도
    """
    pred = to_numpy(pred).squeeze()  # (H, W)
    true = to_numpy(true).squeeze()
    
    # Error 계산
    error = (pred - true) * 100  # %
    
    # 유효 마스크
    valid_mask = (true >= 0) & (true <= 1)
    pred_masked = np.where(valid_mask, pred, np.nan)
    true_masked = np.where(valid_mask, true, np.nan)
    error_masked = np.where(valid_mask, error, np.nan)
    
    # 플롯
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    
    # Prediction
    im1 = axes[0].imshow(pred_masked, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    axes[0].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude Index')
    axes[0].set_ylabel('Latitude Index')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='SIC')
    
    # Ground Truth
    im2 = axes[1].imshow(true_masked, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longitude Index')
    axes[1].set_ylabel('Latitude Index')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='SIC')
    
    # Error
    im3 = axes[2].imshow(error_masked, cmap='RdBu_r', vmin=-50, vmax=50, aspect='auto')
    axes[2].set_title('Error', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Longitude Index')
    axes[2].set_ylabel('Latitude Index')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='Error (%)')
    
    # 전체 제목
    if date:
        fig.suptitle(f'Sea Ice Concentration Comparison\nDate: {date}', 
                     fontsize=14, fontweight='bold', y=1.02)
    else:
        fig.suptitle('Sea Ice Concentration Comparison', 
                     fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================
# Metric Summary
# ============================

def create_metric_table(metrics_dict: Dict[str, float],
                       save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Metric 요약 테이블 생성 및 CSV 저장
    
    Args:
        metrics_dict: calc_metrics() 반환값
        save_path: CSV 저장 경로
    
    Returns:
        pandas.DataFrame
    """
    df = pd.DataFrame([metrics_dict])
    
    # 컬럼 순서 정렬
    column_order = ['RMSE', 'MAE', 'R2', 'SIE_pred', 'SIE_true', 'SIE_error_pct']
    df = df[[col for col in column_order if col in df.columns]]
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, float_format='%.6f')
    
    return df


# ============================
# Best Metric 업데이트
# ============================

def update_best_metrics(current_metrics: Dict[str, float],
                       best_metrics: Dict[str, float],
                       save_dir: str,
                       model: Optional[torch.nn.Module] = None,
                       epoch: int = 0) -> Dict[str, float]:
    """
    Best metrics 업데이트 및 저장
    
    Args:
        current_metrics: 현재 epoch의 metrics
        best_metrics: 지금까지의 best metrics
        save_dir: 저장 디렉토리
        model: PyTorch 모델 (저장할 경우)
        epoch: 현재 epoch
    
    Returns:
        업데이트된 best_metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Best 업데이트 (RMSE 기준)
    updated = False
    if current_metrics['RMSE'] < best_metrics.get('RMSE', float('inf')):
        best_metrics = current_metrics.copy()
        best_metrics['best_epoch'] = epoch
        updated = True
        
        # 모델 저장
        if model is not None:
            model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
    
    # Metrics 저장
    metrics_path = os.path.join(save_dir, 'best_metrics.npy')
    np.save(metrics_path, best_metrics)
    
    # CSV로도 저장
    csv_path = os.path.join(save_dir, 'best_metrics.csv')
    create_metric_table(best_metrics, csv_path)
    
    return best_metrics


# ============================
# 사용 예시 (주석)
# ============================

"""
Example usage:

import torch
from metric_visualizer import *

# 예측 및 실제 데이터
pred = torch.randn(2, 7, 1, 448, 304)  # (B, T, 1, H, W)
true = torch.randn(2, 7, 1, 448, 304)
mask = torch.ones(2, 448, 304).bool()  # (B, H, W)

# 평가지표 계산
metrics = calc_metrics(pred, true, mask)
print(metrics)
# {'RMSE': 0.234, 'MAE': 0.189, 'R2': 0.856, 
#  'SIE_pred': 12.34, 'SIE_true': 12.56, 'SIE_error_pct': -1.75}

# 시각화
plot_spatial_comparison(pred[0, -1, 0], true[0, -1, 0], 
                       date='2020-01-15', 
                       save_path='results/spatial_comparison.png')

plot_timeseries(pred[0, :, 0], true[0, :, 0], 
               model_name='CNN_2D', 
               seq_output=7,
               save_path='results/timeseries.png')

plot_rmse_map(pred[0, :, 0], true[0, :, 0], 
             save_path='results/rmse_map.png')

# SIE Trend
sie_pred_list = [calc_sie(pred[0, t, 0]) for t in range(7)]
sie_true_list = [calc_sie(true[0, t, 0]) for t in range(7)]
plot_sie_trend(sie_pred_list, sie_true_list, 
              time_axis=list(range(1, 8)),
              save_path='results/sie_trend.png')

# Metric Table
df = create_metric_table(metrics, save_path='results/metrics.csv')
print(df)

# Best 업데이트
best = {'RMSE': float('inf'), 'MAE': float('inf')}
best = update_best_metrics(metrics, best, 'results/', model=None, epoch=10)
"""

