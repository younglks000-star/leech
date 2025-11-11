# Metric_node.py - 해빙 공간 시각화 추가 버전
# ✅ 기존 코드 + 공간 지도 시각화 함수 추가

from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, balanced_accuracy_score
import numpy as np
from skimage.metrics import structural_similarity
from torchmetrics import MeanAbsolutePercentageError
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import os
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# ============================
# 🧊 해빙 예측 전용 평가 지표들
# ============================

def _mape(pred, target):
    return MeanAbsolutePercentageError()(pred.reshape(-1), target.reshape(-1)) 

def _mse(outputs, targets):
    return mean_squared_error(outputs.reshape(-1), targets.reshape(-1))

def _mae(outputs, targets):
    return mean_absolute_error(outputs.reshape(-1), targets.reshape(-1))

def _rmse(outputs, targets):
    """Root Mean Square Error"""
    mse = mean_squared_error(outputs.reshape(-1), targets.reshape(-1))
    return np.sqrt(mse)

def _bacc(outputs, targets, threshold=15.0):
    """
    Binary Accuracy (BACC) for Sea Ice Concentration
    - 15% SIC threshold (일반적으로 사용되는 해빙/무빙 경계)
    - Balanced accuracy considering class imbalance
    """
    # 해빙/무빙 이진 분류
    pred_binary = (outputs.reshape(-1) >= threshold).astype(int)
    true_binary = (targets.reshape(-1) >= threshold).astype(int)
    
    # NaN 값 제거
    valid_mask = ~(np.isnan(pred_binary) | np.isnan(true_binary))
    pred_binary = pred_binary[valid_mask]
    true_binary = true_binary[valid_mask]
    
    if len(pred_binary) == 0:
        return 0.0
    
    return balanced_accuracy_score(true_binary, pred_binary)

def _ssim(outputs, targets):
    """
    Structural Similarity Index (SSIM)
    - 공간적 구조 유사성 측정
    - 해빙 패턴의 구조적 특성 평가
    """
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    # 2D 이미지로 변환 (batch, height, width)
    if len(outputs.shape) == 3:  # (batch, features, time)
        # features를 height*width로 변환
        batch_size = outputs.shape[0]
        features = outputs.shape[1]
        time_steps = outputs.shape[2]
        
        # 정사각형에 가까운 형태로 reshape
        sqrt_features = int(np.sqrt(features))
        if sqrt_features * sqrt_features == features:
            h, w = sqrt_features, sqrt_features
        else:
            # 가장 가까운 제곱수로 패딩
            h = int(np.ceil(np.sqrt(features)))
            w = h
            pad_size = h * w - features
            outputs = np.pad(outputs, ((0, 0), (0, pad_size), (0, 0)), mode='constant')
            targets = np.pad(targets, ((0, 0), (0, pad_size), (0, 0)), mode='constant')
        
        outputs_2d = outputs.reshape(batch_size, h, w, time_steps)
        targets_2d = targets.reshape(batch_size, h, w, time_steps)
    else:
        outputs_2d = outputs
        targets_2d = targets
    
    ssim_values = []
    for i in range(outputs_2d.shape[0]):
        for j in range(outputs_2d.shape[-1]):
            pred_img = outputs_2d[i, :, :, j]
            true_img = targets_2d[i, :, :, j]
            
            # NaN 값 처리
            valid_mask = ~(np.isnan(pred_img) | np.isnan(true_img))
            if valid_mask.sum() < 10:  # 유효한 픽셀이 너무 적으면 스킵
                continue
                
            pred_img = np.where(valid_mask, pred_img, 0)
            true_img = np.where(valid_mask, true_img, 0)
            
            # SSIM 계산
            ssim_val = structural_similarity(
                pred_img, true_img, 
                data_range=100.0,  # SIC 범위 0-100%
                win_size=min(7, min(pred_img.shape[0], pred_img.shape[1]))
            )
            
            if not np.isnan(ssim_val):
                ssim_values.append(ssim_val)
    
    return np.mean(ssim_values) if ssim_values else 0.0

def CORR_uni(pred, true):
    pred = np.sum(pred, axis=(-1))
    true = np.sum(true, axis=(-1))
    
    corrs = []
    for i in range(pred.shape[0]):
        corr = np.corrcoef(pred[i, :], true[i, :])[0, 1]
        if not np.isnan(corr).any():  
            corrs.append(corr)

    return np.mean(corrs)

def metric(outputs, actuals, n_features, normalize=False):
    """
    🧊 해빙 예측 전용 평가 지표
    - RMSE: Root Mean Square Error
    - BACC: Binary Accuracy (15% SIC threshold)
    - SSIM: Structural Similarity Index
    """
    outputs = outputs.detach().cpu().numpy()
    actuals = actuals.detach().cpu().numpy()

    if normalize:
        mean = np.mean(actuals)
        std = np.std(actuals)
        outputs = (outputs - mean) / std
        actuals = (actuals - mean) / std

    # 새로운 평가 지표들
    rmse = _rmse(outputs, actuals)
    bacc = _bacc(outputs, actuals, threshold=15.0)
    ssim = _ssim(outputs, actuals)
    
    # 기존 지표들도 유지 (비교용)
    mse = _mse(outputs, actuals)
    mae = _mae(outputs, actuals)
    cor = CORR_uni(outputs, actuals)

    return [rmse, bacc, ssim, mse, mae, cor]

def metric_ice_specific(pred_maps, true_maps, land_mask=None, threshold=15.0):
    """
    🧊 해빙 특화 평가 지표 (공간 지도용)
    
    Parameters:
    -----------
    pred_maps: [N, H, W, C] - 예측 지도
    true_maps: [N, H, W, C] - 실제 지도  
    land_mask: [H, W] - 육지 마스크
    threshold: 해빙/무빙 경계 (기본 15%)
    
    Returns:
    --------
    dict: 각 지표별 값들
    """
    if torch.is_tensor(pred_maps):
        pred_maps = pred_maps.detach().cpu().numpy()
        true_maps = true_maps.detach().cpu().numpy()
    
    # 육지 마스킹
    if land_mask is not None:
        pred_masked = pred_maps * land_mask[..., None]
        true_masked = true_maps * land_mask[..., None]
    else:
        pred_masked = pred_maps
        true_masked = true_maps
    
    # NaN 값 처리
    valid_mask = ~(np.isnan(pred_masked) | np.isnan(true_masked))
    pred_valid = pred_masked[valid_mask]
    true_valid = true_masked[valid_mask]
    
    if len(pred_valid) == 0:
        return {
            'rmse': 0.0, 'bacc': 0.0, 'ssim': 0.0,
            'mse': 0.0, 'mae': 0.0, 'cor': 0.0
        }
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(pred_valid, true_valid))
    
    # BACC (Binary Accuracy)
    pred_binary = (pred_valid >= threshold).astype(int)
    true_binary = (true_valid >= threshold).astype(int)
    bacc = balanced_accuracy_score(true_binary, pred_binary)
    
    # SSIM (공간적 구조 유사성)
    ssim_values = []
    for i in range(pred_maps.shape[0]):
        for j in range(pred_maps.shape[-1]):
            pred_img = pred_masked[i, :, :, j]
            true_img = true_masked[i, :, :, j]
            
            # 유효한 픽셀만 사용
            valid_pixels = ~(np.isnan(pred_img) | np.isnan(true_img))
            if valid_pixels.sum() < 10:
                continue
                
            pred_img = np.where(valid_pixels, pred_img, 0)
            true_img = np.where(valid_pixels, true_img, 0)
            
            try:
                ssim_val = structural_similarity(
                    pred_img, true_img,
                    data_range=100.0,
                    win_size=min(7, min(pred_img.shape[0], pred_img.shape[1]))
                )
                if not np.isnan(ssim_val):
                    ssim_values.append(ssim_val)
            except:
                continue
    
    ssim = np.mean(ssim_values) if ssim_values else 0.0
    
    # 기존 지표들
    mse = mean_squared_error(pred_valid, true_valid)
    mae = mean_absolute_error(pred_valid, true_valid)
    
    # Correlation
    if len(pred_valid) > 1:
        cor = np.corrcoef(pred_valid, true_valid)[0, 1]
        cor = cor if not np.isnan(cor) else 0.0
    else:
        cor = 0.0
    
    return {
        'rmse': rmse, 'bacc': bacc, 'ssim': ssim,
        'mse': mse, 'mae': mae, 'cor': cor
    }

def update(now, save, model, best, metric, model_name, seq_output, epoch):
    """
    🧊 새로운 평가 지표에 맞게 수정된 update 함수
    - RMSE, BACC, SSIM, MSE, MAE, COR 순서
    """
    folder_path = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 새로운 평가 지표 순서: [rmse, bacc, ssim, mse, mae, cor]
    if best[0] > metric[0]: best[0] = metric[0]  # RMSE (낮을수록 좋음)
    if best[1] < metric[1]: best[1] = metric[1]  # BACC (높을수록 좋음)
    if best[2] < metric[2]: best[2] = metric[2]  # SSIM (높을수록 좋음)
    if best[3] > metric[3]: best[3] = metric[3]  # MSE (낮을수록 좋음)
    if best[4] > metric[4]: best[4] = metric[4]  # MAE (낮을수록 좋음)
    if best[5] < metric[5]: best[5] = metric[5]  # COR (높을수록 좋음)

    metric = np.array([best[0], best[1], best[2], best[3], best[4], best[5], epoch])
    np.save(folder_path + f'/{model_name}_{seq_output}.npy', metric)

    return best

# ============================
# 기존 시계열 플롯 (유지)
# ============================

def plot(pred, true, model_name, seq_output, now):
    """기존 시계열 플롯 (유지)"""
    folder_path = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    os.makedirs(folder_path, exist_ok=True)
    
    true = true.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    
    idx = [0, pred.shape[-1]*1//4, pred.shape[-1]*1//2, pred.shape[-1]*3//4]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    titles = ["0", "1/4", "1/2", "3/4"]
    for i, ax in enumerate(axs.flat):
        ax.plot(pred[int(pred.shape[0]*1/2), :, idx[i]], label="prediction", color="r")
        ax.plot(true[int(true.shape[0]*1/2), :, idx[i]], label="actual", color="b")
        ax.set_title(titles[i])
    plt.tight_layout()
    plt.legend()
    plt.title('Cluster Time flow')
    plt.savefig(folder_path+f"/{model_name}_cluster.png")
    plt.close()
    
    pred = np.sum(pred, axis=(-1))
    true = np.sum(true, axis=(-1))
    
    idx = [0, pred.shape[0]*1//4, pred.shape[0]*1//2, pred.shape[0]*3//4]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    titles = ["0", "1/4", "1/2", "3/4"]
    for i, ax in enumerate(axs.flat):
        ax.plot(pred[idx[i]], label="prediction", color="r")
        ax.plot(true[idx[i]], label="actual", color="b")
        ax.set_title(titles[i])
    plt.tight_layout()
    plt.legend()
    plt.title('All Cluster Time flow')
    plt.savefig(folder_path+f"/{model_name}.png")
    plt.close()

# ============================
# 🧊 해빙 예측 전용 직관적 시각화 함수들
# ============================

def plot_ice_prediction_dashboard(pred_maps, true_maps, land_mask, 
                                 model_name, seq_output, now, 
                                 lead_days=[1, 7, 14], sample_idx=0):
    """
    🧊 해빙 예측 성능 대시보드 (직관적 시각화)
    
    주요 특징:
    1. 성능 지표를 한눈에 볼 수 있는 대시보드
    2. 해빙 경계선 비교 (15% SIC)
    3. 지역별 오차 분석
    4. 시간별 성능 변화
    """
    folder_path = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    os.makedirs(folder_path, exist_ok=True)
    
    if torch.is_tensor(pred_maps):
        pred_maps = pred_maps.detach().cpu().numpy()
        true_maps = true_maps.detach().cpu().numpy()
    
    # 성능 지표 계산
    metrics = metric_ice_specific(pred_maps, true_maps, land_mask)
    
    # NSIDC 스타일 컬러맵
    colors = ['#08519c', '#2171b5', '#4292c6', '#6baed6', 
              '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff']
    cmap = ListedColormap(colors)
    cmap.set_bad(color='lightgray')
    
    # Lead days 선택
    lead_indices = [d - 1 for d in lead_days if d <= pred_maps.shape[-1]]
    n_leads = len(lead_indices)
    
    # 대시보드 생성: 2x3 레이아웃
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. 성능 지표 요약 (상단)
    ax_metrics = fig.add_subplot(gs[0, :2])
    ax_metrics.axis('off')
    
    # 성능 지표 텍스트
    metrics_text = f"""
    🧊 해빙 예측 성능 대시보드
    
    📊 주요 지표:
    • RMSE: {metrics['rmse']:.3f}% (낮을수록 좋음)
    • BACC: {metrics['bacc']:.3f} (높을수록 좋음) 
    • SSIM: {metrics['ssim']:.3f} (높을수록 좋음)
    
    📈 추가 지표:
    • MSE: {metrics['mse']:.3f} | MAE: {metrics['mae']:.3f}% | COR: {metrics['cor']:.3f}
    
    🎯 SIFNet 목표: MAE 4.69% | BACC 95.16% | SSIM 95.13%
    """
    
    ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # 2. 성능 지표 막대 그래프 (상단 오른쪽)
    ax_bar = fig.add_subplot(gs[0, 2:])
    metrics_names = ['RMSE', 'BACC', 'SSIM', 'MSE', 'MAE', 'COR']
    metrics_values = [metrics['rmse'], metrics['bacc'], metrics['ssim'], 
                     metrics['mse'], metrics['mae'], metrics['cor']]
    
    # 정규화 (0-1 범위)
    normalized_values = []
    for i, val in enumerate(metrics_values):
        if i in [1, 2, 5]:  # BACC, SSIM, COR (높을수록 좋음)
            normalized_values.append(val)
        else:  # RMSE, MSE, MAE (낮을수록 좋음)
            normalized_values.append(1.0 - min(val/50.0, 1.0))  # 50% 기준으로 정규화
    
    bars = ax_bar.bar(metrics_names, normalized_values, 
                     color=['red' if i in [0, 3, 4] else 'green' for i in range(6)])
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel('정규화된 성능 (1.0 = 최고)')
    ax_bar.set_title('성능 지표 비교', fontweight='bold')
    
    # 값 표시
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. 해빙 지도 비교 (중간)
    for i, lead_idx in enumerate(lead_indices):
        pred = pred_maps[sample_idx, :, :, lead_idx]
        true = true_maps[sample_idx, :, :, lead_idx]
        
        pred_masked = np.where(land_mask > 0, pred, np.nan)
        true_masked = np.where(land_mask > 0, true, np.nan)
        error = pred_masked - true_masked
        
        # 예측 지도
        ax_pred = fig.add_subplot(gs[1, i*2])
        im_pred = ax_pred.imshow(pred_masked, cmap=cmap, vmin=0, vmax=100, 
                                interpolation='nearest', origin='lower')
        ax_pred.set_title(f'예측 (Day +{lead_days[i]})', fontweight='bold')
        ax_pred.axis('off')
        ax_pred.contour(pred_masked, levels=[15], colors='red', 
                       linewidths=2, linestyles='--', alpha=0.8)
        
        # 실제 지도
        ax_true = fig.add_subplot(gs[1, i*2+1])
        im_true = ax_true.imshow(true_masked, cmap=cmap, vmin=0, vmax=100, 
                                interpolation='nearest', origin='lower')
        ax_true.set_title(f'실제 (Day +{lead_days[i]})', fontweight='bold')
        ax_true.axis('off')
        ax_true.contour(true_masked, levels=[15], colors='red', 
                       linewidths=2, linestyles='--', alpha=0.8)
        
        # Colorbar
        if i == 0:
            cbar = plt.colorbar(im_pred, ax=[ax_pred, ax_true], 
                              fraction=0.046, pad=0.04)
            cbar.set_label('해빙 농도 (%)', fontsize=10)
    
    # 4. 오차 분석 (하단)
    ax_error = fig.add_subplot(gs[2, :2])
    
    # 모든 lead days의 오차 히스토그램
    all_errors = []
    for i in range(pred_maps.shape[0]):
        for j in range(pred_maps.shape[-1]):
            pred = pred_maps[i, :, :, j]
            true = true_maps[i, :, :, j]
            error = pred - true
            error_masked = error[land_mask > 0]
            all_errors.extend(error_masked[~np.isnan(error_masked)])
    
    all_errors = np.array(all_errors)
    ax_error.hist(all_errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax_error.axvline(0, color='red', linestyle='--', linewidth=2, label='완벽한 예측')
    ax_error.set_xlabel('예측 오차 (%)', fontsize=12)
    ax_error.set_ylabel('빈도', fontsize=12)
    ax_error.set_title('오차 분포', fontweight='bold')
    ax_error.legend()
    ax_error.grid(alpha=0.3)
    
    # 통계 정보
    error_stats = f'평균: {np.mean(all_errors):.2f}%\n표준편차: {np.std(all_errors):.2f}%\nMAE: {np.mean(np.abs(all_errors)):.2f}%'
    ax_error.text(0.02, 0.98, error_stats, transform=ax_error.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 5. 지역별 성능 (하단 오른쪽)
    ax_region = fig.add_subplot(gs[2, 2:])
    
    # 위도별 평균 오차
    mean_error = np.mean(pred_maps - true_maps, axis=(0, -1))
    mean_error = np.where(land_mask > 0, mean_error, np.nan)
    
    im_region = ax_region.imshow(mean_error, cmap='RdBu_r', vmin=-20, vmax=20,
                                interpolation='nearest', origin='lower')
    ax_region.set_title('지역별 평균 오차', fontweight='bold')
    ax_region.axis('off')
    
    cbar_region = plt.colorbar(im_region, ax=ax_region, fraction=0.046, pad=0.04)
    cbar_region.set_label('평균 오차 (%)', fontsize=10)
    
    # 해빙 경계선 표시
    mean_ice = np.mean(true_maps, axis=(0, -1))
    mean_ice = np.where(land_mask > 0, mean_ice, np.nan)
    ax_region.contour(mean_ice, levels=[15], colors='black', 
                     linewidths=1, linestyles='-', alpha=0.7)
    
    plt.suptitle(f'🧊 {model_name} - 해빙 예측 성능 대시보드', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(folder_path + f'/{model_name}_dashboard.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_ice_edge_comparison(pred_maps, true_maps, land_mask,
                            model_name, seq_output, now,
                            lead_days=[1, 7, 14], sample_idx=0):
    """
    🧊 해빙 경계선 비교 시각화 (15% SIC 기준)
    - 해빙/무빙 경계선이 얼마나 정확하게 예측되었는지 직관적으로 보여줌
    """
    folder_path = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    os.makedirs(folder_path, exist_ok=True)
    
    if torch.is_tensor(pred_maps):
        pred_maps = pred_maps.detach().cpu().numpy()
        true_maps = true_maps.detach().cpu().numpy()
    
    # Lead days 선택
    lead_indices = [d - 1 for d in lead_days if d <= pred_maps.shape[-1]]
    n_leads = len(lead_indices)
    
    fig, axes = plt.subplots(2, n_leads, figsize=(4*n_leads, 8))
    if n_leads == 1:
        axes = axes.reshape(2, -1)
    
    for i, lead_idx in enumerate(lead_indices):
        pred = pred_maps[sample_idx, :, :, lead_idx]
        true = true_maps[sample_idx, :, :, lead_idx]
        
        pred_masked = np.where(land_mask > 0, pred, np.nan)
        true_masked = np.where(land_mask > 0, true, np.nan)
        
        # 상단: 예측 + 실제 경계선
        ax1 = axes[0, i]
        im1 = ax1.imshow(pred_masked, cmap='Blues', vmin=0, vmax=100, 
                        interpolation='nearest', origin='lower', alpha=0.7)
        ax1.contour(pred_masked, levels=[15], colors='red', 
                   linewidths=3, linestyles='-', label='예측 경계선')
        ax1.contour(true_masked, levels=[15], colors='yellow', 
                   linewidths=2, linestyles='--', label='실제 경계선')
        ax1.set_title(f'Day +{lead_days[i]} - 경계선 비교', fontweight='bold')
        ax1.legend()
        ax1.axis('off')
        
        # 하단: 오차 맵 + 경계선
        ax2 = axes[1, i]
        error = pred_masked - true_masked
        im2 = ax2.imshow(error, cmap='RdBu_r', vmin=-30, vmax=30,
                        interpolation='nearest', origin='lower')
        ax2.contour(pred_masked, levels=[15], colors='red', 
                   linewidths=2, linestyles='-', alpha=0.8)
        ax2.contour(true_masked, levels=[15], colors='yellow', 
                   linewidths=2, linestyles='--', alpha=0.8)
        ax2.set_title(f'Day +{lead_days[i]} - 오차 분석', fontweight='bold')
        ax2.axis('off')
        
        # Colorbar
        if i == 0:
            cbar1 = plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            cbar1.set_label('해빙 농도 (%)')
            cbar2 = plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            cbar2.set_label('오차 (%)')
    
    plt.suptitle('🧊 해빙 경계선 예측 정확도 분석 (15% SIC 기준)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(folder_path + f'/{model_name}_ice_edge.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_evolution(pred_maps, true_maps, land_mask,
                              model_name, seq_output, now):
    """
    🧊 시간에 따른 성능 변화 시각화
    - 각 lead day별로 성능이 어떻게 변하는지 보여줌
    """
    folder_path = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    os.makedirs(folder_path, exist_ok=True)
    
    if torch.is_tensor(pred_maps):
        pred_maps = pred_maps.detach().cpu().numpy()
        true_maps = true_maps.detach().cpu().numpy()
    
    # 각 lead day별 성능 계산
    lead_days = list(range(1, pred_maps.shape[-1] + 1))
    rmse_values = []
    bacc_values = []
    ssim_values = []
    mae_values = []
    
    for i in range(pred_maps.shape[-1]):
        pred_single = pred_maps[:, :, :, i:i+1]
        true_single = true_maps[:, :, :, i:i+1]
        
        metrics = metric_ice_specific(pred_single, true_single, land_mask)
        rmse_values.append(metrics['rmse'])
        bacc_values.append(metrics['bacc'])
        ssim_values.append(metrics['ssim'])
        mae_values.append(metrics['mae'])
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE
    axes[0, 0].plot(lead_days, rmse_values, 'o-', color='red', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Lead Time (days)')
    axes[0, 0].set_ylabel('RMSE (%)')
    axes[0, 0].set_title('RMSE vs Lead Time', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # BACC
    axes[0, 1].plot(lead_days, bacc_values, 'o-', color='green', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Lead Time (days)')
    axes[0, 1].set_ylabel('BACC')
    axes[0, 1].set_title('Binary Accuracy vs Lead Time', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # SSIM
    axes[1, 0].plot(lead_days, ssim_values, 'o-', color='blue', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Lead Time (days)')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].set_title('Structural Similarity vs Lead Time', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # MAE
    axes[1, 1].plot(lead_days, mae_values, 'o-', color='orange', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Lead Time (days)')
    axes[1, 1].set_ylabel('MAE (%)')
    axes[1, 1].set_title('Mean Absolute Error vs Lead Time', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # SIFNet 목표선 표시
    axes[0, 1].axhline(y=0.9516, color='red', linestyle='--', alpha=0.7, label='SIFNet BACC')
    axes[1, 0].axhline(y=0.9513, color='red', linestyle='--', alpha=0.7, label='SIFNet SSIM')
    axes[1, 1].axhline(y=4.69, color='red', linestyle='--', alpha=0.7, label='SIFNet MAE')
    
    axes[0, 1].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()
    
    plt.suptitle('🧊 해빙 예측 성능의 시간적 변화', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(folder_path + f'/{model_name}_performance_evolution.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

# ============================
# 기존 시각화 함수들 (유지)
# ============================

def plot_spatial_comparison(pred_maps, true_maps, land_mask, 
                           model_name, seq_output, now, 
                           lead_days=[1, 7, 14], sample_idx=0):
    """
    ✅ 해빙 농도 공간 지도 시각화 (논문용)
    
    Parameters:
    -----------
    pred_maps: [N, H, W, C] - 예측 지도
    true_maps: [N, H, W, C] - 실제 지도
    land_mask: [H, W] - 육지 마스크
    lead_days: 시각화할 lead time (일)
    sample_idx: 샘플 인덱스
    """
    folder_path = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    os.makedirs(folder_path, exist_ok=True)
    
    # NumPy 변환
    if torch.is_tensor(pred_maps):
        pred_maps = pred_maps.detach().cpu().numpy()
        true_maps = true_maps.detach().cpu().numpy()
    
    # ✅ NSIDC 스타일 컬러맵 (0% = 진한 파랑, 100% = 흰색)
    colors = ['#08519c', '#2171b5', '#4292c6', '#6baed6', 
              '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff']
    cmap = ListedColormap(colors)
    cmap.set_bad(color='lightgray')  # NaN (육지) = 회색
    
    # Lead days 선택
    lead_indices = [d - 1 for d in lead_days if d <= pred_maps.shape[-1]]
    n_leads = len(lead_indices)
    
    # 그림 생성: [행 = lead days, 열 = Pred/True/Error]
    fig, axes = plt.subplots(n_leads, 3, figsize=(15, 5*n_leads), 
                             constrained_layout=True)
    
    if n_leads == 1:
        axes = axes.reshape(1, -1)
    
    for i, lead_idx in enumerate(lead_indices):
        pred = pred_maps[sample_idx, :, :, lead_idx]
        true = true_maps[sample_idx, :, :, lead_idx]
        
        # 육지 마스킹
        pred_masked = np.where(land_mask > 0, pred, np.nan)
        true_masked = np.where(land_mask > 0, true, np.nan)
        error = pred_masked - true_masked
        
        # 1열: 예측
        im1 = axes[i, 0].imshow(pred_masked, cmap=cmap, vmin=0, vmax=100, 
                                interpolation='nearest', origin='lower')
        axes[i, 0].set_title(f'Prediction (Day +{lead_days[i]})', 
                            fontsize=14, fontweight='bold')
        axes[i, 0].axis('off')
        
        # 해빙 경계선 (15% SIC)
        axes[i, 0].contour(pred_masked, levels=[15], colors='red', 
                          linewidths=1.5, linestyles='--')
        
        # 2열: 실제
        im2 = axes[i, 1].imshow(true_masked, cmap=cmap, vmin=0, vmax=100, 
                                interpolation='nearest', origin='lower')
        axes[i, 1].set_title(f'Ground Truth (Day +{lead_days[i]})', 
                            fontsize=14, fontweight='bold')
        axes[i, 1].axis('off')
        axes[i, 1].contour(true_masked, levels=[15], colors='red', 
                          linewidths=1.5, linestyles='--')
        
        # 3열: 오차 (Difference)
        im3 = axes[i, 2].imshow(error, cmap='RdBu_r', vmin=-50, vmax=50, 
                                interpolation='nearest', origin='lower')
        axes[i, 2].set_title(f'Prediction Error (Day +{lead_days[i]})', 
                            fontsize=14, fontweight='bold')
        axes[i, 2].axis('off')
        
        # Colorbar
        cbar1 = plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
        cbar1.set_label('SIC (%)', fontsize=12)
        
        cbar2 = plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
        cbar2.set_label('SIC (%)', fontsize=12)
        
        cbar3 = plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
        cbar3.set_label('Error (%)', fontsize=12)
    
    # 범례 추가
    legend_elements = [
        mpatches.Patch(facecolor='lightgray', label='Land'),
        mpatches.Patch(facecolor='#08519c', label='High Ice (80-100%)'),
        mpatches.Patch(facecolor='#6baed6', label='Medium Ice (40-80%)'),
        mpatches.Patch(facecolor='#f7fbff', label='Low Ice (0-40%)'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='15% SIC Contour')
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
              ncol=5, fontsize=11, bbox_to_anchor=(0.5, 0.98))
    
    plt.savefig(folder_path + f'/{model_name}_spatial_map.png', 
                dpi=300, bbox_inches='tight')
    plt.show()  # Spyder에서 바로 보기

def plot_spatial_temporal(pred_maps, true_maps, land_mask,
                         model_name, seq_output, now,
                         sample_idx=0, n_timesteps=7):
    """
    ✅ 시간에 따른 해빙 변화 애니메이션 스타일 (Static)
    
    Day 1, 3, 5, 7, 14, 21 등을 한 눈에 비교
    """
    folder_path = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    os.makedirs(folder_path, exist_ok=True)
    
    if torch.is_tensor(pred_maps):
        pred_maps = pred_maps.detach().cpu().numpy()
        true_maps = true_maps.detach().cpu().numpy()
    
    # 시각화할 시점 선택
    total_days = pred_maps.shape[-1]
    if total_days <= 7:
        day_indices = list(range(total_days))
    else:
        day_indices = [0, 2, 4, 6, 13, 20]  # Day 1, 3, 5, 7, 14, 21
        day_indices = [d for d in day_indices if d < total_days]
    
    n_days = len(day_indices)
    
    # 컬러맵
    colors = ['#08519c', '#2171b5', '#4292c6', '#6baed6', 
              '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff']
    cmap = ListedColormap(colors)
    cmap.set_bad(color='lightgray')
    
    # 그림 생성
    fig, axes = plt.subplots(2, n_days, figsize=(3*n_days, 6), 
                             constrained_layout=True)
    
    for i, day_idx in enumerate(day_indices):
        pred = pred_maps[sample_idx, :, :, day_idx]
        true = true_maps[sample_idx, :, :, day_idx]
        
        pred_masked = np.where(land_mask > 0, pred, np.nan)
        true_masked = np.where(land_mask > 0, true, np.nan)
        
        # 상단: 예측
        im1 = axes[0, i].imshow(pred_masked, cmap=cmap, vmin=0, vmax=100,
                                interpolation='nearest', origin='lower')
        axes[0, i].set_title(f'Pred Day +{day_idx+1}', fontsize=12)
        axes[0, i].axis('off')
        axes[0, i].contour(pred_masked, levels=[15], colors='red', 
                          linewidths=1, linestyles='--')
        
        # 하단: 실제
        im2 = axes[1, i].imshow(true_masked, cmap=cmap, vmin=0, vmax=100,
                                interpolation='nearest', origin='lower')
        axes[1, i].set_title(f'True Day +{day_idx+1}', fontsize=12)
        axes[1, i].axis('off')
        axes[1, i].contour(true_masked, levels=[15], colors='red', 
                          linewidths=1, linestyles='--')
    
    # Colorbar (마지막 축에만)
    cbar = plt.colorbar(im1, ax=axes[:, -1], fraction=0.046, pad=0.04)
    cbar.set_label('SIC (%)', fontsize=12)
    
    plt.savefig(folder_path + f'/{model_name}_temporal_evolution.png', 
                dpi=300, bbox_inches='tight')
    plt.show()  # Spyder에서 바로 보기

def plot_error_statistics(pred_maps, true_maps, land_mask,
                         model_name, seq_output, now):
    """
    ✅ 오차 통계 시각화
    - 히스토그램: 오차 분포
    - 공간 오차 맵: 어느 지역에서 오차가 큰지
    """
    folder_path = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    os.makedirs(folder_path, exist_ok=True)
    
    if torch.is_tensor(pred_maps):
        pred_maps = pred_maps.detach().cpu().numpy()
        true_maps = true_maps.detach().cpu().numpy()
    
    # 오차 계산 (육지 제외)
    errors = []
    for i in range(pred_maps.shape[0]):
        for j in range(pred_maps.shape[-1]):
            pred = pred_maps[i, :, :, j]
            true = true_maps[i, :, :, j]
            error = pred - true
            error_masked = error[land_mask > 0]
            errors.extend(error_masked.flatten())
    
    errors = np.array(errors)
    
    # 공간 평균 오차
    mean_error = np.mean(pred_maps - true_maps, axis=(0, -1))
    mean_error = np.where(land_mask > 0, mean_error, np.nan)
    
    # 그림 생성
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 오차 히스토그램
    axes[0].hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error (%)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 통계 정보 추가
    textstr = f'Mean: {np.mean(errors):.2f}%\nStd: {np.std(errors):.2f}%\nMAE: {np.mean(np.abs(errors)):.2f}%'
    axes[0].text(0.02, 0.98, textstr, transform=axes[0].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. 공간 평균 오차 맵
    im = axes[1].imshow(mean_error, cmap='RdBu_r', vmin=-20, vmax=20,
                       interpolation='nearest', origin='lower')
    axes[1].set_title('Spatial Mean Error', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Mean Error (%)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(folder_path + f'/{model_name}_error_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()  # Spyder에서 바로 보기

# ============================
# 기존 함수들 (유지)
# ============================

def plot_uni(pred, true):
    """기존 함수 유지"""
    true = true.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    
    a = np.sum(true, axis=(1))
    p = np.sum(pred, axis=(1))
    
    idx = [0, p.shape[0] * 1 // 4, p.shape[0] * 1 // 2, p.shape[0] * 3 // 4]
    fig, axs = plt.subplots(2, 4, figsize=(12, 8))
    for i in range(4):
        im1 = axs[0, i].imshow(p[idx[i]], cmap="viridis", aspect="auto")
        im2 = axs[1, i].imshow(a[idx[i]], cmap="viridis", aspect="auto")
        axs[0, i].set_title(f"Prediction {i}/4")
        axs[1, i].set_title(f"Actual {i}/4")
        fig.colorbar(im1, ax=axs[0, i], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axs[1, i], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    
    pred = np.sum(pred, axis=(-1, -2))
    true = np.sum(true, axis=(-1, -2))
    
    idx = [0, pred.shape[0]*1//4, pred.shape[0]*1//2, pred.shape[0]*3//4]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    titles = ["0", "1/4", "1/2", "3/4"]
    for i, ax in enumerate(axs.flat):
        ax.plot(pred[idx[i]], label="prediction", color="r")
        ax.plot(true[idx[i]], label="actual", color="b")
        ax.set_title(titles[i])
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_results(out, act, model_name):
    """기존 함수 유지"""
    plt.figure(figsize=(8, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    
    plt.plot(out, label='Prediction', linestyle='-', color='#E74C3C', linewidth=2.0, alpha=0.8)
    plt.plot(act, label='Ground Truth', linestyle='--', color='#3498DB', linewidth=2.0, alpha=0.8)

    plt.title(f'{model_name}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Time Step', fontsize=12, labelpad=10)
    plt.ylabel('Aggregated Value', fontsize=12, labelpad=10)
    plt.legend(fontsize=10, loc='upper right', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()

def find_files_with_prefix_and_suffix(folder_path, prefix, suffix='.npy'):
    """기존 함수 유지"""
    matching_files = []
    for root, _, files in os.walk(folder_path):
        folder_name = os.path.basename(root)
        if folder_name.startswith(prefix):
            for file in files:
                if file.endswith(suffix):
                    matching_files.append(os.path.join(root, file))
    return matching_files