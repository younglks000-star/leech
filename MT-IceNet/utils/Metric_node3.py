# Metric_node.py - 해빙 공간 시각화 추가 버전
# ✅ 기존 코드 + 공간 지도 시각화 함수 추가

from sklearn.metrics import mean_squared_error, mean_absolute_error
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
# 기존 지표 계산 함수들
# ============================

def _mape(pred, target):
    return MeanAbsolutePercentageError()(pred.reshape(-1), target.reshape(-1)) 

def _mse(outputs, targets):
    return mean_squared_error(outputs.reshape(-1), targets.reshape(-1))

def _mae(outputs, targets):
    return mean_absolute_error(outputs.reshape(-1), targets.reshape(-1))

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
    outputs = outputs.detach().cpu().numpy()
    actuals = actuals.detach().cpu().numpy()

    if normalize:
        mean = np.mean(actuals)
        std = np.std(actuals)
        outputs = (outputs - mean) / std
        actuals = (actuals - mean) / std

    mse = _mse(outputs, actuals)
    mae = _mae(outputs, actuals)
    cor = CORR_uni(outputs, actuals)

    return [mse, mae, cor]

def update(now, save, model, best, metric, model_name, seq_output, epoch):
    folder_path = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if best[0] > metric[0]: best[0] = metric[0]
    if best[1] > metric[1]: best[1] = metric[1]
    if best[2] < metric[2]: best[2] = metric[2]

    metric = np.array([best[0], best[1], best[2], epoch])
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
# ✅ 새로운 공간 시각화 함수들
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