# Metric_node.py (수정 버전)
# plot 함수를 해빙 시각화 스타일로 변경

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from skimage.metrics import structural_similarity
from torchmetrics import MeanAbsolutePercentageError
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import seaborn as sns
import os

# ============================
# 개별 지표 계산 함수들
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

# ============================
# Metric 집계 함수
# ============================

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

# ============================
# Best metric 업데이트 및 저장
# ============================

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
# 수정된 Plotting 함수
# ============================

def plot(pred, true, model_name, seq_output, now, land_mask=None):
    """
    해빙 예측 시각화 (수정 버전)
    - pred, true: [B, pred_len, N] 형태 (0~1 스케일)
    - land_mask: [H, W] 형태 (옵션, 없으면 자동 생성)
    """
    folder_path = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    os.makedirs(folder_path, exist_ok=True)
    
    print(f"[DEBUG] plot 함수 호출됨")
    print(f"  pred shape: {pred.shape}, true shape: {true.shape}")
    
    true = true.detach().cpu().numpy()  # [B, pred_len, N]
    pred = pred.detach().cpu().numpy()
    
    B, pred_len, N = pred.shape
    
    # ✅ 0~100 스케일로 변환 (중요!)
    true = true * 100.0
    pred = pred * 100.0
    
    print(f"  스케일 변환 후 - pred: min={pred.min():.2f}, max={pred.max():.2f}")
    print(f"  스케일 변환 후 - true: min={true.min():.2f}, max={true.max():.2f}")
    
    # 이미지 크기 추정
    H = int(np.sqrt(N * 224 / 152))
    W = int(N / H)
    
    print(f"  이미지 크기: {H} × {W}")
    
    # Reshape: [B, pred_len, N] → [B, pred_len, H, W]
    pred_img = pred.reshape(B, pred_len, H, W)
    true_img = true.reshape(B, pred_len, H, W)
    
    # Land mask 처리
    if land_mask is None:
        print(f"  land_mask 없음 → 자동 생성")
        all_data = true_img.reshape(-1, H, W)
        land_mask = (all_data.max(axis=0) == 0).astype(float)
        ocean_mask = 1 - land_mask
    else:
        print(f"  land_mask 제공됨: {land_mask.shape}")
        ocean_mask = land_mask
        land_mask = 1 - ocean_mask
    
    print(f"  육지 픽셀: {land_mask.sum():.0f}, 바다 픽셀: {ocean_mask.sum():.0f}")
    
    # ========================================
    # 시각화 1: 개별 시점 비교 (2x4 그리드)
    # ========================================
    sample_idx = B // 2
    time_indices = [0, pred_len // 4, pred_len // 2, pred_len * 3 // 4]
    
    print(f"  시각화 생성 중... (샘플 {sample_idx}, 시점 {time_indices})")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'{model_name} - Sea Ice Prediction Visualization', 
                 fontsize=16, fontweight='bold')
    
    for i, t_idx in enumerate(time_indices):
        # 상단: Ground Truth + Prediction 오버레이
        ax_overlay = axes[0, i]
        create_ice_overlay(ax_overlay, true_img[sample_idx, t_idx], 
                          pred_img[sample_idx, t_idx], 
                          land_mask, ocean_mask,
                          title=f't={t_idx+1} (Overlay)')
        
        # 하단: Error Map
        ax_error = axes[1, i]
        error = np.abs(true_img[sample_idx, t_idx] - pred_img[sample_idx, t_idx])
        create_error_map(ax_error, error, land_mask, ocean_mask,
                        title=f't={t_idx+1} (|Error|)')
    
    plt.tight_layout()
    save_path_1 = folder_path + f"/{model_name}_ice_visualization.png"
    plt.savefig(save_path_1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 저장: {save_path_1}")
    
    # ========================================
    # 시각화 2: 시계열 합산 (기존 스타일)
    # ========================================
    pred_sum = np.sum(pred, axis=(-1))
    true_sum = np.sum(true, axis=(-1))
    
    idx = [0, pred_sum.shape[0] * 1 // 4, pred_sum.shape[0] * 1 // 2, pred_sum.shape[0] * 3 // 4]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    titles = ["Sample 0", "Sample 1/4", "Sample 1/2", "Sample 3/4"]
    
    for i, ax in enumerate(axs.flat):
        ax.plot(pred_sum[idx[i]], label="Prediction", color="r", linewidth=2, alpha=0.8)
        ax.plot(true_sum[idx[i]], label="Actual", color="b", linewidth=2, alpha=0.8)
        ax.set_title(titles[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Aggregated SIC')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path_2 = folder_path + f"/{model_name}_timeseries.png"
    plt.savefig(save_path_2, dpi=150)
    plt.close()
    print(f"  ✓ 저장: {save_path_2}")
    
    print(f"✓ Visualizations saved to: {folder_path}\n")

def create_ice_overlay(ax, true_frame, pred_frame, land_mask, ocean_mask, title=''):
    """
    해빙 오버레이 시각화 생성
    - 바다(파랑) + 육지(회색) + 실제 해빙(흰색) + 예측 해빙(반투명 빨강)
    """
    H, W = true_frame.shape
    
    # RGBA 이미지 생성 [H, W, 4]
    base_layer = np.zeros((H, W, 4))
    
    # 1. 바다 레이어 (파란색)
    base_layer[ocean_mask > 0, 0] = 0.1  # R
    base_layer[ocean_mask > 0, 1] = 0.3  # G
    base_layer[ocean_mask > 0, 2] = 0.7  # B
    base_layer[ocean_mask > 0, 3] = 1.0  # Alpha
    
    # 2. 육지 레이어 (회색)
    base_layer[land_mask > 0, 0] = 0.5  # R
    base_layer[land_mask > 0, 1] = 0.5  # G
    base_layer[land_mask > 0, 2] = 0.5  # B
    base_layer[land_mask > 0, 3] = 1.0  # Alpha
    
    ax.imshow(base_layer, aspect='auto')
    
    # 3. 실제 해빙 (흰색, 0-100 스케일)
    ice_true = np.copy(true_frame)
    ice_true[land_mask > 0] = np.nan  # 육지는 투명
    
    # 흰색 colormap
    white_cmap = LinearSegmentedColormap.from_list(
        'white_ice', 
        [(1, 1, 1, 0), (1, 1, 1, 1)],
        N=256
    )
    
    ax.imshow(ice_true, cmap=white_cmap, vmin=0, vmax=100, 
              aspect='auto', interpolation='bilinear')
    
    # 4. 예측 해빙 (빨간색, 반투명)
    ice_pred = np.copy(pred_frame)
    ice_pred[land_mask > 0] = np.nan
    
    # 빨간색 colormap (투명도 조절)
    red_cmap = LinearSegmentedColormap.from_list(
        'red_ice',
        [(1, 0, 0, 0), (1, 0, 0, 0.6)],  # 투명 → 반투명 빨강
        N=256
    )
    
    ax.imshow(ice_pred, cmap=red_cmap, vmin=0, vmax=100,
              aspect='auto', interpolation='bilinear')
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # 범례 추가
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(1, 1, 1, 1), label='Actual Ice (White)'),
        Patch(facecolor=(1, 0, 0, 0.6), label='Predicted Ice (Red)'),
        Patch(facecolor=(0.1, 0.3, 0.7, 1), label='Ocean (Blue)'),
        Patch(facecolor=(0.5, 0.5, 0.5, 1), label='Land (Gray)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

def create_error_map(ax, error, land_mask, ocean_mask, title=''):
    """
    오차 맵 시각화
    """
    H, W = error.shape
    
    # 베이스 레이어
    base_layer = np.zeros((H, W, 4))
    base_layer[ocean_mask > 0] = [0.1, 0.3, 0.7, 1.0]  # 바다
    base_layer[land_mask > 0] = [0.5, 0.5, 0.5, 1.0]   # 육지
    
    ax.imshow(base_layer, aspect='auto')
    
    # 오차 레이어 (빨간색 그라데이션)
    error_display = np.copy(error)
    error_display[land_mask > 0] = np.nan
    
    # 오차 colormap (노랑 → 빨강)
    error_cmap = LinearSegmentedColormap.from_list(
        'error',
        [(1, 1, 0, 0), (1, 1, 0, 0.8), (1, 0, 0, 1)],  # 투명 → 노랑 → 빨강
        N=256
    )
    
    im = ax.imshow(error_display, cmap=error_cmap, vmin=0, vmax=50,
                   aspect='auto', interpolation='bilinear')
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='|Error|')

# ============================
# 기타 함수들 (그대로 유지)
# ============================

def plot_uni(pred, true):
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
    matching_files = []
    for root, _, files in os.walk(folder_path):
        folder_name = os.path.basename(root)
        if folder_name.startswith(prefix):
            for file in files:
                if file.endswith(suffix):
                    matching_files.append(os.path.join(root, file))
    return matching_files