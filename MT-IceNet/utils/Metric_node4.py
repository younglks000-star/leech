# Metric_node_final.py — Cursor + GPT 통합 안정형 버전
# ✅ 해빙 예측용 확장형 Metric & 시각화 모듈 (2025.10 업데이트)

from sklearn.metrics import mean_squared_error, mean_absolute_error, balanced_accuracy_score
import numpy as np
from skimage.metrics import structural_similarity
from torchmetrics import MeanAbsolutePercentageError
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import os
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# ==========================================================
# 🧊 내부 유틸 (안정성 향상)
# ==========================================================

def _to_numpy(x):
    """Tensor → Numpy 자동 변환"""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)

def _ensure_4d(x):
    """입력 배열을 [N, H, W, C] 형태로 보정"""
    x = _to_numpy(x)
    if x.ndim == 2:
        x = x[None, :, :, None]
    elif x.ndim == 3:
        x = x[:, :, :, None] if x.shape[-1] < 10 else x[None, :, :, :]
    return x

def _apply_land_mask(data, land_mask):
    """육지 마스크 적용: land_mask=1인 부분만 유지"""
    if land_mask is None:
        return np.nan_to_num(data)
    return np.where(land_mask[..., None] > 0, data, np.nan)

# ==========================================================
# 🧊 Metric Functions
# ==========================================================

def _mape(pred, target):
    return MeanAbsolutePercentageError()(torch.tensor(pred).flatten(),
                                         torch.tensor(target).flatten()).item()

def _mse(outputs, targets):
    return mean_squared_error(outputs.reshape(-1), targets.reshape(-1))

def _mae(outputs, targets):
    return mean_absolute_error(outputs.reshape(-1), targets.reshape(-1))

def _rmse(outputs, targets):
    return np.sqrt(mean_squared_error(outputs.reshape(-1), targets.reshape(-1)))

def _bacc(outputs, targets, threshold=15.0, land_mask=None):
    outputs, targets = _to_numpy(outputs), _to_numpy(targets)
    if land_mask is not None:
        mask = land_mask[..., None] > 0
        outputs, targets = outputs[mask], targets[mask]
    pred_bin = (outputs >= threshold).astype(int)
    true_bin = (targets >= threshold).astype(int)
    valid = ~(np.isnan(pred_bin) | np.isnan(true_bin))
    if valid.sum() == 0:
        return 0.0
    return balanced_accuracy_score(true_bin[valid], pred_bin[valid])

def _ssim_mean(pred_maps, true_maps, land_mask=None):
    """다중 시점 SSIM 평균"""
    pred_maps = _ensure_4d(pred_maps)
    true_maps = _ensure_4d(true_maps)
    if land_mask is not None:
        pred_maps = _apply_land_mask(pred_maps, land_mask)
        true_maps = _apply_land_mask(true_maps, land_mask)

    vals = []
    for n in range(pred_maps.shape[0]):
        for t in range(pred_maps.shape[-1]):
            p = pred_maps[n, :, :, t]
            g = true_maps[n, :, :, t]
            valid = ~(np.isnan(p) | np.isnan(g))
            if valid.sum() < 10:
                continue
            try:
                val = structural_similarity(
                    np.nan_to_num(p), np.nan_to_num(g),
                    data_range=100.0,
                    win_size=min(7, min(p.shape))
                )
                if not np.isnan(val):
                    vals.append(val)
            except:
                continue
    return np.mean(vals) if vals else 0.0

def CORR_uni(pred, true):
    pred, true = _to_numpy(pred), _to_numpy(true)
    pred_sum, true_sum = np.sum(pred, axis=-1), np.sum(true, axis=-1)
    vals = []
    for i in range(pred_sum.shape[0]):
        c = np.corrcoef(pred_sum[i, :], true_sum[i, :])[0, 1]
        if not np.isnan(c):
            vals.append(c)
    return np.mean(vals) if vals else 0.0

# ==========================================================
# 🧊 Metric Aggregator
# ==========================================================

def metric(outputs, actuals, n_features=None, normalize=False, land_mask=None):
    outputs, actuals = _to_numpy(outputs), _to_numpy(actuals)
    if normalize:
        m, s = np.mean(actuals), np.std(actuals)
        outputs, actuals = (outputs - m) / s, (actuals - m) / s

    rmse = _rmse(outputs, actuals)
    bacc = _bacc(outputs, actuals, threshold=15.0, land_mask=land_mask)
    ssim = _ssim_mean(outputs, actuals, land_mask)
    mse = _mse(outputs, actuals)
    mae = _mae(outputs, actuals)
    cor = CORR_uni(outputs, actuals)
    return [rmse, bacc, ssim, mse, mae, cor]

# ==========================================================
# ✅ Spatial & Temporal Visualization
# ==========================================================

def plot_spatial_comparison(pred_maps, true_maps, land_mask,
                           model_name, seq_output, now,
                           lead_days=[1, 7, 14], sample_idx=0):
    folder = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    os.makedirs(folder, exist_ok=True)

    pred_maps, true_maps = _ensure_4d(pred_maps), _ensure_4d(true_maps)

    cmap = ListedColormap(['#08519c','#2171b5','#4292c6','#6baed6',
                           '#9ecae1','#c6dbef','#deebf7','#f7fbff'])
    cmap.set_bad('lightgray')

    leads = [d-1 for d in lead_days if d <= pred_maps.shape[-1]]
    fig, axes = plt.subplots(len(leads), 3, figsize=(15, 5*len(leads)),
                             constrained_layout=True)
    if len(leads) == 1:
        axes = axes.reshape(1, -1)

    for i, li in enumerate(leads):
        p = np.where(land_mask>0, pred_maps[sample_idx,:,:,li], np.nan)
        g = np.where(land_mask>0, true_maps[sample_idx,:,:,li], np.nan)
        diff = p - g

        im1 = axes[i,0].imshow(p, cmap=cmap, vmin=0, vmax=100)
        axes[i,0].set_title(f'Prediction +{lead_days[i]}d')
        axes[i,0].axis('off')
        axes[i,0].contour(p, levels=[15], colors='red', lw=1.5, ls='--')

        im2 = axes[i,1].imshow(g, cmap=cmap, vmin=0, vmax=100)
        axes[i,1].set_title(f'Ground Truth +{lead_days[i]}d')
        axes[i,1].axis('off')
        axes[i,1].contour(g, levels=[15], colors='red', lw=1.5, ls='--')

        im3 = axes[i,2].imshow(diff, cmap='RdBu_r', vmin=-50, vmax=50)
        axes[i,2].set_title(f'Error +{lead_days[i]}d')
        axes[i,2].axis('off')

        for im, ax, lab in zip([im1,im2,im3], axes[i,:], ['SIC','SIC','Error']):
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(lab+" (%)")

    legend = [
        mpatches.Patch(fc='lightgray', label='Land'),
        mpatches.Patch(fc='#08519c', label='High Ice'),
        mpatches.Patch(fc='#6baed6', label='Medium Ice'),
        mpatches.Patch(fc='#f7fbff', label='Low Ice'),
        plt.Line2D([0],[0], color='red', lw=2, ls='--', label='15% Contour')
    ]
    fig.legend(handles=legend, loc='upper center', ncol=5, fontsize=11)
    plt.savefig(f'{folder}/{model_name}_spatial_map.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================================
# 🧊 결과 저장 및 관리
# ==========================================================

def update(now, save, model, best, metric, model_name, seq_output, epoch):
    folder = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    os.makedirs(folder, exist_ok=True)
    # [rmse, bacc, ssim, mse, mae, cor]
    if best[0] > metric[0]: best[0] = metric[0]
    if best[1] < metric[1]: best[1] = metric[1]
    if best[2] < metric[2]: best[2] = metric[2]
    if best[3] > metric[3]: best[3] = metric[3]
    if best[4] > metric[4]: best[4] = metric[4]
    if best[5] < metric[5]: best[5] = metric[5]
    np.save(f'{folder}/{model_name}_{seq_output}.npy',
            np.array(best+[epoch]))
    return best
