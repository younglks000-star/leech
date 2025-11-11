# Metric_node5.py
# ✅ Paper-accurate metrics (SSIM Eq.12, BACC = 1 - IIEE/AAGCR) + Minimal visualizations (2 kinds)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity
import torch
import os

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _to_numpy(x):
    """Accept torch.Tensor or np.ndarray and return np.ndarray (float32)."""
    try:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy().astype(np.float32)
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)

def _valid_mask(*arrs):
    """Return boolean mask where all arrays are finite."""
    mask = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        mask &= np.isfinite(a)
    return mask

# ─────────────────────────────────────────────────────────────────────────────
# Metrics (paper-accurate)
# ─────────────────────────────────────────────────────────────────────────────
def _mse(outputs, targets):
    return float(mean_squared_error(outputs.reshape(-1), targets.reshape(-1)))

def _mae(outputs, targets):
    return float(np.mean(np.abs(outputs.reshape(-1) - targets.reshape(-1))))

def _rmse(outputs, targets):
    return float(np.sqrt(_mse(outputs, targets)))

def _corr_uni(outputs, targets):
    """
    Aggregate over features → per-sample vectors → mean Pearson r.
    outputs/targets: [N, ..., T] (T = lead)
    """
    # sum over spatial dims if present
    out = np.sum(outputs, axis=tuple(range(1, outputs.ndim-1)))
    tru = np.sum(targets, axis=tuple(range(1, targets.ndim-1)))
    corrs = []
    for i in range(out.shape[0]):
        r = np.corrcoef(out[i], tru[i])[0, 1]
        if np.isfinite(r):
            corrs.append(r)
    return float(np.mean(corrs)) if corrs else 0.0

def _bacc_paper(outputs, targets, threshold=15.0):
    """
    BACC per paper: 1 - IIEE/AAGCR = |A ∩ B| / |A ∪ B| (IoU of ice (>15%)).
    Works on ANY shape; computed over all elements. Returns 0..1.
    """
    yhat = (outputs.reshape(-1) >= threshold)
    y    = (targets.reshape(-1) >= threshold)

    valid = _valid_mask(outputs.reshape(-1), targets.reshape(-1))
    if valid.sum() == 0:
        return 1.0  # degenerate equal-empties → perfect

    A = yhat[valid]
    B = y[valid]
    union = np.count_nonzero(A | B)
    inter = np.count_nonzero(A & B)

    if union == 0:  # both empty everywhere
        return 1.0
    return float(inter / union)

def _ssim_maps(outputs, targets, assume_percent=True):
    """
    SSIM (Eq.12) via skimage (luminance*contrast*structure).
    - If inputs are 0..100 (%), data_range=100.
    - Accepts [N,H,W,C] or any -> will try to reshape feature dim to square.
    """
    out = outputs
    tru = targets

    # Try to map to [N,H,W,C]
    if out.ndim == 3:  # [N, F, C] or [N, H*W, C]
        N, F, C = out.shape
        h = int(np.floor(np.sqrt(F)))
        w = int(np.ceil(F / h))
        if h * w != F:
            pad = h * w - F
            out = np.pad(out, ((0,0),(0,max(0,pad)),(0,0)), mode='constant')
            tru = np.pad(tru, ((0,0),(0,max(0,pad)),(0,0)), mode='constant')
        out = out.reshape(N, h, w, C)
        tru = tru.reshape(N, h, w, C)

    assert out.ndim == 4, "SSIM expects [N,H,W,C] after reshape."

    vals = []
    data_range = 100.0 if assume_percent else float(np.nanmax(tru) - np.nanmin(tru) + 1e-6)

    for i in range(out.shape[0]):
        for k in range(out.shape[-1]):
            a = out[i, :, :, k]
            b = tru[i, :, :, k]
            v = _valid_mask(a, b)
            if v.sum() < 25:
                continue
            a = np.where(v, a, 0.0)
            b = np.where(v, b, 0.0)
            win = min(7, int(min(a.shape)))
            if win % 2 == 0:
                win = max(3, win-1)
            try:
                s = structural_similarity(a, b, data_range=data_range, win_size=win)
                if np.isfinite(s):
                    vals.append(s)
            except Exception:
                pass

    return float(np.mean(vals)) if vals else 0.0

def metric(outputs, actuals, n_features, normalize=False):
    """
    Interface kept for backward-compatibility with your training loops.
    inputs:
      outputs, actuals: torch.Tensor or np.ndarray (e.g., [N,C,HW] or [N,HW,C]).
    returns:
      [rmse, bacc, ssim, mse, mae, cor]
    """
    out = _to_numpy(outputs)
    tru = _to_numpy(actuals)

    # Try to standardize to [N, F, C]
    if out.ndim == 3:
        # Accept [N, C, F] too → make [N, F, C]
        if out.shape[1] < out.shape[2]:  # [N,C,F]
            out = np.transpose(out, (0, 2, 1))
            tru = np.transpose(tru, (0, 2, 1))
    else:
        # flatten features if needed
        out = out.reshape(out.shape[0], -1, out.shape[-1])
        tru = tru.reshape(tru.shape[0], -1, tru.shape[-1])

    if normalize:
        m, s = np.nanmean(tru), np.nanstd(tru) + 1e-6
        out = (out - m) / s
        tru = (tru - m) / s

    # Elementwise metrics use all elements
    rmse = _rmse(out, tru)
    bacc = _bacc_paper(out, tru, threshold=15.0)
    ssim = _ssim_maps(out, tru)
    mse  = _mse(out, tru)
    mae  = _mae(out, tru)
    cor  = _corr_uni(out, tru)

    return [rmse, bacc, ssim, mse, mae, cor]

def update(now, save, model, best, met, model_name, seq_output, epoch):
    """
    Keep 'best' as [rmse, bacc, ssim, mse, mae, cor] (same order).
    """
    folder_path = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    os.makedirs(folder_path, exist_ok=True)

    if best[0] > met[0]: best[0] = met[0]  # RMSE↓
    if best[1] < met[1]: best[1] = met[1]  # BACC↑
    if best[2] < met[2]: best[2] = met[2]  # SSIM↑
    if best[3] > met[3]: best[3] = met[3]  # MSE↓
    if best[4] > met[4]: best[4] = met[4]  # MAE↓
    if best[5] < met[5]: best[5] = met[5]  # COR↑

    arr = np.array([best[0], best[1], best[2], best[3], best[4], best[5], epoch], dtype=np.float32)
    np.save(os.path.join(folder_path, f'{model_name}_{seq_output}.npy'), arr)
    return best

# ─────────────────────────────────────────────────────────────────────────────
# VIS #1: Skill vs Lead curves (RMSE, BACC, SSIM)
# ─────────────────────────────────────────────────────────────────────────────
def plot_skill_vs_lead(pred_maps, true_maps, land_mask=None, title="Skill vs Lead"):
    """
    pred_maps, true_maps: [N,H,W,C] in % (0..100)
    land_mask: [H,W] 1=ocean, 0=land (optional)
    """
    P = _to_numpy(pred_maps)
    T = _to_numpy(true_maps)

    if land_mask is not None:
        LM = np.asarray(land_mask, dtype=np.float32)
        P = P * LM[..., None]
        T = T * LM[..., None]

    C = P.shape[-1]
    leads = np.arange(1, C+1)

    rmse_list, bacc_list, ssim_list = [], [], []
    for k in range(C):
        p = P[:, :, :, k]
        t = T[:, :, :, k]
        v = _valid_mask(p, t)
        if v.sum() == 0:
            rmse_list.append(0.0); bacc_list.append(0.0); ssim_list.append(0.0); continue
        rmse_list.append(_rmse(p[v], t[v]))
        bacc_list.append(_bacc_paper(p[v], t[v], threshold=15.0))
        # SSIM per lead uses maps
        ssim_list.append(_ssim_maps(P[:, :, :, k:k+1], T[:, :, :, k:k+1]))

    fig, axes = plt.subplots(1, 3, figsize=(14,4), constrained_layout=True)

    axes[0].plot(leads, rmse_list, 'o-', linewidth=2)
    axes[0].set_title('RMSE vs Lead'); axes[0].set_xlabel('Lead (day)'); axes[0].set_ylabel('RMSE (%)'); axes[0].grid(alpha=0.3)

    axes[1].plot(leads, bacc_list, 'o-', linewidth=2)
    axes[1].set_title('BACC (1 - IIEE/AAGCR)'); axes[1].set_xlabel('Lead (day)'); axes[1].set_ylim(0,1); axes[1].grid(alpha=0.3)

    axes[2].plot(leads, ssim_list, 'o-', linewidth=2)
    axes[2].set_title('SSIM vs Lead'); axes[2].set_xlabel('Lead (day)'); axes[2].set_ylim(0,1); axes[2].grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# VIS #2: Error Overlay (white=good, red=bad) on blue ocean & grey land
# ─────────────────────────────────────────────────────────────────────────────
def plot_error_overlay(pred_maps, true_maps, land_mask,
                       lead_days=(1,7,14,21), sample_idx=0,
                       vmax=30.0, title='Absolute Error Overlay'):
    """
    - Base: land=grey, ocean=blue
    - Overlay: |pred-true|  (white → red). Big error = strong red.
    """
    P = _to_numpy(pred_maps)
    T = _to_numpy(true_maps)
    LM = np.asarray(land_mask, dtype=np.float32) if land_mask is not None else np.ones(P.shape[1:3], np.float32)

    leads = [d for d in lead_days if 1 <= d <= P.shape[-1]]
    n = len(leads)
    if n == 0: return

    # Background colors
    land_color  = np.array([0.8, 0.8, 0.8])  # grey
    ocean_color = np.array([0.2, 0.4, 0.8])  # blue

    fig, axes = plt.subplots(1, n, figsize=(4.2*n, 4.5), constrained_layout=True)

    if n == 1:
        axes = [axes]

    for idx, d in enumerate(leads):
        k = d - 1
        pred = P[sample_idx, :, :, k]
        true = T[sample_idx, :, :, k]

        # absolute error on ocean only
        err = np.abs(pred - true)
        err = np.where(LM > 0, err, np.nan)

        # draw background: land grey, ocean blue
        base = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.float32)
        base[LM <= 0] = land_color
        base[LM > 0]  = ocean_color

        ax = axes[idx]
        ax.imshow(base, origin='lower', interpolation='nearest')
        # overlay error (white→red). colormap trick: map 0 to white, vmax to red.
        # We do this by creating an RGBA layer with alpha proportional to error/vmax
        norm = np.clip(err / vmax, 0, 1)
        # color = red; blend against white → red*(alpha) + white*(1-alpha)
        red = np.array([1.0, 0.0, 0.0])[None, None, :]
        white = np.ones((1,1,3), dtype=np.float32)
        overlay_rgb = white * (1 - norm[..., None]) + red * norm[..., None]
        # mask land & NaN
        overlay_rgb[~np.isfinite(err)] = np.nan
        ax.imshow(overlay_rgb, origin='lower', interpolation='nearest')
        ax.set_title(f'Day +{d}', fontweight='bold')
        ax.axis('off')

    fig.suptitle(title + " (white=good, red=bad)", fontsize=13, fontweight='bold')
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible wrappers (your old calls now render only the 2 visuals)
# ─────────────────────────────────────────────────────────────────────────────
def plot_ice_prediction_dashboard(pred_maps, true_maps, land_mask,
                                  model_name, seq_output, now,
                                  lead_days=(1,7,14), sample_idx=0):
    # 1) Lead-time curves
    plot_skill_vs_lead(pred_maps, true_maps, land_mask,
                       title=f'{model_name} | Lead-time Skill (+{seq_output} days)')
    # 2) Error overlay
    plot_error_overlay(pred_maps, true_maps, land_mask,
                       lead_days=lead_days, sample_idx=sample_idx,
                       title=f'{model_name} | Error Overlay (+{seq_output} days)')

def plot_ice_edge_comparison(pred_maps, true_maps, land_mask,
                             model_name, seq_output, now,
                             lead_days=(1,7,14), sample_idx=0):
    # Use the new overlay map
    plot_error_overlay(pred_maps, true_maps, land_mask,
                       lead_days=lead_days, sample_idx=sample_idx,
                       title=f'{model_name} | Error Overlay (+{seq_output} days)')

def plot_performance_evolution(pred_maps, true_maps, land_mask,
                               model_name, seq_output, now):
    # Use the new lead-time curves
    plot_skill_vs_lead(pred_maps, true_maps, land_mask,
                       title=f'{model_name} | Lead-time Skill (+{seq_output} days)')

# ─────────────────────────────────────────────────────────────────────────────
# Legacy time-series plot placeholders (kept for safety; no-ops in minimal set)
# ─────────────────────────────────────────────────────────────────────────────
def plot(pred, true, model_name, seq_output, now):
    # Optional: keep empty or simple print to avoid breaking imports
    try:
        p = _to_numpy(pred); t = _to_numpy(true)
        # lightweight time-series glimpse (middle sample, 4 evenly spaced features)
        if p.ndim == 3 and p.shape[2] >= 4:
            idx = [0, p.shape[2]//4, p.shape[2]//2, 3*p.shape[2]//4]
            fig, axs = plt.subplots(1, 4, figsize=(12,3), constrained_layout=True)
            for i, ax in enumerate(axs):
                ax.plot(p[p.shape[0]//2, :, idx[i]], label="pred")
                ax.plot(t[t.shape[0]//2, :, idx[i]], label="true")
                ax.set_title(f'feat {idx[i]}')
                ax.grid(alpha=0.3)
            axs[0].legend()
            fig.suptitle(f'{model_name} | Quick TS glimpse', fontsize=12)
            plt.show()
    except Exception:
        pass
