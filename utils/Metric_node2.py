# 필수 라이브러리 import
from sklearn.metrics import mean_squared_error, mean_absolute_error  # MSE/MAE 계산용
import numpy as np
from skimage.metrics import structural_similarity  # SSIM (구조적 유사도, 현재는 사용 안 됨)
from torchmetrics import MeanAbsolutePercentageError  # MAPE 지표 계산
import matplotlib.pyplot as plt  # 시각화
import torch
import seaborn as sns  # 고급 시각화
import os  # 파일/폴더 조작

# ============================
# 개별 지표 계산 함수들
# ============================

def _mape(pred, target):
    """
    MAPE (Mean Absolute Percentage Error) 계산 함수
    - 예측값과 실제값의 상대적인 오차 비율을 측정
    - 주의: target이 0일 때 division by zero 발생 가능
    """
    # torchmetrics MAPE 객체를 이용해 계산
    # reshape(-1) → 1차원 벡터로 변환
    return MeanAbsolutePercentageError()(pred.reshape(-1), target.reshape(-1)) 

def _mse(outputs, targets):
    """
    MSE (Mean Squared Error) 계산
    - 오차 제곱 평균
    """
    return mean_squared_error(outputs.reshape(-1), targets.reshape(-1))

def _mae(outputs, targets):
    """
    MAE (Mean Absolute Error) 계산
    - 오차 절대값 평균
    """
    return mean_absolute_error(outputs.reshape(-1), targets.reshape(-1))

def CORR_uni(pred, true):
    """
    Correlation (상관계수) 계산
    - 모든 feature를 합산한 후, 예측과 실제의 시계열 상관관계를 계산
    - 문제: feature 차원을 합치므로 지역별 특성이 사라질 수 있음
    """
    # feature 차원을 합산
    pred = np.sum(pred, axis=(-1))
    true = np.sum(true, axis=(-1))
    
    corrs = []
    # 배치 단위로 상관계수 계산
    for i in range(pred.shape[0]):
        corr = np.corrcoef(pred[i, :], true[i, :])[0, 1]  # 피어슨 상관계수
        if not np.isnan(corr).any():  
            corrs.append(corr)

    return np.mean(corrs)  # 배치 평균 상관계수 반환

# ============================
# Metric 집계 함수
# ============================

def metric(outputs, actuals, n_features, normalize=False):
    """
    모델 출력과 실제값을 받아서 MSE, MAE, Correlation 계산
    - normalize=True일 경우 Z-score 정규화 수행 (모델 학습 안정화를 위한 것이 아닌 평가지표 계산 값의 크기를 줄이는 용도)
    """
    outputs = outputs.detach().cpu().numpy()  # torch → numpy
    actuals = actuals.detach().cpu().numpy()

    if normalize:
        # Z-score 정규화 (평균=0, 표준편차=1)
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
    """
    현재 성능(metric)을 이전 best와 비교하여 업데이트 후 저장
    - now: 현재 시각(datetime)
    - save: 모델 저장 여부 (현재는 주석 처리)
    - best: 지금까지의 best 성능 리스트 [mse, mae, corr]
    - metric: 이번 epoch 성능
    - model_name: 모델 이름
    - seq_output: 출력 시퀀스 길이
    - epoch: 현재 epoch
    """

    folder_path = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 모델 저장 부분 (현재 주석 처리됨)
    # if save:
    #     torch.save(model.state_dict(), folder_path+f'/{model_name}_{seq_output}')

    # Best 성능 갱신 (MSE/MAE는 작을수록 좋음, Corr은 클수록 좋음)
    if best[0] > metric[0]: best[0] = metric[0]
    if best[1] > metric[1]: best[1] = metric[1]
    if best[2] < metric[2]: best[2] = metric[2]

    # 성능 + epoch 저장
    metric = np.array([best[0], best[1], best[2], epoch])
    np.save(folder_path + f'/{model_name}_{seq_output}.npy', metric)

    return best

# ============================
# Plotting 함수들
# ============================

def plot(pred, true, model_name, seq_output, now):
    """
    예측값 vs 실제값 시계열 비교 플롯
    - cluster별 시계열 / 전체 시계열 2가지 그림 저장
    """
    folder_path = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'

    true = true.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    
    # 특정 feature index 선택 (0, 1/4, 1/2, 3/4 지점) 여기는 0번째 클러스터 8번째 클러스터 16번째 클러스터 32번째 클러스터
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
    plt.show()
    
    # feature 합산 후 전체 시계열 비교 여기서 샘플 인덱스는 전체 batch중에서 첫번째 1/4위치 중간 3/4위치
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
    plt.show()

def plot_uni(pred, true):
    """
    예측값과 실제값을 heatmap + line plot으로 비교
    """
    true = true.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    
    # feature 차원 합산
    a = np.sum(true, axis=(1))
    p = np.sum(pred, axis=(1))
    
    # Heatmap 비교
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
    
    # 전체 시계열 합산 후 라인 플롯
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
    """
    논문용 고급 시각화 (Prediction vs Ground Truth)
    - seaborn 스타일 적용
    - 선 스타일/색상 조정
    """
    plt.figure(figsize=(8, 6), dpi=300)  # 고해상도
    sns.set_theme(style="whitegrid")
    
    plt.plot(out, label='Prediction', linestyle='-', color='#E74C3C', linewidth=2.0, alpha=0.8)
    plt.plot(act, label='Ground Truth', linestyle='--', color='#3498DB', linewidth=2.0, alpha=0.8)

    plt.title(f'{model_name}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Time Step', fontsize=12, labelpad=10)
    plt.ylabel('Aggregated Value', fontsize=12, labelpad=10)
    plt.legend(fontsize=10, loc='upper right', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    # plt.savefig(f"{model_name}_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

# ============================
# 파일 탐색 함수
# ============================

def find_files_with_prefix_and_suffix(folder_path, prefix, suffix='.npy'):
    """
    특정 prefix를 가진 폴더 내부에서, 특정 suffix(.npy) 파일을 찾는 함수
    - 현재: folder_name이 prefix로 시작하는 폴더 내 모든 .npy 파일을 반환
    - 원래 의도: 파일 이름 검사까지 필요할 수 있음
    """
    matching_files = []
    for root, _, files in os.walk(folder_path):
        folder_name = os.path.basename(root)
        if folder_name.startswith(prefix):
            for file in files:
                if file.endswith(suffix):
                    matching_files.append(os.path.join(root, file))
    return matching_files


# ============================
# (추가) 전체 32개 클러스터를 4개씩 묶어 저장하는 플롯 함수
#       + 10에폭마다 저장 트리거
# ============================

def plot_clusters_all(pred, true, model_name, seq_output, now, epoch,
                      batch_idx=None, group_size=4, rows=2, cols=2,
                      figsize=(12, 8)):
    """
    pred/true: torch.Tensor [B, L_out, C]
    C개(예: 32) 클러스터를 4개씩(2x2) 묶어 여러 장으로 저장.
    저장 경로: STMA_node/{model_name}/models/{model_name}_{seq_output}_<timestamp>/epoch_XXXX/
    """
    # 저장 폴더
    base_dir = f'STMA_node/{model_name}/models/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    epoch_dir = os.path.join(base_dir, f'epoch_{epoch:04d}')
    os.makedirs(epoch_dir, exist_ok=True)

    # 텐서를 numpy로
    pred = pred.detach().cpu().numpy()
    true = true.detach().cpu().numpy()

    B, L, C = pred.shape
    if batch_idx is None:
        batch_idx = B // 2  # 가운데 배치 하나 선택(기존 plot과 비슷한 감각)

    # 그룹단위로 4개씩 그림
    n_per_fig = rows * cols
    assert group_size == n_per_fig, "rows*cols가 group_size와 같아야 합니다 (기본 2x2=4)."

    for start in range(0, C, group_size):
        end = min(start + group_size, C)
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axs = axs.flat if hasattr(axs, "flat") else [axs]  # 2x2 -> iterator

        for i in range(group_size):
            ax = axs[i]
            cl = start + i
            ax.axis('off')  # 빈 칸 기본 off
            if cl < C:
                ax.plot(pred[batch_idx, :, cl], label="prediction", color="r", linewidth=1.5)
                ax.plot(true[batch_idx, :, cl], label="actual", color="b", linewidth=1.2, alpha=0.8)
                ax.set_title(f"Cluster {cl}", fontsize=11)
                ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
                ax.legend(fontsize=8, loc='upper right')
                ax.axis('on')

        fig.suptitle(f"{model_name} | seq_out={seq_output} | epoch={epoch} | clusters {start}-{end-1}",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        save_path = os.path.join(epoch_dir, f"{model_name}_clusters_{start:02d}-{end-1:02d}_ep{epoch:04d}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)  # 메모리 누수 방지

def maybe_plot_every_k(pred, true, model_name, seq_output, now, epoch,
                       k=10, **kwargs):
    """
    epoch이 k의 배수일 때만 plot_clusters_all을 호출.
    kwargs는 plot_clusters_all로 그대로 전달.
    사용 예:
        maybe_plot_every_k(pred, true, model_name, seq_output, now, epoch, k=10)
    """
    if (epoch % k) == 0:
        plot_clusters_all(pred, true, model_name, seq_output, now, epoch, **kwargs)
