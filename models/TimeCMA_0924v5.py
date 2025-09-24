import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal


class Dual(nn.Module):
    """
    TimeCMA (Temporal → Spatial Cross) + v4 강화 + 멀티-호라이즌 헤드 + KD 지원

    출력 모드
    --------
    - multi_head=True, return_dict=True   → dict {180:[B,180,N], ..., 1800:[B,1800,N]}
    - multi_head=True, return_dict=False  → [B, pred_len, N]  (*기본값: 기존 루프 호환)
    - multi_head=False                    → [B, pred_len, N]  (단일 헤드)

    옵션 요약
    --------
    - Fourier 주입, Prompt 길이 정규화, Modality Dropout, Cross-LN, RevIN
    - FiLM(prompt→gamma,beta), Image-K 기반 그래프 스무딩, Depthwise Conv 시계열 리파인
    - 멀티-호라이즌 헤드(ModuleDict), 지식증류를 위한 헬퍼 함수 제공
    """

    def __init__(
        self,
        device="cuda:7",
        channel=32,          # C
        num_nodes=7,         # N
        seq_len=96,          # L_in
        pred_len=96,         # L_out (single-head 또는 multi_head에서 텐서로 받을 때 사용)
        dropout_n=0.1,
        d_llm=768,
        e_layer=1,
        d_layer=1,
        d_ff=32,
        head=8,

        # Fourier
        use_time_feat=True,
        time_fourier_periods=(7.0, 30.4375, 91.3125, 365.25),

        # Modality dropout
        p_drop_prompt=0.15,
        p_drop_image=0.15,

        # v4 강화
        use_film=True,
        film_hidden=64,
        film_scale=0.1,

        use_graph_smooth=True,
        graph_alpha=0.25,
        graph_sim='cosine',   # 'cosine' | 'dot'

        use_temporal_refine=True,
        refine_kernel=5,

        # 멀티-호라이즌
        multi_head=True,
        pred_set=(180, 360, 720, 1080, 1440, 1800),
        return_dict=False      # ← 기본값을 텐서 반환으로 바꿈 (기존 루프 호환)
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n = dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.d_ff = d_ff
        self.head = head

        self.use_time_feat = use_time_feat
        self.time_fourier_periods = time_fourier_periods
        self.p_drop_prompt = float(p_drop_prompt)
        self.p_drop_image = float(p_drop_image)

        self.use_film = use_film
        self.film_hidden = film_hidden
        self.film_scale = float(film_scale)

        self.use_graph_smooth = use_graph_smooth
        self.graph_sim = graph_sim
        self.graph_alpha = nn.Parameter(torch.tensor(float(graph_alpha)))

        self.use_temporal_refine = use_temporal_refine
        self.refine_kernel = int(refine_kernel) if refine_kernel % 2 == 1 else int(refine_kernel) + 1

        self.multi_head = bool(multi_head)
        # pred_len이 pred_set에 반드시 포함되도록 보정
        _pred_set = list(pred_set)
        if self.multi_head and (self.pred_len not in _pred_set):
            _pred_set.append(self.pred_len)
        self.pred_set = sorted(set(_pred_set))
        self.return_dict = bool(return_dict)

        # --- RevIN ---
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)

        # --- Fourier → N 차원 ---
        if self.use_time_feat:
            self.time_ff = nn.Sequential(
                nn.Linear(2 * len(self.time_fourier_periods), self.num_nodes, bias=False),
                nn.GELU()
            ).to(self.device)
            nn.init.xavier_uniform_(self.time_ff[0].weight)

        # --- [B,N,L] → [B,N,C] ---
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.ln_after_len2feat = nn.LayerNorm(self.channel).to(self.device)

        # --- TS Encoder ---
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(
            self.ts_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # --- Prompt Encoder ---
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(
            self.prompt_encoder_layer, num_layers=self.e_layer
        ).to(self.device)
        self.prompt_pool = nn.AdaptiveAvgPool1d(self.d_llm)
        self.ln_prompt_in = nn.LayerNorm(self.d_llm).to(self.device)

        # --- Temporal Cross (Prompt) ---
        self.cross = CrossModal(
            d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
            activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)
        self.ln_q_temporal_in = nn.LayerNorm(self.num_nodes).to(self.device)

        # --- Spatial Cross (Image) ---
        self.c_to_nodes = nn.Linear(self.channel, self.num_nodes, bias=False).to(self.device)
        self.nodes_to_c = nn.Linear(self.num_nodes, self.channel, bias=False).to(self.device)
        self.ln_q_spatial_in = nn.LayerNorm(self.num_nodes).to(self.device)

        self.img_k_proj = nn.Linear(self.num_nodes, self.num_nodes, bias=False).to(self.device)
        self.img_v_proj = nn.Linear(self.num_nodes, self.num_nodes, bias=False).to(self.device)
        self.ln_img_kv = nn.LayerNorm(self.num_nodes).to(self.device)

        self.cross_spatial = CrossModal(
            d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
            activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)

        # --- Residual 게이트 ---
        self.g_t = nn.Parameter(torch.tensor(0.1))
        self.g_s = nn.Parameter(torch.tensor(0.1))

        # --- FiLM (prompt → gamma,beta) ---
        if self.use_film:
            self.film_mlp = nn.Sequential(
                nn.Linear(self.d_llm, self.film_hidden),
                nn.GELU(),
                nn.Linear(self.film_hidden, 2 * self.channel)
            ).to(self.device)

        # --- Decoder & Projection ---
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=self.d_layer
        ).to(self.device)

        # 단일 헤드(뒤호환)
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)
        self.ln_after_spatial = nn.LayerNorm(self.channel).to(self.device)

        # 멀티-호라이즌 헤드
        if self.multi_head:
            self.heads = nn.ModuleDict({
                str(h): nn.Linear(self.channel, h, bias=True).to(self.device)
                for h in self.pred_set
            })

        # Temporal refinement (depthwise conv over time)
        if self.use_temporal_refine:
            pad = self.refine_kernel // 2
            self.refine_conv = nn.Conv1d(
                in_channels=self.num_nodes, out_channels=self.num_nodes,
                kernel_size=self.refine_kernel, padding=pad, groups=self.num_nodes, bias=True
            ).to(self.device)

        # Graph smoothing projection
        if self.use_graph_smooth:
            self.graph_proj = nn.Linear(self.channel, self.channel, bias=False).to(self.device)

    # ===== utils =====
    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _ensure_prompt_shape(p):
        if p is None: return None
        return p.squeeze(-1) if (p.dim() == 4) else p

    @staticmethod
    def _build_time_fourier(L: int, device, dtype, periods):
        # [L, 2K]
        t = torch.arange(L, device=device, dtype=dtype)
        feats = []
        for p in periods:
            w = 2.0 * torch.pi * t / p
            feats.append(torch.sin(w))
            feats.append(torch.cos(w))
        tf = torch.stack(feats, dim=-1)  # [L, 2K]
        return tf

    def _prep_image_kv(self, emb_image, B, N, device):
        if emb_image is None:
            return None, None
        assert isinstance(emb_image, dict) and ('K' in emb_image) and ('V' in emb_image), \
            "emb_image must be dict with keys 'K' and 'V'"
        K, V = emb_image['K'], emb_image['V']
        if not torch.is_tensor(K): K = torch.as_tensor(K)
        if not torch.is_tensor(V): V = torch.as_tensor(V)
        K = K.to(device).float(); V = V.to(device).float()
        if K.dim() == 2: K = K.unsqueeze(0).expand(B, -1, -1)
        if V.dim() == 2: V = V.unsqueeze(0).expand(B, -1, -1)
        assert K.size(1) == N and V.size(1) == N, f"K/V must have N={N} at dim=1, got {K.size()} / {V.size()}"
        K = K.transpose(1, 2).contiguous()  # [B, d, N]
        V = V.transpose(1, 2).contiguous()
        return K, V

    def _build_graph_from_imageK(self, emb_image, B, N, device):
        if (emb_image is None) or ('K' not in emb_image):
            return None
        K = emb_image['K']
        if not torch.is_tensor(K): K = torch.as_tensor(K)
        if K.dim() == 2: K = K.unsqueeze(0).expand(B, -1, -1)  # [B,N,d]
        K = K.to(device).float()
        if self.graph_sim == 'cosine':
            K = F.normalize(K, dim=-1)
            S = torch.matmul(K, K.transpose(1, 2))
            S = S.clamp_min(0)
        else:
            S = torch.matmul(K, K.transpose(1, 2))
            S = F.relu(S)
        S = 0.5 * (S + S.transpose(1, 2))
        S = S / (S.sum(dim=-1, keepdim=True).clamp_min(1e-6))
        return S

    def gate_values(self):
        return float(F.softplus(self.g_t).detach().cpu()), float(F.softplus(self.g_s).detach().cpu())

    # ===== core forward (shared representation) =====
    def _forward_shared(self, input_data, input_data_mark, emb_prompt, emb_image):
        """
        Returns:
          dec_hid : [B, N, C]  (decoder 은닉 표현)
          state   : dict (모니터링용)
        """
        x = input_data.float()   # [B, L, N]
        p = self._ensure_prompt_shape(emb_prompt)

        B, L, N = x.size()
        assert N == self.num_nodes, f"N mismatch: input {N} vs model {self.num_nodes}"

        # RevIN
        x = self.normalize_layers(x, 'norm')    # [B, L, N]

        # Fourier
        if self.use_time_feat:
            tf = self._build_time_fourier(L, x.device, x.dtype, self.time_fourier_periods)  # [L,2K]
            tf_proj = self.time_ff(tf)          # [L, N]
            x = x + tf_proj.unsqueeze(0)        # [B, L, N]

        # [B,L,N] -> [B,N,L] -> L→C
        x = x.permute(0, 2, 1)                  # [B, N, L]
        x = self.length_to_feature(x)           # [B, N, C]
        x = self.ln_after_len2feat(x)

        # Prompt 인코딩
        if p is not None:
            p = p.permute(0, 2, 1)              # [B, N, E]
            p = self.prompt_pool(p)             # [B, N, d_llm]
            p = self.ln_prompt_in(p)
            p_enc = self.prompt_encoder(p)      # [B, N, d_llm]
            p_kv = p_enc.permute(0, 2, 1)       # [B, d_llm, N]
            if self.use_film:
                p_sum = p_enc.mean(dim=1)       # [B, d_llm]
                gamma_beta = self.film_mlp(p_sum)  # [B,2C]
                gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
                gamma = 1.0 + self.film_scale * torch.tanh(gamma)
                beta  = self.film_scale * torch.tanh(beta)
        else:
            p_kv = None
            gamma = beta = None

        # Modality Dropout (prompt)
        if self.training and (p_kv is not None) and (torch.rand(1).item() < self.p_drop_prompt):
            p_kv = None

        # TS encoder
        enc_nodes = self.ts_encoder(x)          # [B, N, C]
        if self.use_film and (gamma is not None):
            enc_nodes = enc_nodes * gamma.unsqueeze(1) + beta.unsqueeze(1)

        q_t = enc_nodes.permute(0, 2, 1)        # [B, C, N]
        q_t = self.ln_q_temporal_in(q_t)

        # Temporal Cross
        if p_kv is not None:
            delta_t = self.cross(q_t, p_kv, p_kv)  # [B, C, N]
            z_t = q_t + F.softplus(self.g_t) * delta_t
        else:
            z_t = q_t
        z_t = z_t.permute(0, 2, 1)              # [B, N, C]

        # Spatial Cross
        q_s = self.c_to_nodes(z_t)              # [B, N, N]
        q_s = self.ln_q_spatial_in(q_s)

        k_img, v_img = self._prep_image_kv(emb_image, B=B, N=self.num_nodes, device=z_t.device)
        if self.training and (k_img is not None) and (torch.rand(1).item() < self.p_drop_image):
            k_img, v_img = None, None

        if (k_img is not None) and (v_img is not None):
            k_img = self.img_k_proj(k_img); v_img = self.img_v_proj(v_img)
            k_img = self.ln_img_kv(k_img); v_img = self.ln_img_kv(v_img)
            delta_s = self.cross_spatial(q_s, k_img, v_img)  # [B, N, N]
            z_s = q_s + F.softplus(self.g_s) * delta_s
        else:
            z_s = q_s

        z = self.nodes_to_c(z_s)                # [B, N, C]
        z = self.ln_after_spatial(z)

        # Graph smoothing
        if self.use_graph_smooth:
            S = self._build_graph_from_imageK(emb_image, B=B, N=self.num_nodes, device=z.device)
            if S is not None:
                z = z + torch.clamp(self.graph_alpha, 0.0, 1.0) * self.graph_proj(torch.bmm(S, z))

        # Decoder 공통
        dec_hid = self.decoder(z, z)            # [B, N, C]
        state = {"enc_nodes": enc_nodes, "z": z, "dec_hid": dec_hid}
        return dec_hid, state

    # ===== public forward =====
    def forward(self, input_data, input_data_mark, emb_prompt, emb_image,
                target_horizon=None):
        """
        Returns:
          - multi_head=True, return_dict=True   → dict {h: [B,h,N]}
          - multi_head=True, return_dict=False  → [B, pred_len(or target), N]
          - multi_head=False                    → [B, pred_len, N]
        """
        dec_hid, _ = self._forward_shared(input_data, input_data_mark, emb_prompt, emb_image)  # [B,N,C]

        if self.multi_head:
            outs = {}
            for h in self.pred_set:
                y = self.heads[str(h)](dec_hid)    # [B, N, h]
                if self.use_temporal_refine:
                    y = self.refine_conv(y)        # depthwise conv: [B,N,h]
                y = y.permute(0, 2, 1)             # [B, h, N]
                y = self.normalize_layers(y, 'denorm')
                outs[h] = y

            if self.return_dict:
                return outs

            # 텐서 반환 모드: target_horizon가 있으면 우선, 없으면 pred_len 선택
            h_sel = target_horizon if (target_horizon is not None) else self.pred_len
            if h_sel not in outs:
                raise ValueError(f"Requested horizon {h_sel} not in pred_set {self.pred_set}. "
                                 f"Include it or pass a valid target_horizon.")
            return outs[h_sel]  # [B, h_sel, N]

        else:
            # 단일 헤드(뒤호환)
            y = self.c_to_length(dec_hid)          # [B, N, pred_len]
            if self.use_temporal_refine:
                y = self.refine_conv(y)            # [B,N,pred_len]
            y = y.permute(0, 2, 1)                 # [B, pred_len, N]
            y = self.normalize_layers(y, 'denorm')
            return y


# =========================
# ===== Loss Utilities ====
# =========================

def seasonal_baseline(past, horizon):
    """past: [B,L_in,N] → baseline: [B,horizon,N]. 아주 단순한 '마지막 값 유지'."""
    return past[:, -1:, :].repeat(1, horizon, 1)


def freq_loss(yh, yt, topk_ratio=0.15):  # [B,L,N]
    YH, YT = torch.fft.rfft(yh, dim=1), torch.fft.rfft(yt, dim=1)
    mag_h = (YH.real**2 + YH.imag**2).sqrt()
    mag_t = (YT.real**2 + YT.imag**2).sqrt()
    k = max(1, int(mag_h.size(1) * float(topk_ratio)))
    return ((mag_h[:, :k] - mag_t[:, :k])**2).mean()


def diff_loss(yh, yt):  # 고주파 민감(1차 차분)
    return F.mse_loss(yh[:,1:,:] - yh[:,:-1,:], yt[:,1:,:] - yt[:,:-1,:])


def build_laplacian_from_imageK(emb_image, device):
    """이미지 K로 그래프 라플라시안 L 생성 (batch 무시, 정적 사용)."""
    if emb_image is None or ('K' not in emb_image):
        return None
    K = emb_image['K']
    if not torch.is_tensor(K): K = torch.as_tensor(K)
    if K.dim()==3: K = K[0]
    K = K.to(device).float()
    K = F.normalize(K, dim=-1)
    S = (K @ K.T).clamp_min(0)
    S.fill_diagonal_(0.)
    D = torch.diag(S.sum(dim=1))
    L = D - S
    return L


def laplacian_reg(y, L):  # y:[B,L,N], L:[N,N]
    if L is None: return 0.0 * y.mean()
    Y = y.transpose(1,2)  # [B,N,L]
    return (Y @ L @ Y.transpose(1,2)).mean()


def multi_horizon_loss(
    outs_dict,         # dict {h: [B,h,N]}
    y_targets_dict,    # dict {h: [B,h,N]}
    x_past,            # [B,L_in,N]
    emb_image=None,
    # 가중치
    H_w={180:1.0, 360:1.0, 720:1.1, 1080:1.2, 1440:1.4, 1800:1.6},
    alpha_diff={180:0.5, 360:0.4, 720:0.3, 1080:0.25, 1440:0.2, 1800:0.2},
    alpha_freq={180:0.2, 360:0.25, 720:0.3, 1080:0.4, 1440:0.5, 1800:0.6},
    lambda_lap=5e-3,
    lambda_cons=0.2,
    # 지식 증류(선생 모델 출력 dict 또는 함수)
    teacher_outs_dict=None,
    kd_weight=0.2
):
    device = x_past.device
    Lmat = build_laplacian_from_imageK(emb_image, device=device)

    loss = 0.0
    loss_items = {}

    # per-horizon losses
    for h, y_hat in outs_dict.items():
        y_true = y_targets_dict[h]
        base = seasonal_baseline(x_past, h)
        err  = (y_hat - base)
        tgt  = (y_true - base)

        mse  = F.mse_loss(err, tgt)
        dlf  = diff_loss(err, tgt)
        frq  = freq_loss(err, tgt)

        lap  = laplacian_reg(y_hat, Lmat)

        h_loss = H_w[h] * (mse + alpha_diff[h]*dlf + alpha_freq[h]*frq) + lambda_lap*lap
        loss += h_loss

        loss_items[f"mse_{h}"] = mse.detach()
        loss_items[f"diff_{h}"] = dlf.detach()
        loss_items[f"freq_{h}"] = frq.detach()
        loss_items[f"lap_{h}"]  = lap.detach()

        # KD (teacher outs가 있으면)
        if teacher_outs_dict is not None and (h in teacher_outs_dict):
            kd = F.mse_loss(y_hat, teacher_outs_dict[h].to(device))
            loss += kd_weight * kd
            loss_items[f"kd_{h}"] = kd.detach()

    # consistency (긴 호라이즌의 앞부분 ↔ 짧은 호라이즌 전체)
    pairs = [(360,180),(720,360),(1080,720),(1440,1080),(1800,1440)]
    for hi, hj in pairs:
        if (hi in outs_dict) and (hj in outs_dict):
            cons = F.mse_loss(outs_dict[hi][:,:hj,:], outs_dict[hj])
            loss += lambda_cons * cons
            loss_items[f"cons_{hi}_{hj}"] = cons.detach()

    return loss, loss_items
