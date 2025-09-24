import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal


class Dual(nn.Module):
    """
    TimeCMA 스타일의 2-단 크로스 모달 구조 (Temporal → Spatial).

    v4 강화점(기본 ON)
    -----------------
    - FiLM-style Prompt Conditioning (prompt → gamma,beta)
    - Image-K 기반 1-hop Graph Smoothing (z ← z + α·S z W)
    - Depthwise Conv1d Temporal Refinement (예측 후 얕은 스무딩)

    v3 대비 변경 없는 점
    -------------------
    - Fourier 주입, Prompt 길이 정규화, Modality Dropout, Cross-LN, RevIN, 해더/입출력 규격
    """

    def __init__(
        self,
        device="cuda:7",
        channel=32,          # C
        num_nodes=7,         # N
        seq_len=96,          # L_in
        pred_len=96,         # L_out
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

        # === v4 추가 플래그 ===
        use_film=True,
        film_hidden=64,
        film_scale=0.1,          # gamma/beta 스케일 안정화

        use_graph_smooth=True,
        graph_alpha=0.25,        # 스무딩 강도(학습가능 파라메터로 둠)
        graph_sim='cosine',      # 'cosine' | 'dot'

        use_temporal_refine=True,
        refine_kernel=5          # depthwise conv 커널(홀수)
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
        self.graph_alpha = nn.Parameter(torch.tensor(float(graph_alpha)))  # 학습 가능

        self.use_temporal_refine = use_temporal_refine
        self.refine_kernel = int(refine_kernel) if refine_kernel % 2 == 1 else int(refine_kernel) + 1

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

        # Prompt 길이 정규화(pool to d_llm)
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
        self.g_t = nn.Parameter(torch.tensor(0.1))  # temporal
        self.g_s = nn.Parameter(torch.tensor(0.1))  # spatial

        # --- FiLM (prompt summary → gamma,beta) ---
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

        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)
        self.ln_after_spatial = nn.LayerNorm(self.channel).to(self.device)

        # --- Temporal refinement (depthwise conv over time) ---
        if self.use_temporal_refine:
            pad = self.refine_kernel // 2
            # 입력: [B,N,L] → Conv1d(ch=N, groups=N) → [B,N,L]
            self.refine_conv = nn.Conv1d(
                in_channels=self.num_nodes, out_channels=self.num_nodes,
                kernel_size=self.refine_kernel, padding=pad, groups=self.num_nodes, bias=True
            ).to(self.device)

        # --- Graph smoothing projection ---
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
        t = torch.arange(L, device=device, dtype=dtype)
        feats = []
        for p in periods:
            w = 2.0 * torch.pi * t / p
            feats.extend([torch.sin(w), torch.cos(w)])
        return torch.stack(feats, dim=1).transpose(0, 1).transpose(0, 1)  # [L,2K]

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
        """이미지 K로 노드 유사도 그래프 S[B,N,N] 생성 (row-normalized, symmetric)."""
        if (emb_image is None) or ('K' not in emb_image):
            return None
        K = emb_image['K']
        if not torch.is_tensor(K): K = torch.as_tensor(K)
        if K.dim() == 2: K = K.unsqueeze(0).expand(B, -1, -1)  # [B,N,d]
        K = K.to(device).float()
        if self.graph_sim == 'cosine':
            K = F.normalize(K, dim=-1)
            S = torch.matmul(K, K.transpose(1, 2))  # [B,N,N] in [-1,1]
            S = S.clamp_min(0)                      # 음수 유사도 제거
        else:  # 'dot'
            S = torch.matmul(K, K.transpose(1, 2))
            S = F.relu(S)
        S = 0.5 * (S + S.transpose(1, 2))           # 대칭화
        # row-normalize
        S = S / (S.sum(dim=-1, keepdim=True).clamp_min(1e-6))
        return S

    def gate_values(self):
        return float(F.softplus(self.g_t).detach().cpu()), float(F.softplus(self.g_s).detach().cpu())

    # ===== forward =====
    def forward(self, input_data, input_data_mark, emb_prompt, emb_image):
        """
        input_data : [B, L, N]
        emb_prompt : [B, E, N] or [B, E, N, 1] or None
        emb_image  : {'K':[B,N,dk]|[N,dk], 'V':[B,N,dv]|[N,dv]} or None
        return     : [B, L_out, N]
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
            p_kv = p_enc.permute(0, 2, 1)       # [B, d_llm, N]  (K/V)
            # FiLM (global prompt summary)
            if self.use_film:
                p_sum = p_enc.mean(dim=1)       # [B, d_llm]
                gamma_beta = self.film_mlp(p_sum)  # [B,2C]
                gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)  # [B,C],[B,C]
                gamma = 1.0 + self.film_scale * torch.tanh(gamma)
                beta  = self.film_scale * torch.tanh(beta)
        else:
            p_kv = None
            if self.use_film:
                gamma = beta = None

        # Modality Dropout (prompt)
        if self.training and (p_kv is not None) and (torch.rand(1).item() < self.p_drop_prompt):
            p_kv = None

        # TS encoder
        enc_nodes = self.ts_encoder(x)          # [B, N, C]
        if self.use_film and (p is not None):
            # FiLM: [B,N,C] ← [B,C] broadcast
            enc_nodes = enc_nodes * gamma.unsqueeze(1) + beta.unsqueeze(1)

        q_t = enc_nodes.permute(0, 2, 1)        # [B, C, N]
        q_t = self.ln_q_temporal_in(q_t)        # Cross 입력 정규화

        # (1) Temporal Cross (Prompt)
        if p_kv is not None:
            delta_t = self.cross(q_t, p_kv, p_kv)  # [B, C, N]
            z_t = q_t + F.softplus(self.g_t) * delta_t
        else:
            z_t = q_t
        z_t = z_t.permute(0, 2, 1)              # [B, N, C]

        # (2) Spatial Cross (Image)
        q_s = self.c_to_nodes(z_t)              # [B, N, N]
        q_s = self.ln_q_spatial_in(q_s)

        k_img, v_img = self._prep_image_kv(emb_image, B=B, N=self.num_nodes, device=z_t.device)
        if self.training and (k_img is not None) and (torch.rand(1).item() < self.p_drop_image):
            k_img, v_img = None, None

        if (k_img is not None) and (v_img is not None):
            k_img = self.img_k_proj(k_img)      # [B, d, N]
            v_img = self.img_v_proj(v_img)      # [B, d, N]
            k_img = self.ln_img_kv(k_img); v_img = self.ln_img_kv(v_img)
            delta_s = self.cross_spatial(q_s, k_img, v_img)  # [B, N, N]
            z_s = q_s + F.softplus(self.g_s) * delta_s
        else:
            z_s = q_s

        # N→C 복원
        z = self.nodes_to_c(z_s)                # [B, N, C]
        z = self.ln_after_spatial(z)

        # (옵션) Graph smoothing in C-space using image-K graph
        if self.use_graph_smooth:
            S = self._build_graph_from_imageK(emb_image, B=B, N=self.num_nodes, device=z.device)
            if S is not None:
                z = z + torch.clamp(self.graph_alpha, 0.0, 1.0) * self.graph_proj(torch.bmm(S, z))  # [B,N,C]

        # Decoder & Projection
        dec_out = self.decoder(z, z)            # [B, N, C]
        dec_out = self.c_to_length(dec_out)     # [B, N, L_out]

        # (옵션) Temporal refinement (depthwise conv over time)
        if self.use_temporal_refine:
            dec_out = self.refine_conv(dec_out) # [B, N, L_out]

        dec_out = dec_out.permute(0, 2, 1)      # [B, L_out, N]

        # RevIN denorm
        y_hat = self.normalize_layers(dec_out, 'denorm')
        return y_hat
