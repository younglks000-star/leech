import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal


class Dual(nn.Module):
    """
    TimeCMA (Temporal → Spatial) — 0924v2 업그레이드판
    - Cross 용량 확장(bottleneck d_cm, multi-head)
    - Learnable Time2Vec 스타일 시간 특징 (주파수/위상 학습)
    - 입력 사전 리파인(depthwise temporal conv)
    - 나머지 인터페이스/동작은 0924v2와 동일
    """

    def __init__(
        self,
        device="cuda:7",
        channel=32,          # C: 노드 임베딩 채널 수
        num_nodes=7,         # N: 노드/클러스터 개수
        seq_len=96,          # L_in
        pred_len=96,         # L_out
        dropout_n=0.1,
        d_llm=768,           # 프롬프트 임베딩 피처 차원
        e_layer=1,
        d_layer=1,
        d_ff=32,
        head=8,

        # 시간 특징
        use_time_feat=True,
        init_periods=(7.0, 30.4375, 91.3125, 365.25),  # 초기 주기값(학습 대상)
        add_linear_time=False,  # 필요하면 선형 t 성분 추가(기본 off)

        # modality dropout
        p_drop_prompt=0.15,
        p_drop_image=0.15,

        # cross 용량 확장
        d_cm=32,           # cross의 d_model (N -> d_cm -> N)
        cross_heads=4,     # multi-head (d_cm % cross_heads == 0)
    ):
        super().__init__()

        assert d_cm % cross_heads == 0, f"d_cm({d_cm}) % cross_heads({cross_heads}) must be 0"

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
        self.add_linear_time = bool(add_linear_time)
        self.p_drop_prompt = float(p_drop_prompt)
        self.p_drop_image = float(p_drop_image)

        self.d_cm = d_cm
        self.cross_heads = cross_heads

        # ---------------- RevIN ----------------
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)

        # -------- 시간 특징: learnable Time2Vec(sin/cos) --------
        if self.use_time_feat:
            K = len(init_periods)
            omega = 2.0 * torch.pi / torch.tensor(init_periods, dtype=torch.float32)
            self.time_omega = nn.Parameter(omega)              # [K]
            self.time_phase = nn.Parameter(torch.zeros(K))     # [K]
            if self.add_linear_time:
                self.time_w0 = nn.Parameter(torch.randn(1)*0.01)
                self.time_b0 = nn.Parameter(torch.zeros(1))
            in_dim = 2*K + (1 if self.add_linear_time else 0)
            self.time_ff = nn.Sequential(
                nn.Linear(in_dim, self.num_nodes, bias=False),
                nn.GELU()
            ).to(self.device)
            nn.init.xavier_uniform_(self.time_ff[0].weight)

        # -------- 사전 리파인: depthwise conv over time --------
        self.pre_conv = nn.Conv1d(
            in_channels=self.num_nodes, out_channels=self.num_nodes,
            kernel_size=5, padding=2, groups=self.num_nodes, bias=True
        ).to(self.device)
        self.pre_conv_gain = nn.Parameter(torch.tensor(0.1))

        # -------- [B,N,L] → [B,N,C] --------
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.ln_after_len2feat = nn.LayerNorm(self.channel).to(self.device)

        # -------- Time-Series Encoder --------
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(
            self.ts_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # -------- Prompt Encoder --------
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(
            self.prompt_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # 프롬프트 길이 정규화(E → d_llm)
        self.prompt_pool = nn.AdaptiveAvgPool1d(self.d_llm)
        self.ln_prompt_in = nn.LayerNorm(self.d_llm).to(self.device)

        # ================== Cross (capacity up: N→d_cm→N) ==================

        # (A) Temporal Cross (Prompt)
        self.temporal_q_in  = nn.Linear(self.num_nodes, self.d_cm, bias=False).to(self.device)
        self.temporal_kv_in = nn.Linear(self.num_nodes, self.d_cm, bias=False).to(self.device)
        self.cross_temporal = CrossModal(
            d_model=self.d_cm, n_heads=self.cross_heads, d_ff=self.d_ff, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
            activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)
        self.temporal_out   = nn.Linear(self.d_cm, self.num_nodes, bias=False).to(self.device)
        self.ln_q_temporal_in = nn.LayerNorm(self.d_cm).to(self.device)

        # (B) Spatial Cross (Image)
        # ✨ 여기 두 줄이 누락돼서 AttributeError가 났던 부분입니다.
        self.c_to_nodes = nn.Linear(self.channel, self.num_nodes, bias=False).to(self.device)   # [B,N,C] -> [B,N,N]
        self.nodes_to_c = nn.Linear(self.num_nodes, self.channel, bias=False).to(self.device)   # [B,N,N] -> [B,N,C]

        self.spatial_q_in   = nn.Linear(self.num_nodes, self.d_cm, bias=False).to(self.device)
        self.spatial_kv_in  = nn.Linear(self.num_nodes, self.d_cm, bias=False).to(self.device)
        self.cross_spatial  = CrossModal(
            d_model=self.d_cm, n_heads=self.cross_heads, d_ff=self.d_ff, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
            activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)
        self.spatial_out    = nn.Linear(self.d_cm, self.num_nodes, bias=False).to(self.device)
        self.ln_q_spatial_in = nn.LayerNorm(self.d_cm).to(self.device)
        self.ln_img_kv      = nn.LayerNorm(self.d_cm).to(self.device)

        # -------- Residual 게이트 --------
        self.g_t = nn.Parameter(torch.tensor(0.1))  # temporal
        self.g_s = nn.Parameter(torch.tensor(0.1))  # spatial

        # -------- Decoder & Projection --------
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=self.d_layer
        ).to(self.device)

        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)
        self.ln_after_spatial = nn.LayerNorm(self.channel).to(self.device)

    # ---------------- utils ----------------
    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _ensure_prompt_shape(p: torch.Tensor) -> torch.Tensor:
        return p.squeeze(-1) if (p is not None and p.dim() == 4) else p

    @staticmethod
    def _build_time2vec(L, device, dtype, omega, phase, add_linear_time=False, w0=None, b0=None):
        t = torch.arange(L, device=device, dtype=dtype)  # [L]
        T = t.unsqueeze(-1)
        sin = torch.sin(T * omega + phase)
        cos = torch.cos(T * omega + phase)
        feats = [sin, cos]
        if add_linear_time and (w0 is not None) and (b0 is not None):
            lin = T.squeeze(-1) * w0 + b0
            feats.append(lin.unsqueeze(-1))
        return torch.cat(feats, dim=-1)  # [L, 2K(+1)]

    def _prep_image_kv(self, emb_image, B, N, device):
        assert isinstance(emb_image, dict) and ('K' in emb_image) and ('V' in emb_image), \
            "emb_image must be dict with keys 'K' and 'V'"
        K, V = emb_image['K'], emb_image['V']
        if not torch.is_tensor(K): K = torch.as_tensor(K)
        if not torch.is_tensor(V): V = torch.as_tensor(V)
        K = K.to(device).float()
        V = V.to(device).float()
        if K.dim() == 2: K = K.unsqueeze(0).expand(B, -1, -1)  # [B,N,d]
        if V.dim() == 2: V = V.unsqueeze(0).expand(B, -1, -1)
        assert K.size(1) == N and V.size(1) == N, f"K/V must have N={N} at dim=1, got {K.size()} / {V.size()}"
        K = K.transpose(1, 2).contiguous()  # [B, d_img, N]
        V = V.transpose(1, 2).contiguous()
        return K, V

    def gate_values(self):
        return float(F.softplus(self.g_t).detach().cpu()), float(F.softplus(self.g_s).detach().cpu())

    # ---------------- forward ----------------
    def forward(self, input_data, input_data_mark, emb_prompt, emb_image):
        """
        input_data : [B, L, N]
        emb_prompt : [B, E, N] or [B, E, N, 1]
        emb_image  : {'K':[B,N,dk]|[N,dk], 'V':[B,N,dv]|[N,dv]}
        return     : [B, L_out, N]
        """
        x = input_data.float()                  # [B, L, N]
        p = self._ensure_prompt_shape(emb_prompt).float() if emb_prompt is not None else None
        B, L, N = x.size()
        assert N == self.num_nodes, f"N mismatch: input {N} vs model {self.num_nodes}"

        # RevIN
        x = self.normalize_layers(x, 'norm')    # [B, L, N]

        # (a) Learnable Time2Vec 주입
        if self.use_time_feat:
            tf = self._build_time2vec(
                L, device=x.device, dtype=x.dtype,
                omega=self.time_omega, phase=self.time_phase,
                add_linear_time=self.add_linear_time,
                w0=(self.time_w0 if self.add_linear_time else None),
                b0=(self.time_b0 if self.add_linear_time else None)
            )                                   # [L, 2K(+1)]
            tf_proj = self.time_ff(tf)          # [L, N]
            x = x + tf_proj.unsqueeze(0)        # [B, L, N]

        # (a2) 입력 사전 리파인 (depthwise conv over time)
        x_ = x.transpose(1, 2)                  # [B, N, L]
        x_ref = self.pre_conv(x_)               # [B, N, L]
        x = x + F.tanh(self.pre_conv_gain) * x_ref.transpose(1, 2)  # [B, L, N]

        # [B,L,N] -> [B,N,L] -> L→C
        x = x.permute(0, 2, 1)                  # [B, N, L]
        x = self.length_to_feature(x)           # [B, N, C]
        x = self.ln_after_len2feat(x)           # 안정화

        # Prompt 인코딩
        if p is not None:
            p = p.permute(0, 2, 1)              # [B, N, E]
            p = self.prompt_pool(p)             # [B, N, d_llm]
            p = self.ln_prompt_in(p)            # 안정화
            p_enc = self.prompt_encoder(p)      # [B, N, d_llm]
            p_enc = p_enc.permute(0, 2, 1)      # [B, d_llm, N]
        else:
            p_enc = None

        # Modality Dropout (학습시에만)
        if self.training and (p_enc is not None) and (torch.rand(1).item() < self.p_drop_prompt):
            p_enc = None

        # TS encoder
        enc_nodes = self.ts_encoder(x)          # [B, N, C]
        q_t = enc_nodes.permute(0, 2, 1)        # [B, C, N]

        # ---------- (1) Temporal Cross (Prompt) : N→d_cm ----------
        q_t_cm = self.temporal_q_in(q_t)        # [B, C, d_cm]
        q_t_cm = self.ln_q_temporal_in(q_t_cm)

        if p_enc is not None:
            k_cm = self.temporal_kv_in(p_enc)   # [B, d_llm, d_cm]
            v_cm = self.temporal_kv_in(p_enc)   # [B, d_llm, d_cm]
            delta_t = self.cross_temporal(q_t_cm, k_cm, v_cm)  # [B, C, d_cm]
            z_t_cm = q_t_cm + F.softplus(self.g_t) * delta_t
        else:
            z_t_cm = q_t_cm

        z_t = self.temporal_out(z_t_cm)         # [B, C, N]
        z_t = z_t.permute(0, 2, 1)              # [B, N, C]

        # ---------- (2) Spatial Cross (Image) : N→d_cm ----------
        q_s = self.c_to_nodes(z_t)              # [B, N, N]
        q_s_cm = self.spatial_q_in(q_s)         # [B, N, d_cm]
        q_s_cm = self.ln_q_spatial_in(q_s_cm)

        k_img, v_img = self._prep_image_kv(emb_image, B=B, N=self.num_nodes, device=z_t.device)  # [B, d_img, N]
        if self.training and (k_img is not None) and (torch.rand(1).item() < self.p_drop_image):
            k_img, v_img = None, None

        if (k_img is not None) and (v_img is not None):
            k_img_cm = self.spatial_kv_in(k_img)  # [B, d_img, d_cm]
            v_img_cm = self.spatial_kv_in(v_img)  # [B, d_img, d_cm]
            k_img_cm = self.ln_img_kv(k_img_cm)
            v_img_cm = self.ln_img_kv(v_img_cm)
            delta_s = self.cross_spatial(q_s_cm, k_img_cm, v_img_cm)  # [B, N, d_cm]
            z_s_cm = q_s_cm + F.softplus(self.g_s) * delta_s
        else:
            z_s_cm = q_s_cm

        z_s = self.spatial_out(z_s_cm)          # [B, N, N]
        z = self.nodes_to_c(z_s)                # [B, N, C]
        z = self.ln_after_spatial(z)

        # Decoder & Projection
        dec_out = self.decoder(z, z)            # [B, N, C]
        dec_out = self.c_to_length(dec_out)     # [B, N, L_out]
        dec_out = dec_out.permute(0, 2, 1)      # [B, L_out, N]

        # RevIN denorm
        y_hat = self.normalize_layers(dec_out, 'denorm')
        return y_hat
