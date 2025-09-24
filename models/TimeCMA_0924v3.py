import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal


class Dual(nn.Module):
    """
    TimeCMA 스타일의 2-단 크로스 모달 구조 (Temporal → Spatial).

    파워업 포인트
    -------------
    - Time Fourier feature 주입(항상 입력 L과 정합)
    - 프롬프트 길이 정규화(AdaptiveAvgPool1d → d_llm)
    - Modality Dropout(prompt/image)
    - Cross 입력 전/후 LayerNorm으로 안정성 강화
    - emb_prompt / emb_image 가 None 이어도 안전하게 스킵
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

        # --- 추가 옵션 ---
        use_time_feat=True,
        time_fourier_periods=(7.0, 30.4375, 91.3125, 365.25),  # 일/월/계절/연 주기(예시)
        p_drop_prompt=0.15,
        p_drop_image=0.15,
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

        # --- RevIN ---
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)

        # --- 시간 포리에 특징을 N 차원으로 사상해 x([B,L,N])에 더함 ---
        if self.use_time_feat:
            self.time_ff = nn.Sequential(
                nn.Linear(2 * len(self.time_fourier_periods), self.num_nodes, bias=False),
                nn.GELU()
            ).to(self.device)
            nn.init.xavier_uniform_(self.time_ff[0].weight)

        # --- [B,N,L] → [B,N,C] ---
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.ln_after_len2feat = nn.LayerNorm(self.channel).to(self.device)

        # --- (A) Time-Series Encoder: [B,N,C] ---
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(
            self.ts_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # --- (B) Prompt Encoder: [B,N,d_llm] ---
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(
            self.prompt_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # 프롬프트 길이 정규화(항상 d_llm로 맞춤). 입력 p: [B,N,E] → pool → [B,N,d_llm]
        self.prompt_pool = nn.AdaptiveAvgPool1d(self.d_llm)
        self.ln_prompt_in = nn.LayerNorm(self.d_llm).to(self.device)

        # --- (C) Temporal Cross (Prompt) ---
        # 마지막 축 d_model = N
        self.cross = CrossModal(
            d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
            activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)
        # Temporal Cross 입력 전 정규화(마지막 축=N)
        self.ln_q_temporal_in = nn.LayerNorm(self.num_nodes).to(self.device)

        # --- (D) Spatial Cross (Image) ---
        self.c_to_nodes = nn.Linear(self.channel, self.num_nodes, bias=False).to(self.device)   # [B,N,C] -> [B,N,N]
        self.nodes_to_c = nn.Linear(self.num_nodes, self.channel, bias=False).to(self.device)   # [B,N,N] -> [B,N,C]
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

        # --- (F) Decoder & Projection ---
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=self.d_layer
        ).to(self.device)

        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)
        self.ln_after_spatial = nn.LayerNorm(self.channel).to(self.device)

    # ---------- utils ----------
    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _ensure_prompt_shape(p):
        # None 허용 + [B,E,N,1] → [B,E,N]
        if p is None:
            return None
        return p.squeeze(-1) if (p.dim() == 4) else p

    @staticmethod
    def _build_time_fourier(L: int, device, dtype, periods):
        # [L, 2K] (sin/cos 쌍을 L 길이에 맞춰 생성)
        t = torch.arange(L, device=device, dtype=dtype)
        feats = []
        for p in periods:
            w = 2.0 * torch.pi * t / p
            feats.extend([torch.sin(w), torch.cos(w)])
        return torch.stack(feats, dim=1).transpose(0, 1).transpose(0, 1)  # [L,2K]

    def _prep_image_kv(self, emb_image, B, N, device):
        # None 허용
        if emb_image is None:
            return None, None

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
        K = K.transpose(1, 2).contiguous()  # [B, d, N]
        V = V.transpose(1, 2).contiguous()
        return K, V

    def gate_values(self):
        return float(F.softplus(self.g_t).detach().cpu()), float(F.softplus(self.g_s).detach().cpu())

    # ---------- forward ----------
    def forward(self, input_data, input_data_mark, emb_prompt, emb_image):
        """
        input_data : [B, L, N]
        emb_prompt : [B, E, N] or [B, E, N, 1] or None
        emb_image  : {'K':[B,N,dk]|[N,dk], 'V':[B,N,dv]|[N,dv]} or None
        return     : [B, L_out, N]
        """
        x = input_data.float()                  # [B, L, N]
        p = self._ensure_prompt_shape(emb_prompt)  # [B,E,N] or None

        B, L, N = x.size()
        assert N == self.num_nodes, f"N mismatch: input {N} vs model {self.num_nodes}"

        # RevIN
        x = self.normalize_layers(x, 'norm')    # [B, L, N]

        # (a) 시간 포리에 특징 주입 (항상 L 일치)
        if self.use_time_feat:
            tf = self._build_time_fourier(L, device=x.device, dtype=x.dtype, periods=self.time_fourier_periods)  # [L,2K]
            tf_proj = self.time_ff(tf)          # [L, N]
            x = x + tf_proj.unsqueeze(0)        # [B, L, N]

        # [B,L,N] -> [B,N,L] -> L→C
        x = x.permute(0, 2, 1)                  # [B, N, L]
        x = self.length_to_feature(x)           # [B, N, C]
        x = self.ln_after_len2feat(x)           # 안정화

        # Prompt: [B,E,N] -> [B,N,E] -> (pool to d_llm) -> LN -> Encoder -> [B,N,d_llm] -> [B,E,N]
        if p is not None:
            p = p.permute(0, 2, 1)              # [B, N, E]
            p = self.prompt_pool(p)             # [B, N, d_llm]
            p = self.ln_prompt_in(p)            # LN
            p_enc = self.prompt_encoder(p)      # [B, N, d_llm]
            p_enc = p_enc.permute(0, 2, 1)      # [B, d_llm, N] (Cross의 K/V)
        else:
            p_enc = None

        # Modality Dropout (학습시에만)
        if self.training and (p_enc is not None) and (torch.rand(1).item() < self.p_drop_prompt):
            p_enc = None

        # TS encoder
        enc_nodes = self.ts_encoder(x)          # [B, N, C]
        q_t = enc_nodes.permute(0, 2, 1)        # [B, C, N]
        q_t = self.ln_q_temporal_in(q_t)        # Temporal Cross 입력 정규화

        # (1) Temporal Cross (Prompt): Q=[B,C,N], KV=[B,E,N]  (없으면 스킵)
        if p_enc is not None:
            delta_t = self.cross(q_t, p_enc, p_enc)  # [B, C, N]
            z_t = q_t + F.softplus(self.g_t) * delta_t
        else:
            z_t = q_t
        z_t = z_t.permute(0, 2, 1)              # [B, N, C]

        # (2) Spatial Cross (Image): Q=C→N, KV=image
        q_s = self.c_to_nodes(z_t)              # [B, N, N]
        q_s = self.ln_q_spatial_in(q_s)         # 안정화

        k_img, v_img = self._prep_image_kv(emb_image, B=B, N=self.num_nodes, device=z_t.device)  # [B, d, N] or (None,None)
        if self.training and (k_img is not None) and (torch.rand(1).item() < self.p_drop_image):
            k_img, v_img = None, None

        if (k_img is not None) and (v_img is not None):
            k_img = self.img_k_proj(k_img)      # [B, d, N]
            v_img = self.img_v_proj(v_img)      # [B, d, N]
            k_img = self.ln_img_kv(k_img)
            v_img = self.ln_img_kv(v_img)
            delta_s = self.cross_spatial(q_s, k_img, v_img)  # [B, N, N]
            z_s = q_s + F.softplus(self.g_s) * delta_s
        else:
            z_s = q_s

        z = self.nodes_to_c(z_s)                # [B, N, C]
        z = self.ln_after_spatial(z)

        # Decoder & Projection
        dec_out = self.decoder(z, z)            # [B, N, C]
        dec_out = self.c_to_length(dec_out)     # [B, N, L_out]
        dec_out = dec_out.permute(0, 2, 1)      # [B, L_out, N]

        # RevIN denorm
        y_hat = self.normalize_layers(dec_out, 'denorm')
        return y_hat
