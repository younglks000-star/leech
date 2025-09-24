# dual_timecma_0923.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.StandardNorm import Normalize
# ⬇️ 새로 저장한 개선판을 사용 (파일명/경로만 네 프로젝트에 맞춰주면 됨)
from layers.Cross_Modal_Align_0923 import CrossModal


class Dual(nn.Module):
    """
    STAEformer 흐름 + TimeCMA 결합 (개선판)

    입력
    ----
    x            : [B, L, C]
    x_mark       : [B, L, d_time]   (현재 미사용)
    emb_prompt   : [B, E, C] 또는 [B, E, C, 1]
    emb_image    : dict {'K':[C, dk] 또는 [B,C,dk], 'V':[C, dv] 또는 [B,C,dv]}

    출력
    ----
    y_hat        : [B, L_out, C]
    """

    def __init__(
        self,
        device="cuda",
        channel=32,           # C (= num_nodes)
        num_nodes=32,         # = C
        seq_len=96,           # L_in
        pred_len=96,          # L_out
        dropout_n=0.1,
        d_llm=4096,           # 프롬프트 토큰 길이 E(라벨 아님; 실제 d_model은 C). 정보: pooling 전 길이 기준
        e_layer=1,            # temporal/prompt encoder layer 수
        d_layer=1,            # decoder layer 수
        d_ff=256,
        head=8,

        # --- 추가: 프롬프트 다운샘플 ---
        prompt_pool=True,
        prompt_pool_stride=16,       # 16배 avg-pool → E -> E/16
        prompt_pool_type="avg",      # 'avg' | 'max'
        min_prompt_len=32,           # 다운샘플 후 최소 길이 보장(너무 과도한 축소 방지)

        # --- 추가: CrossModal 옵션(Temporal/Spatial 각각 설정) ---
        use_pe_temporal=True,
        use_pe_spatial=True,
        pe_type="sincos",
        pe_drop=0.05,
        pe_targets="qk",
        use_gating=True,
        res_attention=True,          # Realformer residual attention
        cross_layers=1,              # CrossModal 내부 레이어 수
        lsa=False,                   # learnable temperature 사용하려면 True (Cross_Modal_Align_0923 내부 지원)
    ):
        super().__init__()

        self.device     = device
        self.channel    = channel
        self.num_nodes  = num_nodes
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.dropout_n  = dropout_n
        self.d_llm      = d_llm
        self.e_layer    = e_layer
        self.d_layer    = d_layer
        self.d_ff       = d_ff
        self.head       = head

        self.prompt_pool       = prompt_pool
        self.prompt_pool_stride= prompt_pool_stride
        self.prompt_pool_type  = prompt_pool_type
        self.min_prompt_len    = min_prompt_len

        assert (self.channel % self.head) == 0, \
            f"d_model(={self.channel}) must be divisible by n_heads(={self.head})"

        # RevIN
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)

        # -------------------------------
        # 1) Temporal encoder over time: x [B, L, C] -> [B, L, C]
        # -------------------------------
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.temporal_encoder = nn.TransformerEncoder(
            self.temporal_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # -------------------------------
        # 2) Prompt encoder: emb_prompt [B, E, C] -> [B, E, C]
        #    (d_model=C 기준으로 정제; K,V로 사용)
        # -------------------------------
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(
            self.prompt_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # -------------------------------
        # 3) Temporal Cross-Attention (Q=x_t, K/V=p_enc)
        # -------------------------------
        self.cross_temporal = CrossModal(
            d_model=self.channel, n_heads=self.head, d_ff=self.d_ff,
            norm='LayerNorm', attn_dropout=self.dropout_n, dropout=self.dropout_n,
            pre_norm=True, activation="gelu",
            res_attention=res_attention, n_layers=cross_layers, store_attn=False,
            use_pe=use_pe_temporal, pe_type=pe_type, pe_drop=pe_drop, pe_targets=pe_targets,
            use_gating=use_gating
        ).to(self.device)

        # -------------------------------
        # 4) Time → Nodes (Linear): [B, L, C] -> [B, N, C]
        # -------------------------------
        self.time_to_nodes = nn.Linear(self.seq_len, self.num_nodes, bias=True).to(self.device)

        # -------------------------------
        # 5) Spatial Cross-Attention (Q=z_n, K/V=image tokens)
        # -------------------------------
        self.cross_spatial = CrossModal(
            d_model=self.channel, n_heads=self.head, d_ff=self.d_ff,
            norm='LayerNorm', attn_dropout=self.dropout_n, dropout=self.dropout_n,
            pre_norm=True, activation="gelu",
            res_attention=res_attention, n_layers=cross_layers, store_attn=False,
            use_pe=use_pe_spatial, pe_type=pe_type, pe_drop=pe_drop, pe_targets=pe_targets,
            use_gating=use_gating
        ).to(self.device)

        # -------------------------------
        # 6) Decoder + Projection
        # -------------------------------
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=self.d_layer
        ).to(self.device)

        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

        # ---- prompt pooling 모듈 준비 (Avg/MaxPool1d) ----
        if self.prompt_pool:
            if self.prompt_pool_type == "max":
                self.prompt_pool_op = nn.AdaptiveMaxPool1d
            else:
                self.prompt_pool_op = nn.AdaptiveAvgPool1d

    # ---------------- util ----------------
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------------- helpers ----------------
    @staticmethod
    def _ensure_prompt_shape(emb_prompt: torch.Tensor) -> torch.Tensor:
        # [B,E,C,1] -> [B,E,C]
        if emb_prompt.ndim == 4:
            emb_prompt = emb_prompt.squeeze(-1)
        return emb_prompt

    def _pool_prompt_tokens(self, p: torch.Tensor) -> torch.Tensor:
        """
        프롬프트 토큰 다운샘플링.
        입력 p: [B, E, C]  → 출력: [B, E', C]
        - 기본: 16배 축소(혹은 stride 지정), 단 최소 길이 보장
        """
        if not self.prompt_pool:
            return p

        B, E, C = p.shape
        if E <= self.min_prompt_len:
            return p  # 너무 짧으면 스킵

        # 목표 길이 = max(E // stride, min_len)
        target = max(E // max(1, int(self.prompt_pool_stride)), self.min_prompt_len)

        # Pool1d는 [B, C, L] 입력을 기대하므로 차원 전환
        # 여기서 "채널" = C, "길이" = E
        x = p.transpose(1, 2)  # [B, C, E]
        pool = self.prompt_pool_op(target)  # Adaptive*Pool1d(output_size=target)
        x = pool(x)  # [B, C, target]
        p_ds = x.transpose(1, 2)  # [B, target, C]
        return p_ds

    def _prep_image_kv(self, emb_image, B, C, device):
        """
        이미지 임베딩(K/V) → CrossModal 형상으로 변환.
        emb_image['K']: [C, dk] 또는 [B, C, dk]
        emb_image['V']: [C, dv] 또는 [B, C, dv]
        반환: k_img, v_img  = [B, dk', C], [B, dv', C]
        """
        if isinstance(emb_image, dict):
            if 'K' not in emb_image and 'Kc' in emb_image:
                emb_image['K'] = emb_image['Kc']
            if 'V' not in emb_image and 'Vc' in emb_image:
                emb_image['V'] = emb_image['Vc']
            assert ('K' in emb_image) and ('V' in emb_image), \
                "emb_image must contain 'K' and 'V' (or 'Kc'/'Vc')."
            K_img = emb_image['K']
            V_img = emb_image['V']
        else:
            raise TypeError("emb_image must be a dict holding 'K' and 'V' tensors")

        if not torch.is_tensor(K_img):
            K_img = torch.as_tensor(K_img)
        if not torch.is_tensor(V_img):
            V_img = torch.as_tensor(V_img)

        K_img = K_img.to(device).float()
        V_img = V_img.to(device).float()

        # [C, d] → [B, C, d] 복제
        if K_img.dim() == 2:
            K_img = K_img.unsqueeze(0).expand(B, -1, -1)
        if V_img.dim() == 2:
            V_img = V_img.unsqueeze(0).expand(B, -1, -1)

        # C 정합성
        if K_img.size(1) != C or V_img.size(1) != C:
            raise RuntimeError(f"[image] C mismatch: K:{K_img.size()} V:{V_img.size()} vs C={C}")

        # 길이 일치
        if K_img.size(2) != V_img.size(2):
            min_d = min(K_img.size(2), V_img.size(2))
            K_img = K_img[:, :, :min_d]
            V_img = V_img[:, :, :min_d]

        # [B, C, d] -> [B, d, C]
        k_img = K_img.permute(0, 2, 1).contiguous()
        v_img = V_img.permute(0, 2, 1).contiguous()
        return k_img, v_img

    # ---------------- forward ----------------
    def forward(self, input_data, input_data_mark, emb_prompt, emb_image):
        """
        Temporal: Q=x_t [B,L,C],       K/V=p_enc [B,E',C]
        Spatial : Q=z_n [B,N,C],       K/V=image [B,d_img,C]
        """
        # ---- 타입/장치 정리 ----
        x = input_data.float()                         # [B, L, C]
        emb_prompt = self._ensure_prompt_shape(emb_prompt).float()  # [B, E, C]

        B, L, C = x.size()
        assert C == self.channel == self.num_nodes, \
            f"C(channel) mismatch: x:{C}, model.channel:{self.channel}, model.num_nodes:{self.num_nodes}"

        # 이미지 임베딩 K/V 준비
        k_img, v_img = self._prep_image_kv(emb_image, B=B, C=C, device=x.device)  # [B, dk, C], [B, dv, C]

        # ---- RevIN ----
        x = self.normalize_layers(x, 'norm')           # [B, L, C]

        # =========================================================
        # (1) Temporal Encoder
        # =========================================================
        x_t = self.temporal_encoder(x)                 # [B, L, C]

        # =========================================================
        # (2) Prompt Encoder (+ 다운샘플)
        # =========================================================
        # 프롬프트 토큰 다운샘플 (E → E')
        emb_prompt = self._pool_prompt_tokens(emb_prompt)   # [B, E', C]
        p_enc = self.prompt_encoder(emb_prompt)             # [B, E', C]

        # =========================================================
        # (3) Temporal Cross-Attention: x_t ⟵ p_enc
        # =========================================================
        z_t = self.cross_temporal(x_t, p_enc, p_enc)        # [B, L, C]

        # =========================================================
        # (4) Time → Nodes (Linear over time)
        # =========================================================
        z_t_T = z_t.transpose(1, 2)                         # [B, C, L]
        z_n = self.time_to_nodes(z_t_T)                     # [B, C, N]
        z_n = z_n.transpose(1, 2)                           # [B, N, C]

        # =========================================================
        # (5) Spatial Cross-Attention: z_n ⟵ image
        # =========================================================
        z_s = self.cross_spatial(z_n, k_img, v_img)         # [B, N, C]

        # =========================================================
        # (6) Decoder + Projection
        # =========================================================
        dec_in = z_s                                        # [B, N, C]
        dec_out = self.decoder(dec_in, dec_in)              # [B, N, C]
        dec_out = self.c_to_length(dec_out)                 # [B, N, L_out]
        dec_out = dec_out.transpose(1, 2)                   # [B, L_out, N(=C)]

        # ---- RevIN 역정규화 ----
        dec_out = self.normalize_layers(dec_out, 'denorm')  # [B, L_out, C]
        return dec_out
