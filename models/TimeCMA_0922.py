import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal


class Dual(nn.Module):
    """
    STAEformer 흐름 + TimeCMA 결합 버전

    입력:
      x            : [B, L, C]           (시계열 값, L=입력 길이, C=노드/채널 수)
      x_mark       : [B, L, d_time]      (현재 미사용)
      emb_prompt   : [B, E, C] 또는 [B, E, C, 1] (프롬프트 임베딩, E=프롬프트 길이)
      emb_image    : dict {'K':[C, dk] 또는 [B, C, dk], 'V':[C, dv] 또는 [B, C, dv], 'meta': {...}}

    출력:
      y_hat        : [B, L_out, C]
    """
    def __init__(
        self,
        device="cuda",
        channel=32,          # C (= num_nodes)
        num_nodes=32,        # C와 동일하게 사용
        seq_len=96,          # L_in
        pred_len=96,         # L_out
        dropout_n=0.1,
        d_llm=768,           # 프롬프트 길이 E (참조용)
        e_layer=1,           # temporal self-attn layer 수
        d_layer=1,           # decoder layer 수
        d_ff=256,
        head=8
    ):
        super().__init__()
        self.device     = device
        self.channel    = channel          # = C (= num_nodes)
        self.num_nodes  = num_nodes        # = C
        self.seq_len    = seq_len          # = L
        self.pred_len   = pred_len         # = L_out
        self.dropout_n  = dropout_n
        self.d_llm      = d_llm
        self.e_layer    = e_layer
        self.d_layer    = d_layer
        self.d_ff       = d_ff
        self.head       = head

        # 멀티헤드 조건
        assert (self.channel % self.head) == 0, \
            f"d_model(={self.channel}) must be divisible by n_heads(={self.head})"

        # RevIN (입력/출력 정규화/역정규화)
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)

        # ------------------------------
        # 1) Temporal encoder (Self-Attention over time)
        #    d_model = channel(C), 입력 [B, L, C]
        # ------------------------------
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.temporal_encoder = nn.TransformerEncoder(
            self.temporal_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # ------------------------------
        # 2) Prompt encoder (프롬프트 임베딩을 d_model=C 기준으로 정제)
        #    입력 [B, E, C] → 출력 [B, E, C]
        # ------------------------------
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(
            self.prompt_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # ------------------------------
        # 3) Temporal Cross-Attention
        #    Q: [B, L, C] (시간축 Query)
        #    K,V: [B, E, C] (프롬프트)
        #    → [B, L, C]
        # ------------------------------
        self.cross_temporal = CrossModal(
            d_model=self.channel, n_heads=self.head, d_ff=self.d_ff,
            norm='LayerNorm', attn_dropout=self.dropout_n, dropout=self.dropout_n,
            pre_norm=True, activation="gelu", res_attention=False, n_layers=1, store_attn=False
        ).to(self.device)

        # ------------------------------
        # 4) Time → Nodes 변환
        #    [B, L, C] --T--> [B, C, L] --Linear(L→N)--> [B, C, N] --T--> [B, N, C]
        # ------------------------------
        self.time_to_nodes = nn.Linear(self.seq_len, self.num_nodes, bias=True).to(self.device)

        # ------------------------------
        # 5) Spatial Cross-Attention
        #    Q: [B, N, C]  (노드 축 Query)
        #    K,V: 이미지 임베딩에서 (B?, C, dk/dv) → [B, dk/dv, C]로 변환(토큰 길이=dk/dv, d_model=C)
        #    → [B, N, C]
        # ------------------------------
        self.cross_spatial = CrossModal(
            d_model=self.channel, n_heads=self.head, d_ff=self.d_ff,
            norm='LayerNorm', attn_dropout=self.dropout_n, dropout=self.dropout_n,
            pre_norm=True, activation="gelu", res_attention=False, n_layers=1, store_attn=False
        ).to(self.device)

        # ------------------------------
        # 6) Transformer Decoder & Projection
        #    디코더 입력/메모리: [B, N, C] → [B, N, C]
        #    선형: channel(C) → pred_len(L_out), 노드별 예측 길이 산출
        # ------------------------------
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=self.d_layer
        ).to(self.device)

        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    # ---- utils ----
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---- helpers ----
    @staticmethod
    def _ensure_prompt_shape(emb_prompt: torch.Tensor) -> torch.Tensor:
        # [B, E, C] or [B, E, C, 1] -> [B, E, C]
        if emb_prompt.ndim == 4:
            emb_prompt = emb_prompt.squeeze(-1)
        return emb_prompt

    def _prep_image_kv(self, emb_image, B, C, device):
        """
        emb_image['K']: [C, dk] 또는 [B, C, dk]
        emb_image['V']: [C, dv] 또는 [B, C, dv]
        반환:
          k_img: [B, dk, C]
          v_img: [B, dv, C]
        """
        # K/V 키명 호환(Kc/Vc로 저장된 경우도 받아줌)
        if isinstance(emb_image, dict):
            if 'K' not in emb_image and 'Kc' in emb_image: emb_image['K'] = emb_image['Kc']
            if 'V' not in emb_image and 'Vc' in emb_image: emb_image['V'] = emb_image['Vc']
            assert ('K' in emb_image) and ('V' in emb_image), \
                "emb_image must be a dict with keys 'K' and 'V' (or 'Kc'/'Vc')."
            K_img = emb_image['K']
            V_img = emb_image['V']
        else:
            raise TypeError("emb_image must be a dict holding 'K' and 'V' tensors")

        if not torch.is_tensor(K_img): K_img = torch.as_tensor(K_img)
        if not torch.is_tensor(V_img): V_img = torch.as_tensor(V_img)

        # CPU -> device
        K_img = K_img.to(device).float()
        V_img = V_img.to(device).float()

        # [C, d] -> [B, C, d],  [B, C, d] -> 그대로
        if K_img.dim() == 2:  # [C, dk]
            K_img = K_img.unsqueeze(0).expand(B, -1, -1)  # [B, C, dk]
        if V_img.dim() == 2:  # [C, dv]
            V_img = V_img.unsqueeze(0).expand(B, -1, -1)  # [B, C, dv]

        # 유효성 체크
        if K_img.size(1) != C or V_img.size(1) != C:
            raise RuntimeError(f"[image] C mismatch: K:{K_img.size()} V:{V_img.size()} vs C={C}")
        if K_img.size(2) != V_img.size(2):
            # K/V 길이가 다르면 최소 길이에 맞춰 잘라서 정합성 보장 (일반적으로 dk==dv)
            min_d = min(K_img.size(2), V_img.size(2))
            K_img = K_img[:, :, :min_d]
            V_img = V_img[:, :, :min_d]

        # CrossModal(d_model=C)에 맞춰 마지막 축을 C로: [B, dk, C] / [B, dv, C]
        k_img = K_img.permute(0, 2, 1).contiguous()
        v_img = V_img.permute(0, 2, 1).contiguous()
        return k_img, v_img

    # ---- forward ----
    def forward(self, input_data, input_data_mark, emb_prompt, emb_image):
        """
        input_data     : [B, L, C]
        input_data_mark: [B, L, d_time]  (현재 미사용)
        emb_prompt     : [B, E, C] or [B, E, C, 1]
        emb_image      : {'K':[C, dk] or [B, C, dk], 'V':[C, dv] or [B, C, dv], 'meta': {...}}
        """
        # ---- 타입/장치 정리 ----
        x = input_data.float()                       # [B, L, C]
        # x_mark = input_data_mark.float()           # 필요 시 사용

        emb_prompt = self._ensure_prompt_shape(emb_prompt).float()   # [B, E, C]

        B, L, C = x.size()
        assert C == self.channel == self.num_nodes, \
            f"C(channel) mismatch: x:{C}, model.channel:{self.channel}, model.num_nodes:{self.num_nodes}"

        # 이미지 임베딩 K/V 정리 (배치/디바이스/차원)
        k_img, v_img = self._prep_image_kv(emb_image, B=B, C=C, device=x.device)  # [B, dk, C], [B, dv, C]

        # ---- RevIN ----
        x = self.normalize_layers(x, 'norm')         # [B, L, C]

        # =========================================================
        # (1) Temporal Encoder (Self-Attention over time)
        # =========================================================
        x_t = self.temporal_encoder(x)               # [B, L, C]

        # =========================================================
        # (2) Prompt Encoder (d_model=C)
        # =========================================================
        p_enc = self.prompt_encoder(emb_prompt)      # [B, E, C]

        # =========================================================
        # (3) Temporal Cross-Attention: Q=x_t (time), KV=prompt
        # =========================================================
        z_t = self.cross_temporal(x_t, p_enc, p_enc) # [B, L, C]

        # =========================================================
        # (4) Time → Nodes 변환: [B, L, C] → [B, N, C]
        # =========================================================
        z_t_T = z_t.transpose(1, 2)                  # [B, C, L]
        z_n = self.time_to_nodes(z_t_T)              # [B, C, N]
        z_n = z_n.transpose(1, 2)                    # [B, N, C]

        # =========================================================
        # (5) Spatial Cross-Attention: Q=z_n (nodes), KV=image
        # =========================================================
        # CrossModal은 [B, q_len, d_model] 형식(d_model=C) 가정.
        # 여기서 q_len=N, k_len=v_len=dk(=dv)
        z_s = self.cross_spatial(z_n, k_img, v_img)  # [B, N, C]

        # =========================================================
        # (6) Decoder + Projection
        # =========================================================
        dec_in = z_s                                  # [B, N, C]
        dec_out = self.decoder(dec_in, dec_in)        # [B, N, C] (tgt=memory)
        dec_out = self.c_to_length(dec_out)           # [B, N, L_out]
        dec_out = dec_out.transpose(1, 2)             # [B, L_out, N(=C)]

        # ---- RevIN 역정규화 ----
        dec_out = self.normalize_layers(dec_out, 'denorm')  # [B, L_out, C]
        return dec_out
