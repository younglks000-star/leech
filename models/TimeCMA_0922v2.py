import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal


class Dual(nn.Module):
    """
    STAEformer 흐름 + TimeCMA 결합 버전

    입력 텐서/임베딩 포맷
    ----------------------
    x            : [B, L, C]            시계열 값 (B=batch, L=input length, C=노드/채널 수)
    x_mark       : [B, L, d_time]       (현재 미사용: 필요 시 시간 인코더에 추가 가능)
    emb_prompt   : [B, E, C] 또는 [B, E, C, 1]
                   - '프롬프트(텍스트) 임베딩' 토큰 길이=E, 채널축=C
                   - Temporal Cross-Attn의 K,V로 사용됨
    emb_image    : dict {'K':[C, dk] 또는 [B, C, dk], 'V':[C, dv] 또는 [B, C, dv], 'meta': {...}}
                   - '이미지 임베딩' (예: ViT/DINO 등)
                   - Spatial Cross-Attn의 K,V로 사용됨
                   - K/V는 노드축= C 가 반드시 일치해야 함

    출력
    ----
    y_hat : [B, L_out, C]   (L_out = pred_len)

    핵심 메커니즘(요약)
    -------------------
    1) Temporal Self-Attention (STAEformer의 시간 블록):
       - 입력 x: [B, L, C] → 시간축(L)에서 멀티헤드 self-attn → x_t: [B, L, C]
    2) Temporal Cross-Attention (TimeCMA 아이디어 적용):
       - Q = x_t (시간 축 표현)
       - K,V = prompt encoder의 출력 p_enc (프롬프트 임베딩)
       - z_t = CrossAttn(Q=x_t, KV=p_enc): [B, L, C]
    3) Time→Nodes 선형 매핑:
       - z_t: [B, L, C] → z_t^T: [B, C, L] → Linear(L→N) → [B, C, N] → z_n: [B, N, C]
         (각 채널별로 시간열 L을 노드 수 N으로 요약)
    4) Spatial Cross-Attention:
       - Q = z_n (노드 축 표현)
       - K,V = 이미지 임베딩(k_img, v_img)로부터 만든 토큰열 (길이 dk/dv, d_model=C)
       - z_s = CrossAttn(Q=z_n, KV=image): [B, N, C]
    5) Decoder + Projection:
       - 디코더(옵션 성격)로 z_s 정제 후, Linear(C→L_out)로 각 노드별 예측 길이 산출

    주의
    ----
    - CrossModal은 입력 형식을 [B, seq_len, d_model]로 가정한다.
      여기서 d_model은 항상 C(채널 수)로 고정.
    - 프롬프트/이미지 임베딩의 마지막 차원은 반드시 d_model=C가 되도록 맞춘다(여기선 이미 그렇게 로드/전처리함).
    """

    def __init__(
        self,
        device="cuda",
        channel=32,          # C (= num_nodes)
        num_nodes=32,        # C와 동일하게 사용
        seq_len=96,          # L_in
        pred_len=96,         # L_out
        dropout_n=0.1,
        d_llm=768,           # 프롬프트 길이 E (참조용: 실제 d_model은 C)
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

        # 멀티헤드 조건(check): d_model % n_heads == 0 이어야 함
        assert (self.channel % self.head) == 0, \
            f"d_model(={self.channel}) must be divisible by n_heads(={self.head})"

        # RevIN (입력/출력 정규화/역정규화)
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)

        # ---------------------------------------------------------------------
        # 1) Temporal encoder (Self-Attention over time)
        #    d_model = channel(C), 입력 x: [B, L, C]  (L=시계열 길이)
        #    출력 x_t: [B, L, C]
        # ---------------------------------------------------------------------
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.temporal_encoder = nn.TransformerEncoder(
            self.temporal_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # ---------------------------------------------------------------------
        # 2) Prompt encoder
        #    입력 emb_prompt: [B, E, C] → 출력 p_enc: [B, E, C]
        #    (프롬프트를 d_model=C 기준으로 정제하여 K,V가 될 준비)
        # ---------------------------------------------------------------------
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(
            self.prompt_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # ---------------------------------------------------------------------
        # 3) Temporal Cross-Attention
        #    Q: x_t [B, L, C]  /  K,V: p_enc [B, E, C]
        #    출력: z_t [B, L, C]
        # ---------------------------------------------------------------------
        self.cross_temporal = CrossModal(
            d_model=self.channel, n_heads=self.head, d_ff=self.d_ff,
            norm='LayerNorm', attn_dropout=self.dropout_n, dropout=self.dropout_n,
            pre_norm=True, activation="gelu", res_attention=False, n_layers=1, store_attn=False
        ).to(self.device)

        # ---------------------------------------------------------------------
        # 4) Time → Nodes 선형 매핑
        #    z_t: [B, L, C] → (transpose) [B, C, L] → Linear(L→N) = [B, C, N] → (transpose) [B, N, C]
        #    각 채널(노드)별 시간열 L을 길이 N으로 요약(노드표현으로 치환)
        # ---------------------------------------------------------------------
        self.time_to_nodes = nn.Linear(self.seq_len, self.num_nodes, bias=True).to(self.device)

        # ---------------------------------------------------------------------
        # 5) Spatial Cross-Attention
        #    Q: z_n [B, N, C]
        #    K,V: 이미지 임베딩(k_img, v_img)에서 만든 토큰열 [B, dk/dv, C]
        #         (여기서 'seq_len'은 dk/dv, d_model은 C)
        #    출력: z_s [B, N, C]
        # ---------------------------------------------------------------------
        self.cross_spatial = CrossModal(
            d_model=self.channel, n_heads=self.head, d_ff=self.d_ff,
            norm='LayerNorm', attn_dropout=self.dropout_n, dropout=self.dropout_n,
            pre_norm=True, activation="gelu", res_attention=False, n_layers=1, store_attn=False
        ).to(self.device)

        # ---------------------------------------------------------------------
        # 6) Transformer Decoder & Projection
        #    디코더: [B, N, C] → [B, N, C]
        #    선형: channel(C) → pred_len(L_out)
        #    최종 출력: [B, L_out, N(=C)]
        # ---------------------------------------------------------------------
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=self.d_layer
        ).to(self.device)

        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    # ---------------- util ----------------
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------------- helpers ----------------
    @staticmethod
    def _ensure_prompt_shape(emb_prompt: torch.Tensor) -> torch.Tensor:
        """
        프롬프트 임베딩을 [B, E, C]로 강제 (뒤에 붙은 dummy 축이 있으면 제거)
        """
        if emb_prompt.ndim == 4:  # [B, E, C, 1]
            emb_prompt = emb_prompt.squeeze(-1)
        return emb_prompt

    def _prep_image_kv(self, emb_image, B, C, device):
        """
        이미지 임베딩(K/V)을 CrossModal 입력(K,V)에 맞는 형상으로 변환.

        입력
        ----
        emb_image['K']: [C, dk] 또는 [B, C, dk]
        emb_image['V']: [C, dv] 또는 [B, C, dv]

        처리
        ----
        1) 배치 축 보정: [C, d] → [B, C, d]
        2) C(노드 수) 정합성 검사
        3) (안전) dk != dv면 공통 최소길이에 맞춰 잘라냄
        4) CrossModal(d_model=C)에 맞춰 마지막 축을 C로 변환:
           [B, C, d] --permute--> [B, d, C]

        반환
        ----
        k_img: [B, dk', C]
        v_img: [B, dv', C]
        """
        # K/V 키명 호환(Kc/Vc로 저장된 경우도 받아줌)
        if isinstance(emb_image, dict):
            if 'K' not in emb_image and 'Kc' in emb_image:
                emb_image['K'] = emb_image['Kc']
            if 'V' not in emb_image and 'Vc' in emb_image:
                emb_image['V'] = emb_image['Vc']
            assert ('K' in emb_image) and ('V' in emb_image), \
                "emb_image must be a dict with keys 'K' and 'V' (or 'Kc'/'Vc')."
            K_img = emb_image['K']
            V_img = emb_image['V']
        else:
            raise TypeError("emb_image must be a dict holding 'K' and 'V' tensors")

        # numpy → torch
        if not torch.is_tensor(K_img):
            K_img = torch.as_tensor(K_img)
        if not torch.is_tensor(V_img):
            V_img = torch.as_tensor(V_img)

        # device/dtype
        K_img = K_img.to(device).float()
        V_img = V_img.to(device).float()

        # [C, d] → [B, C, d] (배치 복제), 이미 [B,C,d]면 그대로
        if K_img.dim() == 2:  # [C, dk]
            K_img = K_img.unsqueeze(0).expand(B, -1, -1)  # [B, C, dk]
        if V_img.dim() == 2:  # [C, dv]
            V_img = V_img.unsqueeze(0).expand(B, -1, -1)  # [B, C, dv]

        # C 정합성
        if K_img.size(1) != C or V_img.size(1) != C:
            raise RuntimeError(f"[image] C mismatch: K:{K_img.size()} V:{V_img.size()} vs C={C}")

        # dk != dv일 때 공통 최소길이에 맞춤 (대부분 dk==dv)
        if K_img.size(2) != V_img.size(2):
            min_d = min(K_img.size(2), V_img.size(2))
            K_img = K_img[:, :, :min_d]
            V_img = V_img[:, :, :min_d]

        # CrossModal(d_model=C) 입력 규격으로 맞춤: [B, d, C]
        k_img = K_img.permute(0, 2, 1).contiguous()  # [B, dk, C]
        v_img = V_img.permute(0, 2, 1).contiguous()  # [B, dv, C]
        return k_img, v_img

    # ---------------- forward ----------------
    def forward(self, input_data, input_data_mark, emb_prompt, emb_image):
        """
        Q/K/V가 어디서 오는지 (정확한 흐름)
        -----------------------------------
        시간 블록(Temporal):
          - Q := x_t              (temporal encoder 출력, [B, L, C])
          - K, V := p_enc         (prompt encoder 출력, [B, E, C])

        공간 블록(Spatial):
          - Q := z_n              (time→nodes 매핑 결과, [B, N, C])
          - K, V := (k_img, v_img)  이미지 K/V 변환 결과, [B, d_img, C]
                                     (여기서 d_img = dk = dv 토큰 길이)
        """
        # ---- 타입/장치 정리 ----
        x = input_data.float()                       # [B, L, C]
        # x_mark = input_data_mark.float()           # (현재 미사용)

        emb_prompt = self._ensure_prompt_shape(emb_prompt).float()   # [B, E, C]

        B, L, C = x.size()
        # 채널 수 일치성 검증 (모델 파라미터와 입력 데이터)
        assert C == self.channel == self.num_nodes, \
            f"C(channel) mismatch: x:{C}, model.channel:{self.channel}, model.num_nodes:{self.num_nodes}"

        # 이미지 임베딩 K/V 정리 (배치/디바이스/차원 → CrossModal 규격)
        k_img, v_img = self._prep_image_kv(emb_image, B=B, C=C, device=x.device)  # [B, dk, C], [B, dv, C]

        # ---- RevIN ----
        x = self.normalize_layers(x, 'norm')         # [B, L, C]

        # =========================================================
        # (1) Temporal Encoder (Self-Attention over time)
        #     입력 x: [B, L, C] → x_t: [B, L, C]
        # =========================================================
        x_t = self.temporal_encoder(x)

        # =========================================================
        # (2) Prompt Encoder (프롬프트를 d_model=C 기준으로 정제)
        #     emb_prompt: [B, E, C] → p_enc: [B, E, C]
        # =========================================================
        p_enc = self.prompt_encoder(emb_prompt)

        # =========================================================
        # (3) Temporal Cross-Attention
        #     Q=x_t([B,L,C]), K=p_enc([B,E,C]), V=p_enc([B,E,C])  → z_t: [B, L, C]
        #     = 시간축 표현을 '프롬프트' 쪽 정보로 보강
        # =========================================================
        z_t = self.cross_temporal(x_t, p_enc, p_enc)  # [B, L, C]

        # =========================================================
        # (4) Time → Nodes 변환
        #     z_t: [B, L, C] → z_t^T: [B, C, L] → Linear(L→N) → [B, C, N] → z_n: [B, N, C]
        #     = 시간 정보를 노드 축 표현으로 요약
        # =========================================================
        z_t_T = z_t.transpose(1, 2)                   # [B, C, L]
        z_n = self.time_to_nodes(z_t_T)               # [B, C, N]
        z_n = z_n.transpose(1, 2)                     # [B, N, C]

        # =========================================================
        # (5) Spatial Cross-Attention
        #     Q=z_n([B,N,C]), K=k_img([B,dk,C]), V=v_img([B,dv,C]) → z_s: [B, N, C]
        #     = 노드 간 표현을 '이미지' 쪽 정보로 보강
        # =========================================================
        z_s = self.cross_spatial(z_n, k_img, v_img)   # [B, N, C]

        # =========================================================
        # (6) Decoder + Projection
        #     디코더로 한 번 더 정제(선택적 성격) → Linear(C→L_out) → transpose
        # =========================================================
        dec_in = z_s                                   # [B, N, C]
        dec_out = self.decoder(dec_in, dec_in)         # [B, N, C] (tgt=memory 동일: residual 정제 용도)
        dec_out = self.c_to_length(dec_out)            # [B, N, L_out]
        dec_out = dec_out.transpose(1, 2)              # [B, L_out, N(=C)]

        # ---- RevIN 역정규화 ----
        dec_out = self.normalize_layers(dec_out, 'denorm')  # [B, L_out, C]
        return dec_out
