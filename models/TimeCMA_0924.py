import torch
import torch.nn as nn
from einops import rearrange
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal
import torch.nn.functional as F


class Dual(nn.Module):
    """
    TimeCMA 스타일의 2-단 크로스 모달 구조 (Temporal → Spatial).

    -------------
    1) RevIN으로 시계열 표준화 후, [B,N,L] → Linear → [B,N,C] 로 "노드별 임베딩"을 만든다.
    2) Temporal-Cross (Prompt):
       - Q  : 시계열 인코딩 결과를 N-공간(feature)으로 본 [B,C,N]
       - K,V: 프롬프트 인코딩을 N-공간(feature)으로 본 [B,E,N]
       - d_model = N (마지막 축)로 맞춰 CrossModal 수행 → Δ_t
       - Residual + gate(softplus(g_t))로 주입: z_t = q_t + g_t * Δ_t
    3) Spatial-Cross (Image):
       - Q: z_t를 C→N으로 사상한 [B,N,N] (토큰=N개, feature=N)
       - K,V: 클러스터(노드)별 이미지/지형 설명자 임베딩 [B,E_img,N]
       - CrossModal로 Δ_s 계산 후 residual + gate(softplus(g_s))로 주입
       - 다시 N→C로 복원하여 디코더 투입
    4) Transformer Decoder로 [B,N,C] → [B,N,L_out] 예측 후, RevIN 역정규화.

    장점
    ----
    - Prompt는 "시간적 단서(서술/지식)"를, Image는 "공간적/지형적 단서"를
      각각 다른 단계에서 안정적으로 주입(게이트로 세기 제어).
    - 마지막 축을 N(노드 공간)으로 고정해 CrossModal의 '특징 차원'을 일관되게 둠.
    """

    def __init__(
        self,
        device="cuda:7",
        channel=32,          # C: 노드 임베딩 채널 수(작게 두어 안정/경량)
        num_nodes=7,         # N: 노드/클러스터 개수
        seq_len=96,          # L_in: 입력 길이
        pred_len=96,         # L_out: 출력 길이
        dropout_n=0.1,
        d_llm=768,           # 프롬프트 임베딩 입력 차원(E의 feature차원)
        e_layer=1,
        d_layer=1,
        d_ff=32,
        head=8,
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

        # --- RevIN: 샘플별 정규화/역정규화로 분포 드리프트 완화 ---
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)

        # --- 길이(L) 축을 채널(C)로 압축: [B,N,L] → [B,N,C] ---
        #     (Temporal 패턴을 노드별 고정 길이 표현으로 변환)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)

        # --- (A) Time-Series Encoder: 노드 토큰(N개), 피처 C ---
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(
            self.ts_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # --- (B) Prompt Encoder: 노드 토큰(N개), 프롬프트 피처 d_llm ---
        #     (프롬프트는 노드별로 정렬된 텍스트 임베딩 토큰 [B,N,E]를 입력으로 가정)
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(
            self.prompt_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # --- (C) Temporal Cross (Prompt) ---
        # CrossModal은 마지막 차원(d_model)을 feature로 본다 → d_model=N 으로 통일.
        # Q: [B,C,N], K/V: [B,E,N] 꼴로 두기 위해 마지막 축=N으로 맞춘다.
        self.cross = CrossModal(
            d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
            activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)

        # --- (D) Spatial Cross (Image) ---
        # Q를 C→N으로 사상해 [B,N,N]로 만든 뒤, 이미지 K/V와 교차정렬.
        self.c_to_nodes = nn.Linear(self.channel, self.num_nodes, bias=False).to(self.device)   # [B,N,C] -> [B,N,N]
        self.nodes_to_c = nn.Linear(self.num_nodes, self.channel, bias=False).to(self.device)   # [B,N,N] -> [B,N,C]

        # 이미지 K,V는 [B,E_img,N] (E_img=이미지 토큰 수, 마지막 축=N)로 Cross 입력
        # 아래 Linear는 N→N 투영(안정화용; scale/회전 조정)
        self.img_k_proj = nn.Linear(self.num_nodes, self.num_nodes, bias=False).to(self.device)
        self.img_v_proj = nn.Linear(self.num_nodes, self.num_nodes, bias=False).to(self.device)

        self.cross_spatial = CrossModal(
            d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
            activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)

        # --- (E) Residual 주입 강도 게이트(softplus) ---
        #     (학습 중 자연스럽게 주입 세기가 조절됨. 초기 0.1 → 과한 주입 방지)
        self.g_t = nn.Parameter(torch.tensor(0.1))  # Temporal-Cross 용
        self.g_s = nn.Parameter(torch.tensor(0.1))  # Spatial-Cross  용

        # --- (F) Decoder & Projection ---
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=self.d_layer
        ).to(self.device)

        # [B,N,C] → [B,N,L_out]
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    # ===== 유틸 =====
    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _ensure_prompt_shape(p: torch.Tensor) -> torch.Tensor:
        # 프롬프트가 [B,E,N,1]로 들어오면 마지막 1차원 제거
        return p.squeeze(-1) if (p is not None and p.dim() == 4) else p

    def _prep_image_kv(self, emb_image, B, N, device):
        """
        emb_image: dict {'K':[B,N,dk]|[N,dk], 'V':[B,N,dv]|[N,dv]}
          - N축(두 번째 차원)이 클러스터 순서와 일치해야 함.
          - 배치 없이 [N,d]로 저장된 정적 설명자도 허용(자동으로 B로 broadcast).
        반환:
          K_img, V_img: [B, E_img, N] (E_img=dk 또는 dv)
        """
        assert isinstance(emb_image, dict) and ('K' in emb_image) and ('V' in emb_image), \
            "emb_image must be dict with keys 'K' and 'V'"

        K, V = emb_image['K'], emb_image['V']
        if not torch.is_tensor(K): K = torch.as_tensor(K)
        if not torch.is_tensor(V): V = torch.as_tensor(V)
        K = K.to(device).float()
        V = V.to(device).float()

        # 배치 축 보정: [N,d] → [B,N,d]
        if K.dim() == 2:  # 정적 K (배치 무관)
            K = K.unsqueeze(0).expand(B, -1, -1)
        if V.dim() == 2:
            V = V.unsqueeze(0).expand(B, -1, -1)

        # N축 검사 (노드 정렬 불일치 시 성능 급락)
        assert K.size(1) == N and V.size(1) == N, \
            f"emb_image K/V must have N={N} at dim=1, got {tuple(K.size())}, {tuple(V.size())}"

        # CrossModal 입력 규격: 마지막 축 = d_model = N.
        # 따라서 [B,N,d] → [B,d,N] (E_img가 '토큰 수'가 되고, N이 feature가 된다)
        K = K.transpose(1, 2).contiguous()  # [B, E_img, N]
        V = V.transpose(1, 2).contiguous()  # [B, E_img, N]
        return K, V

    def gate_values(self):
        # 모니터링용: 현재 주입 게이트 세기(softplus 변환 후)
        return float(F.softplus(self.g_t).detach().cpu()), float(F.softplus(self.g_s).detach().cpu())

    # ===== Forward =====
    def forward(self, input_data, input_data_mark, emb_prompt, emb_image):
        """
        input_data : [B, L, N]  — 시계열 (시간 먼저, 노드 뒤)
        emb_prompt : [B, E, N] 또는 [B, E, N, 1] — 노드별 프롬프트 토큰열
        emb_image  : {'K':[B,N,dk]|[N,dk], 'V':[B,N,dv]|[N,dv]} — 노드별 이미지/지형 설명자
        반환       : [B, L_out, N]
        """
        # 0) 타입/모양 체크
        x = input_data.float()                   # [B, L, N]
        _ = input_data_mark                     # 현재 미사용(자리만)
        p = self._ensure_prompt_shape(emb_prompt).float()

        B, L, N = x.size()
        assert N == self.num_nodes, f"N mismatch: input {N} vs model {self.num_nodes}"

        # 1) RevIN 정규화 (샘플별 통계)
        x = self.normalize_layers(x, 'norm')     # [B, L, N]

        # 2) [B,L,N] → [B,N,L] → 길이→채널 사상 → [B,N,C]
        x = x.permute(0, 2, 1)                   # [B, N, L]
        x = self.length_to_feature(x)            # [B, N, C]

        # 3) 프롬프트 인코딩: [B,E,N] → [B,N,E] → Encoder → [B,N,E] → Cross용 [B,E,N]
        p = p.permute(0, 2, 1)                   # [B, N, E]
        p_enc = self.prompt_encoder(p)           # [B, N, E]
        p_enc = p_enc.permute(0, 2, 1)           # [B, E, N]

        # 4) TS 인코딩: [B,N,C] → [B,C,N] (Cross의 Q 규격에 맞춤)
        enc_nodes = self.ts_encoder(x)           # [B, N, C]
        q_t = enc_nodes.permute(0, 2, 1)         # [B, C, N]

        # 5) Temporal-Cross (Prompt): Q=[B,C,N], K/V=[B,E,N], d_model=N
        delta_t = self.cross(q_t, p_enc, p_enc)  # [B, C, N]
        z_t = q_t + F.softplus(self.g_t) * delta_t   # residual + gate
        z_t = z_t.permute(0, 2, 1)               # [B, N, C]

        # 6) Spatial-Cross (Image):
        #    Q를 C→N으로 사상해 [B,N,N] (토큰=N, feature=N)
        q_s = self.c_to_nodes(z_t)               # [B, N, N]

        #    이미지 K,V 준비: [B,N,d] → [B,E_img,N] (토큰=E_img, feature=N)
        k_img, v_img = self._prep_image_kv(emb_image, B=B, N=self.num_nodes, device=z_t.device)
        k_img = self.img_k_proj(k_img)           # [B, E_img, N] (안정 투영)
        v_img = self.img_v_proj(v_img)           # [B, E_img, N]

        #    Cross: 노드 토큰(N개)이 이미지 토큰(E_img개)을 참고해 N-공간에서 보정
        delta_s = self.cross_spatial(q_s, k_img, v_img)  # [B, N, N]
        z_s = q_s + F.softplus(self.g_s) * delta_s       # residual + gate

        #    다시 N→C로 복원 → 디코더 입력
        z = self.nodes_to_c(z_s)                 # [B, N, C]

        # 7) 디코더 & 길이 투영: [B,N,C] → [B,N,L_out] → [B,L_out,N]
        dec_out = self.decoder(z, z)             # [B, N, C]
        dec_out = self.c_to_length(dec_out)      # [B, N, L_out]
        dec_out = dec_out.permute(0, 2, 1)       # [B, L_out, N]

        # 8) RevIN 역정규화
        y_hat = self.normalize_layers(dec_out, 'denorm')
        return y_hat
