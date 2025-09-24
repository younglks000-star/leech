import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal

class Dual(nn.Module):
    """
    TimeCMA(Dual) - Enhanced (주석 상세 버전)
   
    - [TEMPO]   경량 시계열 분해(Trend/Seasonal/Residual) 후 각 성분을 개별 임베딩 → 재합성
    - [Time-LLM / AutoTimes / SMETimes]
        · 타임스탬프/도메인 메타데이터, 통계 요약(mean/std)을 프롬프트로 주입(FiLM-like)
        · 시간 인덱스(타임스탬프)를 텍스트 토큰처럼 전달하는 아이디어를 수치 임베딩으로 구현
        · 작은 모델에도 효과적이도록 간단한 통계 프롬프트 채택
    - [CALF]    프롬프트/이미지 임베딩(LLM 공간)을 TS 특성 공간(channel=C)으로 사상 + 정렬 손실
    - (기타)    기존 I/O, Cross-Modal 구조, 디코더/프로젝션 인터페이스는 그대로 유지

    입력/출력 텐서 규약
    - input_data:      [B, L_in, N]   시계열 원본 (노드/클러스터 N개)
    - input_data_mark: [B, L_in, Dm]  시간/달력/위치 등 메타(선택)
    - embeddings:      [B, E, N] 혹은 [B, E, N, 1]  (프롬프트/이미지/텍스트 임베딩)
    - 출력:            [B, L_out, N]
    """
    def __init__(
        self,
        device="cuda:0",
        channel=32,          # C: TS 특성 차원 (트랜스포머 d_model)
        num_nodes=7,         # N: 노드(클러스터) 수
        seq_len=96,          # L_in: 입력 길이
        pred_len=96,         # L_out: 예측 길이
        dropout_n=0.1,
        d_llm=768,           # E: LLM 임베딩(프롬프트/이미지) 차원
        e_layer=1,
        d_layer=1,
        d_ff=64,
        head=8,
        # --- 추가 하이퍼파라미터 ---
        k_trend=31,          # [TEMPO] 추세 추출 커널(홀수 권장, 넓을수록 저주파)
        k_season=7,          # [TEMPO] 계절성 추출 커널(상대적으로 좁게)
        use_alignment_loss=True,  # [CALF] 정렬 손실 사용 여부
        ts_prompt=True,      # [Time-LLM/AutoTimes/SMETimes] 타임스탬프/통계 프롬프트 사용
        stats_pool='meanstd' # 통계 프롬프트 종류: 'mean' | 'meanstd'
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
        self.use_alignment_loss = use_alignment_loss
        self.ts_prompt = ts_prompt
        self.stats_pool = stats_pool

        # -------- Normalization (RevIN) --------
        # [공통] 분포 이동을 제거해 모델 안정화. 학습/추론 시 동일 인터페이스 유지.
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)

        # -------- Light Decomposition [TEMPO 간소화] --------
        # 입력: [B, L, N] → Conv1d는 [B, N, L]을 기대하므로 축 변경 후 depthwise 1D conv 적용
        # - trend : 넓은 커널 평균(저주파) 근사
        # - season: 좁은 커널로 고주파 강조(중심 임펄스 - 평균)
        # - resid : 원신호 - (trend + season)
        padding_tr = k_trend // 2
        padding_se = k_season // 2
        # depthwise = groups=num_nodes (노드별로 독립 필터: 공간 상호 간섭 없이 시간축 패턴 추출)
        self.conv_trend = nn.Conv1d(self.num_nodes, self.num_nodes, k_trend, padding=padding_tr,
                                    groups=self.num_nodes, bias=False).to(self.device)
        self.conv_season = nn.Conv1d(self.num_nodes, self.num_nodes, k_season, padding=padding_se,
                                     groups=self.num_nodes, bias=False).to(self.device)
        # 초기화 전략: trend는 평균 필터, season은 (delta - 평균)로 band-pass 느낌
        nn.init.constant_(self.conv_trend.weight, 1.0 / k_trend)
        with torch.no_grad():
            self.conv_season.weight.zero_()
            center = k_season // 2
            self.conv_season.weight[:, :, center] = 1.0
            self.conv_season.weight[:, :, :] -= 1.0 / k_season

        # 분해 성분 가중 결합용 gate (softmax로 가중치 정규화)
        self.comp_gate = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))  # [trend, season, resid]

        # -------- Length -> Feature 변환 (성분별 투영 후 결합) --------
        # [B, L, N] → [B, N, L] → Linear(L→C) → [B, N, C]
        # 원신호(main)와 세 성분(tr/season/resid)을 각각 투영 후 가중 합성
        self.length_to_feature_main = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.length_to_feature_tr = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.length_to_feature_se = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.length_to_feature_re = nn.Linear(self.seq_len, self.channel).to(self.device)

        # -------- TS Encoder (Transformer) --------
        # [공통] 시계열 노드 차원 N을 시퀀스 길이로 보고, 채널 C를 d_model로 사용
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers=self.e_layer).to(self.device)

        # -------- Prompt(LLM) Encoder --------
        # [공통] 프롬프트/이미지 임베딩을 LLM 차원(E)에서 문맥화. 이후 TS 특성 차원(C)로 사상.
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers=self.e_layer).to(self.device)

        # -------- Cross-Modality Alignment [CALF] --------
        # (1) LLM 임베딩(E) → TS 특성 공간(C)으로 투영 (공간 정렬을 위한 선형 사상)
        self.proj_kv_llm_to_c = nn.Linear(self.d_llm, self.channel).to(self.device)
        # (2) alignment loss 계산 시 노드 축 평균을 위한 풀러 (node-wise mean: [B, C, N] → [B, C, 1])
        self.align_reduce = nn.AdaptiveAvgPool1d(1)

        # 기존 CrossModal 유지 (Q: TS, K/V: Prompt)
        # CrossModal은 입력을 [B, C, N] 형태로 기대하므로 forward에서 permute로 맞춰줌
        self.cross = CrossModal(
            d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff,
            norm='LayerNorm', attn_dropout=self.dropout_n,
            dropout=self.dropout_n, pre_norm=True,
            activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)

        # -------- Decoder (Transformer) --------
        # Cross 후의 결합 표현을 디코더로 후처리. 소규모 층수로 비용 최소화.
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.d_layer).to(self.device)

        # -------- Projection to output length --------
        # [B, N, C] → Linear(C→L_out) → [B, N, L_out] → [B, L_out, N]
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

        # -------- Time/Stats Prompt [Time-LLM / AutoTimes / SMETimes] --------
        if self.ts_prompt:
            # input_data_mark: [B, L, Dm] → L 평균 풀링 → [B, Dm] → Linear(Dm→N) → [B, N]
            #   → 채널 차원 C로 broadcast하여 [B, N, C]
            # 주의: Dm은 런타임에만 알 수 있어 lazy init 사용 (초기 in_features는 placeholder)
            self.mark_to_node = nn.Linear(in_features=0 + 1, out_features=self.num_nodes, bias=True).to(self.device)
            # 통계 프롬프트: mean 또는 mean+std를 채널 C로 사상하여 TS 특성에 더함(FiLM-like)
            stats_in = 2 if self.stats_pool == 'meanstd' else 1
            self.stats_to_c = nn.Linear(stats_in, self.channel).to(self.device)

        # 디버깅/로깅용 보조 출력 저장 딕셔너리
        self.last_aux = {}

    # --- cosine alignment loss [CALF] ---
    # TS 인코딩과 프롬프트 인코딩을 동일 공간(C)에서 코사인 유사도로 정렬
    def _cos_align_loss(self, a, b, eps=1e-6):
        # a, b: [B, C]  (node-wise mean 이후)
        a = F.normalize(a, dim=-1, eps=eps)
        b = F.normalize(b, dim=-1, eps=eps)
        return (1.0 - (a * b).sum(dim=-1)).mean()

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_data, input_data_mark, embeddings):
        """
        Args
        - input_data:      [B, L, N]
        - input_data_mark: [B, L, Dm] (optional)
        - embeddings:      [B, E, N] 또는 [B, E, N, 1]

        Returns
        - y_hat:           [B, L_out, N]
        - (훈련 시) self.last_aux['alignment_loss']에 정렬 손실 저장
        """
        B, L, N = input_data.shape
        assert N == self.num_nodes, f"num_nodes mismatch: got {N}, expected {self.num_nodes}"

        x = input_data.float()
        mark = input_data_mark.float() if input_data_mark is not None else None
        emb = embeddings.float()

        # -------- RevIN --------
        # [공통] 도메인 시프트를 줄이고 수렴 안정화. denorm에서 원 스케일 복구.
        x = self.normalize_layers(x, 'norm')  # [B, L, N]

        # -------- Light Decomposition [TEMPO] --------
        # Conv1d를 위해 채널-길이 축 교환: [B, L, N] → [B, N, L]
        x_c = x.permute(0, 2, 1)  # [B, N, L]
        trend = self.conv_trend(x_c)          # [B, N, L]  (저주파)
        season = self.conv_season(x_c)        # [B, N, L]  (고주파)
        resid = x_c - (trend + season)        # [B, N, L]  (잔차)
        # 다시 [B, L, N]로 복귀
        trend = trend.permute(0, 2, 1)
        season = season.permute(0, 2, 1)
        resid = resid.permute(0, 2, 1)

        # 성분 가중 결합 (softmax 게이팅)
        gate = F.softmax(self.comp_gate, dim=0)  # [3] (trend/season/resid 비율)
        x_decomp = gate[0] * trend + gate[1] * season + gate[2] * resid  # [B, L, N]

        # -------- Length -> Feature (성분별 투영) --------
        # 각 신호/성분을 [B, N, L]로 변환 후 Linear(L→C)로 투영 → [B, N, C]
        def len2feat(linear_layer, z):  # z: [B, L, N]
            z = z.permute(0, 2, 1)      # [B, N, L]
            z = linear_layer(z)         # [B, N, C]
            return z

        f_main = len2feat(self.length_to_feature_main, x)      # 원신호 투영
        f_tr   = len2feat(self.length_to_feature_tr,   trend)  # 추세 투영
        f_se   = len2feat(self.length_to_feature_se,   season) # 계절 투영
        f_re   = len2feat(self.length_to_feature_re,   resid)  # 잔차 투영

        # 성분 결합 + 원신호 보강 (residual 합성)
        f_ts = f_main + (gate[0] * f_tr + gate[1] * f_se + gate[2] * f_re)  # [B, N, C]

        # -------- Time/Stats Prompt [Time-LLM / AutoTimes / SMETimes] --------
        if self.ts_prompt:
            # (1) 통계 프롬프트: 노드별 mean(/std)을 채널 C로 사상 후 TS 특성에 합산
            if self.stats_pool == 'meanstd':
                mean = x.mean(dim=1)                 # [B, N]
                std = x.std(dim=1, unbiased=False)   # [B, N]
                s = torch.stack([mean, std], dim=-1) # [B, N, 2]
            else:
                s = x.mean(dim=1, keepdim=False).unsqueeze(-1)  # [B, N, 1]
            s = self.stats_to_c(s)                   # [B, N, C]

            # (2) 타임스탬프/메타 프롬프트: mark를 노드별 요약으로 변환 후 채널에 broadcast
            t_prompt = 0
            if mark is not None:
                # lazy init: 실제 Dm을 확인해 Linear(Dm→N) 재정의
                if isinstance(self.mark_to_node, nn.Linear) and self.mark_to_node.in_features == 1:
                    in_f = mark.shape[-1]
                    self.mark_to_node = nn.Linear(in_f, self.num_nodes, bias=True).to(self.device)
                # 시간축 평균 풀링으로 전역 요약 (원하면 최근 W 구간만 평균도 가능)
                m_pool = mark.mean(dim=1)            # [B, Dm]
                m_nodes = self.mark_to_node(m_pool)  # [B, N]
                t_prompt = m_nodes.unsqueeze(-1).expand(-1, -1, self.channel)  # [B, N, C]

            # FiLM-like 주입: f_ts에 통계/메타 정보를 additive로 더해 패턴 해석을 유도
            f_ts = f_ts + s + (t_prompt if isinstance(t_prompt, torch.Tensor) else 0)

        # -------- TS Encoder (Q 생성) --------
        enc_out = self.ts_encoder(f_ts)           # [B, N, C]
        enc_out_c_first = enc_out.permute(0, 2, 1)  # [B, C, N]  (Cross에서 Q로 사용)

        # -------- Prompt / Image Encoder (K/V 생성) --------
        # embeddings: [B, E, N] 또는 [B, E, N, 1] 대응
        if emb.dim() == 4 and emb.shape[-1] == 1:
            emb = emb.squeeze(-1)                 # [B, E, N]
        emb = emb.permute(0, 2, 1)                # [B, N, E]
        emb_enc = self.prompt_encoder(emb)        # [B, N, E]  (LLM 차원에서 문맥화)
        kv_c = self.proj_kv_llm_to_c(emb_enc)     # [B, N, C]  (TS 특성 공간으로 사상)
        kv_c_c_first = kv_c.permute(0, 2, 1)      # [B, C, N]  (Cross에서 K/V로 사용)

        # -------- Cross-Modal Attention [CALF spirit + 기존 CrossModal] --------
        # Q: enc_out_c_first(시계열), K/V: kv_c_c_first(프롬프트/이미지)
        cross_out = self.cross(enc_out_c_first, kv_c_c_first, kv_c_c_first)  # [B, C, N]
        cross_out = cross_out.permute(0, 2, 1)  # [B, N, C]

        # -------- Decoder (후처리) --------
        dec_out = self.decoder(cross_out, cross_out)  # [B, N, C]

        # -------- Projection to output length --------
        dec_len = self.c_to_length(dec_out)  # [B, N, L_out]
        dec_len = dec_len.permute(0, 2, 1)   # [B, L_out, N]

        # -------- denorm (RevIN inverse) --------
        y_hat = self.normalize_layers(dec_len, 'denorm')  # [B, L_out, N]

        # -------- Alignment Loss [CALF] (옵션, 학습 시에만) --------
        # TS 인코딩(Q)과 프롬프트 인코딩(K/V)을 같은 C 공간에서 코사인 유사도로 가깝게 만듦
        self.last_aux = {}
        if self.use_alignment_loss and self.training:
            # node 평균으로 [B, C]로 축소하여 시계열/프롬프트 전역 정렬
            q_red = self.align_reduce(enc_out_c_first).squeeze(-1)   # [B, C]
            k_red = self.align_reduce(kv_c_c_first).squeeze(-1)      # [B, C]
            align_loss = self._cos_align_loss(q_red, k_red)
            self.last_aux['alignment_loss'] = align_loss
            # 사용 예: total_loss = main_loss + lambda_align * model.last_aux['alignment_loss']

        return y_hat
