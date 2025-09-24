import torch
import torch.nn as nn
from einops import rearrange
from layers.StandardNorm import Normalize  # RevIN


class Dual(nn.Module):
    """
    변수 축을 토큰으로 보고(iTransformer 철학), LLM 프롬프트와 직접 Cross-Attn:
      - Stage 1 (Variable Self-Attn):  Q=K=V=Var  → 변수 간 상호작용 학습
      - Stage 2 (Var ← Prompt Cross-Attn): Q=Var, K=V=Prompt → 변수별 외부 지식 주입
      - Decoder: 학습 가능한 미래 시간 쿼리로 [B, pred_len, N] 생성

    입력/출력:
      input_data:      [B, L, N]
      input_data_mark: [B, L, *]  (미사용; 인터페이스 유지용)
      embeddings:      [B, E, N] or [B, N, E] or [B, E, N, 1]  (프롬프트 임베딩, 변수 정렬)
      output:          [B, pred_len, N]
    """
    def __init__(
        self,
        device="cuda:0",
        channel=32,         # == N (variables/clusters)
        num_nodes=32,       # 호환성 유지를 위한 alias (channel과 동일하게 씀)
        seq_len=360,
        pred_len=360,
        dropout_n=0.1,
        d_llm=4096,         # LLM hidden (e.g., LLaMA 4096, GPT2 768 등)
        e_layer=1,          # 인코더(Var/Prompt) 레이어 수
        d_layer=1,          # 디코더 레이어 수
        d_ff=512,
        head=8,
        d_model=256         # 공통 임베딩 차원 (Var/Prompt를 여기에 맞춤)
    ):
        super().__init__()

        self.device    = device
        self.N         = channel
        self.num_nodes = num_nodes
        self.L         = seq_len
        self.pred_len  = pred_len
        self.drop_p    = dropout_n
        self.d_llm     = d_llm
        self.e_layer   = e_layer
        self.d_layer   = d_layer
        self.d_ff      = d_ff
        self.head      = head
        self.d_model   = d_model

        # ---------------------------
        # RevIN (입력 표준화/역정규화)
        # ---------------------------
        self.normalize_layers = Normalize(self.N, affine=False).to(self.device)

        # ------------------------------------------------------------------
        # [변경/핵심] 1) Var encoder (Q=K=V=Var로 쓸 변수 토큰 만들기)
        #    입력 [B, L, N] → transpose → [B, N, L] → proj: Linear(L -> d_model)
        #    → TransformerEncoder → [B, N, d_model]
        #    - 여기서 나온 [B,N,D]가 곧 "변수 토큰" (Stage 1의 Q=K=V, Stage 2의 Q)
        # ------------------------------------------------------------------
        self.var_proj = nn.Linear(self.L, self.d_model).to(self.device)
        self.var_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.head, batch_first=True,
            dim_feedforward=self.d_ff, norm_first=True, dropout=self.drop_p
        ).to(self.device)
        self.var_encoder = nn.TransformerEncoder(self.var_enc_layer, num_layers=self.e_layer).to(self.device)

        # ------------------------------------------------------------------
        # [변경/핵심] 2) Prompt encoder (K=V=Prompt)
        #    입력 embeddings: [B, *, N] → [B, N, d_llm] 정렬 → proj(LLM→d_model)
        #    → TransformerEncoder → [B, N, d_model]
        #    - 변수 정렬(N과 일치)된 프롬프트 토큰 (Stage 2의 K,V)
        # ------------------------------------------------------------------
        self.prompt_proj = nn.Linear(self.d_llm, self.d_model).to(self.device)
        self.prompt_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.head, batch_first=True,
            dim_feedforward=self.d_ff, norm_first=True, dropout=self.drop_p
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_enc_layer, num_layers=self.e_layer).to(self.device)

        # ------------------------------------------------------------------
        # [변경/핵심] 3) Stage 1: Variable Self-Attention (Q=K=V=Var)
        #    - MultiheadAttention으로 한 번 더 mixing (원한다면 생략 가능)
        #    - 목적: 변수 간 상호작용을 명시적으로 강화
        # ------------------------------------------------------------------
        self.var_self_attn = nn.MultiheadAttention(self.d_model, num_heads=self.head,
                                                   batch_first=True, dropout=self.drop_p).to(self.device)
        self.var_ln = nn.LayerNorm(self.d_model).to(self.device)
        self.var_ffn = nn.Sequential(                # 간단한 FFN 추가 (Post-Attn)
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.drop_p),
            nn.Linear(self.d_ff, self.d_model),
        ).to(self.device)

        # ------------------------------------------------------------------
        # [변경/핵심] 4) Stage 2: Cross-Attention  (Q=Var, K=Prompt, V=Prompt)
        #    - 변수 토큰을 쿼리로, 프롬프트 토큰에서 컨텍스트를 끌어옴
        # ------------------------------------------------------------------
        self.cross_attn = nn.MultiheadAttention(self.d_model, num_heads=self.head,
                                                batch_first=True, dropout=self.drop_p).to(self.device)
        self.cross_ln_q = nn.LayerNorm(self.d_model).to(self.device)
        self.cross_ln_k = nn.LayerNorm(self.d_model).to(self.device)
        self.cross_ln_v = nn.LayerNorm(self.d_model).to(self.device)
        self.cross_ffn = nn.Sequential(              # Cross 후 안정화용 FFN
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.drop_p),
            nn.Linear(self.d_ff, self.d_model),
        ).to(self.device)

        # ------------------------------------------------------------------
        # 5) Temporal decoder to future horizon:
        #    - 학습 가능한 미래 시간 쿼리 [1, pred_len, d_model] → B로 broadcast
        #    - TransformerDecoder(TGT=future_q, MEM=variable_context)
        #    결과: [B, pred_len, d_model]
        #    * 메모리는 변수 컨텍스트 [B, N, d_model]이므로, 디코더가 "변수 메모리"를 참조해
        #      각 미래 시점의 표현을 구성 → 마지막에 [D→N] 투영
        # ------------------------------------------------------------------
        self.future_queries = nn.Parameter(torch.randn(1, self.pred_len, self.d_model))
        self.dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=self.head, batch_first=True,
            dim_feedforward=self.d_ff, norm_first=True, dropout=self.drop_p
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(self.dec_layer, num_layers=self.d_layer).to(self.device)

        # ------------------------------------------------------------------
        # 6) Output projection: [B, pred_len, d_model] → [B, pred_len, N]
        # ------------------------------------------------------------------
        self.out_head = nn.Linear(self.d_model, self.N, bias=True).to(self.device)

    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------- 유틸: 프롬프트 정렬/인코딩 ----------
    def _prep_prompt(self, embeddings):
        """
        embeddings 다양한 입력 형태를 흡수해서 [B, N, d_llm]로 맞춘 뒤
        prompt_proj → prompt_encoder → [B, N, d_model] 반환
        """
        # ex) [B, E, N, 1] → squeeze → [B, E, N]
        if embeddings.dim() == 4 and embeddings.size(-1) == 1:
            embeddings = embeddings.squeeze(-1)
        # [B, E, N] → [B, N, E]
        if embeddings.dim() == 3 and embeddings.size(1) != self.N and embeddings.size(2) == self.N:
            embeddings = embeddings.permute(0, 2, 1).contiguous()
        # 이제 [B, N, E] 가정
        assert embeddings.size(1) == self.N, f"Prompt embeddings must align with N (got {embeddings.shape})"
        V = self.prompt_proj(embeddings)        # [B, N, d_model]
        V = self.prompt_encoder(V)              # [B, N, d_model]
        return V

    def forward(self, input_data, input_data_mark, embeddings):
        """
        input_data:      [B, L, N]
        input_data_mark: [B, L, *]  (not used)
        embeddings:      see _prep_prompt
        return:          [B, pred_len, N]
        """
        x   = input_data.float()     # [B, L, N]
        emb = embeddings.float()

        # ----------------
        # RevIN normalize
        # ----------------
        x = self.normalize_layers(x, 'norm')                     # [B, L, N]

        # -------------------------
        # Stage 1) Variable tokens (Q=K=V=Var) 생성 + 자기어텐션
        # -------------------------
        x_var = x.transpose(1, 2).contiguous()                   # [B, N, L]
        var_tokens = self.var_proj(x_var)                        # [B, N, d_model]
        var_tokens = self.var_encoder(var_tokens)                # [B, N, d_model] (1차 인코딩)

        # 추가 자기어텐션으로 변수 mixing 강화 (선택적이지만 효과적)
        vq = self.var_ln(var_tokens)
        var_mixed, _ = self.var_self_attn(query=vq, key=vq, value=vq)  # [B, N, d_model]
        var_tokens = var_tokens + var_mixed                           # Residual
        var_tokens = var_tokens + self.var_ffn(var_tokens)            # FFN Residual

        # -------------------------
        # Stage 2) Cross-Attn: Q=Var, K=V=Prompt
        # -------------------------
        prompt_tokens = self._prep_prompt(emb)                 # [B, N, d_model]

        qn = self.cross_ln_q(var_tokens)
        kn = self.cross_ln_k(prompt_tokens)
        vn = self.cross_ln_v(prompt_tokens)
        var_enhanced, _ = self.cross_attn(query=qn, key=kn, value=vn)  # [B, N, d_model]
        var_context = var_tokens + var_enhanced                         # Residual
        var_context = var_context + self.cross_ffn(var_context)         # FFN Residual
        # var_context: 디코더가 참조할 "변수 메모리"  [B, N, d_model]

        # -------------------------
        # Temporal decoder to future horizon
        # -------------------------
        B = x.size(0)
        tgt_queries = self.future_queries.expand(B, -1, -1)             # [B, pred_len, d_model]
        dec_ctx = self.decoder(tgt=tgt_queries, memory=var_context)      # [B, pred_len, d_model]

        # -------------------------
        # Output head: per-time output over variables
        # -------------------------
        y = self.out_head(dec_ctx)                                       # [B, pred_len, N]

        # --------------
        # RevIN denorm
        # --------------
        y = self.normalize_layers(y, 'denorm')                           # [B, pred_len, N]
        return y
