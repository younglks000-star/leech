import torch
import torch.nn as nn
from einops import rearrange
from layers.StandardNorm import Normalize  # RevIN
# from layers.Cross_Modal_Align import CrossModal  # ← 사용하지 않음(새 Cross-Attn으로 대체)


class Dual(nn.Module):
    """
    Reworked TimeCMA-style dual branch with:
      - Q = time-based embeddings (Time tokens)
      - K = variable-based embeddings (Variable tokens)
      - V = prompt embeddings (projected to common dim)
    Cross-Attention: Attn(Q_time, K_var, V_prompt)
    Then decode to future horizon with learnable future queries.

    Shapes:
      input_data:       [B, L, N]
      input_data_mark:  [B, L, *]  (unused here but kept for API parity)
      embeddings:       [B, E, N] or [B, N, E] or [B, E, N, 1]  (we squeeze/permute)
      output:           [B, pred_len, N]
    """
    def __init__(
        self,
        device="cuda:0",
        channel=32,         # == N (variables/clusters)
        num_nodes=32,       # kept for compatibility; == channel
        seq_len=360,
        pred_len=360,
        dropout_n=0.1,
        d_llm=4096,         # LLaMA hidden (예: 4096) / GPT2(768) 등
        e_layer=1,
        d_layer=1,
        d_ff=512,
        head=8,
        d_model=256         # 공통 임베딩 차원 (Time/Var/Prompt를 모두 여기에 맞춤)
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
        # 1) Time encoder (Q 생성): 시간 토큰 = L개, 토큰 차원 = d_model
        #    입력 [B, L, N] → proj: Linear(N -> d_model) → TE layers → [B, L, d_model]
        # ------------------------------------------------------------------
        self.time_proj = nn.Linear(self.N, self.d_model).to(self.device)
        self.time_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.head, batch_first=True,
            dim_feedforward=self.d_ff, norm_first=True, dropout=self.drop_p
        ).to(self.device)
        self.time_encoder = nn.TransformerEncoder(self.time_enc_layer, num_layers=self.e_layer).to(self.device)

        # ------------------------------------------------------------------
        # 2) Var encoder (K 생성): 변수 토큰 = N개, 토큰 차원 = d_model
        #    입력 [B, L, N] → transpose → [B, N, L] → proj: Linear(L -> d_model)
        #    → VE layers → [B, N, d_model]
        # ------------------------------------------------------------------
        self.var_proj = nn.Linear(self.L, self.d_model).to(self.device)
        self.var_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.head, batch_first=True,
            dim_feedforward=self.d_ff, norm_first=True, dropout=self.drop_p
        ).to(self.device)
        self.var_encoder = nn.TransformerEncoder(self.var_enc_layer, num_layers=self.e_layer).to(self.device)

        # ------------------------------------------------------------------
        # 3) Prompt encoder (V 생성): LLM 임베딩[E] → 공통 차원 d_model
        #    입력 embeddings: [B, *, N] → 표준화/정렬 → [B, N, d_model]
        # ------------------------------------------------------------------
        self.prompt_proj = nn.Linear(self.d_llm, self.d_model).to(self.device)
        self.prompt_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.head, batch_first=True,
            dim_feedforward=self.d_ff, norm_first=True, dropout=self.drop_p
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_enc_layer, num_layers=self.e_layer).to(self.device)

        # ------------------------------------------------------------------
        # 4) Cross-Attention: Attn(Q_time [B,L,D], K_var [B,N,D], V_prompt [B,N,D]) → [B,L,D]
        #    PyTorch MultiheadAttention로 직접 구현
        # ------------------------------------------------------------------
        self.cross_attn = nn.MultiheadAttention(self.d_model, num_heads=self.head, batch_first=True, dropout=self.drop_p).to(self.device)
        self.cross_ln_q = nn.LayerNorm(self.d_model).to(self.device)
        self.cross_ln_k = nn.LayerNorm(self.d_model).to(self.device)
        self.cross_ln_v = nn.LayerNorm(self.d_model).to(self.device)

        # ------------------------------------------------------------------
        # 5) Temporal decoder to future horizon:
        #    - learnable future queries [1, pred_len, d_model] → broadcast to B
        #    - TransformerDecoder(TGT=future_q, MEM=fused_time_ctx)
        #    결과: [B, pred_len, d_model]
        # ------------------------------------------------------------------
        self.future_queries = nn.Parameter(torch.randn(1, self.pred_len, self.d_model))
        self.dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=self.head, batch_first=True,
            dim_feedforward=self.d_ff, norm_first=True, dropout=self.drop_p
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(self.dec_layer, num_layers=self.d_layer).to(self.device)

        # ------------------------------------------------------------------
        # 6) Output projection: [B, pred_len, d_model] → [B, pred_len, N]
        # ----------------------------------------------------------------
        self.out_head = nn.Linear(self.d_model, self.N, bias=True).to(self.device)

    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _prep_prompt(self, embeddings):
        """
        embeddings 입력 형태 다양한 경우를 흡수해서 [B, N, d_llm]로 맞춘 뒤
        prompt_proj → prompt_encoder → [B, N, d_model] 반환
        """
        # 가능한 입력 케이스 흡수
        # ex) [B, E, N, 1] → squeeze → [B, E, N]
        if embeddings.dim() == 4 and embeddings.size(-1) == 1:
            embeddings = embeddings.squeeze(-1)
        # [B, E, N] → [B, N, E]
        if embeddings.dim() == 3 and embeddings.size(1) != self.N and embeddings.size(2) == self.N:
            embeddings = embeddings.permute(0, 2, 1).contiguous()
        # 이제 [B, N, E] 가정
        assert embeddings.size(1) == self.N, f"Prompt embeddings must align with N (got {embeddings.shape})"
        # LLM → 공통차원
        V = self.prompt_proj(embeddings)                 # [B, N, d_model]
        V = self.prompt_encoder(V)                       # [B, N, d_model]
        return V

    def forward(self, input_data, input_data_mark, embeddings):
        """
        input_data:      [B, L, N]
        input_data_mark: [B, L, *]  (not used here)
        embeddings:      see _prep_prompt
        return:          [B, pred_len, N]
        """
        x = input_data.float()
        emb = embeddings.float()

        # ----------------
        # RevIN normalize
        # ----------------
        x = self.normalize_layers(x, 'norm')             # [B, L, N]

        # -------------------------
        # Q: Time encoder (time tokens)
        # -------------------------
        q_time = self.time_proj(x)                       # [B, L, d_model]
        q_time = self.time_encoder(q_time)               # [B, L, d_model]

        # -------------------------
        # K: Var encoder (variable tokens)
        # -------------------------
        # [B, L, N] -> [B, N, L]
        x_var = x.transpose(1, 2).contiguous()
        k_var = self.var_proj(x_var)                     # [B, N, d_model]
        k_var = self.var_encoder(k_var)                  # [B, N, d_model]

        # -------------------------
        # V: Prompt encoder (prompt tokens aligned to variables)
        # -------------------------
        v_pr = self._prep_prompt(emb)                    # [B, N, d_model]

        # -------------------------
        # Cross-Attention: Attn(Q_time, K_var, V_prompt) → [B, L, d_model]
        # -------------------------
        qn = self.cross_ln_q(q_time)
        kn = self.cross_ln_k(k_var)
        vn = self.cross_ln_v(v_pr)
        # nn.MultiheadAttention(batch_first=True) expects shapes [B, S, D]
        fused_time, _ = self.cross_attn(query=qn, key=kn, value=vn)  # [B, L, d_model]

        # -------------------------
        # Temporal decoder to future horizon
        # -------------------------
        B = x.size(0)
        tgt_queries = self.future_queries.expand(B, -1, -1)         # [B, pred_len, d_model]
        dec_ctx = self.decoder(tgt=tgt_queries, memory=fused_time)  # [B, pred_len, d_model]

        # -------------------------
        # Output head: per-time output over variables
        # -------------------------
        y = self.out_head(dec_ctx)                                  # [B, pred_len, N]

        # --------------
        # RevIN denorm
        # --------------
        y = self.normalize_layers(y, 'denorm')                      # [B, pred_len, N]
        return y
