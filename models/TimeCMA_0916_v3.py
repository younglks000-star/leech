import torch
import torch.nn as nn
from einops import rearrange
from layers.StandardNorm import Normalize  # RevIN


class Dual(nn.Module):
    """
    Q = Var tokens (변수축 시계열 임베딩)
    K = V = Prompt tokens (프롬프트 임베딩; 변수 인덱스 N에 정렬)

    - Stage 0: RevIN
    - Stage 1 (선택): Var Self-Attn  -> Q=K=V=Var (변수 간 상호작용 강화)
    - Stage 2: Cross-Attn            -> Q=Var, K=V=Prompt (외부 지식 주입)
    - Decoder: 미래 시간 쿼리로 [B, pred_len, N] 생성

    입력/출력:
      input_data:      [B, L, N]
      input_data_mark: [B, L, *]  (미사용; 인터페이스 유지)
      embeddings:      [B, E, N] or [B, N, E] or [B, E, N, 1] (Prompt, N에 정렬)
      output:          [B, pred_len, N]
    """
    def __init__(
        self,
        device="cuda:0",
        channel=32,         # == N (variables/clusters)
        num_nodes=32,       # alias (channel과 동일)
        seq_len=360,
        pred_len=360,
        dropout_n=0.1,
        d_llm=4096,         # LLM hidden (e.g., LLaMA 4096, GPT-2 768 등)
        e_layer=1,          # 인코더 레이어 수(Var/Prompt 공통)
        d_layer=1,          # 디코더 레이어 수
        d_ff=512,
        head=8,
        d_model=256,        # 공통 임베딩 차원 (Var/Prompt 투영 목표)
        use_var_self_attn=True,  # 변수 자기어텐션 사용 여부
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
        self.use_var_self_attn = use_var_self_attn

        # ---------------------------
        # RevIN (입력 표준화/역정규화)
        # ---------------------------
        self.normalize_layers = Normalize(self.N, affine=False).to(self.device)

        # ------------------------------------------------------------------
        # 1) Var encoder  (Q=Var의 원천 토큰)
        #    [B, L, N] → [B, N, L] → Linear(L→D) → TransformerEncoder → [B, N, D]
        # ------------------------------------------------------------------
        self.var_proj = nn.Linear(self.L, self.d_model).to(self.device)
        self.var_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.head, batch_first=True,
            dim_feedforward=self.d_ff, norm_first=True, dropout=self.drop_p
        ).to(self.device)
        self.var_encoder = nn.TransformerEncoder(self.var_enc_layer, num_layers=self.e_layer).to(self.device)

        # (선택) 변수 자기어텐션 블록
        if self.use_var_self_attn:
            self.var_self_attn = nn.MultiheadAttention(
                self.d_model, num_heads=self.head, batch_first=True, dropout=self.drop_p
            ).to(self.device)
            self.var_ln = nn.LayerNorm(self.d_model).to(self.device)
            self.var_ffn = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_ff), nn.GELU(), nn.Dropout(self.drop_p),
                nn.Linear(self.d_ff, self.d_model),
            ).to(self.device)

        # ------------------------------------------------------------------
        # 2) Prompt encoder  (K=V=Prompt의 원천 토큰)
        #    embeddings: [B, *, N] → [B, N, d_llm] 정렬 후 proj(LLM→D) → Encoder
        # ------------------------------------------------------------------
        self.prompt_proj = nn.Linear(self.d_llm, self.d_model).to(self.device)
        self.prompt_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.head, batch_first=True,
            dim_feedforward=self.d_ff, norm_first=True, dropout=self.drop_p
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_enc_layer, num_layers=self.e_layer).to(self.device)

        # ------------------------------------------------------------------
        # 3) Cross-Attention  (핵심: Q=Var, K=V=Prompt)
        # ------------------------------------------------------------------
        self.cross_attn = nn.MultiheadAttention(
            self.d_model, num_heads=self.head, batch_first=True, dropout=self.drop_p
        ).to(self.device)
        self.cross_ln_q = nn.LayerNorm(self.d_model).to(self.device)
        self.cross_ln_k = nn.LayerNorm(self.d_model).to(self.device)
        self.cross_ln_v = nn.LayerNorm(self.d_model).to(self.device)
        self.cross_ffn = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_ff), nn.GELU(), nn.Dropout(self.drop_p),
            nn.Linear(self.d_ff, self.d_model),
        ).to(self.device)

        # ------------------------------------------------------------------
        # 4) Temporal decoder (미래 시간 쿼리)
        # ------------------------------------------------------------------
        self.future_queries = nn.Parameter(torch.randn(1, self.pred_len, self.d_model))
        self.dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=self.head, batch_first=True,
            dim_feedforward=self.d_ff, norm_first=True, dropout=self.drop_p
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(self.dec_layer, num_layers=self.d_layer).to(self.device)

        # ------------------------------------------------------------------
        # 5) Output projection: [B, pred_len, D] → [B, pred_len, N]
        # ------------------------------------------------------------------
        self.out_head = nn.Linear(self.d_model, self.N, bias=True).to(self.device)

    # ------------------ utils ------------------
    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def _shape_info(self, x):
        return f"[{x.shape[0]}, {x.shape[1]}, {x.shape[2]}]"

    # ---------- 프롬프트 정렬/인코딩 ----------
    def _prep_prompt(self, embeddings):
        """
        embeddings를 [B, N, d_llm]로 정렬/변환 후
        prompt_proj → prompt_encoder → [B, N, d_model] 반환
        """
        # ex) [B, E, N, 1] → [B, E, N]
        if embeddings.dim() == 4 and embeddings.size(-1) == 1:
            embeddings = embeddings.squeeze(-1)
        # [B, E, N] → [B, N, E]
        if embeddings.dim() == 3 and embeddings.size(1) != self.N and embeddings.size(2) == self.N:
            embeddings = embeddings.permute(0, 2, 1).contiguous()

        assert embeddings.dim() == 3 and embeddings.size(1) == self.N, \
            f"Prompt embeddings must align to N. Got {embeddings.shape} with N={self.N}"

        Vp = self.prompt_proj(embeddings)   # [B, N, D]
        Vp = self.prompt_encoder(Vp)        # [B, N, D]
        return Vp

    # ------------------ forward ------------------
    def forward(self, input_data, input_data_mark, embeddings):
        """
        input_data:      [B, L, N]
        input_data_mark: [B, L, *] (unused)
        embeddings:      prompt embeddings (정렬 필요 시 _prep_prompt가 처리)
        return:          [B, pred_len, N]
        """
        x   = input_data.float()            # [B, L, N]
        emb = embeddings.float()

        # Stage 0) RevIN
        x = self.normalize_layers(x, 'norm')           # [B, L, N]

        # 1) Var tokens (Q 소스)
        x_var = x.transpose(1, 2).contiguous()         # [B, N, L]
        var_tokens = self.var_proj(x_var)              # [B, N, D]
        var_tokens = self.var_encoder(var_tokens)      # [B, N, D]

        if self.use_var_self_attn:
            vq = self.var_ln(var_tokens)
            var_mixed, _ = self.var_self_attn(query=vq, key=vq, value=vq)  # [B, N, D]
            var_tokens = var_tokens + var_mixed
            var_tokens = var_tokens + self.var_ffn(var_tokens)

        # 2) Prompt tokens (K=V 소스)
        prompt_tokens = self._prep_prompt(emb)         # [B, N, D]

        # 3) Cross-Attn: Q=Var, K=V=Prompt  (요청 사양)
        qn = self.cross_ln_q(var_tokens)
        kn = self.cross_ln_k(prompt_tokens)
        vn = self.cross_ln_v(prompt_tokens)
        var_enhanced, _ = self.cross_attn(query=qn, key=kn, value=vn)      # [B, N, D]
        var_context = var_tokens + var_enhanced
        var_context = var_context + self.cross_ffn(var_context)             # [B, N, D]

        # 4) Temporal decoder to future horizon
        B = x.size(0)
        tgt_queries = self.future_queries.expand(B, -1, -1)                 # [B, pred_len, D]
        dec_ctx = self.decoder(tgt=tgt_queries, memory=var_context)          # [B, pred_len, D]

        # 5) Output head
        y = self.out_head(dec_ctx)                                           # [B, pred_len, N]

        # Stage 0') RevIN denorm
        y = self.normalize_layers(y, 'denorm')                               # [B, pred_len, N]
        return y
