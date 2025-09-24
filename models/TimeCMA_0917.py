# models/timecma_dual.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from layers.StandardNorm import Normalize  # RevIN


class TimeCMA_Dual(nn.Module):
    """
   
    A robust two-stream model with a dedicated fusion block.
      1) Temporal Stream : Processes time-series dynamics.
         x -> time_tokens -> t_attn(Q=time, K,V=prompt) -> t_ctx
      2) Spatial Stream  : Processes variable-specific features.
         x -> var_tokens
      3) Fusion Block    : Intelligently fuses temporal summary into spatial tokens.
         t_summary = mean(t_ctx)
         fused_tok = gate(var_tok) * t_summary + var_tok
      4) Spatial-CMA     : Aligns the fused representation with the prompt.
         s_attn(Q=fused_tok, K,V=prompt) -> v_ctx
      5) Decoder         : Decodes the final context for future prediction.
        
    """

    def __init__(
        self,
        device="cuda:0",
        channel=32,          # N (number of variables / clusters)
        seq_len=180,         # L
        pred_len=360,
        d_llm=768,           # prompt embedding dim before projection
        d_model=256,         # common hidden dim used across blocks
        head=8,
        d_ff=512,
        e_layer=1,
        d_layer=1,
        dropout=0.1,
        use_revin=True,
        revin_affine=False,
        # ------ 과거 코드 호환용 alias ------
        num_nodes=None,
        dropout_n=None,
    ):
        super().__init__()

        # ---- alias 흡수 ----
        if num_nodes is not None:
            channel = num_nodes
        if dropout_n is not None:
            dropout = float(dropout_n)

        self.device   = device
        self.N        = int(channel)
        self.L        = int(seq_len)
        self.pred_len = int(pred_len)
        self.d_llm    = int(d_llm)
        self.d_model  = int(d_model)
        self.head     = int(head)
        self.d_ff     = int(d_ff)
        self.e_layer  = int(e_layer)
        self.d_layer  = int(d_layer)
        self.drop_p   = float(dropout)
        self.use_revin = bool(use_revin)

        # ---------------------------
        # RevIN (표준화/역정규화)
        # ---------------------------
        if self.use_revin:
            self.revin = Normalize(self.N, affine=revin_affine).to(self.device)
        else:
            self.revin = None

        # ============================================================
        # 1) TEMPORAL ENCODER: time tokens (L tokens)
        # ============================================================
        self.time_proj = nn.Linear(self.N, self.d_model).to(self.device)
        self.time_pos  = nn.Parameter(torch.zeros(1, self.L, self.d_model))
        self.time_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.head, batch_first=True,
            dim_feedforward=self.d_ff, dropout=self.drop_p, norm_first=True
        ).to(self.device)
        self.time_encoder = nn.TransformerEncoder(
            self.time_enc_layer, num_layers=self.e_layer
        ).to(self.device)

        # ============================================================
        # PROMPT ENCODER (shared)
        # ============================================================
        self.prompt_proj = nn.Linear(self.d_llm, self.d_model).to(self.device)
        self.prompt_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.head, batch_first=True,
            dim_feedforward=self.d_ff, dropout=self.drop_p, norm_first=True
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(
            self.prompt_enc_layer, num_layers=self.e_layer
        ).to(self.device)

        # ============================================================
        # TEMPORAL CMA: Q=, K=V= (prompt)
        # ============================================================
        self.t_attn = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=self.head, batch_first=True, dropout=self.drop_p
        ).to(self.device)
        self.t_ln_q = nn.LayerNorm(self.d_model).to(self.device)
        self.t_ln_k = nn.LayerNorm(self.d_model).to(self.device)
        self.t_ln_v = nn.LayerNorm(self.d_model).to(self.device)
        self.t_ffn = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_ff), nn.GELU(), nn.Dropout(self.drop_p),
            nn.Linear(self.d_ff, self.d_model),
        ).to(self.device)

        # ============================================================
        # 2) SPATIAL ENCODER: variable tokens (N tokens)
        # ============================================================
        self.var_proj = nn.Linear(self.L, self.d_model).to(self.device)
        self.var_pos  = nn.Parameter(torch.zeros(1, self.N, self.d_model))
        self.var_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.head, batch_first=True,
            dim_feedforward=self.d_ff, dropout=self.drop_p, norm_first=True
        ).to(self.device)
        self.var_encoder = nn.TransformerEncoder(
            self.var_enc_layer, num_layers=self.e_layer
        ).to(self.device)

        # --- ---
        # The aggressive 'Time -> Var injection' block has been removed.
        # self.tv_attn, self.tv_ln_q/k/v, self.tv_ffn are all deleted.

        # ============================================================
        # --- --- 3) FUSION BLOCK: Gated fusion of Time into Var
        # ============================================================
        # This gate learns how much temporal summary should be added to each variable token.
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid()
        ).to(self.device)
        self.fusion_ln = nn.LayerNorm(self.d_model).to(self.device)

        # ============================================================
        # 4) SPATIAL CMA: Q= (fused), K=V= (prompt)
        # ============================================================
        self.s_attn = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=self.head, batch_first=True, dropout=self.drop_p
        ).to(self.device)
        self.s_ln_q = nn.LayerNorm(self.d_model).to(self.device)
        self.s_ln_k = nn.LayerNorm(self.d_model).to(self.device)
        self.s_ln_v = nn.LayerNorm(self.d_model).to(self.device)
        self.s_ffn = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_ff), nn.GELU(), nn.Dropout(self.drop_p),
            nn.Linear(self.d_ff, self.d_model),
        ).to(self.device)
        # The gate for the residual connection in Spatial CMA is kept, as it's a good practice.
        self.s_gate = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid()
        ).to(self.device)

        # ============================================================
        # 5) DECODER
        # ============================================================
        self.future_queries = nn.Parameter(torch.randn(1, self.pred_len, self.d_model))
        self.dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=self.head, batch_first=True,
            dim_feedforward=self.d_ff, dropout=self.drop_p, norm_first=True
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(self.dec_layer, num_layers=self.d_layer).to(self.device)

        # ============================================================
        # OUTPUT PROJECTION
        # ============================================================
        self.out_head = nn.Linear(self.d_model, self.N, bias=True).to(self.device)

    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _prep_prompt(self, embeddings: torch.Tensor, B: int):
        emb = embeddings
        if emb.dim() == 4:
            if emb.size(-1)!= 1: raise ValueError(f"Expected trailing singleton in 4D prompt, got {tuple(emb.shape)}")
            emb = emb.squeeze(-1)
        if emb.dim() == 2:
            emb = emb.unsqueeze(0).expand(B, -1, -1)
        if emb.dim()!= 3:
            raise ValueError(f"Prompt embeddings must be 3D (got {tuple(emb.shape)})")
        if emb.size(1) == self.d_llm and emb.size(2) == self.N:
            emb = emb.permute(0, 2, 1).contiguous()
        elif emb.size(1) == self.N and emb.size(2) == self.d_llm:
            pass
        else:
            raise ValueError(f"Prompt shape must align with N={self.N}, E={self.d_llm}; got {tuple(emb.shape)}")
        p = self.prompt_proj(emb)
        p = self.prompt_encoder(p)
        return p

    def forward(self, input_data, input_data_mark, embeddings):
        x   = input_data.float()
        emb = embeddings.float()
        B   = x.size(0)

        if self.revin is not None:
            x = self.revin(x, 'norm')

        # ---------------------------
        # PROMPT TOKENS
        # ---------------------------
        prompt_tok = self._prep_prompt(emb, B)

        # ---------------------------
        # (1) TEMPORAL STREAM
        # ---------------------------
        t_tok = self.time_proj(x) + self.time_pos[:, :x.size(1), :]
        t_tok = self.time_encoder(t_tok)

        tq = self.t_ln_q(t_tok)
        tk = self.t_ln_k(prompt_tok)
        tv = self.t_ln_v(prompt_tok)
        t_ctx, _ = self.t_attn(tq, tk, tv)
        t_ctx = t_tok + t_ctx
        t_ctx = t_ctx + self.t_ffn(t_ctx)  # Final temporal context:

        # ---------------------------
        # (2) SPATIAL STREAM
        # ---------------------------
        v_in  = x.transpose(1, 2).contiguous()
        v_tok = self.var_proj(v_in) + self.var_pos[:, :v_in.size(1), :]
        v_tok = self.var_encoder(v_tok)  # Pure variable tokens:

        # --- ---
        # The entire 'Time -> Var injection' block is replaced by this new Fusion Block.
        # ---------------------------
        # (3) FUSION BLOCK
        # ---------------------------
        # Summarize temporal context across the time dimension.
        t_summary = t_ctx.mean(dim=1, keepdim=True)  #

        # Calculate a gate based on variable tokens to control the information flow.
        gate = self.fusion_gate(v_tok)  #

        # Fuse with a gated mechanism and apply LayerNorm for stability.
        # t_summary is broadcasted to during the operation.
        fused_tok = self.fusion_ln(v_tok + gate * t_summary)

        # ---------------------------
        # (4) SPATIAL CMA
        # ---------------------------
        # Use the fused tokens as the query to align with the prompt.
        sq = self.s_ln_q(fused_tok)
        sk = self.s_ln_k(prompt_tok)
        sv = self.s_ln_v(prompt_tok)
        v_att, _ = self.s_attn(sq, sk, sv)

        # Gated residual connection for stability.
        g = self.s_gate(fused_tok)
        v_ctx = fused_tok + g * v_att
        v_ctx = v_ctx + self.s_ffn(v_ctx)  # Final variable context:

        # ---------------------------
        # (5) DECODER
        # ---------------------------
        tgt = self.future_queries.expand(B, -1, -1)
        dec = self.decoder(tgt=tgt, memory=v_ctx)
        y   = self.out_head(dec)

        if self.revin is not None:
            y = self.revin(y, 'denorm')

        return y