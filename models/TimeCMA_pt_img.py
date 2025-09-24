import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal

class Dual(nn.Module):
    """
    Sequential Cross-Attention:
      Q = TS
      1st pass:  Q <- Q + σ(gp) * Cross(Q, Kp, Vp)  (prompt)
      2nd pass:  Q <- Q + σ(gi) * Cross(Q, Ki, Vi)  (image)
      (각 pass 뒤 LayerNorm)
    """
    def __init__(
        self,
        device="cuda:0",
        channel=32,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.1,
        d_llm=768,       # prompt embedding dim
        e_layer=1,
        d_layer=1,
        d_ff=32,
        head=8,
        cross_order="prompt-first",  # or "image-first"
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
        self.cross_order = cross_order

        # -------- RevIN & L->C --------
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)

        # -------- TS Encoder --------
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers=self.e_layer).to(self.device)

        # -------- Prompt Encoder & proj(E->C) --------
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers=self.e_layer).to(self.device)
        self.proj_kv_prompt = nn.Linear(self.d_llm, self.channel).to(self.device)

        # -------- Image proj (lazy init) --------
        self.proj_img_k = None  # for dict['K']: dk -> C
        self.proj_img_v = None  # for dict['V']: dv -> C
        self.proj_img_any = None  # for tensor [B, E_img, N]: E_img -> C

        # -------- Cross modules (공유/분리 가능) --------
        # 하나를 두 번 써도 되고, 모달별로 분리해도 됨. 여기선 분리해서 약간의 용량/유연성 확보.
        self.cross_prompt = CrossModal(
            d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff,
            norm='LayerNorm', attn_dropout=self.dropout_n, dropout=self.dropout_n,
            pre_norm=True, activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)
        self.cross_image = CrossModal(
            d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff,
            norm='LayerNorm', attn_dropout=self.dropout_n, dropout=self.dropout_n,
            pre_norm=True, activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)

        # -------- Residual gates (learnable) --------
        self.gate_prompt = nn.Parameter(torch.tensor(1.0))
        self.gate_image  = nn.Parameter(torch.tensor(1.0))
        self.sigmoid = nn.Sigmoid()

        # -------- Post-LayerNorms (채널 축 기준) --------
        # Q shape: [B, C, N] → LN은 [B, N, C]로 바꿔 채널 기준 정규화
        self.post_ln_prompt = nn.LayerNorm(self.channel).to(self.device)
        self.post_ln_image  = nn.LayerNorm(self.channel).to(self.device)

        # -------- Decoder --------
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.d_layer).to(self.device)

        # -------- Projection C -> L_out --------
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    def param_num(self):
        return sum(p.nelement() for p in self.parameters())
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---- helpers ----
    def _ln_on_Q(self, Q, ln: nn.LayerNorm):
        # Q: [B,C,N] -> [B,N,C] -> LN -> [B,C,N]
        Qn = Q.permute(0, 2, 1).contiguous()
        Qn = ln(Qn)
        Qn = Qn.permute(0, 2, 1).contiguous()
        return Qn

    def _prep_prompt_kv(self, prompt_emb: torch.Tensor):
        # prompt_emb: [B, E, N] or [B, E, N,1]
        if prompt_emb.dim() == 4 and prompt_emb.shape[-1] == 1:
            prompt_emb = prompt_emb.squeeze(-1)       # [B, E, N]
        X = prompt_emb.permute(0, 2, 1)               # [B, N, E]
        enc = self.prompt_encoder(X)                  # [B, N, E]
        kv = self.proj_kv_prompt(enc)                 # [B, N, C]
        kv = kv.permute(0, 2, 1).contiguous()         # [B, C, N]
        return kv, kv  # K=V

    def _prep_image_kv(self, img):
        # dict {'K':[B,N,dk], 'V':[B,N,dv]} or tensor [B,E_img,N(,1)]
        if isinstance(img, dict) and ('K' in img or 'V' in img):
            K = img.get('K', None)
            V = img.get('V', None)
            if K is None and V is None:
                raise ValueError("image dict must contain 'K' and/or 'V'.")

            if K is not None:
                dk = K.shape[-1]
                if self.proj_img_k is None:
                    self.proj_img_k = nn.Linear(dk, self.channel).to(self.device)
                Kc = self.proj_img_k(K).permute(0, 2, 1).contiguous()  # [B,C,N]
            else:
                Kc = None

            if V is not None:
                dv = V.shape[-1]
                if self.proj_img_v is None:
                    self.proj_img_v = nn.Linear(dv, self.channel).to(self.device)
                Vc = self.proj_img_v(V).permute(0, 2, 1).contiguous()  # [B,C,N]
            else:
                Vc = Kc

            if Kc is None:
                Kc = Vc
            return Kc, Vc

        if torch.is_tensor(img):
            ten = img
            if ten.dim() == 4 and ten.shape[-1] == 1:
                ten = ten.squeeze(-1)                  # [B,E_img,N]
            ten = ten.permute(0, 2, 1)                 # [B,N,E_img]
            E_img = ten.shape[-1]
            if self.proj_img_any is None:
                self.proj_img_any = nn.Linear(E_img, self.channel).to(self.device)
            Vc = self.proj_img_any(ten).permute(0, 2, 1).contiguous()  # [B,C,N]
            return Vc, Vc
        raise TypeError("image embeddings must be dict({'K','V'}) or a tensor.")

    def forward(self, input_data, input_data_mark=None, embeddings=None, image_embeddings=None):
        """
        input_data:        [B, L, N]
        embeddings:        (optional) prompt embeddings [B,E_text,N(,1)] or image dict (하위호환)
        image_embeddings:  (optional) image embeddings (dict{'K','V'} or [B,E_img,N(,1)])
        """
        x = input_data.float()                            # [B,L,N]
        x = self.normalize_layers(x, 'norm')              # RevIN

        # L→C & TS encoder
        ts = x.permute(0, 2, 1)                           # [B,N,L]
        ts = self.length_to_feature(ts)                   # [B,N,C]
        ts_enc = self.ts_encoder(ts)                      # [B,N,C]
        Q = ts_enc.permute(0, 2, 1).contiguous()          # [B,C,N]

        # 입력 분기
        prompt_emb = None
        img_emb = None
        if image_embeddings is not None:
            prompt_emb = embeddings if (embeddings is not None and torch.is_tensor(embeddings)) else None
            img_emb = image_embeddings
        else:
            if embeddings is not None:
                if isinstance(embeddings, dict) and ('K' in embeddings or 'V' in embeddings):
                    img_emb = embeddings
                else:
                    prompt_emb = embeddings

        # 준비된 K/V
        Kp = Vp = None
        Ki = Vi = None
        if prompt_emb is not None:
            Kp, Vp = self._prep_prompt_kv(prompt_emb)     # [B,C,N]
        if img_emb is not None:
            Ki, Vi = self._prep_image_kv(img_emb)         # [B,C,N]

        # ===== Sequential Cross =====
        def pass_prompt(Q):
            if Kp is None: return Q
            gp = self.sigmoid(self.gate_prompt)
            Zp = self.cross_prompt(Q, Kp, Vp)             # [B,C,N]
            Q = Q + gp * Zp
            Q = self._ln_on_Q(Q, self.post_ln_prompt)
            return Q

        def pass_image(Q):
            if Ki is None: return Q
            gi = self.sigmoid(self.gate_image)
            Zi = self.cross_image(Q, Ki, Vi)              # [B,C,N]
            Q = Q + gi * Zi
            Q = self._ln_on_Q(Q, self.post_ln_image)
            return Q

        if (Kp is not None) and (Ki is not None):
            if self.cross_order.lower().startswith("prompt"):
                Q = pass_prompt(Q)
                Q = pass_image(Q)
            else:
                Q = pass_image(Q)
                Q = pass_prompt(Q)
        elif (Kp is not None):
            Q = pass_prompt(Q)
        elif (Ki is not None):
            Q = pass_image(Q)
        # else: 둘 다 없으면 TS-only (Q 그대로)

        # Decoder & Projection
        dec_in = Q.permute(0, 2, 1).contiguous()          # [B,N,C]
        dec_out = self.decoder(dec_in, dec_in)            # [B,N,C]
        y = self.c_to_length(dec_out)                     # [B,N,L_out]
        y = y.permute(0, 2, 1).contiguous()               # [B,L_out,N]

        # RevIN inverse
        y = self.normalize_layers(y, 'denorm')
        return y
