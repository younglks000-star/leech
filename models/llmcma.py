import torch
import torch.nn as nn
from einops import rearrange
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal
from transformers import GPT2Model

class GPT2Decoder(nn.Module):
    """
    Minimal GPT-2 decoder drop-in:
      in : x ∈ [B, N, C]  (각 노드의 C-차원 특징)
      out: y ∈ [B, N, C]
    내부:
      - C → d_llm(=768) 선형투영
      - learnable soft prompt P개 + 노드 토큰(1개) → GPT-2(inputs_embeds)
      - 마지막 토큰 히든을 다시 d_llm → C로 투영
    토크나이저/문장 프롬프트 없이 TimeLLM의 "LLM로 예측단 처리" 아이디어만 반영.
    """
    def __init__(
        self,
        d_in: int,                  # = channel (C)
        d_llm: int = 768,           # GPT-2 hidden size
        soft_n: int = 8,            # soft prompt 길이
        llm_name: str = "openai-community/gpt2",
        device: str = "cuda:0",
        device_map=None,            # None: 단일 GPU(FP16), "auto": 오프로딩
        torch_dtype: torch.dtype = torch.float16,
        freeze_llm: bool = True
    ):
        super().__init__()
        self.d_in = d_in
        self.d_llm = d_llm
        self.soft_n = soft_n

        # GPT-2 본체(토크나이저 불필요)
        self.llm = GPT2Model.from_pretrained(
            llm_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        # device_map=None(단일 GPU)일 때만 명시 이동
        if device_map is None:
            self.llm.to(device)
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
        self.llm.eval()

        # C ↔ d_llm 사상 + soft prompt 파라미터
        self.in_proj  = nn.Linear(d_in, d_llm)
        self.out_proj = nn.Linear(d_llm, d_in)
        self.soft_prompt = nn.Parameter(torch.randn(soft_n, d_llm) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C]
        return: [B, N, C]
        """
        B, N, C = x.shape
        dev = self.llm.device

        # 노드별 토큰 생성: C → d_llm
        h = self.in_proj(x)                      # [B, N, d_llm]
        h = h.reshape(B * N, 1, self.d_llm)      # [B*N, 1, d_llm]

        # soft prompt 복제
        sp = self.soft_prompt.unsqueeze(0).repeat(B * N, 1, 1)  # [B*N, soft_n, d_llm]

        # concat 후 GPT-2 실행
        inputs_embeds = torch.cat([sp.to(dev), h.to(dev)], dim=1)   # [B*N, soft_n+1, d_llm]
        out = self.llm(inputs_embeds=inputs_embeds)
        last_h = out.last_hidden_state[:, -1, :]                    # [B*N, d_llm]

        # d_llm → C
        y = self.out_proj(last_h)                                   # [B*N, C]
        y = y.view(B, N, C)                                         # [B, N, C]
        return y


class Dual(nn.Module):
    def __init__(
        self,
        device = "cuda:7",
        channel = 32,
        num_nodes = 7,
        seq_len = 96,
        pred_len = 96,
        dropout_n = 0.1,
        d_llm = 768,      # GPT-2 hidden size
        e_layer = 1,
        d_layer = 1,      # (사용 안 해도 시그니처 유지)
        d_ff=32,
        head =8,
        # ↓↓↓ 추가 옵션(기본값 유지로 기존 호출 안 깨짐)
        gpt2_soft_n = 8,
        gpt2_name = "openai-community/gpt2",
        gpt2_dtype = torch.float16,
        gpt2_device_map = None,   # 4090이면 None 권장
        gpt2_freeze = True,
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n= dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.d_ff = d_ff
        self.head = head

        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)

        # Time Series Encoder
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers=self.e_layer).to(self.device)

        # Prompt Encoder
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers=self.e_layer).to(self.device)

        # Cross-modality alignment (Q=TS(C), K/V=LLM(E), node 축 정렬)
        self.cross = CrossModal(
            d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n,
            pre_norm=True, activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)

        # ---------------------------
        # 🔁 Decoder 교체: GPT-2 기반
        # ---------------------------
        self.llm_decoder = GPT2Decoder(
            d_in=self.channel,          # C
            d_llm=self.d_llm,           # 768
            soft_n=gpt2_soft_n,
            llm_name=gpt2_name,
            device=self.device,
            device_map=gpt2_device_map,
            torch_dtype=gpt2_dtype,
            freeze_llm=gpt2_freeze
        )

        # Projection (그대로 유지)
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_data, input_data_mark, embeddings):
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()
        embeddings = embeddings.float()

        # RevIN
        input_data = self.normalize_layers(input_data, 'norm')

        # [B, L, N] -> [B, N, L] -> Linear(L->C) -> [B, N, C]
        x = input_data.permute(0,2,1)
        x = self.length_to_feature(x)

        # embeddings: [B, E, N, 1] or [B, E, N] -> [B, N, E]
        emb = embeddings.squeeze(-1) if embeddings.dim()==4 and embeddings.size(-1)==1 else embeddings
        emb = emb.permute(0,2,1)

        # Encoder
        enc_out = self.ts_encoder(x)         # [B, N, C]
        enc_out = enc_out.permute(0,2,1)     # [B, C, N]
        emb = self.prompt_encoder(emb)       # [B, N, E]
        emb = emb.permute(0,2,1)             # [B, E, N]

        # Cross (Q=TS, K/V=LLM Token-G)
        cross_out = self.cross(enc_out, emb, emb)  # [B, C, N]
        cross_out = cross_out.permute(0,2,1)       # [B, N, C]

        # 🔁 Decoder (GPT-2)
        dec_feat = self.llm_decoder(cross_out)     # [B, N, C]

        # Projection → [B, N, L] → [B, L, N]
        dec_out = self.c_to_length(dec_feat)       # [B, N, L]
        dec_out = dec_out.permute(0,2,1)           # [B, L, N]

        # denorm
        dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out
