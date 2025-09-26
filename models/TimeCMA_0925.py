import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal
from transformers import LlamaModel, LlamaConfig, AutoTokenizer
import warnings


class LLMTimeCMA(nn.Module):
    """
    LLM-Enhanced TimeCMA with LLaMA-2-7B Decoder
    
    주요 변경사항:
    - 기존 Transformer Decoder → LLaMA-2-7B 기반 디코더
    - 시계열 데이터를 LLM이 이해할 수 있는 형태로 변환
    - LLM 출력을 시계열 예측으로 변환하는 projection head
    """

    def __init__(
        self,
        device="cuda",
        channel=32,          # C: 노드 임베딩 채널 수
        num_nodes=32,        # N: 노드/클러스터 개수 (수정: 7→32)
        seq_len=96,          # L_in
        pred_len=96,         # L_out
        dropout_n=0.1,
        d_llm=4096,          # 프롬프트 임베딩 입력 차원
        e_layer=1,
        d_layer=1,
        d_ff=32,
        head=8,

        # --- LLM 관련 설정 ---
        llm_model_name="meta-llama/Llama-2-7b-hf",
        llm_hidden_size=4096,    # LLaMA-2-7B hidden size
        freeze_llm_layers=True,  # LLM 레이어 고정 여부
        llm_adapter_dim=128,     # LoRA 어댑터 차원 (선택사항)

        # --- 기존 옵션 ---
        use_time_feat=True,
        time_fourier_periods=(7.0, 30.4375, 91.3125, 365.25),
        p_drop_prompt=0.15,
        p_drop_image=0.15,
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

        self.use_time_feat = use_time_feat
        self.time_fourier_periods = time_fourier_periods
        self.p_drop_prompt = float(p_drop_prompt)
        self.p_drop_image = float(p_drop_image)
        
        self.llm_model_name = llm_model_name
        self.llm_hidden_size = llm_hidden_size
        self.freeze_llm_layers = freeze_llm_layers

        # --- RevIN ---
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)

        # --- 시간 포리에 특징 ---
        if self.use_time_feat:
            self.time_ff = nn.Sequential(
                nn.Linear(2 * len(self.time_fourier_periods), self.num_nodes, bias=False),
                nn.GELU()
            ).to(self.device)
            nn.init.xavier_uniform_(self.time_ff[0].weight)

        # --- [B,N,L] → [B,N,C] ---
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.ln_after_len2feat = nn.LayerNorm(self.channel).to(self.device)

        # --- (A) Time-Series Encoder: [B,N,C] ---
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(
            self.ts_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # --- (B) Prompt Encoder: [B,N,d_llm] ---
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm, nhead=self.head, batch_first=True,
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(
            self.prompt_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # 프롬프트 길이 정규화
        self.prompt_pool = nn.AdaptiveAvgPool1d(self.d_llm)
        self.ln_prompt_in = nn.LayerNorm(self.d_llm).to(self.device)

        # --- (C) Temporal Cross (Prompt) ---
        self.cross = CrossModal(
            d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
            activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)

        # --- (D) Spatial Cross (Image) ---
        self.c_to_nodes = nn.Linear(self.channel, self.num_nodes, bias=False).to(self.device)
        self.nodes_to_c = nn.Linear(self.num_nodes, self.channel, bias=False).to(self.device)
        self.ln_q_spatial_in = nn.LayerNorm(self.num_nodes).to(self.device)

        self.img_k_proj = nn.Linear(self.num_nodes, self.num_nodes, bias=False).to(self.device)
        self.img_v_proj = nn.Linear(self.num_nodes, self.num_nodes, bias=False).to(self.device)
        self.ln_img_kv = nn.LayerNorm(self.num_nodes).to(self.device)

        self.cross_spatial = CrossModal(
            d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
            activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)

        # --- Residual 게이트 ---
        self.g_t = nn.Parameter(torch.tensor(0.1))  # temporal
        self.g_s = nn.Parameter(torch.tensor(0.1))  # spatial

        # ===========================================
        # LLM 기반 디코더 (핵심 변경사항)
        # ===========================================
        
        # LLaMA-2 모델 로드
        try:
            print(f"Loading LLM model: {self.llm_model_name}")
            self.llm_model = LlamaModel.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float32,  # 또는 torch.float16
                device_map=None,  # 수동으로 device 관리
                trust_remote_code=True
            )
            
            # LLM 파라미터 고정 (선택사항)
            if self.freeze_llm_layers:
                for param in self.llm_model.parameters():
                    param.requires_grad = False
                print("LLM parameters frozen")
            
        except Exception as e:
            print(f"Warning: Could not load LLM model {self.llm_model_name}")
            print(f"Error: {e}")
            print("Falling back to standard Transformer decoder")
            self.llm_model = None
            
        # 시계열 데이터를 LLM 입력으로 변환
        self.ts_to_llm_proj = nn.Linear(self.channel, self.llm_hidden_size).to(self.device)
        self.ts_pos_embedding = nn.Parameter(
            torch.randn(1, self.num_nodes, self.llm_hidden_size) * 0.02
        )
        
        # LLM 출력을 시계열 예측으로 변환
        self.llm_to_ts_proj = nn.Sequential(
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_n),
            nn.Linear(self.llm_hidden_size // 2, self.pred_len),
        ).to(self.device)
        
        # 백업: 일반 Transformer Decoder (LLM 로딩 실패시)
        if self.llm_model is None:
            self.fallback_decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.channel, nhead=self.head, batch_first=True,
                norm_first=True, dropout=self.dropout_n
            ).to(self.device)
            self.fallback_decoder = nn.TransformerDecoder(
                self.fallback_decoder_layer, num_layers=self.d_layer
            ).to(self.device)
            self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)
        
        self.ln_after_spatial = nn.LayerNorm(self.channel).to(self.device)

    # ---------- utils ----------
    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _ensure_prompt_shape(p: torch.Tensor) -> torch.Tensor:
        return p.squeeze(-1) if (p is not None and p.dim() == 4) else p

    @staticmethod
    def _build_time_fourier(L: int, device, dtype, periods):
        t = torch.arange(L, device=device, dtype=dtype)
        feats = []
        for p in periods:
            w = 2.0 * torch.pi * t / p
            feats.extend([torch.sin(w), torch.cos(w)])
        return torch.stack(feats, dim=1).transpose(0, 1).transpose(0, 1)

    def _prep_image_kv(self, emb_image, B, N, device):
        assert isinstance(emb_image, dict) and ('K' in emb_image) and ('V' in emb_image), \
            "emb_image must be dict with keys 'K' and 'V'"
        K, V = emb_image['K'], emb_image['V']
        if not torch.is_tensor(K): K = torch.as_tensor(K)
        if not torch.is_tensor(V): V = torch.as_tensor(V)
        K = K.to(device).float()
        V = V.to(device).float()
        if K.dim() == 2: K = K.unsqueeze(0).expand(B, -1, -1)
        if V.dim() == 2: V = V.unsqueeze(0).expand(B, -1, -1)
        assert K.size(1) == N and V.size(1) == N, f"K/V must have N={N} at dim=1, got {K.size()} / {V.size()}"
        K = K.transpose(1, 2).contiguous()
        V = V.transpose(1, 2).contiguous()
        return K, V

    def gate_values(self):
        return float(F.softplus(self.g_t).detach().cpu()), float(F.softplus(self.g_s).detach().cpu())

    def _llm_decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        LLM 기반 디코딩
        
        Args:
            z: [B, N, C] - Cross-modal attention 후 특성
            ㄴㄴ
        Returns:
            predictions: [B, L_out, N] - 시계열 예측 결과
        """
        if self.llm_model is None:
            # LLM 로딩 실패시 백업 디코더 사용
            dec_out = self.fallback_decoder(z, z)  # [B, N, C]
            dec_out = self.c_to_length(dec_out)    # [B, N, L_out]
            return dec_out.permute(0, 2, 1)        # [B, L_out, N]
        
        B, N, C = z.shape
        
        # 1. 시계열 특성을 LLM 차원으로 투영
        llm_input = self.ts_to_llm_proj(z)  # [B, N, llm_hidden_size]
        
        # 2. Position embedding 추가
        llm_input = llm_input + self.ts_pos_embedding  # [B, N, llm_hidden_size]
        
        # 3. LLM forward pass
        try:
            with torch.set_grad_enabled(not self.freeze_llm_layers):
                # attention_mask 생성 (모든 토큰에 attend)
                attention_mask = torch.ones(B, N, dtype=torch.long, device=self.device)
                
                llm_outputs = self.llm_model(
                    inputs_embeds=llm_input,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # 마지막 hidden state 사용
                llm_hidden = llm_outputs.last_hidden_state  # [B, N, llm_hidden_size]
                
        except Exception as e:
            print(f"LLM forward pass failed: {e}")
            # 백업 방법: 단순 linear projection
            llm_hidden = llm_input
        
        # 4. LLM 출력을 시계열 예측으로 변환
        predictions = self.llm_to_ts_proj(llm_hidden)  # [B, N, pred_len]
        predictions = predictions.permute(0, 2, 1)     # [B, pred_len, N]
        
        return predictions

    # ---------- forward ----------
    def forward(self, input_data, input_data_mark, emb_prompt, emb_image):
        """
        input_data : [B, L, N]
        emb_prompt : [B, E, N] or [B, E, N, 1]
        emb_image  : {'K':[B,N,dk]|[N,dk], 'V':[B,N,dv]|[N,dv]}
        return     : [B, L_out, N]
        """
        x = input_data.float()
        p = self._ensure_prompt_shape(emb_prompt).float()
        B, L, N = x.size()
        assert N == self.num_nodes, f"N mismatch: input {N} vs model {self.num_nodes}"

        # RevIN
        x = self.normalize_layers(x, 'norm')

        # 시간 포리에 특징 주입
        if self.use_time_feat:
            tf = self._build_time_fourier(L, device=x.device, dtype=x.dtype, periods=self.time_fourier_periods)
            tf_proj = self.time_ff(tf)
            x = x + tf_proj.unsqueeze(0)

        # [B,L,N] -> [B,N,L] -> L→C
        x = x.permute(0, 2, 1)
        x = self.length_to_feature(x)
        x = self.ln_after_len2feat(x)

        # Prompt 처리
        p = p.permute(0, 2, 1)
        p = self.prompt_pool(p)
        p = self.ln_prompt_in(p)
        p_enc = self.prompt_encoder(p)
        p_enc = p_enc.permute(0, 2, 1)

        # Modality Dropout
        if self.training and (torch.rand(1).item() < self.p_drop_prompt):
            p_enc = None

        # TS encoder
        enc_nodes = self.ts_encoder(x)
        q_t = enc_nodes.permute(0, 2, 1)

        # (1) Temporal Cross (Prompt)
        if p_enc is not None:
            delta_t = self.cross(q_t, p_enc, p_enc)
            z_t = q_t + F.softplus(self.g_t) * delta_t
        else:
            z_t = q_t
        z_t = z_t.permute(0, 2, 1)

        # (2) Spatial Cross (Image)
        q_s = self.c_to_nodes(z_t)
        q_s = self.ln_q_spatial_in(q_s)

        k_img, v_img = self._prep_image_kv(emb_image, B=B, N=self.num_nodes, device=z_t.device)
        k_img = self.img_k_proj(k_img)
        v_img = self.img_v_proj(v_img)
        k_img = self.ln_img_kv(k_img)
        v_img = self.ln_img_kv(v_img)

        if self.training and (torch.rand(1).item() < self.p_drop_image):
            k_img, v_img = None, None

        if (k_img is not None) and (v_img is not None):
            delta_s = self.cross_spatial(q_s, k_img, v_img)
            z_s = q_s + F.softplus(self.g_s) * delta_s
        else:
            z_s = q_s

        z = self.nodes_to_c(z_s)
        z = self.ln_after_spatial(z)

        # ===========================================
        # LLM 기반 디코딩 (핵심 변경사항)
        # ===========================================
        dec_out = self._llm_decode(z)  # [B, L_out, N]

        # RevIN denorm
        y_hat = self.normalize_layers(dec_out, 'denorm')
        return y_hat

# 이전 버전과의 호환성을 위한 Dual 클래스 별칭
Dual = LLMTimeCMA