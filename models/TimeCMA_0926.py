
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal


class Dual(nn.Module):
    """
    TimeCMA_0926 - Enhanced Multi-Modal Time Series Forecasting
    
    주요 개선사항:
    - Parallel Multi-Modal Processing (기존 순차 → 병렬)
    - Multi-Scale Temporal Modeling (다중 시간 스케일)
    - Enhanced Skip Connections
    - Dynamic Fusion Mechanism
    - Improved Decoder with Autoregressive Elements
    """

    def __init__(
        self,
        device="cuda",
        channel=32,          # C: 노드 임베딩 채널 수
        num_nodes=32,        # N: 노드/클러스터 개수
        seq_len=96,          # L_in
        pred_len=96,         # L_out
        dropout_n=0.1,
        d_llm=4096,          # 프롬프트 임베딩 입력 차원
        e_layer=2,           # 인코더 레이어 증가
        d_layer=2,           # 디코더 레이어 증가
        d_ff=64,             # FFN 차원 증가
        head=8,

        # --- 개선된 옵션들 ---
        use_time_feat=False,  # Fourier 비활성화 (성능 차이 없음)
        use_multi_scale=True, # 다중 시간 스케일 사용
        time_scales=[1, 7, 30], # 일/주/월 스케일
        p_drop_prompt=0.1,    # 드롭아웃 감소
        p_drop_image=0.1,
        use_skip_connections=True,  # Skip connection 활성화
        fusion_method="dynamic",    # "fixed" or "dynamic"
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
        self.use_multi_scale = use_multi_scale
        self.time_scales = time_scales
        self.p_drop_prompt = float(p_drop_prompt)
        self.p_drop_image = float(p_drop_image)
        self.use_skip_connections = use_skip_connections
        self.fusion_method = fusion_method

        # --- RevIN ---
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)

        # --- Multi-Scale Temporal Processing ---
        if self.use_multi_scale:
            self.multi_scale_convs = nn.ModuleList([
                nn.Conv1d(self.num_nodes, self.num_nodes, kernel_size=scale, 
                         padding='same', groups=self.num_nodes)  # 'same' padding으로 길이 보장
                for scale in self.time_scales
            ]).to(self.device)
            self.scale_fusion = nn.Linear(len(self.time_scales) * self.num_nodes, self.num_nodes).to(self.device)

        # --- Enhanced Length to Feature ---
        self.length_to_feature = nn.Sequential(
            nn.Linear(self.seq_len, self.channel * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_n),
            nn.Linear(self.channel * 2, self.channel)
        ).to(self.device)
        self.ln_after_len2feat = nn.LayerNorm(self.channel).to(self.device)

        # --- (A) Enhanced Time-Series Encoder ---
        self.ts_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.channel, nhead=self.head, dim_feedforward=self.d_ff,
                batch_first=True, norm_first=True, dropout=self.dropout_n
            ) for _ in range(self.e_layer)
        ]).to(self.device)

        # --- (B) Enhanced Prompt Encoder ---
        self.prompt_pool = nn.AdaptiveAvgPool1d(self.d_llm)
        self.ln_prompt_in = nn.LayerNorm(self.d_llm).to(self.device)
        
        self.prompt_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_llm, nhead=self.head, dim_feedforward=self.d_ff * 4,
                batch_first=True, norm_first=True, dropout=self.dropout_n
            ) for _ in range(self.e_layer)
        ]).to(self.device)

        # --- (C) Parallel Cross-Modal Attention ---
        # Temporal Cross-Attention (Prompt)
        self.cross_temporal = CrossModal(
            d_model=self.num_nodes, n_heads=4, d_ff=self.d_ff, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
            activation="gelu", res_attention=True, n_layers=2, store_attn=False
        ).to(self.device)

        # Spatial Cross-Attention (Image)
        self.c_to_nodes = nn.Linear(self.channel, self.num_nodes, bias=False).to(self.device)
        self.nodes_to_c = nn.Linear(self.num_nodes, self.channel, bias=False).to(self.device)
        self.ln_q_spatial_in = nn.LayerNorm(self.num_nodes).to(self.device)

        self.img_k_proj = nn.Sequential(
            nn.Linear(self.num_nodes, self.num_nodes),
            nn.GELU(),
            nn.Dropout(self.dropout_n)
        ).to(self.device)
        
        self.img_v_proj = nn.Sequential(
            nn.Linear(self.num_nodes, self.num_nodes),
            nn.GELU(), 
            nn.Dropout(self.dropout_n)
        ).to(self.device)
        
        self.ln_img_kv = nn.LayerNorm(self.num_nodes).to(self.device)

        self.cross_spatial = CrossModal(
            d_model=self.num_nodes, n_heads=4, d_ff=self.d_ff, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
            activation="gelu", res_attention=True, n_layers=2, store_attn=False
        ).to(self.device)

        # --- Dynamic Fusion Mechanism ---
        if self.fusion_method == "dynamic":
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.channel, self.channel // 2),
                nn.GELU(),
                nn.Linear(self.channel // 2, 3),  # 3개 컴포넌트 (original, temporal, spatial)
                nn.Softmax(dim=-1)
            ).to(self.device)
        else:
            # Fixed gates (기존 방식)
            self.g_t = nn.Parameter(torch.tensor(0.1))
            self.g_s = nn.Parameter(torch.tensor(0.1))

        # --- Skip Connection for Multi-Scale ---
        if self.use_skip_connections:
            self.skip_proj = nn.Linear(self.num_nodes, self.channel).to(self.device)

        # --- Enhanced Decoder with Autoregressive Elements ---
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.channel, nhead=self.head, dim_feedforward=self.d_ff,
                batch_first=True, norm_first=True, dropout=self.dropout_n
            ) for _ in range(self.d_layer)
        ]).to(self.device)

        # Projection Head with Residual
        self.projection_head = nn.Sequential(
            nn.Linear(self.channel, self.channel * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_n),
            nn.Linear(self.channel * 2, self.pred_len)
        ).to(self.device)

        self.ln_after_spatial = nn.LayerNorm(self.channel).to(self.device)
        
        # --- Learnable Position Embeddings ---
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_nodes, self.channel) * 0.02)

    # ---------- utils ----------
    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _ensure_prompt_shape(p: torch.Tensor) -> torch.Tensor:
        return p.squeeze(-1) if (p is not None and p.dim() == 4) else p

    def _multi_scale_processing(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-scale temporal processing"""
        if not self.use_multi_scale:
            return x
        
        B, L, N = x.shape
        x_conv = x.transpose(1, 2)  # [B, N, L]
        
        scale_outputs = []
        for conv in self.multi_scale_convs:
            scale_out = conv(x_conv)  # [B, N, L]
            scale_outputs.append(scale_out)
        
        # Concatenate and fuse
        concat_scales = torch.cat(scale_outputs, dim=1)  # [B, N*scales, L]
        fused = self.scale_fusion(concat_scales.transpose(1, 2)).transpose(1, 2)  # [B, N, L]
        
        return fused.transpose(1, 2)  # [B, L, N]

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
        if self.fusion_method == "dynamic":
            return "dynamic", "dynamic"
        else:
            return float(F.softplus(self.g_t).detach().cpu()), float(F.softplus(self.g_s).detach().cpu())

    # ---------- forward ----------
    def forward(self, input_data, input_data_mark, emb_prompt, emb_image):
        """
        Enhanced forward pass with parallel processing
        """
        x = input_data.float()
        p = self._ensure_prompt_shape(emb_prompt).float()
        B, L, N = x.size()
        assert N == self.num_nodes, f"N mismatch: input {N} vs model {self.num_nodes}"

        # RevIN normalization
        x = self.normalize_layers(x, 'norm')

        # Multi-scale temporal processing
        if self.use_multi_scale:
            x_multi = self._multi_scale_processing(x)
            x = x + 0.1 * x_multi  # Residual connection

        # Transform: [B,L,N] -> [B,N,L] -> [B,N,C]
        x = x.permute(0, 2, 1)
        x = self.length_to_feature(x)
        x = self.ln_after_len2feat(x)
        
        # Skip connection storage (after length_to_feature)
        if self.use_skip_connections:
            x_skip = x.clone()  # [B, N, C]
        
        # Add positional embedding
        x = x + self.pos_embedding

        # Enhanced TS Encoder with skip connections
        ts_outputs = [x]
        for layer in self.ts_encoder_layers:
            x = layer(x)
            if self.use_skip_connections:
                ts_outputs.append(x)
        
        # Combine all encoder outputs
        if self.use_skip_connections and len(ts_outputs) > 1:
            x = x + 0.1 * sum(ts_outputs[:-1]) / len(ts_outputs[:-1])

        # Prompt processing
        p = p.permute(0, 2, 1)  # [B, N, E]
        p = self.prompt_pool(p)  # [B, N, d_llm]
        p = self.ln_prompt_in(p)

        # Enhanced prompt encoder
        for layer in self.prompt_encoder_layers:
            p = layer(p)
        p_enc = p.permute(0, 2, 1)  # [B, d_llm, N]

        # Modality Dropout
        if self.training and (torch.rand(1).item() < self.p_drop_prompt):
            p_enc = None

        # Original features for skip connection
        enc_nodes = x
        q_original = enc_nodes.permute(0, 2, 1)  # [B, C, N]

        # ===== PARALLEL CROSS-MODAL PROCESSING =====
        
        # Branch 1: Temporal Cross-Attention (Prompt)
        if p_enc is not None:
            delta_t = self.cross_temporal(q_original, p_enc, p_enc)
        else:
            delta_t = torch.zeros_like(q_original)

        # Branch 2: Spatial Cross-Attention (Image) 준비
        q_spatial = self.c_to_nodes(enc_nodes)  # [B, N, N]
        q_spatial = self.ln_q_spatial_in(q_spatial)

        k_img, v_img = self._prep_image_kv(emb_image, B=B, N=self.num_nodes, device=x.device)
        k_img = self.img_k_proj(k_img)
        v_img = self.img_v_proj(v_img)
        k_img = self.ln_img_kv(k_img)
        v_img = self.ln_img_kv(v_img)

        if self.training and (torch.rand(1).item() < self.p_drop_image):
            k_img, v_img = None, None

        if (k_img is not None) and (v_img is not None):
            delta_s_raw = self.cross_spatial(q_spatial, k_img, v_img)  # [B, N, N]
        else:
            delta_s_raw = torch.zeros_like(q_spatial)

        delta_s = self.nodes_to_c(delta_s_raw).permute(0, 2, 1)  # [B, C, N]

        # ===== DYNAMIC FUSION =====
        if self.fusion_method == "dynamic":
            # Learn fusion weights dynamically
            fusion_input = enc_nodes.mean(dim=1)  # [B, C] - Global pooling
            fusion_weights = self.fusion_gate(fusion_input)  # [B, 3]
            
            w_orig = fusion_weights[:, 0:1].unsqueeze(-1)  # [B, 1, 1]
            w_temp = fusion_weights[:, 1:2].unsqueeze(-1)  # [B, 1, 1]
            w_spat = fusion_weights[:, 2:3].unsqueeze(-1)  # [B, 1, 1]
            
            z = (w_orig * q_original + w_temp * delta_t + w_spat * delta_s)
        else:
            # Fixed fusion (original method)
            z = q_original + F.softplus(self.g_t) * delta_t + F.softplus(self.g_s) * delta_s

        z = z.permute(0, 2, 1)  # [B, N, C]
        
        # Skip connection from encoder
        if self.use_skip_connections:
            z = z + 0.1 * x_skip
        
        z = self.ln_after_spatial(z)

        # ===== ENHANCED DECODER =====
        decoder_input = z
        for layer in self.decoder_layers:
            decoder_input = layer(decoder_input, decoder_input)

        # Final projection with residual
        dec_out = self.projection_head(decoder_input)  # [B, N, L_out]
        
        # Add residual connection if dimensions match
        if self.use_skip_connections and dec_out.shape[-1] == z.shape[-1]:
            dec_out = dec_out + 0.1 * z

        dec_out = dec_out.permute(0, 2, 1)  # [B, L_out, N]

        # RevIN denormalization
        y_hat = self.normalize_layers(dec_out, 'denorm')
        return y_hat