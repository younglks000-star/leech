# storage/gen_prompt_llama_emb.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class GenPromptEmb(nn.Module):
    """
    LLaMA-2 7B 기반 프롬프트 임베딩 생성기
    - device_map="auto": 가중치 오프로딩/샤딩 (모델에 .to(...) 금지)
    - device_map=None + torch_dtype=torch.float16: 단일 GPU(FP16)로 전부 탑재
    - 출력: 마지막 토큰 임베딩 [B, d_model, C]
    """
    def __init__(
        self,
        data_path='FRED',
        model_name="meta-llama/Llama-2-7b-hf",
        device='cuda:0',
        input_len=360,
        d_model=None,          # None이면 backbone hidden_size 사용 (LLaMA-2 7B=4096)
        layer=None,            # (미사용) 호환성 유지
        divide='train',
        load_in_4bit=False,    # bitsandbytes 없으면 False 권장
        load_in_8bit=False,
        torch_dtype=None,      # 예: torch.float16
        device_map=None,       # 단일 4090이면 None 권장(FP16); 오프로딩이면 "auto"
        low_cpu_mem_usage=True # True 권장
    ):
        super().__init__()
        self.data_path = data_path
        self.input_len = input_len
        self.model_name = model_name
        self.layer = layer
        self.len = self.input_len - 1

        # device 정규화
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.device_map = device_map

        # ----- Tokenizer -----
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            # 수동 패딩 전략 쓰지만, 경고 방지를 위해 pad를 eos로 맞춰둠
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ----- Model -----
        model_kwargs = dict(
            device_map=device_map,            # None 또는 "auto"
            trust_remote_code=False,
            low_cpu_mem_usage=low_cpu_mem_usage
        )
        # 양자화(옵션) — bitsandbytes 설치되어 있을 때만 True로
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        # dtype
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

        # 단일 GPU에 모두 올리는 모드(FP16 등)일 때만 to(device)
        if self.device_map is None:
            # 주: device_map="auto"일 때는 절대 .to(...) 하지 말 것 (meta tensor 에러)
            self.model.to(self.device)

        # d_model 자동 설정 (LLaMA-2 7B = 4096)
        self.d_model = int(self.model.config.hidden_size) if d_model is None else int(d_model)

    def _format_prompt(self, input_template, in_data, in_data_mark, i, j):
        # values → "v1, v2, ..., vT"
        values = in_data[i, :, j].flatten().tolist()
        values_str = ", ".join([str(int(v)) for v in values])

        # 간단 trend 합
        trends = torch.sum(torch.diff(in_data[i, :, j].flatten()))
        trends_str = f"{trends.item():0f}"

        # 날짜 문자열
        if self.data_path in ['FRED', 'ILI']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d}"
            end_date   = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d}"
        elif self.data_path in ['ETTh1', 'ETTh2', 'ECL']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:00"
            end_date   = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:00"
        else:  # ETTm1, ETTm2, Weather
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:{int(in_data_mark[i,0,5]):02d}"
            end_date   = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:{int(in_data_mark[i,self.len,5]):02d}"

        # 템플릿 채우기
        in_prompt = (input_template
                     .replace("value1, ..., valuen", values_str)
                     .replace("Trends", trends_str)
                     .replace("[t1]", start_date).replace("[t2]", end_date))
        return in_prompt

    def _tokenize(self, prompt: str):
        # 수동 패딩/트렁케이션 안 함 (문장 길이를 그대로 임베딩)
        out = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        # device_map="auto"면 CPU에 둬도 HF가 알아서 라우팅
        # 단일 GPU(FP16) 모드이면 입력만 GPU로 보냄
        if self.device_map is None:
            out = {k: v.to(self.device) for k, v in out.items()}
        return out

    @torch.no_grad()
    def forward(self, tokenized_inputs):
        # outputs.last_hidden_state: [B=1, L, d_model]
        outputs = self.model(**tokenized_inputs)
        return outputs.last_hidden_state

    def generate_embeddings(self, in_data, in_data_mark):
        # 작업별 템플릿
        input_templates = {
            'FRED':   "From [t1] to [t2], the values were value1, ..., valuen every month. The total trend value was Trends",
            'ILI':    "From [t1] to [t2], the values were value1, ..., valuen every week. The total trend value was Trends",
            'ETTh1':  "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            'ETTh2':  "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            'ECL':    "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            'ETTm1':  "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
            'ETTm2':  "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
            'Weather':"From [t1] to [t2], the values were value1, ..., valuen every 10 minutes. The total trend value was Trends"
        }
        input_template = input_templates.get(self.data_path, input_templates['FRED'])

        B, T, C = in_data.shape
        tokenized_list = []
        max_len = 0

        # 1) 모든 윈도우에 대해 토크나이즈 + 최대 길이 파악
        for i in range(B):
            for j in range(C):
                prompt = self._format_prompt(input_template, in_data, in_data_mark, i, j)
                tok = self._tokenize(prompt)  # dict of tensors
                seq_len = int(tok["input_ids"].shape[1])
                max_len = max(max_len, seq_len)
                tokenized_list.append((i, j, tok))

        # 2) 임베딩 버퍼 할당: [B, Lmax, d_model, C]
        # dtype은 일단 float32로 두되, 모델 출력이 fp16이면 그걸 그대로 copy할 때 캐스팅됨
        dev_for_buf = self.device if self.device_map is None else torch.device("cpu")
        in_prompt_emb = torch.zeros((B, max_len, self.d_model, C), dtype=torch.float32, device=dev_for_buf)

        # 3) 모델 추론 + 우측 패딩(마지막 토큰 반복)
        for (i, j, tok) in tokenized_list:
            hs = self.forward(tok)  # [1, L, d_model]
            pad = max_len - hs.shape[1]
            if pad > 0:
                last = hs[:, -1, :].unsqueeze(1)           # [1,1,d]
                hs = torch.cat([hs, last.repeat(1, pad, 1)], dim=1)
            # 버퍼 디바이스로 이동 (필요 시)
            if hs.device != in_prompt_emb.device:
                hs = hs.to(in_prompt_emb.device)
            in_prompt_emb[i, :max_len, :, j] = hs

        # 4) 마지막 토큰만 슬라이스 → [B, 1, d_model, C] → squeeze(1) → [B, d_model, C]
        last_token_emb = in_prompt_emb[:, max_len-1:max_len, :, :].squeeze(1)
        return last_token_emb
