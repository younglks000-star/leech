import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

class GenPromptEmb(nn.Module):
    def __init__(
        self,
        data_path='FRED',
        model_name="gpt2",
        device='cuda:0',
        input_len=180,
        d_model=768,
        layer=12,
        divide='train'
    ):
        super(GenPromptEmb, self).__init__()
        self.data_path = data_path
        self.device = device
        self.input_len = input_len
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer
        self.len = self.input_len - 1

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)

    # -------------------------------
    # (추가) 안전 유틸: 추세 합계
    # -------------------------------
    def _trend_sum(self, series_1d):
        """
        series_1d: 1D (numpy or torch)
        returns: sum(diff(series)) as float
        """
        t = torch.as_tensor(series_1d, dtype=torch.float32)
        if t.numel() <= 1:
            return 0.0
        return (t[1:] - t[:-1]).sum().item()

    # -------------------------------
    # (추가) 클러스터 prior 생성
    # -------------------------------
    def _cluster_prior(self, cid: int):
        # 0–7: 북부/늦은 해빙, 8–15: 북중반/중간, 16–23: 남중반/이른, 24–31: 남부/이른
        if 0 <= cid <= 7:
            region, melt, desc = "North", "Late", "far north; later seasonal melt"
        elif 8 <= cid <= 15:
            region, melt, desc = "Mid-North", "Mid", "mid-northern; moderate melt timing"
        elif 16 <= cid <= 23:
            region, melt, desc = "Mid-South", "Early", "mid-southern; earlier seasonal melt"
        else:
            region, melt, desc = "South", "Early", "southern; fastest/earliest melt"
        tag = f"[REGION={region}][MELT={melt}] Cluster {cid}. "
        sent = f"{desc}. "
        return tag + sent

    # -------------------------------
    # (추가) 프롬프트 문자열 생성(사람이 읽을 수 있는 텍스트)
    # -------------------------------
    def make_prompt_text(self, input_template, in_data, in_data_mark, i, j):
        """
        기존 _prepare_prompt는 토큰 텐서를 반환함.
        이 함수는 같은 로직으로 '문자열'을 생성해서 돌려줌.
        """
        # values
        values = torch.as_tensor(in_data[i, :, j]).flatten().tolist()
        values_str = ", ".join([str(int(v)) for v in values])

        # trend
        trends_str = f"{self._trend_sum(values):0f}"

        # 날짜 안전 추출 (넘파이/토치 모두 처리)
        def get_int(*idx):
            return int(torch.as_tensor(in_data_mark[idx], dtype=torch.int64).item())

        # 데이터셋별 날짜 포맷 (기존 분기 유지)
        if self.data_path in ['FRED', 'ILI']:
            start_date = f"{get_int(i,0,2):02d}/{get_int(i,0,1):02d}/{get_int(i,0,0):04d}"
            end_date   = f"{get_int(i,self.len,2):02d}/{get_int(i,self.len,1):02d}/{get_int(i,self.len,0):04d}"
        elif self.data_path in ['ETTh1', 'ETTh2', 'ECL']:
            start_date = f"{get_int(i,0,2):02d}/{get_int(i,0,1):02d}/{get_int(i,0,0):04d} {get_int(i,0,4):02d}:00"
            end_date   = f"{get_int(i,self.len,2):02d}/{get_int(i,self.len,1):02d}/{get_int(i,self.len,0):04d} {get_int(i,self.len,4):02d}:00"
        else:  # ETTm1, ETTm2, Weather
            start_date = f"{get_int(i,0,2):02d}/{get_int(i,0,1):02d}/{get_int(i,0,0):04d} {get_int(i,0,4):02d}:{get_int(i,0,5):02d}"
            end_date   = f"{get_int(i,self.len,2):02d}/{get_int(i,self.len,1):02d}/{get_int(i,self.len,0):04d} {get_int(i,self.len,4):02d}:{get_int(i,self.len,5):02d}"

        # 템플릿 치환
        text = input_template.replace("value1, ..., valuen", values_str)
        text = text.replace("Trends", trends_str)
        text = text.replace("[t1]", start_date).replace("[t2]", end_date)

        # 클러스터 prior 프리픽스
        prefix = self._cluster_prior(j)
        text = prefix + text
        return text

    # -------------------------------
    # (기존) 토큰 텐서 생성
    # -------------------------------
    def _prepare_prompt(self, input_template, in_data, in_data_mark, i, j):
        """
        기존 호출부 호환: 토큰 텐서만 반환
        """
        text = self.make_prompt_text(input_template, in_data, in_data_mark, i, j)
        tokenized_prompt = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        return tokenized_prompt

    def forward(self, tokenized_prompt):
        with torch.no_grad():
            prompt_embeddings = self.model(tokenized_prompt).last_hidden_state
        return prompt_embeddings

    def generate_embeddings(self, in_data, in_data_mark):
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

        tokenized_prompts = []
        max_token_count = 0
        B = len(in_data)
        C = in_data.shape[2]

        for i in range(B):
            for j in range(C):
                tokenized_prompt = self._prepare_prompt(input_template, in_data, in_data_mark, i, j).to(self.device)
                max_token_count = max(max_token_count, tokenized_prompt.shape[1])
                tokenized_prompts.append((i, tokenized_prompt.to(self.device), j))

        in_prompt_emb = torch.zeros(
            (B, max_token_count, self.d_model, C),
            dtype=torch.float32, device=self.device
        )

        for i, tokenized_prompt, j in tokenized_prompts:
            prompt_embeddings = self.forward(tokenized_prompt)  # [1, T, d_model]
            padding_length = max_token_count - tokenized_prompt.shape[1]
            if padding_length > 0:
                last_token_embedding = prompt_embeddings[:, -1, :].unsqueeze(1)
                padding = last_token_embedding.repeat(1, padding_length, 1)
                prompt_embeddings_padded = torch.cat([prompt_embeddings, padding], dim=1)
            else:
                prompt_embeddings_padded = prompt_embeddings

            in_prompt_emb[i, :max_token_count, :, j] = prompt_embeddings_padded
            last_token_emb = in_prompt_emb[:, max_token_count-1:max_token_count, :, :]
            last_token_emb = last_token_emb.squeeze()

        return last_token_emb

    # -------------------------------
    # (추가) 프리뷰: 원문 프롬프트 n개 출력용
    # -------------------------------
    def preview_prompts(self, in_data, in_data_mark, input_template=None, max_examples=10):
        """
        사람이 읽을 수 있는 원문 프롬프트 문자열을 최대 max_examples개 반환
        return: List[(i, j, text)]
        """
        if input_template is None:
            # generate_embeddings에서 쓰는 템플릿 선택 로직과 동일
            templates = {
                'FRED':   "From [t1] to [t2], the values were value1, ..., valuen every month. The total trend value was Trends",
                'ILI':    "From [t1] to [t2], the values were value1, ..., valuen every week. The total trend value was Trends",
                'ETTh1':  "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ETTh2':  "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ECL':    "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ETTm1':  "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
                'ETTm2':  "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
                'Weather':"From [t1] to [t2], the values were value1, ..., valuen every 10 minutes. The total trend value was Trends"
            }
            input_template = templates.get(self.data_path, templates['FRED'])

        previews = []
        B = len(in_data)
        C = in_data.shape[2]
        for i in range(B):
            for j in range(C):
                text = self.make_prompt_text(input_template, in_data, in_data_mark, i, j)
                previews.append((i, j, text))
                if len(previews) >= max_examples:
                    return previews
        return previews
