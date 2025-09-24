# store_emb.py
import os
import time
import h5py
import argparse
import torch
from data_provider.data_loader_save_0908 import Dataset_Custom
from gen_prompt_emb import GenPromptEmb

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    # 네 CSV 경로 지정
    p.add_argument("--csv_path", type=str, required=True)
    # 윈도우 길이
    p.add_argument("--input_len", type=int, default=96)
    p.add_argument("--output_len", type=int, default=96)
    # split 비율
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    # 임베딩 모델
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--d_model", type=int, default=768)
    # divide: 'train' | 'val' | 'test'
    p.add_argument("--divide", type=str, default="train")
    # 저장 루트
    p.add_argument("--save_root", type=str, default="./Embeddings")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 데이터셋 (임베딩 생성 전용)
    ds = Dataset_Custom(
        csv_path=args.csv_path,
        flag=args.divide,
        size=[args.input_len, 0, args.output_len],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        features='M',
        scale=False,     # 프롬프트엔 원본값이 더 자연스러움
        timeenc=0,       # 연/월/일 조합 문자열 생성에 맞춤
        freq='D'
    )

    # 프롬프트 임베딩 생성기
    # data_path는 날짜 포맷용 분기 키: 일 단위면 'FRED' 또는 'ILI' 중 하나로 두면 됨
    gen = GenPromptEmb(
        data_path='FRED',
        model_name=args.model_name,
        device=device,
        input_len=args.input_len,
        d_model=args.d_model,
        layer=12,
        divide=args.divide
    ).to(device)

    # 저장 경로: ./Embeddings/<csv_basename>/<divide>/
    base = os.path.splitext(os.path.basename(args.csv_path))[0]
    save_dir = os.path.join(args.save_root, base, args.divide)
    os.makedirs(save_dir, exist_ok=True)

    t1 = time.time()
    for idx in range(len(ds)):
        x, y, x_mark, y_mark = ds[idx]           # (L_in,C), (L_out,C), (L_in,dt), (L_out,dt)
        # 배치 차원 추가(B=1)
        x_b = x.unsqueeze(0).to(device)
        x_mark_b = x_mark.unsqueeze(0).to(device)

        # 임베딩 생성: (B, d_model, C) → B==1이면 (d_model, C)
        emb = gen.generate_embeddings(x_b, x_mark_b).squeeze(0).detach().cpu().numpy()

        # 파일명: 윈도우 시작 인덱스 s
        s = int(ds.indices[idx])
        fpath = os.path.join(save_dir, f"{s}.h5")
        with h5py.File(fpath, "w") as hf:
            hf.create_dataset("embeddings", data=emb)  # (E, C)

        # 진행 표시 (선택)
        # if idx % 100 == 0: print(f"{args.divide}: saved {idx}/{len(ds)}")

    print(f"[{args.divide}] Saved {len(ds)} files to {save_dir}")
    print(f"Time: {(time.time()-t1)/60:.2f} min")

if __name__ == "__main__":
    main()
