 #!/bin/bash
export PYTHONPATH=/mnt/sfs-common/cxliu/TimeCMA:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4

data_paths=("ECL")
divides=("train" "val" "test")
num_nodes=321
input_len=96
output_len=96

for data_path in "${data_paths[@]}"; do
  for divide in "${divides[@]}"; do
    log_file="./Results/emb_logs/${data_path}_${divide}.log"
    nohup python storage/store_emb.py --divide $divide --data_path $data_path --num_nodes $num_nodes --input_len $input_len --output_len $output_len > $log_file &
  done
done