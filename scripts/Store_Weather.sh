 #!/bin/bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH

data_paths=("Weather")
divides=("train" "val" "test")
num_nodes=21
input_len=96
output_len=96

for data_path in "${data_paths[@]}"; do
  for divide in "${divides[@]}"; do
    log_file="./Results/emb_logs/${data_path}_${divide}.log"
    nohup python storage/store_emb.py --divide $divide --data_path $data_path --device $device --num_nodes $num_nodes --input_len $input_len --output_len $output_len > $log_file &
  done
done