model_name=DLinear
seq_len=512

mkdir -p ./logs
mkdir -p ./logs/DLinear

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --con_cls_1 0.05 \
  --con_cls_2 0.05 \
  --con_time 0.1 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 >logs/DLinear/ETTm2-96.log

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --con_cls_1 0.05 \
  --con_cls_2 0.05 \
  --con_time 0.1 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 >logs/DLinear/ETTm2-192.log

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --con_cls_1 0.05 \
  --con_cls_2 0.05 \
  --con_time 0.1 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 >logs/DLinear/ETTm2-336.log

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --con_cls_1 0.05 \
  --con_cls_2 0.05 \
  --con_time 0.1 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 >logs/DLinear/ETTm2-720.log