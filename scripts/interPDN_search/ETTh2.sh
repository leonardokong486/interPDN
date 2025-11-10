model_name=interPDN
ma_type=ema
alpha=0.3
beta=0.3
seq_len=512

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/"$ma_type ]; then
    mkdir ./logs/$ma_type
fi

pred_len=96
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 100 \
    --batch_size 1024 \
    --learning_rate 0.0001 \
    --lradj 'sigmoid'\
    --con_cls_1 0.02 \
    --con_cls_2 0.02 \
    --con_time 0.1 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_ETTh2_'$seq_len'_'$pred_len.log

pred_len=192
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 100 \
    --batch_size 1024 \
    --learning_rate 0.0001 \
    --lradj 'sigmoid'\
    --con_cls_1 0.05 \
    --con_cls_2 0.05 \
    --con_time 0.2 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_ETTh2_'$seq_len'_'$pred_len.log

pred_len=336
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 100 \
    --batch_size 1024 \
    --learning_rate 0.0001 \
    --lradj 'sigmoid'\
    --con_cls_1 0.02 \
    --con_cls_2 0.02 \
    --con_time 0.3 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_ETTh2_'$seq_len'_'$pred_len.log

pred_len=720
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 100 \
    --batch_size 1024 \
    --learning_rate 0.0001 \
    --lradj 'sigmoid'\
    --con_cls_1 0.02 \
    --con_cls_2 0.02 \
    --con_time 0.4 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_ETTh2_'$seq_len'_'$pred_len.log