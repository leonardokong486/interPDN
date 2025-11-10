model_name=interPDN
ma_type=ema
alpha=0.3
beta=0.3
seq_len=720

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/"$ma_type ]; then
    mkdir ./logs/$ma_type
fi

pred_len=96
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len $pred_len \
    --enc_in 862 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 120 \
    --batch_size 64 \
    --learning_rate 0.005 \
    --lradj 'sigmoid'\
    --con_cls_1 0.1 \
    --con_cls_2 0.1 \
    --con_time 0.2 \
    --patch_len 24 \
    --stride 12 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_traffic_'$seq_len'_'$pred_len.log 

pred_len=192
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len $pred_len \
    --enc_in 862 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 120 \
    --batch_size 64 \
    --learning_rate 0.005 \
    --lradj 'sigmoid'\
    --con_cls_1 0.1 \
    --con_cls_2 0.1 \
    --con_time 0.2 \
    --patch_len 24 \
    --stride 12 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_traffic_'$seq_len'_'$pred_len.log 

pred_len=336
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len $pred_len \
    --enc_in 862 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 120 \
    --batch_size 64 \
    --learning_rate 0.005 \
    --lradj 'sigmoid'\
    --con_cls_1 0.05 \
    --con_cls_2 0.05 \
    --con_time 0.2 \
    --patch_len 24 \
    --stride 12 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_traffic_'$seq_len'_'$pred_len.log 

pred_len=720
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len $pred_len \
    --enc_in 862 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 120 \
    --batch_size 64 \
    --learning_rate 0.005 \
    --lradj 'sigmoid'\
    --con_cls_1 0.05 \
    --con_cls_2 0.05 \
    --con_time 0.3 \
    --patch_len 24 \
    --stride 12 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_traffic_'$seq_len'_'$pred_len.log 