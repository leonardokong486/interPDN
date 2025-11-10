seq_len=36
model_name=interPDN
ma_type=reg
alpha=0.3
beta=0.3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/"$ma_type ]; then
    mkdir ./logs/$ma_type
fi

pred_len=24
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --model_id ili_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --lradj 'type3'\
    --train_epochs 100 \
    --con_cls_1 0.1 \
    --con_cls_2 0.1 \
    --con_time 0.1 \
    --patch_len 6 \
    --stride 3 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_ili_'$seq_len'_'$pred_len.log

pred_len=36
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --model_id ili_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --lradj 'type3'\
    --train_epochs 100 \
    --con_cls_1 0.1 \
    --con_cls_2 0.1 \
    --con_time 0.3 \
    --patch_len 6 \
    --stride 3 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_ili_'$seq_len'_'$pred_len.log

pred_len=48
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --model_id ili_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --lradj 'type3'\
    --train_epochs 100 \
    --con_cls_1 0.1 \
    --con_cls_2 0.1 \
    --con_time 0.3 \
    --patch_len 6 \
    --stride 3 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_ili_'$seq_len'_'$pred_len.log

pred_len=60
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --model_id ili_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --lradj 'type3'\
    --train_epochs 100 \
    --con_cls_1 0.1 \
    --con_cls_2 0.1 \
    --con_time 0.1 \
    --patch_len 6 \
    --stride 3 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_ili_'$seq_len'_'$pred_len.log