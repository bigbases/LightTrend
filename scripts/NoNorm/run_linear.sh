if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/NoNorm" ]; then
  mkdir ./logs/NoNorm
fi

if [ ! -d "./logs/NoNorm/DLinear" ]; then
  mkdir ./logs/NoNorm/DLinear
fi

gpu=0
use_norm=none
features=M

for model_name in DLinear; do
  for pred_len in 96 192 336 720; do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets \
      --data_path exchange.csv \
      --model_id $use_norm'_'exchange_336_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --learning_rate 0.001 \
      --itr 1 >logs/NoNorm/$model_name/exch_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/electricity \
      --data_path electricity.csv \
      --model_id $use_norm'_'electricity_336_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --learning_rate 0.001 \
      --itr 1 >logs/NoNorm/$model_name/elec_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/traffic \
      --data_path traffic.csv \
      --model_id $use_norm'_'traffic_336_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --itr 1 \
      --gpu $gpu \
      --learning_rate 0.05 \
      --use_norm $use_norm >logs/NoNorm/$model_name/traf_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/weather \
      --data_path weather.csv \
      --model_id $use_norm'_'weather_336_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --itr 1 \
      --gpu $gpu \
      --learning_rate 0.0001 \
      --use_norm $use_norm >logs/NoNorm/$model_name/wea_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTh1.csv \
      --model_id $use_norm'_'ETTh1_336_$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --learning_rate 0.005 \
      --itr 1 >logs/NoNorm/$model_name/eh1_$pred_len.log
    #
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTh2.csv \
      --model_id $use_norm'_'ETTh2_336_$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --learning_rate 0.05 \
      --itr 1 >logs/NoNorm/$model_name/eh2_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTm1.csv \
      --model_id $use_norm'_'ETTm1_336_$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --learning_rate 0.0001 \
      --itr 1 >logs/NoNorm/$model_name/em1_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTm2.csv \
      --model_id $use_norm'_'ETTm2_336_$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --learning_rate 0.01 \
      --itr 1 >logs/NoNorm/$model_name/em2_$pred_len.log
  done
done

for model_name in DLinear; do
  for pred_len in 24 36 48 60; do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/illness \
      --data_path national_illness.csv \
      --model_id $use_norm'_'ili_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm None \
      --learning_rate 0.01 \
      --itr 1 >logs/NoNorm/$model_name/ili_$pred_len.log
  done
done