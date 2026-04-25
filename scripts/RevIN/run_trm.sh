if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/RevIN" ]; then
  mkdir ./logs/RevIN
fi

if [ ! -d "./logs/RevIN/Autoformer" ]; then
  mkdir ./logs/RevIN/Autoformer
fi

if [ ! -d "./logs/RevIN/FEDformer" ]; then
  mkdir ./logs/RevIN/FEDformer
fi

if [ ! -d "./logs/RevIN/iTransformer" ]; then
  mkdir ./logs/RevIN/iTransformer
fi

use_norm=revin
features=M
gpu=0

for model_name in Autoformer FEDformer iTransformer; do

    # model_name=iTransformer -> seq_len=720, label_len=168
    if [ $model_name == "iTransformer" ]; then
        seq_len=720
        label_len=168
    else
        seq_len=96
        label_len=48
    fi

  for pred_len in 96 192 336 720; do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets \
      --data_path exchange.csv \
      --model_id $use_norm'_'exchange'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --station_lr 0.001 \
      --itr 1 >logs/RevIN/$model_name/exch_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets \
      --data_path electricity.csv \
      --model_id $use_norm'_'electricity'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --itr 1 >logs/RevIN/$model_name/elec_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets \
      --data_path traffic.csv \
      --model_id $use_norm'_'traffic'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --itr 1 >logs/RevIN/$model_name/traf_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets \
      --data_path weather.csv \
      --model_id $use_norm'_'weather'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --itr 1 >logs/RevIN/$model_name/wea_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTh1.csv \
      --model_id $use_norm'_'ETTh1'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --itr 1 >logs/RevIN/$model_name/eh1_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTh2.csv \
      --model_id $use_norm'_'ETTh2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --itr 1 >logs/RevIN/$model_name/eh2_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTm1.csv \
      --model_id $use_norm'_'ETTm1'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --itr 1 >logs/RevIN/$model_name/em1_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTm2.csv \
      --model_id $use_norm'_'ETTm2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --itr 1 >logs/RevIN/$model_name/em2_$pred_len.log
  done
done

for model_name in Autoformer FEDformer iTransformer; do
  for pred_len in 24 36 48 60; do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/illness \
      --data_path national_illness.csv \
      --model_id $use_norm'_'ili_36_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 36 \
      --label_len 18 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --itr 1 >logs/RevIN/$model_name/ili_$pred_len.log
  done
done