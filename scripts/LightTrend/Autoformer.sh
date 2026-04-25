if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LightTrend" ]; then
  mkdir ./logs/LightTrend
fi

if [ ! -d "./logs/LightTrend/Autoformer" ]; then
  mkdir ./logs/LightTrend/Autoformer
fi

seq_len=96
label_len=48
features=M
gpu=0
model_name=Autoformer
use_norm=lt
use_mlp=0
t_norm=1

for station_lr in 0.0001 ; do
  for learning_rate in 0.0001; do
    for pred_len in 96 192 336 720; do
      CUDA_VISIBLE_DEVICES=$gpu \
      python -u run.py \
        --is_training 1 \
        --use_norm $use_norm \
        --root_path ./datasets \
        --data_path traffic.csv \
        --model_id $use_norm'_'traffic_96_$pred_len \
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
        --itr 1 \
        --num_workers 0 \
        --learning_rate $learning_rate \
        --t_norm $t_norm \
        --station_lr $station_lr \
        --use_mlp $use_mlp >logs/LightTrend/$model_name/traf_$pred_len.log
      done

    for pred_len in 96 192 336 720; do
      CUDA_VISIBLE_DEVICES=$gpu \
      python -u run.py \
        --is_training 1 \
        --use_norm $use_norm \
        --root_path ./datasets \
        --data_path electricity.csv \
        --model_id $use_norm'_'electricity_96_$pred_len \
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
        --itr 1 \
        --num_workers 0 \
        --learning_rate $learning_rate \
        --t_norm $t_norm \
        --station_lr $station_lr \
        --use_mlp $use_mlp >logs/LightTrend/$model_name/elec_$pred_len.log
      done

    for pred_len in 96 192 336 720; do
      CUDA_VISIBLE_DEVICES=$gpu \
      python -u run.py \
        --is_training 1 \
        --use_norm $use_norm \
        --root_path ./datasets \
        --data_path weather.csv \
        --model_id $use_norm'_'weather_96_$pred_len \
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
        --itr 1 \
        --learning_rate $learning_rate \
        --t_norm $t_norm \
        --station_lr $station_lr \
        --use_mlp $use_mlp >logs/LightTrend/$model_name/wea_$pred_len.log
      done

    for pred_len in 96 192 336 720; do
      CUDA_VISIBLE_DEVICES=$gpu \
      python -u run.py \
        --is_training 1 \
        --use_norm $use_norm \
        --root_path ./dataset/ETT-small \
        --data_path ETTh1.csv \
        --model_id $use_norm'_'ETTh1_96_$pred_len \
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
        --itr 1 \
        --learning_rate $learning_rate \
        --t_norm $t_norm \
        --station_lr $station_lr \
        --use_mlp $use_mlp >logs/LightTrend/$model_name/ETTh1_$pred_len.log
      done

    for pred_len in 96 192 336 720; do
      CUDA_VISIBLE_DEVICES=$gpu \
      python -u run.py \
        --is_training 1 \
        --use_norm $use_norm \
        --root_path ./dataset/ETT-small \
        --data_path ETTh2.csv \
        --model_id $use_norm'_'ETTh2_96_$pred_len \
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
        --itr 1 \
        --learning_rate $learning_rate \
        --t_norm $t_norm \
        --station_lr $station_lr \
        --use_mlp $use_mlp >logs/LightTrend/$model_name/ETTh2_$pred_len.log
      done

    for pred_len in 96 192 336 720; do
      CUDA_VISIBLE_DEVICES=$gpu \
      python -u run.py \
        --is_training 1 \
        --use_norm $use_norm \
        --root_path ./dataset/ETT-small \
        --data_path ETTm1.csv \
        --model_id $use_norm'_'ETTm1_96_$pred_len \
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
        --itr 1 \
        --learning_rate $learning_rate \
        --t_norm $t_norm \
        --station_lr $station_lr \
        --use_mlp $use_mlp >logs/LightTrend/$model_name/ETTm1_$pred_len.log
      done

    for pred_len in 336 720 96 192; do
      CUDA_VISIBLE_DEVICES=$gpu \
      python -u run.py \
        --is_training 1 \
        --use_norm $use_norm \
        --root_path ./dataset/ETT-small \
        --data_path ETTm2.csv \
        --model_id $use_norm'_'ETTm2_96_$pred_len \
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
        --itr 1 \
        --learning_rate $learning_rate \
        --t_norm $t_norm \
        --station_lr $station_lr \
        --use_mlp $use_mlp >logs/LightTrend/$model_name/ETTm2_$pred_len.log
      done
    done
  done