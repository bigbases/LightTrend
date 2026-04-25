if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LightTrend" ]; then
  mkdir ./logs/LightTrend
fi

if [ ! -d "./logs/LightTrend/DLinear" ]; then
  mkdir ./logs/LightTrend/DLinear
fi

gpu=0
features=M
model_name=DLinear
use_norm=lt
use_mlp=1

for learning_rate in 0.001 0.00005 0.0001 0.0005; do
  for station_lr in 0.0005 0.0001 0.00005 0.00001; do
    for t_ff in 64 128 256; do
        for pred_len in 96 192 336 720; do
          CUDA_VISIBLE_DEVICES=$gpu \
          python -u run_longExp.py \
            --is_training 1 \
            --use_norm $use_norm \
            --root_path ./datasets \
            --data_path electricity.csv \
            --model_id $use_norm'_'electricity_336_$pred_len$model_name \
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
            --learning_rate $learning_rate \
            --itr 1 \
            --t_ff $t_ff \
            --station_lr $station_lr \
            --use_mlp $use_mlp >logs/LightTrend/$model_name/elc_$pred_len.log
          done

        for pred_len in 96 192 336 720; do
          CUDA_VISIBLE_DEVICES=$gpu \
          python -u run_longExp.py \
            --is_training 1 \
            --use_norm $use_norm \
            --root_path ./datasets \
            --data_path weather.csv \
            --model_id $use_norm'_'weather_336_$pred_len$model_name \
            --model $model_name \
            --data custom \
            --features $features \
            --seq_len 336 \
            --label_len 168 \
            --pred_len $pred_len \
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
            --kernel_size 13 \
            --des 'Exp' \
            --itr 1 \
            --t_ff $t_ff \
            --learning_rate $learning_rate \
            --station_lr $station_lr \
            --use_mlp $use_mlp >logs/LightTrend/$model_name/wea_$pred_len.log
          done

        for pred_len in 96 192 336 720; do
          CUDA_VISIBLE_DEVICES=$gpu \
          python -u run_longExp.py \
            --is_training 1 \
            --use_norm $use_norm \
            --root_path ./datasets \
            --data_path traffic.csv \
            --model_id $use_norm'_'traffic_336_$pred_len$model_name \
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
            --learning_rate $learning_rate \
            --t_ff $t_ff \
            --station_lr $station_lr \
            --use_mlp $use_mlp >logs/LightTrend/$model_name/tra_$pred_len.log
          done

        # for pred_len in 96 192 336 720; do
        #   CUDA_VISIBLE_DEVICES=$gpu \
        #   python -u run_longExp.py \
        #     --is_training 1 \
        #     --use_norm $use_norm \
        #     --root_path ./datasets/ETT-small \
        #     --data_path ETTh1.csv \
        #     --model_id $use_norm'_'ETTh1_336_$pred_len$model_name \
        #     --model $model_name \
        #     --data ETTh1 \
        #     --features $features \
        #     --seq_len 336 \
        #     --label_len 168 \
        #     --pred_len $pred_len \
        #     --enc_in 7 \
        #     --dec_in 7 \
        #     --c_out 7 \
        #     --des 'Exp' \
        #     --t_ff $t_ff \
        #     --itr 1 \
        #     --station_lr $station_lr \
        #     --learning_rate $learning_rate \
        #     --use_mlp $use_mlp >logs/LightTrend/$model_name/eh1_$pred_len.log
        #   done

        # for pred_len in 96 192 336 720; do
        #   CUDA_VISIBLE_DEVICES=$gpu \
        #   python -u run_longExp.py \
        #     --is_training 1 \
        #     --use_norm $use_norm \
        #     --root_path ./datasets/ETT-small \
        #     --data_path ETTh2.csv \
        #     --model_id $use_norm'_'ETTh2_336_$pred_len$model_name \
        #     --model $model_name \
        #     --data ETTh2 \
        #     --features $features \
        #     --seq_len 336 \
        #     --label_len 168 \
        #     --pred_len $pred_len \
        #     --enc_in 7 \
        #     --dec_in 7 \
        #     --c_out 7 \
        #     --des 'Exp' \
        #     --itr 1 \
        #     --t_ff $t_ff \
        #     --station_lr $station_lr \
        #     --learning_rate $learning_rate \
        #     --use_mlp $use_mlp >logs/LightTrend/$model_name/eh2_$pred_len.log
        #   done

        # for pred_len in 96 192 336 720; do
        #   CUDA_VISIBLE_DEVICES=$gpu \
        #   python -u run_longExp.py \
        #     --is_training 1 \
        #     --use_norm $use_norm \
        #     --root_path ./datasets/ETT-small \
        #     --data_path ETTm1.csv \
        #     --model_id $use_norm'_'ETTm1_336_$pred_len$model_name \
        #     --model $model_name \
        #     --data ETTm1 \
        #     --features $features \
        #     --seq_len 336 \
        #     --label_len 168 \
        #     --pred_len $pred_len \
        #     --enc_in 7 \
        #     --dec_in 7 \
        #     --c_out 7 \
        #     --des 'Exp' \
        #     --kernel_size 13 \
        #     --itr 1 \
        #     --t_ff $t_ff \
        #     --station_lr $station_lr \
        #     --learning_rate $learning_rate \
        #     --use_mlp $use_mlp >logs/LightTrend/$model_name/em1_$pred_len.log
        #   done

        # for pred_len in 96 192 336 720; do
        #   CUDA_VISIBLE_DEVICES=$gpu \
        #   python -u run_longExp.py \
        #     --is_training 1 \
        #     --use_norm $use_norm \
        #     --root_path ./datasets/ETT-small \
        #     --data_path ETTm2.csv \
        #     --model_id $use_norm'_'ETTm2_336_$pred_len$model_name \
        #     --model $model_name \
        #     --data ETTm2 \
        #     --features $features \
        #     --seq_len 336 \
        #     --label_len 168 \
        #     --pred_len $pred_len \
        #     --enc_in 7 \
        #     --dec_in 7 \
        #     --c_out 7 \
        #     --des 'Exp' \
        #     --kernel_size 13 \
        #     --itr 1 \
        #     --t_ff $t_ff \
        #     --station_lr $station_lr \
        #     --learning_rate $learning_rate \
        #     --use_mlp $use_mlp >logs/LightTrend/$model_name/em2_$pred_len.log
        #   done
      done
    done
  done