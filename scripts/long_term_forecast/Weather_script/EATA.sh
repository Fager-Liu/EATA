#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -u run.py \
  --model_id weather_96_96 \
  --model EATA \
  --data custom \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 21 \
  --d_model 128 \
  --dropout 0.2 \
  --batch_size 4 \
  --learning_rate 0.002 \
  --train_epochs 15 \
  --patience 3 \
  --k_lookback 48 \
  --method Dynamic \
  --hidden 24 \
  --bias \
  --interact
