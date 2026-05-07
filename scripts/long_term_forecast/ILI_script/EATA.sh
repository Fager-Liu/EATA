#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -u run.py \
  --model_id ili_36_48 \
  --model EATA \
  --data custom \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --enc_in 7 \
  --d_model 64 \
  --dropout 0.25 \
  --batch_size 12 \
  --learning_rate 0.005 \
  --train_epochs 10 \
  --patience 3 \
  --k_lookback 36 \
  --method Dynamic \
  --hidden 28 \
  --bias \
  --interact
