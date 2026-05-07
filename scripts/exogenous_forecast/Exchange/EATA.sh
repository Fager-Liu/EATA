#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -u run.py \
  --model_id Exchange_96_96 \
  --model EATA \
  --data custom \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 8 \
  --d_model 128 \
  --dropout 0.05 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --train_epochs 20 \
  --patience 3 \
  --k_lookback 4 \
  --des EATA-MS
