#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -u run.py \
  --model_id ECL_96_720 \
  --model EATA \
  --data custom \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 321 \
  --d_model 512 \
  --dropout 0.05 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --train_epochs 10 \
  --patience 3 \
  --k_lookback 4 \
  --des EATA-MS
