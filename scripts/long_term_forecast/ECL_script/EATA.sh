#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -u run.py \
  --model_id ECL_96_96 \
  --model EATA \
  --data custom \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 321 \
  --d_model 64 \
  --dropout 0.1 \
  --batch_size 4 \
  --learning_rate 0.0005 \
  --train_epochs 20 \
  --patience 10 \
  --k_lookback 8 \
  --method Dynamic \
  --hidden 10 \
  --bias \
  --interact
