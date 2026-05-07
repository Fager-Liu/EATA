#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -u run.py \
  --model_id ETTm2_96_96 \
  --model EATA \
  --data ETTm2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --d_model 512 \
  --dropout 0.05 \
  --batch_size 16 \
  --learning_rate 0.00035 \
  --train_epochs 3 \
  --patience 3 \
  --k_lookback 8 \
  --des EATA-MS
