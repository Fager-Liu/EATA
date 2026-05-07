#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -u run.py \
  --model_id ETTm1_96_96 \
  --model EATA \
  --data ETTm1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --d_model 256 \
  --dropout 0.3 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 20 \
  --patience 3 \
  --k_lookback 8 \
  --des EATA-MS
