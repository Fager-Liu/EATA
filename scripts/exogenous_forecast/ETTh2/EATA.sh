#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -u run.py \
  --model_id ETTh2_96_96 \
  --model EATA \
  --data ETTh2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --d_model 512 \
  --dropout 0.05 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --patience 3 \
  --k_lookback 96 \
  --des EATA-MS
