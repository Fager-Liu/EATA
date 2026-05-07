#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -u run.py \
  --model_id ETTh1_96_336 \
  --model EATA \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 7 \
  --d_model 64 \
  --dropout 0.2 \
  --batch_size 32 \
  --learning_rate 0.03 \
  --train_epochs 20 \
  --patience 3 \
  --k_lookback 96 \
  --method Dynamic \
  --hidden 32 \
  --bias \
  --interact \
  --des EATA-MS
