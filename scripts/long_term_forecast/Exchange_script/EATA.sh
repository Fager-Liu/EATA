#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for pred_len in 96 192 336 720; do
  case "$pred_len" in
    96)
      d_model=256
      dropout=0.1
      batch_size=16
      learning_rate=0.01
      k_lookback=96
      hidden=24
      ;;
    192)
      d_model=128
      dropout=0.25
      batch_size=16
      learning_rate=0.003
      k_lookback=32
      hidden=32
      ;;
    336)
      d_model=256
      dropout=0.15
      batch_size=16
      learning_rate=0.03
      k_lookback=96
      hidden=64
      ;;
    720)
      d_model=32
      dropout=0.15
      batch_size=16
      learning_rate=0.0001
      k_lookback=64
      hidden=28
      ;;
  esac

  python -u run.py \
    --model_id Exchange_96_${pred_len} \
    --model EATA \
    --data custom \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --enc_in 8 \
    --d_model ${d_model} \
    --dropout ${dropout} \
    --batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --train_epochs 20 \
    --patience 3 \
    --k_lookback ${k_lookback} \
    --method Dynamic \
    --hidden ${hidden} \
    --bias \
    --interact
done
