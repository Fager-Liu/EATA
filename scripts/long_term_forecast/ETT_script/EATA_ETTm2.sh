#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for pred_len in 96 192 336 720; do
  case "$pred_len" in
    96)
      d_model=64
      dropout=0.15
      batch_size=32
      learning_rate=0.05
      hidden=28
      ;;
    192)
      d_model=64
      dropout=0.05
      batch_size=32
      learning_rate=0.05
      hidden=26
      ;;
    336)
      d_model=64
      dropout=0.15
      batch_size=32
      learning_rate=0.05
      hidden=28
      ;;
    720)
      d_model=64
      dropout=0.15
      batch_size=32
      learning_rate=0.05
      hidden=28
      ;;
  esac

  python -u run.py \
    --model_id ETTm2_96_${pred_len} \
    --model EATA \
    --data ETTm2 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --enc_in 7 \
    --d_model ${d_model} \
    --dropout ${dropout} \
    --batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --train_epochs 20 \
    --patience 3 \
    --k_lookback 96 \
    --method Dynamic \
    --hidden ${hidden} \
    --bias \
    --interact
done
