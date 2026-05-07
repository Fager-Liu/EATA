#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for pred_len in 24 36 48 60; do
  case "$pred_len" in
    24)
      dropout=0.05
      batch_size=32
      d_model=64
      learning_rate=0.05
      hidden=26
      ;;
    36)
      dropout=0.15
      batch_size=16
      d_model=64
      learning_rate=0.01
      hidden=26
      ;;
    48)
      dropout=0.25
      batch_size=12
      d_model=64
      learning_rate=0.005
      hidden=28
      ;;
    60)
      dropout=0.15
      batch_size=16
      d_model=64
      learning_rate=0.001
      hidden=32
      ;;
  esac

  python -u run.py \
    --model_id ili_36_${pred_len} \
    --model EATA \
    --data custom \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --features MS \
    --seq_len 36 \
    --label_len 18 \
    --pred_len ${pred_len} \
    --enc_in 7 \
    --d_model ${d_model} \
    --dropout ${dropout} \
    --batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --train_epochs 10 \
    --patience 3 \
    --k_lookback 36 \
    --method Dynamic \
    --hidden ${hidden} \
    --bias \
    --interact \
    --des EATA-MS
done
