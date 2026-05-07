#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for pred_len in 96 192 336 720; do
  case "$pred_len" in
    96)
      dropout=0.2
      batch_size=32
      d_model=64
      learning_rate=0.005
      ;;
    192)
      dropout=0.05
      batch_size=32
      d_model=64
      learning_rate=0.00075
      ;;
    336)
      dropout=0.05
      batch_size=32
      d_model=64
      learning_rate=0.001
      ;;
    720)
      dropout=0.05
      batch_size=32
      d_model=64
      learning_rate=0.001
      ;;
  esac

  python -u run.py \
    --model_id Futures_TA_96_${pred_len} \
    --model EATA \
    --data Futures \
    --root_path ./dataset/futures/ \
    --data_path futures_TA.csv \
    --target LastPrice \
    --features MS \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --enc_in 146 \
    --d_model ${d_model} \
    --dropout ${dropout} \
    --batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --train_epochs 20 \
    --patience 3 \
    --k_lookback 64 \
    --method Dynamic \
    --hidden 28 \
    --bias \
    --interact \
    --des EATA-MS
done
