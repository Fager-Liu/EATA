# EATA-NET

Anonymous code release for `EATA-NET`.

## Scope

This repository keeps only the minimum code path required to run `EATA` for:

- long-term forecasting (`M`)
- exogenous-variable forecasting (`MS`)

Supported benchmark groups:

- `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`
- `Exchange`
- `ECL`
- `Weather`
- `ILI`
- `Futures`

## Repository Layout

```text
.
├── data_provider/
├── dataset/
├── exp/
├── layers/
├── models/
├── scripts/
│   ├── exogenous_forecast/
│   └── long_term_forecast/
├── utils/
├── run.py
└── requirements.txt
```

## Environment

```bash
pip install -r requirements.txt
```

## Data

Place datasets under `./dataset/` following the paths used in each script.

Examples:

- `./dataset/ETT-small/ETTh1.csv`
- `./dataset/exchange_rate/exchange_rate.csv`
- `./dataset/electricity/electricity.csv`
- `./dataset/weather/weather.csv`
- `./dataset/illness/national_illness.csv`
- `./dataset/futures/futures_TA.csv`

## Run

Long-term forecasting on futures:

```bash
bash ./scripts/long_term_forecast/Futures_script/EATA_TA.sh
```

Exogenous-variable forecasting on futures:

```bash
bash ./scripts/exogenous_forecast/Futures/EATA_TA.sh
```

Direct Python entry example:

```bash
python -u run.py \
  --model_id Futures_TA_96_96 \
  --model EATA \
  --data Futures \
  --root_path ./dataset/futures/ \
  --data_path futures_TA.csv \
  --target LastPrice \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 12 \
  --d_model 64 \
  --dropout 0.05 \
  --batch_size 32 \
  --learning_rate 0.005 \
  --train_epochs 20 \
  --patience 3 \
  --k_lookback 96 \
  --method Dynamic \
  --hidden 28 \
  --bias \
  --interact
```
