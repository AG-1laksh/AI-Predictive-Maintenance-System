# Predictive Maintenance (NASA Turbofan) + Kafka + Spark

Beginner-friendly project scaffold for:
1. Load raw NASA turbofan txt data
2. Clean and label data
3. Train RandomForest classifier
4. Save model and run sample predictions
5. Prepare starter Kafka/Spark integration files for later

## Evaluation and correctness notes

- **No data leakage policy:** training uses only `train_FDxxx.txt`; test evaluation uses only `test_FDxxx.txt` + matching `RUL_FDxxx.txt`.
- **RUL threshold rationale:** `failure = 1 if RUL < 30 else 0` is an early-warning window (30 cycles) for maintenance planning.
- **Beyond accuracy:** pipeline now prints precision, recall, F1-score, and confusion matrix for proper imbalance-aware evaluation.

## Project structure

- `run_pipeline.py` — runs full local ML flow (steps 1 to 9) for `FD001` to `FD004`
- `src/data_pipeline.py` — load, clean, add column names, create RUL + failure label
- `src/train_model.py` — train `RandomForestClassifier`, print accuracy, save model
- `src/predict_sample.py` — load saved model and show sample predictions
- `src/kafka/producer_stub.py` — starter producer to stream CSV rows to Kafka
- `src/spark/streaming_stub.py` — starter Spark streaming consumer for Kafka topic
- `.env` — Kafka connection placeholders

## Quick start

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run full pipeline:
   - `python run_pipeline.py`
3. Run single combined model (FD001+FD002+FD003+FD004):
   - `python run_combined_pipeline.py`
4. Launch interactive dashboard:
   - `streamlit run dashboard_app.py`

## What are sensor_1, sensor_2, ...?

NASA CMAPSS raw row format is:
1) `unit`
2) `cycle`
3-5) operational settings
6-26) sensor measurements

In this project we kept a simple naming rule after `unit` and `cycle`:
- `sensor_1` to `sensor_24`

So practically:
- `sensor_1`, `sensor_2`, `sensor_3` = operational settings
- `sensor_4` to `sensor_24` = actual sensor measurements

## Outputs generated

- `outputs/train_FD001_clean.csv`, `outputs/train_FD002_clean.csv`, `outputs/train_FD003_clean.csv`, `outputs/train_FD004_clean.csv`
- `outputs/test_FD001_clean.csv`, `outputs/test_FD002_clean.csv`, `outputs/test_FD003_clean.csv`, `outputs/test_FD004_clean.csv`
- `outputs/x_clean.csv`
- `models/rf_FD001_model.pkl`, `models/rf_FD002_model.pkl`, `models/rf_FD003_model.pkl`, `models/rf_FD004_model.pkl`

## Dashboard

- File: `dashboard_app.py`
- Reads prediction files from `outputs/predictions/`
- Reads overfitting metrics from `outputs/overfitting_metrics.csv`

Dashboard includes:
- Dataset/split selector (`train`/`test`)
- KPI cards (rows, accuracy, precision, recall)
- Accuracy comparison across datasets
- Confusion components (TP, TN, FP, FN)
- Overfitting train vs test charts
- Unit-level timeline charts (`RUL`, sensor trend, predicted failure signal)

## Notes

- The pipeline now uses all provided NASA files: `train_FD001-004.txt`, `test_FD001-004.txt`, `RUL_FD001-004.txt`, and `x.txt`.
- Kafka/Spark scripts are optional now and ready for your next phase.

## Kafka + Spark (next phase)

1) Start Kafka first (use your own installation, or Docker if already configured).

2) Send rows to Kafka topic:
- `python -m src.kafka.producer_stub`

3) Read Kafka stream using Spark:
- `python -m src.spark.streaming_stub`

If Spark cannot find Kafka connector in your setup, run Spark with Kafka package support (version depends on Spark install).

### Current streaming inference mode

- Current Spark stream computes `predicted_failure` using the same business rule on incoming `RUL` (`RUL < 30`).
- This is a **streaming rule-based inference simulation** suitable for demo.
- Future upgrade: load and apply the saved ML model directly inside Spark micro-batches/UDF for fully model-based online inference.
