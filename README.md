# Demand Forecast

Retail demand forecasting and anomaly detection for M5-style sales data. The
project ingests raw sales, calendar, and price CSVs; builds a feature table with
DuckDB; validates the processed dataset; compares RandomForest, LightGBM, and
Temporal Fusion Transformer forecasting approaches; tracks experiments with
MLflow; and produces prediction and anomaly reports that can be served through a
FastAPI endpoint and viewed in a Streamlit dashboard.

## Pipeline Overview

1. Ingest raw CSV files from `data/raw/`.
2. Convert wide daily sales columns into a long time-series table.
3. Join sales with calendar and sell-price data.
4. Build lag, rolling-window, calendar, price, and event features with DuckDB.
5. Validate the processed feature table before training.
6. Train RandomForest and LightGBM regressors on a CA_1 top-item subset.
7. Track model parameters, validation metrics, and model artifacts with MLflow.
8. Compare models with MAE and WRMSSE, then write validation predictions.
9. Detect forecast anomalies from prediction residuals.

## Modeling Experiments

The main training workflow benchmarks RandomForest and LightGBM regressors using
lag, rolling-window, price, calendar, and event features. Both runs are logged to
MLflow with their parameters and validation metrics, including MAE and WRMSSE.
The best-performing tree model is persisted as an MLflow model artifact.

The project also includes a Temporal Fusion Transformer (TFT) experiment in
`src/demand_forecast/tft_model.py`. The TFT workflow is designed for deeper
sequence modeling over grouped retail time series and logs its training
configuration and best validation loss to MLflow, with TensorBoard outputs saved
under `tb_logs/`.

## Repository Layout

```text
.
|-- src/demand_forecast/
|   |-- ingest.py          # Raw CSV loading, reshaping, and parquet output
|   |-- features.py        # DuckDB feature engineering
|   |-- validate.py        # Data quality checks for the feature table
|   |-- train.py           # RandomForest and LightGBM training with MLflow
|   |-- eva.py             # WRMSSE evaluation helper
|   |-- detector.py        # Residual and IsolationForest anomaly detection
|   |-- run_pipeline.py    # Prefect flow tying ingestion through anomaly output
|   |-- inference.py       # Forecast feature construction for serving
|   |-- api.py             # FastAPI forecast endpoint
|   `-- tft_model.py       # TFT deep learning experiment with MLflow tracking
|-- ui.py                  # Streamlit dashboard for forecast visualization
|-- pyproject.toml         # Package metadata and Python dependencies
|-- Dockerfile             # Training and serving image stages
|-- docker-compose.yml     # MLflow and training services
`-- Makefile               # Common local commands
```

## Data

Place the raw M5-style files in `data/raw/`:

```text
data/raw/calendar.csv
data/raw/sell_prices.csv
data/raw/sales_train_validation.csv
```

The pipeline writes generated artifacts locally:

```text
data/processed/sales_long.parquet
data/processed/calendar.parquet
data/processed/prices.parquet
data/processed/feature_table.parquet
reports/val_predictions.csv
reports/anomalies.csv
```

Raw data, processed parquet files, reports, MLflow runs, logs, virtual
environments, and local-only tool settings are intentionally ignored by Git.

## Setup

This project targets Python 3.11 or newer.

```bash
uv venv
uv pip install -e ".[dev]"
```

If you are not using `uv`, install with pip instead:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The dashboard uses packages that are not currently listed in `pyproject.toml`:

```bash
uv pip install streamlit plotly httpx
```

For the TFT experiment, install the deep learning dependencies:

```bash
make install-dl
```

## Run The Pipeline

After installing the package in editable mode, run each stage directly:

```bash
python -m demand_forecast.ingest
python -m demand_forecast.features
python -m demand_forecast.validate
python -m demand_forecast.train
```

Or run the full Prefect flow:

```bash
python -m demand_forecast.run_pipeline
```

The end-to-end flow writes validation predictions to
`reports/val_predictions.csv` and anomaly results to `reports/anomalies.csv`.

## Serve Forecasts

The FastAPI app loads an MLflow model named `demand-forecasting-lgbm`, version
`3`, and uses `data/processed/sales_long.parquet` to build future forecast
features.

```bash
uvicorn demand_forecast.api:app --reload --port 8000
```

Useful endpoints:

```text
GET /health
GET /forecast?item=FOODS_3_090&store=CA_1&horizon=28
```

## Dashboard

Start the API first, then run the Streamlit dashboard:

```bash
streamlit run ui.py --server.port 8501
```

The dashboard calls `http://localhost:8000/forecast`, plots predicted sales, and
highlights forecast anomaly flags returned by the API.

## MLflow And Docker

Training logs model parameters, metrics, and artifacts to local `mlruns/` by
default. RandomForest, LightGBM, and TFT experiments are all tracked through
MLflow, while the Docker Compose file also defines an MLflow service on port
`5000`:

```bash
docker compose up -d mlflow
```

Build and run the training container with:

```bash
docker compose build
docker compose up train
```

## Quality Checks

Linting and tests are wired through the Makefile and CI workflow:

```bash
ruff check src/ tests/
pytest tests/ -v
```

The current CI workflow installs the package, runs Ruff, runs pytest, and starts
a training smoke test with `python -m demand_forecast.run_pipeline`.
