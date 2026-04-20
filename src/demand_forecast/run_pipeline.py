import pandas as pd
from pathlib import Path
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=24))
def ingest_data():
    """Run data ingestion pipeline."""
    from demand_forecast.ingest import run_ingest
    return run_ingest()

@task
def build_features():
    """Build feature table from raw parquet."""
    from demand_forecast.features import build_feature_parquet
    return build_feature_parquet()

@task
def validate_data():
    """Run DuckDB validation checks. Raises ValueError if checks fail."""
    from demand_forecast.validate import validate
    return validate()  # raises on failure

@task
def train_model():
    """Train LightGBM and log to MLflow."""
    from demand_forecast.train import train_model
    return train_model()

@task
def detect_anomalies(predictions_path):
    """Run anomaly detection on latest predictions."""
    from demand_forecast.detector import ForecastAnomalyDetector
    predictions_path = Path(predictions_path)
    df = pd.read_csv(predictions_path)

    detector = ForecastAnomalyDetector()
    anomaly_df = detector.detect(df)

    output_path = Path("reports/anomalies.csv")
    anomaly_df.to_csv(output_path, index=False)
    return str(output_path)



@flow(name="demand-forecasting-pipeline")
def training_pipeline():
    raw = ingest_data()
    features = build_features()
    validated = validate_data()
    model = train_model()
    anomalies = detect_anomalies(model)
    return anomalies


if __name__ == "__main__":
    training_pipeline()