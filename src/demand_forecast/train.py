from pathlib import Path
from datetime import date
import json

import polars as pl
import pandas as pd
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

from eva import compute_wrmsse
from detector import ForecastAnomalyDetector


FEATURE_PATH = Path("data/processed/feature_table.parquet")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = date(2016, 3, 27)
VAL_START = date(2016, 3, 28)
VAL_END = date(2016, 4, 24)

EXPERIMENT_NAME = "demand-forecasting"
REGISTERED_MODEL_NAME = "demand-forecasting-lgbm"


def load_features(path: Path = FEATURE_PATH) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature path does not exist: {path}")
    return pl.read_parquet(path)


def split_data(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = df.filter(pl.col("date") <= pl.lit(TRAIN_END))
    val = df.filter(
        (pl.col("date") >= pl.lit(VAL_START)) & (pl.col("date") <= pl.lit(VAL_END))
    )
    return train, val


def build_prediction_frame(
    val_keys_df: pl.DataFrame,
    predictions,
    prediction_col_name: str = "predicted",
) -> pd.DataFrame:
    pred_df = val_keys_df.select(["item_id", "store_id", "date", "sales"]).to_pandas()
    pred_df = pred_df.rename(columns={"sales": "actual"})
    pred_df[prediction_col_name] = predictions
    return pred_df



def train_model():
    # -----------------------
    # MLflow setup
    # -----------------------
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="lgbm-v1-tweedie") as run:
        df = load_features(FEATURE_PATH)

        filtered = df.filter(
            (pl.col("state_id") == "CA") &
            (pl.col("store_id") == "CA_1")
        )

        top_items = (
            filtered.group_by("item_id")
            .agg(pl.len().alias("cnt"))
            .sort("cnt", descending=True)
            .get_column("item_id")
            .head(100)
            .to_list()
        )

        df = df.filter(
            (pl.col("state_id") == "CA") &
            (pl.col("store_id") == "CA_1") &
            (pl.col("item_id").is_in(top_items))
        )

        train_df, val_df = split_data(df)
        print(f"Train shape: {train_df.shape}")
        print(f"Val shape: {val_df.shape}")

        feature_cols = [
            "sell_price",
            "sales_lag_7",
            "sales_lag_28",
            "rolling_mean_7",
            "day_of_week",
        ]
        target_col = "sales"

        train_df = train_df.drop_nulls(subset=feature_cols + [target_col])
        val_df = val_df.drop_nulls(subset=feature_cols + [target_col])

        train_keys_df = train_df.select(
            ["item_id", "store_id", "date"] + feature_cols + [target_col]
        )
        val_keys_df = val_df.select(
            ["item_id", "store_id", "date"] + feature_cols + [target_col]
        )

        X_train = train_keys_df.select(feature_cols).to_pandas()
        y_train = train_keys_df[target_col].to_pandas()

        X_val = val_keys_df.select(feature_cols).to_pandas()
        y_val = val_keys_df[target_col].to_pandas()

        # -----------------------
        # Log dataset 
        # -----------------------
        mlflow.log_param("state_id", "CA")
        mlflow.log_param("store_id", "CA_1")
        mlflow.log_param("top_n_items", 100)
        mlflow.log_param("feature_path", str(FEATURE_PATH))
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("feature_cols", json.dumps(feature_cols))
        mlflow.log_param("train_end", str(TRAIN_END))
        mlflow.log_param("val_start", str(VAL_START))
        mlflow.log_param("val_end", str(VAL_END))

        mlflow.log_metric("train_rows", train_df.height)
        mlflow.log_metric("val_rows", val_df.height)
        mlflow.log_metric("n_features", len(feature_cols))

        # -----------------------
        # Random Forest
        # -----------------------
        rf_params = {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1,
        }

        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_val)

        rf_pred_df = val_keys_df.select(["item_id", "store_id", "date"]).with_columns(
            pl.Series("pred", rf_preds)
        )
        actual_df = val_keys_df.select(["item_id", "store_id", "date", "sales"])

        rf_wrmsse = compute_wrmsse(
            actual_df=actual_df,
            pred_df=rf_pred_df,
            train_df=train_df,
        )
        rf_mae = mean_absolute_error(y_val, rf_preds)

        print(f"Train rows: {train_df.height}")
        print(f"Val rows: {val_df.height}")
        print("RandomForest results")
        print(f"Validation MAE: {rf_mae:.4f}")
        print(f"WRMSSE: {rf_wrmsse:.5f}")

        mlflow.log_params({f"rf_{k}": v for k, v in rf_params.items()})
        mlflow.log_metric("rf_val_mae", rf_mae)
        mlflow.log_metric("rf_wrmsse", rf_wrmsse)

        rf_signature = infer_signature(X_train, rf_model.predict(X_train.head(5)))
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            artifact_path="rf_model",
            signature=rf_signature,
            input_example=X_train.head(5),
        )

        # -----------------------
        # LightGBM
        # -----------------------
        print("lgb results")
        lgb_params = {
            "objective": "tweedie",
            "tweedie_variance_power": 1.1,
            "learning_rate": 0.05,
            "num_leaves": 127,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 1000,
            "verbosity": -1,
            "random_state": 42,
        }

        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        lgb_preds = lgb_model.predict(X_val)
        lgb_mae = mean_absolute_error(y_val, lgb_preds)

        lgb_pred_df = val_keys_df.select(["item_id", "store_id", "date"]).with_columns(
            pl.Series("pred", lgb_preds)
        )

        lgb_wrmsse = compute_wrmsse(
            actual_df=actual_df,
            pred_df=lgb_pred_df,
            train_df=train_df,
        )

        print(f"Validation MAE: {lgb_mae:.4f}")
        print(f"WRMSSE: {lgb_wrmsse:.5f}")

        mlflow.log_params({f"lgb_{k}": v for k, v in lgb_params.items()})
        mlflow.log_metric("lgb_val_mae", lgb_mae)
        mlflow.log_metric("lgb_wrmsse", lgb_wrmsse)

        # -----------------------
        # Log best model
        # -----------------------
        best_model_name = "lightgbm" if lgb_mae <= rf_mae else "random_forest"
        mlflow.log_param("best_model_by_mae", best_model_name)

        if best_model_name == "lightgbm":
            predictions_df = build_prediction_frame(val_keys_df, lgb_preds)
            signature = infer_signature(X_train, lgb_model.predict(X_train.head(5)))
            model_info = mlflow.lightgbm.log_model(
                lgb_model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(5),
            )

            mlflow.register_model(
                model_uri=model_info.model_uri,
                name=REGISTERED_MODEL_NAME,
            )
        else:
            predictions_df = build_prediction_frame(val_keys_df, rf_preds)
            signature = infer_signature(X_train, rf_model.predict(X_train.head(5)))
            mlflow.sklearn.log_model(
                sk_model=rf_model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(5),
            )

        predictions_path = REPORTS_DIR / "val_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        return str(predictions_path)

if __name__ == "__main__":
    train_model()