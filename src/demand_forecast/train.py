from pathlib import Path
from datetime import date

import polars as pl

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

from eva import compute_wrmsse


FEATURE_PATH = Path("data/processed/feature_table.parquet")

TRAIN_END = date(2016, 3, 27)
VAL_START = date(2016, 3, 28)
VAL_END = date(2016, 4, 24)


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


def main():
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

    # Keep Polars frames with keys for evaluation
    train_keys_df = train_df.select(["item_id", "store_id", "date"] + feature_cols + [target_col])
    val_keys_df = val_df.select(["item_id", "store_id", "date"] + feature_cols + [target_col])

    # Convert only model features/target to pandas/numpy for sklearn / lightgbm
    X_train = train_keys_df.select(feature_cols).to_pandas()
    y_train = train_keys_df[target_col].to_pandas()

    X_val = val_keys_df.select(feature_cols).to_pandas()
    y_val = val_keys_df[target_col].to_pandas()

    # -----------------------
    # Random Forest
    # -----------------------
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    preds = rf_model.predict(X_val)

    pred_df = val_keys_df.select(["item_id", "store_id", "date"]).with_columns(
        pl.Series("pred", preds)
    )

    actual_df = val_keys_df.select(["item_id", "store_id", "date", "sales"])

    rf_wrmsse = compute_wrmsse(
        actual_df=actual_df,
        pred_df=pred_df,
        train_df=train_df,
    )

    rf_mae = mean_absolute_error(y_val, preds)

    print(f"Train rows: {train_df.height}")
    print(f"Val rows: {val_df.height}")

    print("RandomForest results")
    print(f"Validation MAE: {rf_mae:.4f}")
    print(f"WRMSSE: {rf_wrmsse:.5f}")

    # -----------------------
    # LightGBM
    # -----------------------
    print("lgb results")
    default_params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 1000,
        "verbosity": -1,
    }

    lgb_model = lgb.LGBMRegressor(**default_params)
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


if __name__ == "__main__":
    main()