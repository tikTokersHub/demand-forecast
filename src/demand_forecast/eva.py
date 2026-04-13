from __future__ import annotations

import polars as pl

def compute_wrmsse(
        actual_df:pl.DataFrame,
        pred_df:pl.DataFrame,
        train_df:pl.DataFrame
) -> float:
    
    required_train = {"item_id", "store_id", "date", "sales"}
    required_actual = {"item_id", "store_id", "date", "sales"}
    required_pred = {"item_id", "store_id", "date", "pred"}

    if not required_actual.issubset(set(actual_df.columns)):
        missing_actual = required_actual - set(actual_df.columns)
        raise ValueError(f"actual_df missing columns: {missing_actual}")

    if not required_train.issubset(set(train_df.columns)):
        missing_train = required_train - set(train_df.columens)
        raise ValueError(f"train+df missing columsn: {missing_train}")
    
    if not required_pred.issubset(set(pred_df.columns)):
        missing = required_pred - set(pred_df.columns)
        raise ValueError(f"pred_df missing columns: {missing}")
    
    
    series_keys = ['item_id', 'store_id']
    row_keys = ["item_id", "store_id", "date"]

    scales = (
        train_df
        .sort(row_keys)
        .with_columns(
            pl.col('sales')
            .diff()
            .over(series_keys)
            .alias("sales_diff")
        )
        .with_columns(
            (pl.col("sales_diff") ** 2)
            .alias("sq_diff")
        )
        .group_by(series_keys)
        .agg(
            pl.col("sq_diff").mean().alias("scale")
        )
        .with_columns(
            pl.when(
                pl.col('scale').is_null() | (pl.col('scale') <= 0)
            )
            .then(1e-8)
            .otherwise(pl.col("scale"))
            .alias("scale")
        )
    )

    eval_df = actual_df.join(
        pred_df,
        on=row_keys,
        how="inner",
    )

    if eval_df.height == 0:
        raise ValueError(
            "No matching rows between actual_df and pred_df on "
            "['item_id', 'store_id', 'date']."
        )
    
    errors = (
        eval_df
        .with_columns(
            ((pl.col("sales") - pl.col("pred")) ** 2).alias("sq_error")
        )
        .group_by(series_keys)
        .agg(
            pl.col("sq_error").mean().alias("mse")
        )
    )

    if "sell_price" in actual_df.columns:
        weights = (
            actual_df
            .with_columns(
                (pl.col("sales") * pl.col("sell_price")).alias("revenue")
            )
            .group_by(series_keys)
            .agg(
                pl.col("revenue").sum().alias("raw_weight")
            )
        )
    else:
        weights = (
            actual_df
            .group_by(series_keys)
            .agg(
                pl.col("sales").sum().alias("raw_weight")
            )
        )

    total_weight = weights.select(pl.col("raw_weight").sum()).item()

    if total_weight <= 0:
        raise ValueError("Total weight is zero; cannot compute WRMSSE.")

    weights = weights.with_columns(
        (pl.col("raw_weight") / total_weight).alias("weight")
    ).select(series_keys + ['weight'])

    per_series = (
        errors
        .join(scales, on=series_keys, how="inner")
        .join(weights, on=series_keys, how="inner")
        .with_columns(
            (pl.col("mse") / pl.col("scale")).sqrt().alias("rmsse")
        )
        .with_columns(
            (pl.col("rmsse") * pl.col("weight")).alias("weighted_rmsse")
        )
    )

    wrmsse = per_series.select(pl.col("weighted_rmsse").sum()).item()
    return float(wrmsse)