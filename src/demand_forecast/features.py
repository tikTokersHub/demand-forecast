from pathlib import Path

import duckdb

PROCESSED_DIR = Path("data/processed")
INPUT_PATH = PROCESSED_DIR / "sales_long.parquet"
OUTPUT_PATH = PROCESSED_DIR / "feature_table.parquet"

FEATURE_SQL = f"""
COPY (
    SELECT 
        item_id,
        dept_id,
        cat_id,
        store_id,
        state_id,
        day,
        date,
        wm_yr_wk,
        sales,
        sell_price,
        event_name_1,
        event_type_1,
        event_name_2,
        event_type_2,
        snap_CA,
        snap_TX,
        snap_WI,

        LAG(sales, 7) OVER w AS sales_lag_7,
        LAG(sales, 28) OVER w AS sales_lag_28,

        AVG(sales) OVER (
            PARTITION BY item_id, store_id
            ORDER BY date
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) AS rolling_mean_7,

        AVG(sales) OVER (
            PARTITION BY item_id, store_id
            ORDER BY date
            ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
        ) as rolling_mean_28,

        STDDEV_SAMP(sales) OVER (
            PARTITION BY item_id, store_id
            ORDER BY date
            ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
        ) AS rolling_std_28,

        EXTRACT(DOW FROM date) AS day_of_week,
        EXTRACT(MONTH FROM date) AS month,
        EXTRACT(YEAR FROM date) AS year,

        sell_price / NULLIF(
            AVG(sell_price) OVER (
                PARTITION BY item_id, store_id
                ORDER BY date
                ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
            ),
            0
        ) AS price_ratio_28,

        CASE
            WHEN event_name_1 IS NOT NULL OR event_name_2 IS NOT NULL THEN 1
            ELSE 0
        END AS has_event

        FROM read_parquet('{INPUT_PATH.as_posix()}')
        WINDOW w AS (
            PARTITION BY item_id, store_id
            ORDER BY date
        )
    ) TO '{OUTPUT_PATH.as_posix()}';
"""

def build_feature_parquet():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input parquet: {INPUT_PATH}")
    
    con = duckdb.connect()
    try:
        print("Building feature table with DuckDB...")
        con.execute(FEATURE_SQL)
        print(f"Feature table written to: {OUTPUT_PATH}")
    finally:
        con.close()

def main():
    build_feature_parquet()


if __name__ == "__main__":
    main()