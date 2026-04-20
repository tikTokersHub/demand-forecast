from pathlib import Path


import polars as pl

RAW_DIR  = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data():
    sales_path = RAW_DIR / "sales_train_validation.csv"
    cal_path = RAW_DIR / "calendar.csv"
    prices_path = RAW_DIR / "sell_prices.csv"

    sales_df = pl.read_csv(sales_path)
    cal_df = pl.read_csv(cal_path)
    prices_df= pl.read_csv(prices_path)

    return sales_df, cal_df, prices_df

def melt_sales_to_long(sales_df:pl.DataFrame) -> pl.DataFrame:
    id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_df.columns if c.startswith("d_")]

    sales_long_df = sales_df.unpivot(
        index=id_cols,
        on=day_cols,
        variable_name="day",
        value_name="sales"
    )

    return sales_long_df.with_columns(
        pl.col("sales").cast(pl.Int16)
    )

def prepare_calendar(cal_df: pl.DataFrame) -> pl.DataFrame:
    return cal_df.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d")
    )

def join_sales_cal_prices(
        sales_df:pl.DataFrame,
        cal_df:pl.DataFrame,
        prices_df:pl.DataFrame
) -> pl.DataFrame:
    sales_df = sales_df.join(
        cal_df,
        left_on="day",
        right_on="d",
        how='left'
    )

    sales_df = sales_df.join(
        prices_df,
        on=["store_id", "item_id", "wm_yr_wk"],
        how='left',
    )

    return sales_df


def write_output(
        sales_df:pl.DataFrame,
        cal_df:pl.DataFrame,
        prices_df:pl.DataFrame
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    sales_df.write_parquet(PROCESSED_DIR / "sales_long.parquet")
    cal_df.write_parquet(PROCESSED_DIR / "calendar.parquet")
    prices_df.write_parquet(PROCESSED_DIR / "prices.parquet")

def run_ingest() -> dict[str, int]:
    print("Loading raw CSV files...")
    sales_df, cal_df, prices_df = load_raw_data()

    print("Preparing calendar...")
    cal_df = prepare_calendar(cal_df)

    print("Melting sales to long format...")
    sales_long = melt_sales_to_long(sales_df)

    print("Joining calendar and prices...")
    joined_df = join_sales_cal_prices(sales_long, cal_df, prices_df)

    print("Writing parquet files...")
    write_output(joined_df, cal_df, prices_df)

    summary = {
        "sales_long_rows": sales_long.height,
        "joined_rows": joined_df.height,
        "calendar_rows": cal_df.height,
        "prices_rows": prices_df.height,
    }

    print("Done.")
    print(f"sales_long rows: {sales_long.height:,}")
    print(f"joined rows:     {joined_df.height:,}")
    print(f"calendar rows:   {cal_df.height:,}")
    print(f"prices rows:     {prices_df.height:,}")

    return summary

def main() -> None:
    run_ingest()
    
if __name__ == "__main__":
    main()

