from dataclasses import dataclass
from pathlib import Path

import duckdb

@dataclass
class ValidResult:
    name: str
    passed: bool
    detail: str


def validate(feature_path: str) -> list[ValidResult]:
    con = duckdb.connect()
    results: list[ValidResult] = []
    path = Path(feature_path)

    if not path.exists():
        raise FileNotFoundError(f"Feature table not found {path}")
    
    table = f"read_parquet('{path}')"

    def run_check(name:str, sql:str):
        val = con.execute(sql).fetchone()[0]
        passed = bool(val)
        results.append(ValidResult(name=name, passed=passed, detail=str(val)))

    
    run_check(
        "row_count >= 100000",
        f"SELECT COUNT(*) >= 100000 FROM {table}",
    )

    required_cols = [
        "item_id",
        "store_id",
        "date",
        "sales",
        "sell_price",
        "sales_lag_7",
        "sales_lag_28",
        "rolling_mean_7",
        "day_of_week",
    ]

    try:
        col_check = ", ".join(f'"{c}"' for c in required_cols)
        con.execute(f"SELECT {col_check} FROM {table} LIMIT 1")
        results.append(ValidResult("required_columns_exist", True, str(required_cols)))
    except duckdb.BinderException as e:
        results.append(ValidResult("required_columns_exist", False, str(e)))

    for col in ["item_id", "store_id", "date"]:
        run_check(
            f"{col}_no_nulls",
            f'SELECT COUNT(*) = 0 FROM {table} WHERE "{col}" IS NULL',
        )

    run_check(
        "sales_between_0_and_1000",
        f"SELECT MIN(sales) >= 0 AND MAX(sales) <= 1000 FROM {table}",
    ) 
    
    run_check(
        "sell_price_non_negative",
        f"SELECT COUNT(*) = 0 FROM {table} WHERE sell_price IS NOT NULL AND sell_price < 0",
    )

    run_check(
        "lag_features_have_nulls_for_early_rows",
        f"SELECT COUNT(*) > 0 FROM {table} WHERE sales_lag_7 IS NULL",
    )

    run_check(
        "date_range_reasonable",
        f"""
        SELECT MIN(date) >= DATE '2011-01-01'
           AND MAX(date) <= DATE '2016-06-30'
        FROM {table}
        """,
    )

    print("\n=== Feature Table Validation Report ===")
    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {r.detail}")
        if not r.passed:
            all_passed = False

    print(f"\n{'All checks passed.' if all_passed else 'SOME CHECKS FAILED.'}\n")

    if not all_passed:
        failed = [r.name for r in results if not r.passed]
        raise ValueError(f"Validation failed: {failed}")

    return results


if __name__ == "__main__":
    feature_path = "data/processed/feature_table.parquet"
    validate(feature_path)