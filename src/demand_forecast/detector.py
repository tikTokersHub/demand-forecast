import pandas as pd
from sklearn.ensemble import IsolationForest


class ForecastAnomalyDetector:
    def __init__(self, z_threshold=3.0, contamination=0.01):
        self.z_threshold = z_threshold
        self.contamination = contamination
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )

    def compute_residual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["item_id", "store_id", "date"])

        df["residual"] = df["actual"] - df["predicted"]
        df["abs_residual"] = df["residual"].abs()
        df["residual_pct"] = df["residual"] / df["predicted"].clip(lower=0.1)

        grp = df.groupby(["item_id", "store_id"])

        df["residual_rolling_mean"] = grp["residual"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )
        df["residual_rolling_std"] = grp["residual"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).std()
        )

        return df

    def detect_statistical(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["z_score"] = df["abs_residual"] / df["residual_rolling_std"].clip(lower=0.1)
        df["anomaly_stat"] = (df["z_score"] > self.z_threshold).astype(int)
        return df

    def detect_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        feature_cols = [
            "residual",
            "abs_residual",
            "residual_pct",
            "residual_rolling_mean",
            "residual_rolling_std",
        ]
        X = df[feature_cols].fillna(0)
        df["anomaly_iso"] = (self.iso_forest.fit_predict(X) == -1).astype(int)
        return df

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.compute_residual_features(df)
        df = self.detect_statistical(df)
        df = self.detect_isolation_forest(df)
        df["is_anomaly"] = ((df["anomaly_stat"] == 1) | (df["anomaly_iso"] == 1)).astype(int)
        return df