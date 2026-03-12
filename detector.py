#!/usr/bin/env python3
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import IsolationForest
from typing import Optional

logger = logging.getLogger(__name__)


class AnomalyDetector:

    def __init__(self, z_threshold: float = 3.0, contamination: float = 0.05):
        self.z_threshold = z_threshold
        self.contamination = contamination  # expected fraction of anomalies

    def zscore_flag(
        self,
        values: pd.Series,
        mean: float,
        std: float
    ) -> pd.Series:
        """
        Flag values more than z_threshold standard deviations from the
        established baseline mean. Returns a Series of z-scores.
        """
        if std == 0:
            return pd.Series([0.0] * len(values))
        return (values - mean).abs() / std

    def isolation_forest_flag(self, df: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
        """
        Multivariate anomaly detection across all numeric channels simultaneously.
        IsolationForest returns -1 for anomalies, 1 for normal points.
        Scores closer to -1 indicate stronger anomalies.
        """
        try:
            model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            X = df[numeric_cols].fillna(df[numeric_cols].median())
            model.fit(X)
            logger.debug(f"IsolationForest model trained on {len(X)} samples")

            labels = model.predict(X)          # -1 = anomaly, 1 = normal
            scores = model.decision_function(X)  # lower = more anomalous
            
            anomaly_count = (labels == -1).sum()
            logger.debug(f"IsolationForest detected {anomaly_count} anomalies")

            return labels, scores
        except Exception as e:
            logger.error(f"Error in isolation_forest_flag: {str(e)}", exc_info=True)
            raise

    def run(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        baseline: dict,
        method: str = "both"
    ) -> pd.DataFrame:
        try:
            result = df.copy()

            # --- Z-score per channel ---
            if method in ("zscore", "both"):
                for col in numeric_cols:
                    stats = baseline.get(col)
                    if stats and stats["count"] >= 30:  # need enough history to trust baseline
                        z_scores = self.zscore_flag(df[col], stats["mean"], stats["std"])
                        result[f"{col}_zscore"] = z_scores.round(4)
                        result[f"{col}_zscore_flag"] = z_scores > self.z_threshold
                    else:
                        # Not enough baseline history yet — flag as unknown
                        result[f"{col}_zscore"] = None
                        result[f"{col}_zscore_flag"] = None

            # --- IsolationForest across all channels ---
            if method in ("isolation", "both"):
                labels, scores = self.isolation_forest_flag(df, numeric_cols)
                result["if_label"] = labels          # -1 or 1
                result["if_score"] = scores.round(4) # continuous anomaly score
                result["if_flag"] = labels == -1

            # --- Consensus flag: anomalous by at least one method ---
            if method == "both":
                zscore_flags = [
                    result[f"{col}_zscore_flag"]
                    for col in numeric_cols
                    if f"{col}_zscore_flag" in result.columns
                    and result[f"{col}_zscore_flag"].notna().any()
                ]
                if zscore_flags:
                    any_zscore = pd.concat(zscore_flags, axis=1).any(axis=1)
                    result["anomaly"] = any_zscore | result["if_flag"]
                else:
                    result["anomaly"] = result["if_flag"]

            return result
        except Exception as e:
            logger.error(f"Error in detector.run: {str(e)}", exc_info=True)
            raise
