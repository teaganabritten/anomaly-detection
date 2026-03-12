#!/usr/bin/env python3
import json
import math
import boto3
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

s3 = boto3.client("s3")
logger = logging.getLogger(__name__)


class BaselineManager:
    """
    Maintains a per-channel running baseline using Welford's online algorithm,
    which computes mean and variance incrementally without storing all past data.
    """

    def __init__(self, bucket: str, baseline_key: str = "state/baseline.json"):
        self.bucket = bucket
        self.baseline_key = baseline_key

    def load(self) -> dict:
        try:
            response = s3.get_object(Bucket=self.bucket, Key=self.baseline_key)
            return json.loads(response["Body"].read())
        except s3.exceptions.NoSuchKey:
            return {}

    def save(self, baseline: dict):
        try:
            baseline["last_updated"] = datetime.utcnow().isoformat()
            s3.put_object(
                Bucket=self.bucket,
                Key=self.baseline_key,
                Body=json.dumps(baseline, indent=2),
                ContentType="application/json"
            )
            logger.info(f"Baseline updated and saved to s3://{self.bucket}/{self.baseline_key}")
            
            # Sync logs to S3 after baseline is saved
            log_dir = Path(__file__).parent / "submit"
            log_file = log_dir / "app.log"
            if log_file.exists():
                try:
                    s3.upload_file(
                        str(log_file),
                        self.bucket,
                        "logs/app.log",
                        ExtraArgs={"ContentType": "text/plain"}
                    )
                    logger.info(f"Logs synced to s3://{self.bucket}/logs/app.log")
                except Exception as e:
                    logger.error(f"Failed to sync logs: {str(e)}")
        except Exception as e:
            logger.error(f"Error saving baseline: {str(e)}", exc_info=True)
            raise

    def update(self, baseline: dict, channel: str, new_values: list[float]) -> dict:
        """
        Welford's online algorithm for numerically stable mean and variance.
        Each channel tracks: count, mean, M2 (sum of squared deviations).
        Variance = M2 / count, std = sqrt(variance).
        """
        try:
            if channel not in baseline:
                baseline[channel] = {"count": 0, "mean": 0.0, "M2": 0.0}

            state = baseline[channel]
            prev_count = state["count"]

            for value in new_values:
                state["count"] += 1
                delta = value - state["mean"]
                state["mean"] += delta / state["count"]
                delta2 = value - state["mean"]
                state["M2"] += delta * delta2

            # Only compute std once we have enough observations
            if state["count"] >= 2:
                variance = state["M2"] / state["count"]
                state["std"] = math.sqrt(variance)
            else:
                state["std"] = 0.0

            baseline[channel] = state
            logger.debug(f"Updated {channel}: count {prev_count} -> {state['count']}, mean={state['mean']:.4f}, std={state['std']:.4f}")
            return baseline
        except Exception as e:
            logger.error(f"Error updating baseline for {channel}: {str(e)}", exc_info=True)
            raise

    def get_stats(self, baseline: dict, channel: str) -> Optional[dict]:
        return baseline.get(channel)
