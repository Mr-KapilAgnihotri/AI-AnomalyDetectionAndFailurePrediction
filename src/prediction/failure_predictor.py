import numpy as np
from src.config.config import (
    RISK_LOW_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
    RISK_HIGH_THRESHOLD,
    RISK_WINDOW
)

class FailurePredictor:
    """
    Converts anomaly scores into failure risk intelligence.
    """

    def __init__(self):
        pass

    def compute_risk(self, anomaly_scores: np.ndarray) -> dict:
        """
        Analyze anomaly score trends and return failure risk.
        """

        if len(anomaly_scores) < RISK_WINDOW:
            return {
                "failure_risk": "LOW",
                "predicted_failure_window": None
            }

        #1 Focus on recent behavior
        recent_scores = anomaly_scores[-RISK_WINDOW:]

        #2 Compute statistics
        avg_score = np.mean(recent_scores)
        trend = recent_scores[-1] - recent_scores[0]
        high_severity_ratio = np.mean(recent_scores > RISK_HIGH_THRESHOLD)

        #3 Determine risk level
        if avg_score > RISK_HIGH_THRESHOLD and trend > 0:
            risk = "HIGH"
        elif avg_score > RISK_MEDIUM_THRESHOLD:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        #4 Estimate failure window
        failure_window = self._estimate_failure_window(risk)

        return {
            "failure_risk": risk,
            "predicted_failure_window": failure_window,
            "avg_anomaly_score": round(avg_score, 3),
            "trend": round(trend, 3),
            "high_severity_ratio": round(high_severity_ratio, 3)
        }

    def _estimate_failure_window(self, risk: str):
        """
        Map risk level to readable time window.
        """

        if risk == "HIGH":
            return "2-4 hours"
        elif risk == "MEDIUM":
            return "6-12 hours"
        else:
            return None
