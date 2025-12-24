import numpy as np
from sklearn.ensemble import IsolationForest

from src.config.config import (
    IF_N_ESTIMATORS,
    IF_CONTAMINATION,
    RANDOM_SEED
)

class IsolationForestModel:
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            contamination=IF_CONTAMINATION,
            random_state=RANDOM_SEED
        )

    def fit(self, X_train: np.ndarray):
        """
        Train Isolation Forest on normal data only.
        """
        self.model.fit(X_train)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return normalized anomaly scores in range [0, 1].
        Higher score = more anomalous.
        """
        raw_scores = -self.model.score_samples(X)
        normalized_scores = (raw_scores - raw_scores.min()) / (
            raw_scores.max() - raw_scores.min()
        )
        return normalized_scores
