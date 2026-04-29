from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from ..base.base_estimator import BaseEstimator
from ..base.mixins import ClassifierMixin
from ..metrics import Cosine, DistanceMetric, Euclidean, Manhattan
from ..utils.validation import check_array, check_is_fitted, check_X_y

logger = logging.getLogger(__name__)


class KNNClassifier(BaseEstimator, ClassifierMixin):
    """K-nearest neighbors classifier."""

    def __init__(self, k: int = 3, dist: str | type[DistanceMetric] = "euclidean"):
        metric_map: dict[str, type[DistanceMetric]] = {
            "euclidean": Euclidean,
            "manhattan": Manhattan,
            "cosine": Cosine,
        }

        if isinstance(dist, str):
            if dist.lower() not in metric_map:
                raise ValueError(f"Unknown distance metric '{dist}'")
            self.metric_cls = metric_map[dist.lower()]
        else:
            self.metric_cls = dist  # already a DistanceMetric subclass

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k

        self._metric: DistanceMetric | None = None
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None) -> KNNClassifier:
        if y is None:
            raise ValueError("y cannot be None for KNNClassifier")

        X, y = check_X_y(X, y)

        if self.k > X.shape[0]:
            raise ValueError(f"k={self.k} is greater than n_samples={X.shape[0]}")

        self.X_train = X
        self.classes_, self.y_train = np.unique(y, return_inverse=True)
        self._metric = self.metric_cls()

        logger.info("Fitted KNN with %d samples and %d features", X.shape[0], X.shape[1])
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        check_is_fitted(self, ["X_train", "y_train", "_metric"])
        X = check_array(X)

        if (
            self._metric is None
            or self.X_train is None
            or self.y_train is None
            or self.classes_ is None
        ):
            raise RuntimeError("Model is not fitted")

        dists: NDArray[np.float64] = self._metric.compute(self.X_train, X)

        y_pred: list[int] = []
        for i in range(dists.shape[0]):
            idx: NDArray[np.int64] = np.argsort(dists[i])[: self.k]
            nearest: NDArray[np.int64] = self.y_train[idx]
            counts: NDArray[np.int64] = np.bincount(nearest, minlength=len(self.classes_))
            y_pred.append(int(np.argmax(counts)))

        classes_arr: NDArray[np.int64] = np.asarray(self.classes_, dtype=np.int64)
        return classes_arr[np.array(y_pred, dtype=np.int64)]

    def score(self, X: NDArray, y: NDArray | None) -> float:
        """Return the mean accuracy on the given test data and labels."""
        if y is None:
            raise ValueError("y cannot be None for KNNClassifier")
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
