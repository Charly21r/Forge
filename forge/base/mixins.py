from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..base.base_estimator import BaseEstimator
from ..metrics.distances import DistanceMetric


class _Predictor(Protocol):
    def predict(self, X: NDArray) -> NDArray: ...


class _ClusterProtocol(Protocol):
    centroids_: NDArray
    dist: DistanceMetric

    def fit(self, X: NDArray, y: NDArray | None = None) -> "BaseEstimator": ...
    def predict(self, X: NDArray) -> NDArray: ...


class ClassifierMixin:
    """Mixin for classification estimators."""

    def score(self: _Predictor, X: NDArray, y: NDArray) -> float:
        """Return accuracy on the given test data."""
        if y is None:
            raise ValueError("y cannot be None for classification score")

        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


class RegressorMixin:
    """Mixin for regression estimators."""

    def score(self: _Predictor, X: NDArray, y: NDArray) -> float:
        """Return R^2 score."""
        if y is None:
            raise ValueError("y cannot be None for regression score")

        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


class ClusterMixin:
    """Mixin for clustering estimators."""

    def fit_predict(self: _ClusterProtocol, X: NDArray, y: NDArray | None = None) -> NDArray:
        """Fit the model and return cluster labels."""
        return self.fit(X, y).predict(X)
