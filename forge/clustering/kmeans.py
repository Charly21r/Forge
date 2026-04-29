from __future__ import annotations

import logging
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ..base.base_estimator import BaseEstimator
from ..base.mixins import ClusterMixin
from ..metrics import DistanceMetric, Euclidean
from ..utils.validation import check_array, check_is_fitted

logger = logging.getLogger(__name__)


class KMeans(BaseEstimator, ClusterMixin):
    """KMeans Algorithm"""

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        init: Literal["kmeans++", "random"] = "kmeans++",
        random_state: int = 42,
    ):
        if n_clusters <= 1:
            raise ValueError(f"n_clusters must be greater than 1, got {n_clusters}")
        self.n_clusters = n_clusters

        if init not in ["random", "kmeans++"]:
            raise ValueError(f"Unsupported init type: {init}")
        self.init = init

        self.max_iter: int = max_iter
        self.tol: float = tol
        self.random_state: int = random_state
        self.dist: DistanceMetric = Euclidean()
        self.centroids_: NDArray[np.float64] | None = None
        self.labels_: NDArray[np.float64] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> KMeans:  # noqa: ARG002
        X = check_array(X)

        if len(X) < self.n_clusters:
            raise ValueError(
                f"Number of samples ({len(X)}) must be >= n_clusters ({self.n_clusters})."
            )

        # Create a random generator with a fixed seed
        rng = np.random.default_rng(seed=self.random_state)

        # Initialize centroids
        if self.init == "kmeans++":
            centroids = np.empty((self.n_clusters, X.shape[1]), dtype=float)
            idx = rng.choice(len(X))
            centroids[0] = X[idx]

            for i in range(1, self.n_clusters):
                # Only used already assigned centroids
                assigned = centroids[:i]

                H = Euclidean().compute(np.array(assigned), X)
                dists = H.min(axis=1)
                probs = dists**2
                total = probs.sum()

                if total == 0:
                    # Edge case: all points identical
                    idx = rng.choice(len(X))
                else:
                    probs /= total
                    idx = rng.choice(len(X), p=probs)

                centroids[i] = X[idx]

        else:
            indices = rng.choice(len(X), size=self.n_clusters, replace=False)
            centroids = X[indices].copy()

        for _ in range(self.max_iter):
            old_centroids = centroids.copy()

            # Compute the distance
            H = self.dist.compute(centroids, X)  # The order of the arguments is important here

            # Assign centroids (labels)
            labels = np.argmin(H, axis=1)

            # Update centroids
            for i in range(self.n_clusters):
                if np.any(labels == i):
                    centroids[i] = X[labels == i].mean(axis=0)
                else:
                    centroids[i] = X[rng.integers(len(X))]  # empty cluster edge case

            # Check for convergence
            shift = np.linalg.norm(centroids - old_centroids)
            if shift < self.tol:
                break

        self.centroids_ = centroids
        self.labels_ = labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["centroids_"])
        centroids = cast(np.ndarray, self.centroids_)

        X = check_array(X)

        H = self.dist.compute(centroids, X)
        labels = np.argmin(H, axis=1)

        return np.array(labels)

    def score(self, X: np.ndarray, y: np.ndarray | None = None) -> float:  # noqa: ARG002
        """Return the negative inertia on the given test data."""
        centroids = cast(np.ndarray, self.centroids_)
        H = self.dist.compute(centroids, X)
        return float(-np.sum(np.min(H, axis=1)))
