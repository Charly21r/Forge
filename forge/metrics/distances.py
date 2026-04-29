from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class DistanceMetric(ABC):
    """Abstract base class for distance metrics."""

    @abstractmethod
    def compute(
        self, X_train: NDArray[np.float64], X_test: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...


class Euclidean(DistanceMetric):
    def compute(
        self, X_train: NDArray[np.float64], X_test: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        X_train = np.atleast_2d(X_train)
        X_test = np.atleast_2d(X_test)
        diff: NDArray[np.float64] = X_test[:, None, :] - X_train[None, :, :]
        dist: NDArray[np.float64] = np.sqrt(np.sum(diff**2, axis=-1))
        return dist


class Manhattan(DistanceMetric):
    def compute(
        self, X_train: NDArray[np.float64], X_test: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        diff: NDArray[np.float64] = np.abs(X_test[:, None, :] - X_train[None, :, :])
        dist: NDArray[np.float64] = np.sum(diff, axis=-1)
        return dist


class Cosine(DistanceMetric):
    def compute(
        self, X_train: NDArray[np.float64], X_test: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        eps: float = 1e-12
        X_train_norm: NDArray[np.float64] = X_train / (
            np.linalg.norm(X_train, axis=1, keepdims=True) + eps
        )
        X_test_norm: NDArray[np.float64] = X_test / (
            np.linalg.norm(X_test, axis=1, keepdims=True) + eps
        )
        sim: NDArray[np.float64] = X_test_norm @ X_train_norm.T
        return 1 - sim
