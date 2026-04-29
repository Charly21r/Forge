from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from ..base.base_estimator import BaseEstimator
from ..base.mixins import ClassifierMixin
from ..utils.probability import gaussian_pdf
from ..utils.validation import check_is_fitted, check_X_y

logger = logging.getLogger(__name__)


class GaussianNB(BaseEstimator, ClassifierMixin):
    """Gaussian Naive Bayes"""

    classes_: NDArray
    means_: NDArray
    vars_: NDArray
    priors_: NDArray

    def __init__(self) -> None:
        pass

    def fit(self, X: NDArray, y: NDArray | None) -> GaussianNB:
        if y is None:
            raise ValueError("y cannot be None for GaussianNB")

        X, y = check_X_y(X, y)

        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx, :] = X_c.mean(axis=0)
            self.vars_[idx, :] = X_c.var(axis=0)
            self.priors_[idx] = X_c.shape[0] / X.shape[0]

        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["classes_", "means_", "vars_", "priors_"])

        probs_list: list[NDArray] = []
        for idx, _ in enumerate(self.classes_):
            prior = self.priors_[idx]
            likelihood = np.prod(gaussian_pdf(X, self.means_[idx], self.vars_[idx]), axis=1)
            probs_list.append(prior * likelihood)
        probs: NDArray = np.array(probs_list).T
        # normalize to get probabilities
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["classes_", "means_", "vars_", "priors_"])

        probs = self.predict_proba(X)
        result: NDArray = self.classes_[np.argmax(probs, axis=1)]
        return result
