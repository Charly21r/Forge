from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..base.base_estimator import BaseEstimator
from ..base.mixins import RegressorMixin
from ..utils.validation import check_array, check_is_fitted, check_X_y


class LinearRegression(BaseEstimator, RegressorMixin):
    """Linear Regression with optional L2 (ridge) regularization."""

    def __init__(self, fit_intercept: bool = True, alpha: float = 0.0) -> None:
        self.fit_intercept = fit_intercept
        self.alpha = alpha

    def fit(self, X: NDArray[Any], y: NDArray[Any] | None) -> "LinearRegression":
        if y is None:
            raise ValueError("y cannot be None for LinearRegression")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")

        X, y = check_X_y(X, y)

        if self.alpha == 0.0:
            if self.fit_intercept:
                X_b = np.c_[np.ones(X.shape[0]), X]
            else:
                X_b = X
            w, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
            if self.fit_intercept:
                self.intercept_: float = float(w[0])
                self.coef_: NDArray[Any] = w[1:]
            else:
                self.intercept_ = 0.0
                self.coef_ = w
            return self

        # Ridge: don't penalize the intercept — center first, recover after.
        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = float(y.mean())
            Xc = X - X_mean
            yc = y - y_mean
        else:
            Xc = X
            yc = y

        n_features = Xc.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(n_features)
        w = np.linalg.solve(A, Xc.T @ yc)

        self.coef_ = w
        if self.fit_intercept:
            self.intercept_ = float(y_mean - X_mean @ w)
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        check_is_fitted(self, ["coef_", "intercept_"])
        X = check_array(X)
        return np.asarray(X @ self.coef_ + self.intercept_)
