from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..base.base_estimator import BaseEstimator
from ..base.mixins import RegressorMixin
from ..utils.validation import check_array, check_is_fitted, check_X_y


class LinearRegression(BaseEstimator, RegressorMixin):
    """Linear Regression"""

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept

    def fit(self, X: NDArray[Any], y: NDArray[Any] | None) -> "LinearRegression":
        if y is None:
            raise ValueError("y cannot be None for LinearRegression")

        X, y = check_X_y(X, y)
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

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        check_is_fitted(self, ["coef_", "intercept_"])
        X = check_array(X)
        return np.asarray(X @ self.coef_ + self.intercept_)
