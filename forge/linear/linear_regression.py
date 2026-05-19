from ..base.base_estimator import BaseEstimator
from ..base.mixins import RegressorMixin
from ..utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np


class LinearRegression(BaseEstimator, RegressorMixin):
    """Linear Regression"""
    def __init__(self):
        super().__init__()
        self.w = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X_b = np.c_[np.ones(X.shape[0]), X]
        self.w, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        return self

    def predict(self, X):
        check_is_fitted(self, ["w"])
        X = check_array(X)
        X_b = np.c_[np.ones(X.shape[0]), X]
        return X_b @ self.w
    