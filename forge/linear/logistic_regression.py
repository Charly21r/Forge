from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..base.base_estimator import BaseEstimator
from ..base.mixins import ClassifierMixin
from ..utils.probability import sigmoid
from ..utils.validation import check_array, check_is_fitted, check_X_y


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Binary Logistic Regression trained with batch gradient descent.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to learn an intercept term.
    learning_rate : float, default=0.1
        Step size for gradient descent.
    max_iter : int, default=1000
        Maximum number of gradient descent iterations.
    tol : float, default=1e-6
        Convergence tolerance on the change in the loss between iterations.
    C : float, default=1.0
        Inverse of L2 regularization strength. Smaller values mean stronger
        regularization. Set to ``np.inf`` to disable regularization.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
        C: float = 1.0,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.C = C

    def fit(self, X: NDArray[Any], y: NDArray[Any] | None) -> "LogisticRegression":
        if y is None:
            raise ValueError("y cannot be None for LogisticRegression")

        X, y = check_X_y(X, y)

        self.classes_: NDArray[Any] = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(
                f"LogisticRegression supports only binary classification, "
                f"got {self.classes_.size} classes"
            )

        y_binary = (y == self.classes_[1]).astype(np.float64)

        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0

        reg = 0.0 if not np.isfinite(self.C) else 1.0 / self.C
        prev_loss = np.inf

        for _ in range(self.max_iter):
            z = X @ w + b
            p = sigmoid(z)

            eps = 1e-15
            p_clipped = np.clip(p, eps, 1 - eps)
            loss = -np.mean(y_binary * np.log(p_clipped) + (1 - y_binary) * np.log(1 - p_clipped))
            loss = loss + 0.5 * reg * float(w @ w) / n_samples

            error = p - y_binary
            grad_w = X.T @ error / n_samples + reg * w / n_samples
            w -= self.learning_rate * grad_w
            if self.fit_intercept:
                grad_b = float(np.mean(error))
                b -= self.learning_rate * grad_b

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        self.coef_: NDArray[Any] = w
        self.intercept_: float = b if self.fit_intercept else 0.0
        return self

    def decision_function(self, X: NDArray[Any]) -> NDArray[Any]:
        check_is_fitted(self, ["coef_", "intercept_", "classes_"])
        X = check_array(X)
        return np.asarray(X @ self.coef_ + self.intercept_)

    def predict_proba(self, X: NDArray[Any]) -> NDArray[Any]:
        scores = self.decision_function(X)
        p1 = sigmoid(scores)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.asarray(self.classes_[idx])
