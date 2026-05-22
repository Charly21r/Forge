from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..base.base_estimator import BaseEstimator
from ..base.mixins import RegressorMixin
from ..utils.validation import check_array, check_is_fitted, check_X_y

Solver = Literal["lstsq", "gd"]
_VALID_SOLVERS: tuple[Solver, ...] = ("lstsq", "gd")


class LinearRegression(BaseEstimator, RegressorMixin):
    """Linear Regression with optional L2 (ridge) regularization.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to learn an intercept term.
    alpha : float, default=0.0
        L2 regularization strength. The intercept is not penalized.
    solver : {"lstsq", "gd"}, default="lstsq"
        - ``"lstsq"``: closed form. Uses ``np.linalg.lstsq`` for OLS and
          the regularized normal equations for ridge.
        - ``"gd"``: batch gradient descent on the (regularized) MSE.
    learning_rate : float, default=0.01
        Step size for gradient descent. Ignored unless ``solver="gd"``.
    max_iter : int, default=1000
        Maximum number of gradient descent iterations. Ignored unless
        ``solver="gd"``.
    tol : float, default=1e-6
        Convergence tolerance on the change in loss between iterations.
        Ignored unless ``solver="gd"``.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        alpha: float = 0.0,
        solver: Solver = "lstsq",
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: NDArray[Any], y: NDArray[Any] | None) -> "LinearRegression":
        if y is None:
            raise ValueError("y cannot be None for LinearRegression")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.solver not in _VALID_SOLVERS:
            raise ValueError(
                f"solver must be one of {_VALID_SOLVERS}, got {self.solver!r}"
            )

        X, y = check_X_y(X, y)

        if self.solver == "lstsq":
            w, b = self._fit_lstsq(X, y)
        else:
            w, b = self._fit_gd(X, y)

        self.coef_: NDArray[Any] = w
        self.intercept_: float = b
        return self

    def _fit_lstsq(self, X: NDArray[Any], y: NDArray[Any]) -> tuple[NDArray[Any], float]:
        if self.alpha == 0.0:
            if self.fit_intercept:
                X_b = np.c_[np.ones(X.shape[0]), X]
            else:
                X_b = X
            w, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
            if self.fit_intercept:
                return w[1:], float(w[0])
            return w, 0.0

        # Ridge: don't penalize the intercept — center first, recover after.
        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = float(y.mean())
            Xc = X - X_mean
            yc = y - y_mean
        else:
            Xc, yc = X, y

        A = Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1])
        w = np.linalg.solve(A, Xc.T @ yc)
        b = float(y_mean - X_mean @ w) if self.fit_intercept else 0.0
        return w, b

    def _fit_gd(self, X: NDArray[Any], y: NDArray[Any]) -> tuple[NDArray[Any], float]:
        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        prev_loss = np.inf

        for _ in range(self.max_iter):
            pred = X @ w + b
            error = pred - y
            loss = float(np.mean(error**2)) + self.alpha * float(w @ w) / n_samples

            grad_w = 2.0 * (X.T @ error) / n_samples + 2.0 * self.alpha * w / n_samples
            w -= self.learning_rate * grad_w
            if self.fit_intercept:
                b -= self.learning_rate * 2.0 * float(np.mean(error))

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return w, b

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        check_is_fitted(self, ["coef_", "intercept_"])
        X = check_array(X)
        return np.asarray(X @ self.coef_ + self.intercept_)
