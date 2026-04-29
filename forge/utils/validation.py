"""Input validation utilities following scikit-learn patterns."""

import numpy as np
from numpy.typing import NDArray

from ..exceptions import NotFittedError


def check_array(
    X: NDArray,
    ensure_2d: bool = True,
    allow_nd: bool = False,
    dtype: type | None = None,
    ensure_min_samples: int = 1,
    ensure_min_features: int = 1,
) -> NDArray:
    """Validate and convert input to ndarray.

    Parameters
    ----------
    X : array-like
        Input to be validated.
    ensure_2d : bool, default=True
        Whether to raise error if X is not 2D.
    allow_nd : bool, default=False
        Whether to allow N-dimensional arrays.
    dtype : type, optional
        Required dtype. If None, dtype is not checked.
    ensure_min_samples : int, default=1
        Minimum number of samples required.
    ensure_min_features : int, default=1
        Minimum number of features required.

    Returns
    -------
    X_converted : ndarray
        Validated array.

    Raises
    ------
    ValueError
        If validation fails.
    """
    X = np.asarray(X)

    if not allow_nd and X.ndim != 2 and ensure_2d:
        raise ValueError(f"X must be a 2D array, got shape {X.shape} with {X.ndim} dimensions")

    if X.shape[0] < ensure_min_samples:
        raise ValueError(f"X must have at least {ensure_min_samples} samples, got {X.shape[0]}")

    if X.ndim >= 2 and X.shape[1] < ensure_min_features:
        raise ValueError(f"X must have at least {ensure_min_features} features, got {X.shape[1]}")

    if dtype is not None and X.dtype != dtype:
        try:
            X = X.astype(dtype)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert X to dtype {dtype}: {e}") from e

    return X


def check_X_y(
    X: NDArray,
    y: NDArray,
    ensure_2d: bool = True,
    allow_nd: bool = False,
    dtype: type | None = None,
    ensure_min_samples: int = 1,
    ensure_min_features: int = 1,
) -> tuple[NDArray, NDArray]:
    """Validate X and y arrays.

    Ensures that X and y have compatible dimensions and returns
    validated arrays.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target labels.
    ensure_2d : bool, default=True
        Whether to ensure X is 2D.
    allow_nd : bool, default=False
        Whether to allow N-dimensional arrays.
    dtype : type, optional
        Required dtype for X.
    ensure_min_samples : int, default=1
        Minimum number of samples.
    ensure_min_features : int, default=1
        Minimum number of features.

    Returns
    -------
    X_checked : ndarray
        Validated feature matrix.
    y_checked : ndarray
        Validated target labels.

    Raises
    ------
    ValueError
        If validation fails.
    """
    X = check_array(
        X,
        ensure_2d=ensure_2d,
        allow_nd=allow_nd,
        dtype=dtype,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
    )

    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape} with {y.ndim} dimensions")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
        )

    return X, y


def check_is_fitted(
    estimator: object,
    attributes: list[str],
) -> None:
    """Check if estimator has been fitted.

    Parameters
    ----------
    estimator : object
        Estimator instance.
    attributes : list of str
        Attributes that should be present if fitted.

    Raises
    ------
    NotFittedError
        If estimator is not fitted.
    """
    if not hasattr(estimator, "__dict__"):
        raise TypeError(f"estimator must be an object with __dict__, " f"got {type(estimator)}")

    not_fitted_attrs = [
        attr
        for attr in attributes
        if not hasattr(estimator, attr) or getattr(estimator, attr) is None
    ]

    if not_fitted_attrs:
        raise NotFittedError(
            f"{estimator.__class__.__name__} is not fitted. "
            f"Call 'fit' with appropriate arguments before using this estimator. "
            f"Missing attributes: {', '.join(not_fitted_attrs)}"
        )
