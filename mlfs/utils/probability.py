"""Probability utility functions."""

import numpy as np
from numpy.typing import NDArray


def gaussian_pdf(X: NDArray, mean: NDArray, var: NDArray) -> NDArray:
    """Gaussian probability density function.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    mean : array-like of shape (n_features,)
        Mean of the Gaussian distribution.
    var : array-like of shape (n_features,)
        Variance of the Gaussian distribution.

    Returns
    -------
    pdf : array of shape (n_samples, n_features)
        Probability density values.
    """
    eps = 1e-9  # avoid division by zero
    coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
    exponent = np.exp(-((X - mean) ** 2) / (2 * var + eps))
    return np.array(coeff * exponent)
