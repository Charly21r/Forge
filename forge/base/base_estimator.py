"""Base classes for all estimators in MLFS."""

import inspect
from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray
from typing_extensions import Self


class BaseEstimator(ABC):
    """Abstract base class for all estimators in MLFS.

    This base class provides common functionality for all estimators,
    including parameter management and scoring.
    """

    @abstractmethod
    def fit(self, X: NDArray, y: NDArray | None) -> Self:
        """Fit the estimator to training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : BaseEstimator
            Fitted estimator instance.
        """
        ...

    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        """Make predictions on new data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to make predictions on.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels.
        """
        ...

    def score(self, X: NDArray, y: NDArray | None) -> float:
        """Default score method.

        Should be implemented by subclasses or mixins.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a default score method."
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return parameters for nested estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        signature = inspect.signature(type(self).__init__)
        params: dict[str, Any] = {}

        for name, _param in signature.parameters.items():
            if name == "self":
                continue

            if not hasattr(self, name):
                raise AttributeError(
                    f"Estimator {self.__class__.__name__} does not store "
                    f"parameter '{name}' as an attribute."
                )

            value = getattr(self, name)
            params[name] = value

            # Handle nested estimators
            if deep and hasattr(value, "get_params"):
                nested_params = value.get_params(deep=True)
                for sub_name, sub_value in nested_params.items():
                    params[f"{name}__{sub_name}"] = sub_value

        return params

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of this estimator.

        Method chaining is supported for convenience.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : BaseEstimator
            Estimator instance.

        Raises
        ------
        ValueError
            If an invalid parameter is passed.
        """
        if not params:
            return self

        valid_params = self.get_params(deep=False)

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{key}' for estimator "
                    f"{self.__class__.__name__}. "
                    f"Valid parameters are: {sorted(valid_params.keys())}"
                )
            setattr(self, key, value)

        return self

    def __repr__(self) -> str:
        """Return a string representation of the estimator.

        Shows the class name with all parameters and their values.

        Returns
        -------
        repr : str
            String representation of the estimator.
        """
        class_name = self.__class__.__name__
        params = self.get_params(deep=False)

        if not params:
            return f"{class_name}()"

        param_str = ", ".join(f"{key}={value!r}" for key, value in params.items())
        return f"{class_name}({param_str})"
