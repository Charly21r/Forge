"""Custom exceptions for MLFS."""


class NotFittedError(RuntimeError):
    """Exception raised when a method is called on an unfitted estimator.

    This error indicates that a method requiring the estimator to be fitted
    (such as predict) was called before the estimator was fitted.

    Examples
    --------
    >>> from forge import KNNClassifier
    >>> from forge.exceptions import NotFittedError
    >>> knn = KNNClassifier()
    >>> try:
    ...     knn.predict([[1, 2]])
    ... except NotFittedError:
    ...     print("Must call fit() first!")
    Must call fit() first!
    """

    pass


class InvalidMetricError(ValueError):
    """Exception raised when an invalid distance metric is provided.

    This error indicates that an unsupported distance metric was passed
    to an estimator that requires a valid metric.

    Examples
    --------
    >>> from forge import KNNClassifier
    >>> from forge.exceptions import InvalidMetricError
    >>> try:
    ...     KNNClassifier(metric="invalid")
    ... except InvalidMetricError:
    ...     print("Unknown metric!")
    Unknown metric!
    """

    pass
